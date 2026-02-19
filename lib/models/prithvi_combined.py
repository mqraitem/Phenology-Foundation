import torch
import torch.nn as nn

from lib.models.prithvi import PrithviBackbone
from lib.models.lsp_temporal_pixels import Conv1DTemporalBlock, LearnablePositionalEncoding


class PrithviCombined(nn.Module):
    """Combined model: Conv3 pixel temporal path + Prithvi spatial residual.

    Conv3 path (primary signal):
        raw HLS (B, C, T, H, W)
        -> per-pixel: band projection (C -> emb_dim)
        -> positional encoding
        -> N conv1d(k=3) blocks with residual + LayerNorm
        -> mean pool over T
        -> (B, emb_dim, H, W)

    Prithvi path (spatial context residual):
        normalized HLS -> Prithvi backbone -> PixelShuffle
        -> 3x3 conv -> project to emb_dim
        -> (B, emb_dim, H, W)

    Combination:
        conv3_features + α * prithvi_features   (α = learnable scalar, init 0.1)
        -> 1x1 conv -> n_classes -> sigmoid
    """

    def __init__(self,
                 prithvi_params: dict,
                 prithvi_ckpt_path: str = None,
                 reshape: bool = True,
                 n_classes: int = 4,
                 model_size: str = "300m",
                 # Conv3 path params
                 emb_dim: int = 128,
                 n_layers: int = 3,
                 dropout: float = 0.1,
                 # Prithvi path params
                 c_per_t: int = 4,
                 feed_timeloc: bool = False):
        super().__init__()

        if feed_timeloc:
            prithvi_params["coords_encoding"] = ["time", "location"]

        self.n_frames = prithvi_params["num_frames"]
        self.n_bands = prithvi_params["in_chans"]
        self.embed_dim = prithvi_params["embed_dim"]
        self.emb_dim = emb_dim
        self.n_classes = n_classes

        patch_h = prithvi_params["patch_size"][1]  # 16

        if model_size != "300m":
            raise ValueError(f"model_size {model_size} not supported")

        # ==================== Conv3 pixel path ====================
        self.input_proj = nn.Linear(self.n_bands, emb_dim, bias=False)
        self.pos_enc = LearnablePositionalEncoding(emb_dim, max_len=self.n_frames)

        self.conv3_blocks = nn.ModuleList()
        self.conv3_norms = nn.ModuleList()
        for _ in range(n_layers):
            self.conv3_blocks.append(Conv1DTemporalBlock(emb_dim, kernel_size=3))
            self.conv3_norms.append(nn.LayerNorm(emb_dim))

        self.conv3_dropout = nn.Dropout(dropout)

        # ==================== Prithvi path ====================
        self.backbone = PrithviBackbone(prithvi_params, prithvi_ckpt_path, reshape)

        # Per-timestep PixelShuffle (same as blowup)
        pixel_out = c_per_t * patch_h * patch_h
        self.token_proj = nn.Sequential(
            nn.Conv2d(self.embed_dim, pixel_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(pixel_out),
            nn.GELU(),
        )
        self.shuffle = nn.PixelShuffle(patch_h)

        # Light 3x3 conv + project to emb_dim
        prithvi_channels = self.n_frames * c_per_t
        self.prithvi_head = nn.Sequential(
            nn.Conv2d(prithvi_channels, emb_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(emb_dim),
            nn.GELU(),
        )

        # ==================== Combination ====================
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.output_conv = nn.Conv2d(emb_dim, n_classes, kernel_size=1)

        # Print summary
        conv3_params = sum(p.numel() for n, p in self.named_parameters() if not n.startswith('backbone'))
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"PrithviCombined: conv3 path (emb_dim={emb_dim}, {n_layers} layers) "
              f"+ Prithvi residual (c_per_t={c_per_t})")
        print(f"  Conv3 + head params: {conv3_params:,}")
        print(f"  Backbone params: {backbone_params:,}")
        print(f"  α init: {self.alpha.item():.2f}")

    def _conv3_path(self, raw_pixels, H_full, W_full, chunk_size=5000):
        """Process raw pixels through conv3 temporal blocks.

        Args:
            raw_pixels: (B, C, T, H, W) normalized input
            H_full, W_full: target spatial dims (from Prithvi path)
            chunk_size: pixels per chunk for memory efficiency

        Returns: (B, emb_dim, H_full, W_full)
        """
        B = raw_pixels.size(0)
        # (B, C, T, H, W) -> (B, H, W, T, C) -> (B*H*W, T, C)
        x = raw_pixels[:, :, :, :H_full, :W_full]
        x = x.permute(0, 3, 4, 2, 1)
        x = x.reshape(B * H_full * W_full, self.n_frames, self.n_bands)

        # Band projection + positional encoding
        x = self.input_proj(x)       # (N, T, emb_dim)
        x = self.pos_enc(x)

        # Process in chunks for memory efficiency
        N = x.size(0)
        outputs = []
        for i in range(0, N, chunk_size):
            chunk = x[i:i + chunk_size]
            for block, ln in zip(self.conv3_blocks, self.conv3_norms):
                chunk = ln(chunk + block(chunk))
            chunk = chunk.mean(dim=1)            # (chunk, emb_dim)
            chunk = self.conv3_dropout(chunk)
            outputs.append(chunk)

        x = torch.cat(outputs, dim=0)           # (B*H*W, emb_dim)
        x = x.view(B, H_full, W_full, self.emb_dim)
        x = x.permute(0, 3, 1, 2)               # (B, emb_dim, H_full, W_full)
        return x

    def _prithvi_path(self, x, batch_size):
        """Process input through Prithvi backbone + PixelShuffle + light conv.

        Returns: (B, emb_dim, H_full, W_full)
        """
        x = self.backbone(x)                    # (B, embed_dim*T, 21, 21)
        _, _, H, W = x.shape

        # Separate time steps
        x = x.view(batch_size, self.embed_dim, self.n_frames, H, W)
        x = x.permute(0, 2, 1, 3, 4)                                     # (B, T, E, H, W)
        x = x.reshape(batch_size * self.n_frames, self.embed_dim, H, W)  # (B*T, E, H, W)

        # Per-timestep projection + PixelShuffle
        x = self.token_proj(x)                  # (B*T, c_per_t*16*16, H, W)
        x = self.shuffle(x)                     # (B*T, c_per_t, H*16, W*16)

        # Stack time steps
        _, C, H_full, W_full = x.shape
        x = x.view(batch_size, self.n_frames * C, H_full, W_full)

        # Light conv to emb_dim
        x = self.prithvi_head(x)                # (B, emb_dim, H_full, W_full)
        return x, H_full, W_full

    def forward(self, x):
        # Extract raw pixels before backbone
        if isinstance(x, dict):
            batch_size = x["chip"].size(0)
            raw_pixels = x["chip"].cuda()
            x_backbone = {k: v.cuda() for k, v in x.items()}
        else:
            batch_size = x.size(0)
            raw_pixels = x.cuda()
            x_backbone = x.cuda()

        # Prithvi path (get spatial dims from PixelShuffle output)
        prithvi_features, H_full, W_full = self._prithvi_path(x_backbone, batch_size)

        # Conv3 path
        conv3_features = self._conv3_path(raw_pixels, H_full, W_full)

        # Combine: conv3 primary + Prithvi residual
        combined = conv3_features + self.alpha * prithvi_features

        # Output
        out = self.output_conv(combined)         # (B, n_classes, H_full, W_full)
        out = torch.sigmoid(out)
        return out

    def forward_features(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        return x
