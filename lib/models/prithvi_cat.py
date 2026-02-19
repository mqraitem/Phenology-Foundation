import torch
import torch.nn as nn
import math

from lib.models.prithvi import PrithviBackbone


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PrithviSegCat(nn.Module):
    """Dual-stream model: Prithvi spatial context + raw pixel temporal signal.

    Stream A (Prithvi context):
        image -> Prithvi backbone -> separate time steps -> shared 1x1 proj
        -> PixelShuffle(16) -> (B, T, c_per_t, H, W)

    Stream B (Raw pixels):
        image -> permute -> (B, T, n_bands, H, W)
        Fine-grained per-pixel temporal signal preserved at full resolution.

    Concatenation:
        Per pixel per timestep: (c_per_t + n_bands) features.

    Predictor:
        Per-pixel temporal transformer processes the 12-step sequence
        and outputs n_classes predictions per pixel.

    This solves the patch-size granularity problem: Prithvi's temporal attention
    at 16x16 patch level provides spatial context, while the raw pixel bypass
    preserves fine-grained temporal dynamics that vary within a patch.
    """

    def __init__(self,
                 prithvi_params: dict,
                 prithvi_ckpt_path: str = None,
                 reshape: bool = True,
                 n_classes: int = 1,
                 model_size: str = "300m",
                 c_per_t: int = 4,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 chunk_size: int = 5000,
                 feed_timeloc: bool = False):
        super().__init__()

        if feed_timeloc:
            prithvi_params["coords_encoding"] = ["time", "location"]

        self.backbone = PrithviBackbone(prithvi_params, prithvi_ckpt_path, reshape)

        self.embed_dim = prithvi_params["embed_dim"]
        self.n_frames = prithvi_params["num_frames"]
        self.n_bands = prithvi_params["in_chans"]
        self.n_classes = n_classes
        self.c_per_t = c_per_t
        self.chunk_size = chunk_size

        patch_h = prithvi_params["patch_size"][1]

        if model_size == "300m":
            # Stream A: Prithvi context via PixelShuffle
            pixel_out = c_per_t * patch_h * patch_h
            self.token_proj = nn.Sequential(
                nn.Conv2d(self.embed_dim, pixel_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(pixel_out),
                nn.GELU(),
            )
            self.shuffle = nn.PixelShuffle(patch_h)

            # Per-pixel temporal transformer
            # Input: c_per_t (Prithvi context) + n_bands (raw pixels) per timestep
            token_dim = c_per_t + self.n_bands
            self.input_proj = nn.Linear(token_dim, d_model, bias=False)
            self.pos_encoder = PositionalEncoding(d_model, max_len=self.n_frames)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation='relu'
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.temporal_dropout = nn.Dropout(dropout)
            self.output_proj = nn.Linear(d_model, n_classes)

            print(f"Cat head: Prithvi({self.embed_dim}) -> PixelShuffle -> {c_per_t}ch/t "
                  f"+ raw {self.n_bands} bands/t = {token_dim}/t -> "
                  f"transformer(d={d_model}, heads={nhead}, layers={num_layers}) -> {n_classes}")
        else:
            raise ValueError(f"model_size {model_size} not supported")

    def _extract_chip(self, x):
        """Extract the raw image tensor from input (handles dict or tensor)."""
        if isinstance(x, dict):
            return x["chip"]
        return x

    def forward(self, x):
        if isinstance(x, dict):
            batch_size = x["chip"].size(0)
            raw_pixels = x["chip"].cuda()       # (B, C, T, H, W)
            x_backbone = {k: v.cuda() for k, v in x.items()}
        else:
            batch_size = x.size(0)
            raw_pixels = x.cuda()               # (B, C, T, H, W)
            x_backbone = x.cuda()

        # --- Stream A: Prithvi context ---
        ctx = self.backbone(x_backbone)          # (B, embed_dim*T, 21, 21)
        _, _, H, W = ctx.shape

        # Separate time steps
        ctx = ctx.view(batch_size, self.embed_dim, self.n_frames, H, W)
        ctx = ctx.permute(0, 2, 1, 3, 4)                                       # (B, T, E, H, W)
        ctx = ctx.reshape(batch_size * self.n_frames, self.embed_dim, H, W)     # (B*T, E, H, W)

        # Per-timestep projection + PixelShuffle
        ctx = self.token_proj(ctx)               # (B*T, c_per_t*16*16, H, W)
        ctx = self.shuffle(ctx)                  # (B*T, c_per_t, H*16, W*16)

        _, C_ctx, H_full, W_full = ctx.shape
        ctx = ctx.view(batch_size, self.n_frames, C_ctx, H_full, W_full)        # (B, T, c_per_t, H_full, W_full)

        # --- Stream B: Raw pixels ---
        raw = raw_pixels.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)
        raw = raw[:, :, :, :H_full, :W_full]      # match spatial dims

        # --- Concatenate per pixel per timestep ---
        combined = torch.cat([ctx, raw], dim=2)   # (B, T, c_per_t + n_bands, H_full, W_full)

        # Rearrange for per-pixel temporal processing
        combined = combined.permute(0, 3, 4, 1, 2)  # (B, H, W, T, features)
        N_pixels = batch_size * H_full * W_full
        token_dim = C_ctx + self.n_bands
        combined = combined.reshape(N_pixels, self.n_frames, token_dim)  # (N, T, features)

        # Per-pixel temporal transformer with chunking
        combined = self.input_proj(combined)       # (N, T, d_model)
        combined = self.pos_encoder(combined)

        outputs = []
        for i in range(0, N_pixels, self.chunk_size):
            chunk = combined[i:i + self.chunk_size]
            chunk = self.transformer_encoder(chunk)    # (chunk, T, d_model)
            chunk = chunk.mean(dim=1)                  # (chunk, d_model)
            chunk = self.temporal_dropout(chunk)
            chunk = self.output_proj(chunk)            # (chunk, n_classes)
            outputs.append(chunk)

        out = torch.cat(outputs, dim=0)            # (N, n_classes)
        out = out.view(batch_size, H_full, W_full, self.n_classes)
        out = out.permute(0, 3, 1, 2)             # (B, n_classes, H_full, W_full)
        out = torch.sigmoid(out)
        return out

    def forward_features(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        return x
