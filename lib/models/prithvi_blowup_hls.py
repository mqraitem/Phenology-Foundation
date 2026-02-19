import torch
import torch.nn.functional as F
from torch import nn

from lib.models.prithvi import PrithviBackbone


class PrithviSegBlowupHLS(nn.Module):
    """Prithvi segmentation with per-timestep PixelShuffle + raw HLS concatenation.

    Same as PrithviSegBlowup but instead of directly producing the output from
    temporal convolutions, this variant:
    1. Runs the blowup pathway (backbone -> per-timestep PixelShuffle -> temporal convs)
       to produce spatial features at full resolution
    2. Reshapes the raw HLS input from (B, C, T, H, W) to (B, C*T, H, W)
    3. Concatenates the temporal conv features with the raw HLS channels
    4. A stack of 1x1 convolutions maps the concatenated features to the final output

    This preserves fine-grained per-pixel temporal signal from the original
    satellite bands alongside the Prithvi-derived features.
    """

    def __init__(self,
                 prithvi_params: dict,
                 prithvi_ckpt_path: str = None,
                 reshape: bool = True,
                 n_classes: int = 1,
                 model_size: str = "300m",
                 c_per_t: int = 4,
                 hidden_dim: int = 64,
                 n_temporal_layers: int = 1,
                 n_final_layers: int = 1,
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

        patch_h = prithvi_params["patch_size"][1]

        if model_size == "300m":
            # Shared per-timestep projection: embed_dim -> c_per_t * 16 * 16
            pixel_out = c_per_t * patch_h * patch_h
            self.token_proj = nn.Sequential(
                nn.Conv2d(self.embed_dim, pixel_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(pixel_out),
                nn.GELU(),
            )

            self.shuffle = nn.PixelShuffle(patch_h)

            # Temporal convolutions at full resolution (no final output layer)
            temporal_in = self.n_frames * c_per_t  # e.g. 12 * 4 = 48
            layers = []
            in_ch = temporal_in
            for _ in range(n_temporal_layers):
                layers.extend([
                    nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.GELU(),
                ])
                in_ch = hidden_dim
            self.temporal_conv = nn.Sequential(*layers)

            # Final 1x1 conv stack: temporal features + raw HLS channels -> output
            hls_channels = self.n_bands * self.n_frames  # 6 * 12 = 72
            final_layers = []
            final_in = in_ch + hls_channels
            for _ in range(n_final_layers):
                final_layers.extend([
                    nn.Conv2d(final_in, hidden_dim, kernel_size=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.GELU(),
                ])
                final_in = hidden_dim
            final_layers.append(nn.Conv2d(final_in, n_classes, kernel_size=1))
            self.final_conv = nn.Sequential(*final_layers)

            print(f"BlowupHLS head: ({self.embed_dim}, 21, 21) x {self.n_frames} timesteps "
                  f"-> proj {pixel_out} -> PixelShuffle({patch_h}) -> "
                  f"({temporal_in}, 336, 336) -> {n_temporal_layers}x conv({hidden_dim}) -> "
                  f"cat raw HLS ({hls_channels}ch) -> {n_final_layers}x 1x1 conv({hidden_dim}) -> {n_classes}")
        else:
            raise ValueError(f"model_size {model_size} not supported")

    def forward(self, x):
        # Extract raw pixels before backbone processing
        if isinstance(x, dict):
            batch_size = x["chip"].size(0)
            raw_pixels = x["chip"].cuda()  # (B, C, T, H, W)
            x = {k: v.cuda() for k, v in x.items()}
        else:
            batch_size = x.size(0)
            raw_pixels = x.cuda()  # (B, C, T, H, W)
            x = x.cuda()

        # --- Blowup pathway ---
        x = self.backbone(x)                   # (B, embed_dim*T, 21, 21)
        _, _, H, W = x.shape

        # Separate time steps
        x = x.view(batch_size, self.embed_dim, self.n_frames, H, W)
        x = x.permute(0, 2, 1, 3, 4)                                     # (B, T, E, H, W)
        x = x.reshape(batch_size * self.n_frames, self.embed_dim, H, W)  # (B*T, E, H, W)

        # Per-timestep projection + PixelShuffle
        x = self.token_proj(x)                  # (B*T, c_per_t*16*16, H, W)
        x = self.shuffle(x)                     # (B*T, c_per_t, H*16, W*16)

        # Stack time steps into channel dimension
        _, C, H_full, W_full = x.shape
        x = x.view(batch_size, self.n_frames * C, H_full, W_full)  # (B, T*c_per_t, 336, 336)

        # Temporal convolutions at full resolution
        x = self.temporal_conv(x)               # (B, hidden_dim, 336, 336)

        # --- Concatenate raw HLS ---
        # raw_pixels: (B, C, T, H, W) -> (B, C*T, H, W)
        raw = raw_pixels.reshape(batch_size, self.n_bands * self.n_frames, raw_pixels.shape[-2], raw_pixels.shape[-1])
        raw = raw[:, :, :H_full, :W_full]       # match spatial dims

        x = torch.cat([x, raw], dim=1)          # (B, hidden_dim + C*T, H, W)

        # Final 1x1 convolution stack
        x = self.final_conv(x)                  # (B, n_classes, H, W)
        x = torch.sigmoid(x)

        return x

    def forward_features(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        return x
