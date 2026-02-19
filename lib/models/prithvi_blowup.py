import torch
import torch.nn.functional as F
from torch import nn

from lib.models.prithvi import PrithviBackbone


class PrithviSegBlowup(nn.Module):
    """Prithvi segmentation with per-timestep PixelShuffle + temporal convolutions.

    Instead of expensive progressive upsampling (~852M head params), this variant:
    1. Separates the backbone output back into 12 time steps
    2. Per time step: a shared 1x1 conv maps each token's embed_dim to
       c_per_t * 16 * 16, then PixelShuffle(16) blows it up to full resolution
    3. The 12 time steps (each with c_per_t channels) are stacked into a
       (B, 12*c_per_t, 336, 336) tensor
    4. A few Conv2d layers combine temporal information at full resolution
       to produce the final (B, n_classes, 336, 336) prediction

    Head params: ~1-2M (vs ~852M original).  Total model: ~301-302M.
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
                 feed_timeloc: bool = False):
        super().__init__()

        if feed_timeloc:
            prithvi_params["coords_encoding"] = ["time", "location"]

        self.backbone = PrithviBackbone(prithvi_params, prithvi_ckpt_path, reshape)

        self.embed_dim = prithvi_params["embed_dim"]
        self.n_frames = prithvi_params["num_frames"]
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

            # Temporal convolutions at full resolution
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
            layers.append(nn.Conv2d(in_ch, n_classes, kernel_size=3, padding=1))
            self.temporal_conv = nn.Sequential(*layers)

            print(f"Blowup head: ({self.embed_dim}, 21, 21) x {self.n_frames} timesteps "
                  f"-> proj {pixel_out} -> PixelShuffle({patch_h}) -> "
                  f"({temporal_in}, 336, 336) -> {n_temporal_layers}x conv({hidden_dim}) -> {n_classes}")
        else:
            raise ValueError(f"model_size {model_size} not supported")

    def forward(self, x):
        if isinstance(x, dict):
            batch_size = x["chip"].size(0)
            x = {k: v.cuda() for k, v in x.items()}
        else:
            batch_size = x.size(0)
            x = x.cuda()

        x = self.backbone(x)                   # (B, embed_dim*T, 21, 21)
        _, _, H, W = x.shape

        # Separate time steps: channels are [embed0_t0, embed0_t1, ..., embed0_t11, embed1_t0, ...]
        x = x.view(batch_size, self.embed_dim, self.n_frames, H, W)      # (B, E, T, H, W)
        # print(x.shape)
        x = x.permute(0, 2, 1, 3, 4)                                     # (B, T, E, H, W)
        # print(x.shape)
        x = x.reshape(batch_size * self.n_frames, self.embed_dim, H, W)  # (B*T, E, H, W)
        # print(x.shape)

        # Per-timestep projection + PixelShuffle
        x = self.token_proj(x)                  # (B*T, c_per_t*16*16, H, W)
        # print(x.shape)
        x = self.shuffle(x)                     # (B*T, c_per_t, H*16, W*16)
        # print(x.shape)

        # Stack time steps into channel dimension
        _, C, H_full, W_full = x.shape
        x = x.view(batch_size, self.n_frames * C, H_full, W_full)  # (B, T*c_per_t, 336, 336)
        # print(x.shape)

        # Temporal convolutions at full resolution
        x = self.temporal_conv(x)               # (B, n_classes, 336, 336)
        # print(x.shape)
        # quit()
        x = torch.sigmoid(x)

        return x

    def forward_features(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        return x
