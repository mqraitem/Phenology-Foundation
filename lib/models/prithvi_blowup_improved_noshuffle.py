import torch
import torch.nn.functional as F
from torch import nn

from lib.models.prithvi import PrithviBackbone


class ResConvBlock(nn.Module):
    """Conv2d + BN + GELU with residual skip connection.
    Skip uses 1x1 conv to match channels when in_ch != out_ch.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)) + self.skip(x))


class TemporalSEBlock(nn.Module):
    """Squeeze-and-Excitation style temporal attention.

    Given (B, T*c_per_t, H, W), globally pools spatial dims, then uses a
    small FC bottleneck to produce per-channel (i.e. per-timestep-feature)
    attention weights. This lets the model learn which timesteps matter
    most before the temporal conv fusion.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        w = x.mean(dim=(2, 3))            # (B, C)  global avg pool
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * w


class PrithviSegBlowupImprovedNoShuffle(nn.Module):
    """Improved Prithvi blowup segmentation head (no progressive PixelShuffle).

    Uses the original one-shot PixelShuffle(16) from the base blowup, plus:
    1. Residual connections in the temporal conv stack
    2. (Optional) Separate prediction heads per phenology stage
    3. (Optional) SE-style temporal attention before temporal conv fusion
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
                 separate_heads: bool = False,
                 temporal_attention: bool = False,
                 feed_timeloc: bool = False):
        super().__init__()

        if feed_timeloc:
            prithvi_params["coords_encoding"] = ["time", "location"]

        self.backbone = PrithviBackbone(prithvi_params, prithvi_ckpt_path, reshape)

        self.embed_dim = prithvi_params["embed_dim"]
        self.n_frames = prithvi_params["num_frames"]
        self.n_classes = n_classes
        self.c_per_t = c_per_t
        self.separate_heads = separate_heads
        self.temporal_attention = temporal_attention

        patch_h = prithvi_params["patch_size"][1]  # 16

        if model_size == "300m":
            # Shared per-timestep projection: embed_dim -> c_per_t * 16 * 16
            pixel_out = c_per_t * patch_h * patch_h
            self.token_proj = nn.Sequential(
                nn.Conv2d(self.embed_dim, pixel_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(pixel_out),
                nn.GELU(),
            )

            self.shuffle = nn.PixelShuffle(patch_h)

            # --- Temporal attention (optional) ---
            temporal_in = self.n_frames * c_per_t
            self.temporal_se = TemporalSEBlock(temporal_in) if temporal_attention else None

            # --- Temporal conv stack with residual connections ---
            self.input_proj = ResConvBlock(temporal_in, hidden_dim) if temporal_in != hidden_dim else None
            res_layers = []
            for _ in range(n_temporal_layers):
                res_layers.append(ResConvBlock(hidden_dim, hidden_dim))
            self.temporal_conv = nn.Sequential(*res_layers) if res_layers else nn.Identity()

            # --- Output head: shared or separate per stage ---
            if separate_heads:
                self.output_heads = nn.ModuleList([
                    nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
                    for _ in range(n_classes)
                ])
            else:
                self.output_conv = nn.Conv2d(hidden_dim, n_classes, kernel_size=3, padding=1)

            print(f"BlowupImprovedNoShuffle head: ({self.embed_dim}, 21, 21) x {self.n_frames} timesteps")
            print(f"  PixelShuffle({patch_h}) -> {c_per_t}ch/t")
            print(f"  Temporal attention: {temporal_attention}")
            print(f"  Temporal conv: {n_temporal_layers}x ResConvBlock({hidden_dim})")
            print(f"  Output: {'separate heads x' + str(n_classes) if separate_heads else 'shared -> ' + str(n_classes)}")
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

        # --- Temporal attention (optional) ---
        if self.temporal_se is not None:
            x = self.temporal_se(x)

        # --- Residual temporal conv stack ---
        if self.input_proj is not None:
            x = self.input_proj(x)              # (B, hidden_dim, 336, 336)
        x = self.temporal_conv(x)               # (B, hidden_dim, 336, 336)

        # --- Output ---
        if self.separate_heads:
            x = torch.cat([head(x) for head in self.output_heads], dim=1)  # (B, n_classes, H, W)
        else:
            x = self.output_conv(x)

        x = torch.sigmoid(x)
        return x

    def forward_features(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        return x
