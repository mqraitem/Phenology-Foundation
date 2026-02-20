import torch
import torch.nn as nn

from lib.models.prithvi import PrithviBackbone


class PrithviReshape3D(nn.Module):
    """Reshape backbone output to (B, embed_dim, T, H_patches, W_patches)
    keeping the temporal dimension separate from channels."""

    def __init__(self, patch_size, input_size, num_frames):
        super().__init__()
        self.patch_size = patch_size
        self.input_size = input_size
        self.num_frames = num_frames
        self.spatial_size = int(self.input_size / self.patch_size[-1])

    def forward(self, latent):
        # latent: (B, 1 + T*H*W, embed_dim) from ViT
        latent = latent[:, 1:, :]  # remove CLS token: (B, T*H*W, D)
        B, N, D = latent.shape
        H = W = self.spatial_size
        T = self.num_frames

        # Reshape to (B, T, H, W, D) then permute to (B, D, T, H, W)
        latent = latent.reshape(B, T, H, W, D)
        latent = latent.permute(0, 4, 1, 2, 3)  # (B, D, T, H, W)
        return latent


class Conv3DTemporalSpatialBlock(nn.Module):
    """Mirrors the original Upscaler pattern: ConvTranspose first, then Conv.

    ConvTranspose3d(in_ch, out_ch, k=(1,2,2), s=(1,2,2)):
      - Doubles H, W and reduces channels. T unchanged.
    Conv3d(out_ch, out_ch, k=(3,3,3), padding=(1,1,1)):
      - Temporal: k=3, padding=1 → T preserved (local temporal mixing)
      - Spatial:  k=3, padding=1 → same-padding, H,W preserved
    """

    def __init__(self, in_ch, out_ch, dropout=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
            nn.Dropout(0.1) if dropout else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TemporalFusionHead(nn.Module):
    """Merge temporal into channels then process with Conv2d(k=3) layers.

    After 3D upscaling: (B, C, T, H, W) -> flatten to (B, C*T, H, W)
    then n_layers of Conv2d(hidden_dim, hidden_dim, k=3) -> Conv2d -> n_classes.
    """

    def __init__(self, in_ch, n_classes=4, hidden_dim=256, n_layers=2):
        super().__init__()
        layers = []
        # First layer: C*T -> hidden_dim
        layers.extend([
            nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.ReLU(inplace=True),
        ])
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(min(32, hidden_dim), hidden_dim),
                nn.ReLU(inplace=True),
            ])
        # Final projection
        layers.append(nn.Conv2d(hidden_dim, n_classes, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Upscaler3D(nn.Module):
    """Conv3D temporal + ConvTranspose spatial upscaler.

    Takes (B, embed_dim, T, H, W) and produces (B, n_classes, H_out, W_out).
    T is preserved through all 3D blocks, then merged into channels
    and processed by Conv2d layers for temporal fusion.
    """

    def __init__(self, embed_dim, num_frames, n_classes=4, hidden_dim=256,
                 n_layers=2, dropout=True):
        super().__init__()

        # 4 blocks: ConvTranspose(doubles spatial, halves ch) then Conv3d(k=3 same-padding)
        # Ch: 1024 -> 512 -> 256 -> 128 -> 64
        # T:  12 throughout (preserved via padding=1)
        # H:  21 -> 42 -> 84 -> 168 -> 336  (clean 2x upscaling, crop to 330)
        self.blocks = nn.Sequential(
            Conv3DTemporalSpatialBlock(embed_dim, embed_dim // 2, dropout=dropout),
            Conv3DTemporalSpatialBlock(embed_dim // 2, embed_dim // 4, dropout=dropout),
            Conv3DTemporalSpatialBlock(embed_dim // 4, embed_dim // 8, dropout=dropout),
            Conv3DTemporalSpatialBlock(embed_dim // 8, embed_dim // 16, dropout=dropout),
        )

        # After blocks: (B, embed_dim//16, T, H, W) -> flatten -> (B, embed_dim//16 * T, H, W)
        merged_ch = (embed_dim // 16) * num_frames  # 64 * 12 = 768

        # Conv2d fusion head with configurable depth/width
        self.head = TemporalFusionHead(merged_ch, n_classes, hidden_dim, n_layers)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.blocks(x)
        # x: (B, C', T, H_out, W_out) — T still preserved
        B, C, T, H, W = x.shape
        x = x.reshape(B, C * T, H, W)
        # x: (B, C'*T, H_out, W_out)
        x = self.head(x)
        return x


class PrithviSegConv3D(nn.Module):
    def __init__(self,
                 prithvi_params: dict,
                 prithvi_ckpt_path: str = None,
                 reshape: bool = True,
                 n_classes: int = 4,
                 model_size: str = "300m",
                 feed_timeloc: bool = False,
                 hidden_dim: int = 256,
                 n_layers: int = 2):
        super().__init__()

        if feed_timeloc:
            prithvi_params["coords_encoding"] = ["time", "location"]

        # Use the standard backbone but WITHOUT its built-in reshaper
        self.backbone = PrithviBackbone(prithvi_params, prithvi_ckpt_path, reshape=False)

        # Our own 3D reshaper that keeps temporal separate
        self.reshaper = PrithviReshape3D(
            prithvi_params["patch_size"],
            prithvi_params["img_size"],
            prithvi_params["num_frames"],
        )

        embed_dim = prithvi_params["embed_dim"]
        num_frames = prithvi_params["num_frames"]

        if model_size == "300m":
            self.head = Upscaler3D(embed_dim, num_frames, n_classes=n_classes,
                                   hidden_dim=hidden_dim, n_layers=n_layers)
        else:
            raise ValueError(f"model_size {model_size} not supported")

        self.n_frames = num_frames
        self.n_classes = n_classes

    def forward(self, x):
        if isinstance(x, dict):
            batch_size = x["chip"].size(0)
            x = {k: v.cuda() for k, v in x.items()}
        else:
            batch_size = x.size(0)
            x = x.cuda()

        x = self.backbone(x)         # (B, 1+T*H*W, D) since reshape=False
        x = self.reshaper(x)          # (B, D, T, H, W)
        x = self.head(x)              # (B, n_classes, H_out, W_out)
        x = x.view(batch_size, self.n_classes, x.size(2), x.size(3))
        # x = torch.sigmoid(x)
        return x
