import torch
import torch.nn.functional as F
from torch import nn

from lib.models.prithvi import PrithviBackbone, PrithviReshape


class BilinearUpscaler(nn.Module):
    """Upscaler using bilinear interpolation + Conv2d per block.

    Each block does:
        1. Bilinear upsample 2x (0 parameters)
        2. Conv2d(in_ch, out_ch, k=conv_k) for channel reduction
        3. BatchNorm + GELU

    The conv kernel size is configurable:
        - k=1: pointwise channel mixing only (cheapest)
        - k=3: adds local spatial refinement after upsampling

    Channels halve at each block, same schedule as the original Upscaler.
    """

    def __init__(self, embed_dim: int, depth: int, conv_k: int = 1, dropout: bool = True):
        super().__init__()
        padding = (conv_k - 1) // 2  # same-padding: k=1->0, k=3->1

        def build_block(in_ch, out_ch): return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=conv_k, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout(0.1) if dropout else nn.Identity())

        self.upscale_blocks = nn.Sequential(
            *[build_block(int(embed_dim // 2**i), int(embed_dim // 2**(i+1))) for i in range(depth)]
        )

    def forward(self, x):
        return self.upscale_blocks(x)


class PrithviSegUpsample(nn.Module):
    """Prithvi segmentation model using bilinear upsampling instead of ConvTranspose2d.

    Replaces the original ~852M parameter head by using bilinear interpolation
    (0 params) + Conv2d for channel reduction at each upscale step.

    The conv kernel size after each upsample is configurable:
        - conv_k=1: pointwise only (~100M head params)
        - conv_k=3: local spatial refinement (~900M head params, similar cost
                     to original but without checkerboard artifacts)
    """

    def __init__(self,
                 prithvi_params: dict,
                 prithvi_ckpt_path: str = None,
                 reshape: bool = True,
                 n_classes: int = 1,
                 model_size: str = "300m",
                 conv_k: int = 1,
                 feed_timeloc: bool = False):
        super().__init__()

        if feed_timeloc:
            prithvi_params["coords_encoding"] = ["time", "location"]

        self.backbone = PrithviBackbone(prithvi_params, prithvi_ckpt_path, reshape)
        concat_dim = prithvi_params["embed_dim"] * prithvi_params["num_frames"]

        if model_size == "300m":
            print(f"Upsample head dim: {concat_dim} (conv_k={conv_k})")
            self.head = nn.Sequential(
                BilinearUpscaler(concat_dim, 4, conv_k=conv_k),
                nn.Conv2d(in_channels=concat_dim // 2**4, out_channels=n_classes, kernel_size=1),
            )
        else:
            raise ValueError(f"model_size {model_size} not supported")

        self.n_frames = prithvi_params["num_frames"]
        self.n_classes = n_classes

    def forward(self, x):
        if isinstance(x, dict):
            batch_size = x["chip"].size(0)
            x = {k: v.cuda() for k, v in x.items()}
        else:
            batch_size = x.size(0)
            x = x.cuda()

        x = self.backbone(x)
        x = self.head(x)
        x = x.view(batch_size, self.n_classes, x.size(2), x.size(3))
        x = torch.sigmoid(x)
        return x

    def forward_features(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        return x
