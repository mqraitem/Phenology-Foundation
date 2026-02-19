import torch
import torch.nn.functional as F
from torch import nn

from lib.models.prithvi import PrithviBackbone, PrithviReshape, Upscaler


class PrithviSegSimple(nn.Module):
    """Prithvi segmentation model with a 1x1 channel projection before upscaling.

    The original PrithviSeg feeds embed_dim*num_frames (e.g. 12288) channels
    directly into the Upscaler, resulting in ~850M head parameters because the
    conv layers scale quadratically with channel count.

    This variant inserts a 1x1 Conv2d projection to compress channels from
    embed_dim*num_frames down to `proj_dim` before the Upscaler, separating
    channel compression from spatial upscaling.  All temporal information is
    still preserved in the concat representation — it is just linearly
    compressed rather than discarded.
    """

    def __init__(self,
                 prithvi_params: dict,
                 prithvi_ckpt_path: str = None,
                 reshape: bool = True,
                 n_classes: int = 1,
                 model_size: str = "300m",
                 proj_dim: int = 512,
                 feed_timeloc: bool = False):
        super().__init__()

        if feed_timeloc:
            prithvi_params["coords_encoding"] = ["time", "location"]

        self.backbone = PrithviBackbone(prithvi_params, prithvi_ckpt_path, reshape)

        concat_dim = prithvi_params["embed_dim"] * prithvi_params["num_frames"]

        if model_size == "300m":
            # 1x1 projection: 12288 -> proj_dim
            self.proj = nn.Sequential(
                nn.Conv2d(concat_dim, proj_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(proj_dim),
                nn.GELU(),
            )

            # Upscaler: proj_dim -> proj_dim//2 -> ... (4 blocks, 2x spatial each)
            # Final channel count: proj_dim // 2^4
            self.head = nn.Sequential(
                Upscaler(proj_dim, 4),
                nn.Conv2d(in_channels=proj_dim // 2**4, out_channels=n_classes, kernel_size=1),
            )

            print(f"Simple head: {concat_dim} -> proj {proj_dim} -> upscale -> {proj_dim // 16} -> {n_classes}")
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

        x = self.backbone(x)       # (B, embed_dim*T, H_p, W_p)
        x = self.proj(x)           # (B, proj_dim, H_p, W_p)
        x = self.head(x)           # (B, n_classes, H_out, W_out)
        x = x.view(batch_size, self.n_classes, x.size(2), x.size(3))
        x = torch.sigmoid(x)
        return x

    def forward_features(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        x = self.proj(x)
        return x
