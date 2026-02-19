import torch
import torch.nn.functional as F
from torch import nn

from lib.models.prithvi import PrithviBackbone, Upscaler


class PrithviReshapePool(nn.Module):
    """Reshape encoder tokens and pool over the temporal dimension.

    Instead of concatenating all T frames along the channel dim
    (producing embed_dim*T channels), this module first separates the
    temporal and spatial dimensions, then pools over time to yield a
    single (embed_dim, H_p, W_p) feature map per sample.

    Supports three pooling modes:
        - "mean":      simple average over timesteps
        - "max":       element-wise max over timesteps
        - "attention": learned query vector that produces per-spatial-
                       position attention weights over timesteps
    """

    def __init__(self, patch_size, input_size, num_frames, embed_dim, pool_type="mean"):
        super().__init__()
        self.patch_size = patch_size
        self.input_size = input_size
        self.view_size = int(self.input_size / self.patch_size[-1])  # e.g. 336/16 = 21
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.pool_type = pool_type

        if pool_type == "attention":
            self.pool_query = nn.Parameter(torch.randn(1, 1, 1, embed_dim) * 0.02)

    def forward(self, latent):
        # latent: (B, 1 + T*H_p*W_p, D)  [cls token + patch tokens]
        latent = latent[:, 1:, :]  # drop cls → (B, T*H_p*W_p, D)

        B, _, D = latent.shape
        T = self.num_frames
        H = W = self.view_size

        # Separate temporal and spatial dims
        # Token order from PatchEmbed Conv3d is (T, H, W)-major
        latent = latent.reshape(B, T, H * W, D)  # (B, T, H_p*W_p, D)

        if self.pool_type == "mean":
            latent = latent.mean(dim=1)  # (B, H_p*W_p, D)
        elif self.pool_type == "max":
            latent = latent.max(dim=1).values  # (B, H_p*W_p, D)
        elif self.pool_type == "attention":
            # Learned attention pooling: per-spatial-position weights over T
            # pool_query: (1, 1, 1, D),  latent: (B, T, H_p*W_p, D)
            attn = (latent * self.pool_query).sum(dim=-1)  # (B, T, H_p*W_p)
            attn = F.softmax(attn, dim=1)  # softmax over T
            latent = (latent * attn.unsqueeze(-1)).sum(dim=1)  # (B, H_p*W_p, D)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # Reshape to spatial feature map
        latent = latent.transpose(1, 2)  # (B, D, H_p*W_p)
        latent = latent.reshape(B, D, H, W)  # (B, D, H_p, W_p)

        return latent


class PrithviSegPool(nn.Module):
    """Prithvi segmentation model with temporal pooling before upscaling.

    Compared to PrithviSeg which concatenates all T frames into channels
    (embed_dim*T → Upscaler), this variant pools over T first, reducing
    the upscaler input from embed_dim*T to embed_dim channels.
    """

    def __init__(self,
                 prithvi_params: dict,
                 prithvi_ckpt_path: str = None,
                 reshape: bool = True,
                 n_classes: int = 1,
                 model_size: str = "300m",
                 pool_type: str = "mean",
                 feed_timeloc: bool = False):
        super().__init__()

        if feed_timeloc:
            prithvi_params["coords_encoding"] = ["time", "location"]

        # Use reshape=False on backbone — we handle reshaping ourselves
        self.backbone = PrithviBackbone(prithvi_params, prithvi_ckpt_path, reshape=False)

        self.reshaper = PrithviReshapePool(
            patch_size=prithvi_params["patch_size"],
            input_size=prithvi_params["img_size"],
            num_frames=prithvi_params["num_frames"],
            embed_dim=prithvi_params["embed_dim"],
            pool_type=pool_type,
        )

        embed_dim = prithvi_params["embed_dim"]

        if model_size == "300m":
            # embed_dim=1024, depth=4: 1024 → 512 → 256 → 128 → 64
            print(f"Pool head dim: {embed_dim} (pool_type={pool_type})")
            self.head = nn.Sequential(
                Upscaler(embed_dim, 4),
                nn.Conv2d(in_channels=embed_dim // 2**4, out_channels=n_classes, kernel_size=1),
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
        x = self.reshaper(x)
        x = self.head(x)
        x = x.view(batch_size, self.n_classes, x.size(2), x.size(3))
        x = torch.sigmoid(x)
        return x

    def forward_features(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        x = self.reshaper(x)
        return x
