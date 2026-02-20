import torch
import torch.nn as nn
import numpy as np

from lib.models.prithvi_mae import PrithviMAE, PatchEmbed, get_3d_sincos_pos_embed
from lib.models.prithvi import PrithviReshape
from lib.models.lsp_temporal_pixels import TemporalFeatureExtractor


class PrithviSeasonal(nn.Module):
    """Prithvi with pretrained temporal feature extractor as input.

    Architecture:
        1. TemporalFeatureExtractor (pretrained on pixel task):
           raw HLS (B, 6, 12, H, W) -> per-pixel conv3 -> temporal reduce
           -> (B, fe_emb_dim, 4, H, W)

        2. Custom PatchEmbed (learned from scratch):
           (B, fe_emb_dim, 4, H, W) -> Conv3d -> tokens (B, N, embed_dim=1024)

        3. Prithvi transformer blocks (pretrained):
           tokens -> 24 transformer blocks -> (B, N, 1024)

        4. Reshape + PixelShuffle -> project to fe_emb_dim

        5. Residual: seasonal_features (mean over T) + prithvi_features
           -> light 1x1 conv -> sigmoid -> (B, n_classes, 336, 336)

    The feature extractor compresses 12 monthly observations into 4 seasonal
    representations. Prithvi adds spatial context as a residual to the
    temporal features.
    """

    def __init__(self,
                 prithvi_params: dict,
                 prithvi_ckpt_path: str = None,
                 feature_ckpt_path: str = None,
                 n_classes: int = 4,
                 model_size: str = "300m",
                 # Feature extractor params
                 fe_variant: str = "conv3",
                 fe_emb_dim: int = 128,
                 fe_n_layers: int = 4,
                 fe_dropout: float = 0.1,
                 fe_temporal_reduce_factor: int = 3,
                 # PixelShuffle params
                 c_per_t: int = 4):
        super().__init__()

        if model_size != "300m":
            raise ValueError(f"model_size {model_size} not supported")

        self.n_classes = n_classes
        self.embed_dim = prithvi_params["embed_dim"]     # 1024
        self.n_frames = prithvi_params["num_frames"]     # should be 4 after config modification
        self.fe_emb_dim = fe_emb_dim

        patch_size = prithvi_params["patch_size"]        # [1, 16, 16]
        patch_h = patch_size[1]                          # 16
        img_size = prithvi_params["img_size"]            # 336

        # ==================== Feature Extractor ====================
        self.feature_extractor = TemporalFeatureExtractor(
            variant=fe_variant,
            input_channels=prithvi_params["in_chans"],   # 6
            seq_len=12,                                   # always 12 raw months
            emb_dim=fe_emb_dim,
            n_layers=fe_n_layers,
            dropout=fe_dropout,
            temporal_reduce_factor=fe_temporal_reduce_factor,
        )

        if feature_ckpt_path is not None:
            ckpt = torch.load(feature_ckpt_path, weights_only=False)
            self.feature_extractor.load_state_dict(ckpt["feature_extractor_state_dict"])
            print(f"Loaded feature extractor from {feature_ckpt_path}")

        # ==================== Custom Patch Embedding ====================
        # Replaces Prithvi's original PatchEmbed (which expects in_chans=6)
        # Our input is (B, fe_emb_dim, 4, H, W) from the feature extractor
        self.custom_patch_embed = PatchEmbed(
            input_size=(self.n_frames, img_size, img_size),
            patch_size=tuple(patch_size),
            in_chans=fe_emb_dim,
            embed_dim=self.embed_dim,
        )

        # ==================== Prithvi Backbone (transformer blocks only) ====================
        # Create PrithviMAE with the modified config (num_frames=4)
        # We only use its transformer blocks, not its patch_embed
        self.prithvi = PrithviMAE(**prithvi_params)

        if prithvi_ckpt_path is not None:
            checkpoint = torch.load(prithvi_ckpt_path, weights_only=False)

            if "encoder.pos_embed" not in checkpoint.keys():
                key = "model" if "model" in checkpoint.keys() else "state_dict"
                checkpoint = checkpoint[key]

            # Delete keys that won't match:
            # - pos_embed: different num_frames (4 vs original)
            # - patch_embed: different in_chans (fe_emb_dim vs 6)
            # - decoder: not needed (encoder_only)
            keys = list(checkpoint.keys())
            for k in keys:
                if "pos_embed" in k:
                    del checkpoint[k]
                elif "decoder" in k and prithvi_params.get("encoder_only", True):
                    del checkpoint[k]
                elif "patch_embed" in k:
                    # Our custom patch_embed has different in_chans
                    del checkpoint[k]
                elif "prithvi" in k:
                    new_k = k.replace("prithvi.", "")
                    checkpoint[new_k] = checkpoint[k]
                    del checkpoint[k]
                elif k in self.prithvi.state_dict() and checkpoint[k].shape != self.prithvi.state_dict()[k].shape:
                    print(f"Warning: size mismatch for {k}, deleting")
                    del checkpoint[k]

            missing, unexpected = self.prithvi.load_state_dict(checkpoint, strict=False)
            print(f"Prithvi loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")
            if missing:
                print(f"  Missing keys (expected): {[k for k in missing if 'patch_embed' in k or 'pos_embed' in k]}")

        # Reshaper: removes CLS token, reshapes to (B, embed_dim*T, H_patch, W_patch)
        self.reshaper = PrithviReshape(patch_size, img_size)

        # ==================== PixelShuffle upsampling ====================
        pixel_out = c_per_t * patch_h * patch_h
        self.token_proj = nn.Sequential(
            nn.Conv2d(self.embed_dim, pixel_out, kernel_size=1, bias=False),
            nn.GroupNorm(1, pixel_out),
            nn.GELU(),
        )
        self.shuffle = nn.PixelShuffle(patch_h)

        # Project Prithvi PixelShuffle output to fe_emb_dim
        prithvi_channels = self.n_frames * c_per_t  # T_out * c_per_t
        self.prithvi_proj = nn.Sequential(
            nn.Conv2d(prithvi_channels, fe_emb_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, fe_emb_dim),
            nn.GELU(),
        )

        # ==================== Combination + Output ====================
        self.seasonal_norm = nn.GroupNorm(1, fe_emb_dim)
        self.prithvi_norm = nn.GroupNorm(1, fe_emb_dim)

        self.output_head = nn.Sequential(
            nn.Conv2d(fe_emb_dim, fe_emb_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, fe_emb_dim),
            nn.GELU(),
            nn.Conv2d(fe_emb_dim, n_classes, kernel_size=1),
        )

        # ==================== Print summary ====================
        fe_params = sum(p.numel() for p in self.feature_extractor.parameters())
        patch_params = sum(p.numel() for p in self.custom_patch_embed.parameters())
        backbone_params = sum(p.numel() for p in self.prithvi.parameters())
        head_params = sum(p.numel() for p in self.token_proj.parameters()) + \
                      sum(p.numel() for p in self.prithvi_proj.parameters()) + \
                      sum(p.numel() for p in self.output_head.parameters())
        print(f"PrithviSeasonal:")
        print(f"  Feature extractor: {fe_params:,} params (fe_emb_dim={fe_emb_dim}, {fe_n_layers} layers)")
        print(f"  Custom patch_embed: {patch_params:,} params ({fe_emb_dim} -> {self.embed_dim})")
        print(f"  Prithvi backbone: {backbone_params:,} params")
        print(f"  Head (PixelShuffle + proj + output): {head_params:,} params (c_per_t={c_per_t})")

    def _extract_features(self, raw_pixels, H_full, W_full):
        """Run feature extractor per-pixel, reshape to (B, fe_emb_dim, T_out, H, W).

        Args:
            raw_pixels: (B, C, T=12, H, W)
            H_full, W_full: spatial dims to use

        Returns: (B, fe_emb_dim, T_out=4, H_full, W_full)
        """
        B = raw_pixels.size(0)
        T_out = self.feature_extractor.seq_len_out

        # (B, C, T, H, W) -> (B, H, W, T, C) -> (B*H*W, T, C)
        x = raw_pixels[:, :, :, :H_full, :W_full]
        x = x.permute(0, 3, 4, 2, 1)
        x = x.reshape(B * H_full * W_full, 12, raw_pixels.size(1))

        x = self.feature_extractor(x)  # (B*H*W, T_out, fe_emb_dim)

        # Reshape to spatial: (B, H, W, T_out, fe_emb_dim) -> (B, fe_emb_dim, T_out, H, W)
        x = x.view(B, H_full, W_full, T_out, self.fe_emb_dim)
        x = x.permute(0, 4, 3, 1, 2)  # (B, fe_emb_dim, T_out, H_full, W_full)
        return x

    def forward(self, x):
        if isinstance(x, dict):
            batch_size = x["chip"].size(0)
            raw_pixels = x["chip"].cuda()
        else:
            batch_size = x.size(0)
            raw_pixels = x.cuda()

        # Target spatial size from Prithvi's expected input
        img_size = self.custom_patch_embed.input_size[1]  # 336
        H_full = img_size
        W_full = img_size

        # 1. Feature extractor: (B, 6, 12, H, W) -> (B, fe_emb_dim, 4, H, W)
        seasonal_features = self._extract_features(raw_pixels, H_full, W_full)

        # 2. Custom patch embed: (B, fe_emb_dim, 4, H, W) -> (B, N, embed_dim)
        tokens = self.custom_patch_embed(seasonal_features)

        # 3. Prithvi transformer blocks (bypass Prithvi's own patch_embed)
        # Add positional embedding
        sample_shape = seasonal_features.shape[-3:]  # (4, H, W)
        pos_embed = self.prithvi.encoder.interpolate_pos_encoding(sample_shape)
        tokens = tokens + pos_embed[:, 1:, :]

        # Add CLS token
        cls_token = self.prithvi.encoder.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        # Run through transformer blocks
        for block in self.prithvi.encoder.blocks:
            tokens = block(tokens)
        tokens = self.prithvi.encoder.norm(tokens)

        # 4. Reshape: remove CLS, reshape to (B, embed_dim*T, 21, 21)
        x = self.reshaper(tokens)
        _, _, H, W = x.shape

        # 5. PixelShuffle to full resolution
        x = x.view(batch_size, self.embed_dim, self.n_frames, H, W)
        x = x.permute(0, 2, 1, 3, 4)                                    # (B, T, E, H, W)
        x = x.reshape(batch_size * self.n_frames, self.embed_dim, H, W) # (B*T, E, H, W)

        x = self.token_proj(x)         # (B*T, c_per_t*16*16, H, W)
        x = self.shuffle(x)            # (B*T, c_per_t, H*16, W*16)

        _, C, H_out, W_out = x.shape
        x = x.view(batch_size, self.n_frames * C, H_out, W_out)  # (B, T*c_per_t, 336, 336)

        # 6. Project Prithvi output to fe_emb_dim
        prithvi_features = self.prithvi_proj(x)     # (B, fe_emb_dim, 336, 336)

        # 7. Residual: seasonal features (mean over T) + Prithvi spatial features
        seasonal_pooled = seasonal_features.mean(dim=2)  # (B, fe_emb_dim, H, W)
        # combined = self.seasonal_norm(seasonal_pooled) + self.prithvi_norm(prithvi_features)
        combined = self.prithvi_norm(prithvi_features)

        # 8. Light output head
        out = self.output_head(combined)             # (B, n_classes, 336, 336)
        out = torch.sigmoid(out)
        return out
