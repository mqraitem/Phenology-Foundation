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


class PrithviSegBlowupT(nn.Module):
    """Prithvi segmentation with per-timestep PixelShuffle + per-pixel temporal transformer.

    Same as PrithviSegBlowup but replaces the Conv2d temporal fusion with a
    lightweight per-pixel transformer that explicitly reasons about the temporal
    sequence at each pixel location.

    Pipeline:
    1. Backbone: (B, 12288, 21, 21)
    2. Separate time steps + shared 1x1 proj + PixelShuffle(16)
       -> (B, T, c_per_t, 336, 336)
    3. Per pixel: a temporal transformer processes the 12-step sequence
       (each token = c_per_t features) and outputs n_classes predictions
    4. Output: (B, n_classes, 336, 336)
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
        self.n_classes = n_classes
        self.c_per_t = c_per_t
        self.chunk_size = chunk_size

        patch_h = prithvi_params["patch_size"][1]

        if model_size == "300m":
            # Shared per-timestep projection + PixelShuffle
            pixel_out = c_per_t * patch_h * patch_h
            self.token_proj = nn.Sequential(
                nn.Conv2d(self.embed_dim, pixel_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(pixel_out),
                nn.GELU(),
            )
            self.shuffle = nn.PixelShuffle(patch_h)

            # Per-pixel temporal transformer
            self.input_proj = nn.Linear(c_per_t, d_model, bias=False)
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

            print(f"BlowupT head: ({self.embed_dim}, 21, 21) x {self.n_frames}t "
                  f"-> proj {pixel_out} -> PixelShuffle({patch_h}) -> "
                  f"per-pixel transformer(d={d_model}, heads={nhead}, layers={num_layers}) "
                  f"-> {n_classes}")
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
        x = x.view(batch_size, self.embed_dim, self.n_frames, H, W)      # (B, E, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)                                     # (B, T, E, H, W)
        x = x.reshape(batch_size * self.n_frames, self.embed_dim, H, W)  # (B*T, E, H, W)

        # Per-timestep projection + PixelShuffle
        x = self.token_proj(x)                  # (B*T, c_per_t*16*16, H, W)
        x = self.shuffle(x)                     # (B*T, c_per_t, H*16, W*16)

        # Reshape to (B, T, c_per_t, H_full, W_full)
        _, C, H_full, W_full = x.shape
        x = x.view(batch_size, self.n_frames, C, H_full, W_full)

        # Rearrange for per-pixel temporal processing: (B, H, W, T, c_per_t)
        x = x.permute(0, 3, 4, 1, 2)           # (B, H_full, W_full, T, c_per_t)
        x = x.reshape(batch_size * H_full * W_full, self.n_frames, C)  # (N, T, c_per_t)

        # Per-pixel temporal transformer with chunking
        x = self.input_proj(x)                  # (N, T, d_model)
        x = self.pos_encoder(x)

        N = x.size(0)
        outputs = []
        for i in range(0, N, self.chunk_size):
            x_chunk = x[i:i + self.chunk_size]
            x_chunk = self.transformer_encoder(x_chunk)   # (chunk, T, d_model)
            x_chunk = x_chunk.mean(dim=1)                 # (chunk, d_model)
            x_chunk = self.temporal_dropout(x_chunk)
            x_chunk = self.output_proj(x_chunk)           # (chunk, n_classes)
            outputs.append(x_chunk)

        x = torch.cat(outputs, dim=0)           # (B*H*W, n_classes)
        x = x.view(batch_size, H_full, W_full, self.n_classes)
        x = x.permute(0, 3, 1, 2)              # (B, n_classes, H_full, W_full)
        x = torch.sigmoid(x)
        return x

    def forward_features(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        return x
