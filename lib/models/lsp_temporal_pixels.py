import torch
import torch.nn as nn
import math


# ========================= Temporal Blocks =========================

class SETemporalBlock(nn.Module):
    """SE-style temporal attention: pools over features, weights timesteps.

    Squeeze: mean over emb_dim -> (N, T)
    Excitation: FC bottleneck over T -> (N, T)
    Scale: (N, T, 1) * (N, T, emb_dim)

    No sigmoid — weights are unbounded so the network learns freely.
    """
    def __init__(self, emb_dim, n_timesteps, reduction=2):
        super().__init__()
        mid = max(n_timesteps // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(n_timesteps, mid),
            nn.GELU(),
            nn.Linear(mid, n_timesteps),
        )

    def forward(self, x):
        # x: (N, T, emb_dim)
        w = x.mean(dim=2)          # (N, T) — squeeze over features
        w = self.fc(w)             # (N, T) — excitation
        return x * w.unsqueeze(-1) # (N, T, emb_dim)


class Conv1DTemporalBlock(nn.Module):
    """1D temporal convolution: operates along T dimension per pixel."""
    def __init__(self, emb_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(emb_dim, emb_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (N, T, emb_dim)
        x_t = x.transpose(1, 2)              # (N, emb_dim, T)
        x_t = self.act(self.bn(self.conv(x_t)))
        return x_t.transpose(1, 2)           # (N, T, emb_dim)


class TransformerTemporalBlock(nn.Module):
    """Stacked transformer encoder layers for temporal processing.

    Uses nn.TransformerEncoder directly (matching the original
    lsp_transformer_pixels.py) to avoid double-norm issues with
    individual layer wrapping.
    """
    def __init__(self, emb_dim, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (N, T, emb_dim)
        return self.encoder(x)


# ========================= Positional Encoding =========================

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embeddings for T timesteps."""
    def __init__(self, emb_dim, max_len=12):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, emb_dim) * 0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""
    def __init__(self, emb_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ========================= Unified Model =========================

class TemporalPixelModel(nn.Module):
    """Unified per-pixel temporal model with interchangeable temporal blocks.

    Variants:
        "se"          — SE temporal attention (pools features, weights timesteps)
        "conv1"       — 1D conv with kernel_size=1 (pointwise temporal mixing)
        "conv3"       — 1D conv with kernel_size=3 (local temporal patterns)
        "transformer" — Full self-attention over timesteps

    Architecture:
        input (N, T, C=6)
        -> band projection (6 -> emb_dim)
        -> positional encoding
        -> N temporal blocks with residual + LayerNorm
        -> mean pool over T
        -> dropout -> output projection (emb_dim -> num_classes)
    """

    VARIANTS = ("se", "conv1", "conv3", "transformer")

    def __init__(self,
                 variant: str = "se",
                 input_channels: int = 6,
                 seq_len: int = 12,
                 num_classes: int = 4,
                 emb_dim: int = 32,
                 n_layers: int = 3,
                 nhead: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert variant in self.VARIANTS, f"variant must be one of {self.VARIANTS}, got '{variant}'"

        self.variant = variant
        self.seq_len = seq_len

        # Band projection
        self.input_proj = nn.Linear(input_channels, emb_dim, bias=False)

        # Positional encoding
        self.pos_enc = LearnablePositionalEncoding(emb_dim, max_len=seq_len)

        # Temporal blocks
        if variant == "transformer":
            # Transformer uses nn.TransformerEncoder internally (handles its own
            # residual + norm), so we create one block with all layers inside.
            self.transformer = TransformerTemporalBlock(emb_dim, nhead, n_layers, dropout)
        else:
            self.blocks = nn.ModuleList()
            self.layer_norms = nn.ModuleList()
            for _ in range(n_layers):
                if variant == "se":
                    self.blocks.append(SETemporalBlock(emb_dim, seq_len))
                elif variant == "conv1":
                    self.blocks.append(Conv1DTemporalBlock(emb_dim, kernel_size=1))
                elif variant == "conv3":
                    self.blocks.append(Conv1DTemporalBlock(emb_dim, kernel_size=3))
                self.layer_norms.append(nn.LayerNorm(emb_dim))

        # Output
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(emb_dim, num_classes)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"TemporalPixelModel(variant={variant}, emb_dim={emb_dim}, "
              f"n_layers={n_layers}) — {total_params:,} params")

    def _forward_core(self, x):
        """Core forward pass on (N, T, C) input, returns (N, num_classes)."""
        # Band projection + positional encoding
        x = self.input_proj(x)           # (N, T, emb_dim)
        x = self.pos_enc(x)

        # Temporal blocks
        if self.variant == "transformer":
            x = self.transformer(x)
        else:
            for block, ln in zip(self.blocks, self.layer_norms):
                x = ln(x + block(x))     # residual + post-norm

        # Pool over T and predict
        x = x.mean(dim=1)               # (N, emb_dim)
        x = self.dropout(x)
        x = self.output_proj(x)          # (N, num_classes)
        return x

    def forward(self, x, processing_images=True, chunk_size=5000):
        x = x.cuda()

        if processing_images:
            # (B, C, T, H, W) -> per-pixel sequences
            x = x.permute(0, 3, 4, 2, 1)    # (B, H, W, T, C)
            B, H, W, T, C = x.shape
            x = x.reshape(B * H * W, T, C)

        # Chunk processing for large inputs
        N = x.size(0)
        if N > chunk_size:
            outputs = []
            for i in range(0, N, chunk_size):
                outputs.append(self._forward_core(x[i:i + chunk_size]))
            x = torch.cat(outputs, dim=0)
        else:
            x = self._forward_core(x)

        if processing_images:
            x = x.view(B, H, W, -1)
            x = x.permute(0, 3, 1, 2)       # (B, num_classes, H, W)

        return x
