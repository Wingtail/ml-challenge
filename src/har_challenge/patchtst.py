"""The PatchTST sensor encoder, masked pretrainer, and direct classifier."""

import torch
from torch import nn


class PatchTSTEncoder(nn.Module):
    """Encode a `[batch, 9, 128]` sensor window into 1,152 values."""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.channels = 9
        self.window_size = 128
        self.patch_length = 8
        self.patch_count = 16
        self.d_model = 128
        self.embedding_dim = 9 * 128

        self.patch_projection = nn.Linear(8, 128, bias=False)
        self.position = nn.Parameter(torch.empty(1, 16, 128))
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.norm = nn.LayerNorm(128)
        nn.init.trunc_normal_(self.position, std=0.02)

    def make_patches(self, x):
        if x.ndim != 3 or x.shape[1:] != (9, 128):
            raise ValueError(f"expected [batch, 9, 128], got {tuple(x.shape)}")
        return x.unfold(-1, 8, 8)

    def encode_patches(self, patches, positions):
        batch, channels, count, length = patches.shape
        patches = patches.reshape(batch * channels, count, length)
        positions = positions.reshape(batch * channels, count)

        positional = self.position.expand(batch * channels, -1, -1).gather(
            1, positions.unsqueeze(-1).expand(-1, -1, 128)
        )
        tokens = self.patch_projection(patches) + positional
        tokens = self.norm(self.transformer(tokens))
        return tokens.reshape(batch, channels, count, 128)

    def patch_tokens(self, x):
        patches = self.make_patches(x)
        positions = torch.arange(16, device=x.device)
        positions = positions.view(1, 1, 16).expand(x.shape[0], 9, 16)
        return self.encode_patches(patches, positions)

    def forward(self, x):
        # Mean-pool time, then concatenate the nine channel representations.
        return self.patch_tokens(x).mean(dim=2).flatten(start_dim=1)


class MaskedPatchPretrainer(nn.Module):
    """Reconstruct the randomly masked 40% of sensor patches."""

    def __init__(self, encoder, mask_ratio=0.4):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.reconstruction_head = nn.Linear(128, 8)

    def forward(self, x):
        patches = self.encoder.make_patches(x)
        batch, channels, count, length = patches.shape
        rows = batch * channels
        patches = patches.reshape(rows, count, length)

        # Keep the original experiment's random-number sequence. The first
        # draw selected retained positions even though DropPatch was disabled.
        positions = torch.rand(rows, count, device=x.device).argsort(dim=1)
        positions = positions.sort(dim=1).values
        patches = patches.gather(
            1, positions.unsqueeze(-1).expand(-1, -1, length)
        )

        mask_count = max(1, round(count * self.mask_ratio))
        order = torch.rand(rows, count, device=x.device).argsort(dim=1)
        mask = torch.zeros(rows, count, dtype=torch.bool, device=x.device)
        mask.scatter_(1, order[:, :mask_count], True)

        masked = patches.masked_fill(mask.unsqueeze(-1), 0.0)
        tokens = self.encoder.encode_patches(
            masked.reshape(batch, channels, count, length),
            positions.reshape(batch, channels, count),
        )
        reconstruction = self.reconstruction_head(tokens).reshape_as(patches)
        return (reconstruction - patches).square().mean(dim=-1)[mask].mean()


class DirectClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(1152, 6)

    def forward(self, x):
        return self.head(self.encoder(x))
