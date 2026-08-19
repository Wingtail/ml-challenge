from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock1d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.0, pool=False):
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool1d(2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        super().__init__(*layers)


class DCTEncoder(nn.Module):
    """Convolutional Transformer encoder used by the DCTCSS reproduction."""

    embedding_dim = 128

    def __init__(self, in_channels=9):
        super().__init__()
        self.convolutions = nn.Sequential(
            ConvBlock1d(in_channels, 64, 7, pool=True),
            ConvBlock1d(64, 128, 5, pool=True),
            ConvBlock1d(128, 128, 3),
        )
        self.position = nn.Parameter(torch.zeros(1, 32, 128))
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.convolutions(x).transpose(1, 2)
        x = x + self.position[:, : x.shape[1]]
        x = self.transformer(x)
        return self.norm(x.mean(dim=1))


class MLP(nn.Sequential):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )


class BYOL(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projector = MLP(encoder.embedding_dim)
        self.predictor = MLP(128, 256, 128)
        self.target_encoder = deepcopy(encoder)
        self.target_projector = deepcopy(self.projector)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.target_projector.parameters():
            parameter.requires_grad = False

    def online(self, x):
        return self.predictor(self.projector(self.encoder(x)))

    @torch.no_grad()
    def target(self, x):
        return self.target_projector(self.target_encoder(x))

    @torch.no_grad()
    def update_target(self, momentum=0.99):
        online = list(self.encoder.parameters()) + list(self.projector.parameters())
        target = list(self.target_encoder.parameters()) + list(self.target_projector.parameters())
        for online_parameter, target_parameter in zip(online, target, strict=True):
            target_parameter.data.mul_(momentum).add_(online_parameter.data, alpha=1 - momentum)


class TCLHAREncoder(nn.Module):
    """Cross-modality convolutional encoder described by TCLHAR."""

    def __init__(self, modality_count=3, channels_per_modality=3, window_size=128):
        super().__init__()
        self.modality_count = modality_count
        self.channels_per_modality = channels_per_modality
        self.subnets = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock1d(channels_per_modality, 32, 3, dropout=0.1),
                    ConvBlock1d(32, 64, 3, dropout=0.1),
                    ConvBlock1d(64, 128, 3, dropout=0.1),
                )
                for _ in range(modality_count)
            ]
        )
        self.fusion = nn.Sequential(
            self._fusion_block(128, 128, (modality_count, 3)),
            self._fusion_block(128, 128, (modality_count, 3)),
            self._fusion_block(128, 256, (modality_count, 3)),
        )
        self.embedding_dim = 256 * modality_count * window_size

    @staticmethod
    def _fusion_block(in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        modalities = torch.split(x, self.channels_per_modality, dim=1)
        features = [subnet(modality) for subnet, modality in zip(self.subnets, modalities)]
        x = torch.stack(features, dim=2)
        x = self.fusion(x)
        return x.flatten(start_dim=1)


class TCLHAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TCLHAREncoder()
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

    def forward(self, x):
        return self.projector(self.encoder(x))


class AutoCLEncoder(nn.Module):
    """Three-layer FCN encoder from AutoCL."""

    embedding_dim = 128 * 16

    def __init__(self, in_channels=9):
        super().__init__()
        self.network = nn.Sequential(
            ConvBlock1d(in_channels, 64, 8, dropout=0.1, pool=True),
            ConvBlock1d(64, 128, 8, dropout=0.1, pool=True),
            ConvBlock1d(128, 128, 8, dropout=0.1, pool=True),
        )

    def forward(self, x):
        return self.network(x).flatten(start_dim=1)


class AutoCL(nn.Module):
    def __init__(self, channels=9, window_size=128):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.encoder = AutoCLEncoder(channels)
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )
        self.generator_norm = nn.BatchNorm1d(128)
        self.generator = nn.GRU(
            input_size=128,
            hidden_size=2 * channels,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.generator_output = nn.Linear(4 * channels, channels)

    def project(self, x):
        return F.softmax(self.projector(self.encoder(x)), dim=1)

    def forward(self, x):
        original = self.project(x)
        generator_input = F.relu(self.generator_norm(original.detach()))
        generator_input = generator_input.unsqueeze(1).expand(-1, self.window_size, -1)
        generated, _ = self.generator(generator_input)
        augmented = self.generator_output(generated).transpose(1, 2)
        projected_augmented = self.project(augmented)
        return original, projected_augmented, augmented


class MaskCAEEncoder(nn.Module):
    """Four-level fully convolutional encoder used by MaskCAE."""

    embedding_dim = 128 * 8 * 9

    def __init__(self):
        super().__init__()
        channels = (1, 16, 32, 64, 128)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels[index],
                        channels[index + 1],
                        kernel_size=(5, 1),
                        padding=(2, 0),
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels[index + 1]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, 1)),
                )
                for index in range(4)
            ]
        )

    def hierarchical(self, x):
        x = x.transpose(1, 2).unsqueeze(1)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def forward(self, x):
        return self.hierarchical(x)[-1].flatten(start_dim=1)


class MaskCAE(nn.Module):
    def __init__(self, mask_ratio=0.3, patch_size=16):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.encoder = MaskCAEEncoder()
        self.decoder = nn.Sequential(
            self._up_block(128, 64),
            self._up_block(64, 32),
            self._up_block(32, 16),
            nn.ConvTranspose2d(16, 1, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
        )

    @staticmethod
    def _up_block(in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(4, 1),
                stride=(2, 1),
                padding=(1, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        batch, channels, length = x.shape
        patch_count = length // self.patch_size
        keep = max(1, round(patch_count * channels * (1 - self.mask_ratio)))
        noise = torch.rand(batch, patch_count * channels, device=x.device)
        active = torch.zeros_like(noise, dtype=torch.bool)
        active.scatter_(1, noise.argsort(dim=1)[:, :keep], True)
        active = active.view(batch, patch_count, channels)

        mask = active.repeat_interleave(self.patch_size, dim=1)
        masked = x.transpose(1, 2) * mask
        features = self.encoder.hierarchical(masked.transpose(1, 2))
        reconstructed = self.decoder(features[-1]).squeeze(1)

        target_patches = x.transpose(1, 2).reshape(batch, patch_count, self.patch_size, channels)
        target_patches = target_patches.permute(0, 1, 3, 2).reshape(
            batch, patch_count * channels, self.patch_size
        )
        mean = target_patches.mean(dim=-1, keepdim=True)
        std = target_patches.std(dim=-1, keepdim=True).clamp_min(1e-6)
        target_patches = (target_patches - mean) / std

        reconstructed = reconstructed.reshape(batch, patch_count, self.patch_size, channels)
        reconstructed = reconstructed.permute(0, 1, 3, 2).reshape_as(target_patches)
        patch_loss = (reconstructed - target_patches).square().mean(dim=-1)
        masked_patches = ~active.flatten(start_dim=1)
        return patch_loss[masked_patches].mean()


def make_pretrainer(name: str):
    if name == "dctcss":
        model = BYOL(DCTEncoder())
        return model, model.encoder
    if name == "tclhar":
        model = TCLHAR()
        return model, model.encoder
    if name == "autocl":
        model = AutoCL()
        return model, model.encoder
    if name == "maskcae":
        model = MaskCAE()
        return model, model.encoder
    raise ValueError(f"unknown model: {name}")

