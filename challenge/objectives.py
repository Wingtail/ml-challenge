import torch
from torch import nn
from torch.nn import functional as F


def off_diagonal(matrix: torch.Tensor):
    """Return the off-diagonal entries of a square matrix."""
    size = matrix.shape[0]
    if matrix.shape != (size, size):
        raise ValueError("expected a square matrix")
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def variance_loss(embedding: torch.Tensor):
    """Keep every embedding coordinate's batch standard deviation above one."""
    standard_deviation = torch.sqrt(embedding.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1.0 - standard_deviation).mean()


def covariance_loss(embedding: torch.Tensor):
    """VICReg covariance penalty."""
    centered = embedding - embedding.mean(dim=0)
    covariance = centered.T @ centered / max(len(embedding) - 1, 1)
    return off_diagonal(covariance).square().sum() / embedding.shape[1]


def normalized_covariance_loss(embedding: torch.Tensor):
    """VIbCReg's scale-independent normalized covariance penalty."""
    centered = embedding - embedding.mean(dim=0)
    normalized = F.normalize(centered, p=2, dim=0)
    correlation = normalized.T @ normalized
    return off_diagonal(correlation).square().mean()


def joint_embedding_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    covariance: str = "vicreg",
):
    """VICReg or VIbCReg-lite loss for two views of the same batch."""
    invariance = F.mse_loss(first, second)
    variance = variance_loss(first) + variance_loss(second)
    if covariance == "vicreg":
        covariance_term = covariance_loss(first) + covariance_loss(second)
        covariance_weight = 1.0
    elif covariance == "vibcreg":
        covariance_term = (
            normalized_covariance_loss(first) + normalized_covariance_loss(second)
        )
        covariance_weight = 200.0
    else:
        raise ValueError(f"unknown covariance objective: {covariance}")
    total = 25.0 * invariance + 25.0 * variance + covariance_weight * covariance_term
    return total, {
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance_term,
    }


def sensor_view(x: torch.Tensor):
    """A mild, activity-preserving scale-and-jitter view of a sensor window."""
    scale = torch.empty(len(x), 1, 1, device=x.device).uniform_(0.9, 1.1)
    return x * scale + 0.02 * torch.randn_like(x)


class EmbeddingProjector(nn.Module):
    """Loss-only projector that leaves the downstream encoder shape unchanged."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)
