"""
M1: Diagonal Gaussian — Axis-Aligned Ellipsoid

The simplest region model. Each embedding dimension gets an
independent variance, allowing the ellipsoid to stretch or
compress along coordinate axes.

    Σ = diag(exp(d))

Shape: Axis-aligned ellipsoid centered at query.
Parameters: d (embed_dim)
Expressiveness: Per-dimension weighting only; no dimensional interaction.
"""

import torch
import torch.nn as nn
from models.backbone import Backbone


class DiagonalGaussian(nn.Module):
    """
    Diagonal covariance Gaussian for context scoring.

    Given a query, predicts per-dimension log-variance.
    Context scores are the log-density under this diagonal Gaussian.

    Args:
        embed_dim: Embedding dimension (e.g., 1024).
        hidden_dim: Backbone hidden dimension. Default 256.
        clamp: (min, max) for log-variance clamping. Default (-3, 3),
               corresponding to variance range [0.05, 20.1].
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        clamp: tuple[float, float] = (-3.0, 3.0),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.clamp_min, self.clamp_max = clamp

        self.backbone = Backbone(embed_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, embed_dim)

        # Initialize to near-isotropic (all log-variances ≈ 0)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, query_emb: torch.Tensor) -> torch.Tensor:
        """
        Predict log-variance for each dimension.

        Args:
            query_emb: (B, D)

        Returns:
            log_var: (B, D)
        """
        h = self.backbone(query_emb)
        return torch.clamp(self.head(h), self.clamp_min, self.clamp_max)

    def score(self, query_emb: torch.Tensor, context_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute log-density of contexts under the diagonal Gaussian.

        score(x) = -0.5 * [ Σ_d (x_d - μ_d)² / σ²_d + Σ_d log σ²_d ]

        Args:
            query_emb: (B, D) — Gaussian center
            context_embs: (B, N, D) — context paragraph embeddings

        Returns:
            scores: (B, N) — log-density scores (higher = more relevant)
        """
        log_var = self.forward(query_emb)  # (B, D)

        mu = query_emb.unsqueeze(1)  # (B, 1, D)
        lv = log_var.unsqueeze(1)    # (B, 1, D)
        diff = context_embs - mu     # (B, N, D)
        var = torch.exp(lv)          # (B, 1, D)

        # Mahalanobis distance (diagonal case)
        mahal = torch.sum(diff ** 2 / var, dim=-1)  # (B, N)

        # Log normalization
        log_norm = torch.sum(lv, dim=-1)  # (B, 1)

        return -0.5 * (mahal + log_norm)
