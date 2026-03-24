"""
M2: Low-Rank Gaussian — Rotated Ellipsoid

Adds dimensional interaction via a low-rank factor.
The covariance matrix can represent rotated ellipsoids,
not just axis-aligned ones.

    Σ = diag(exp(d)) + L·Lᵀ

The precision matrix Σ⁻¹ is computed efficiently via the
Woodbury identity:
    Σ⁻¹ = D⁻¹ - D⁻¹·L·(I + Lᵀ·D⁻¹·L)⁻¹·Lᵀ·D⁻¹

This avoids O(D³) matrix inversion, reducing to O(D·r²).

Shape: Rotated ellipsoid centered at query.
Parameters: d (embed_dim) + L (embed_dim × rank)
Expressiveness: Captures dimensional interactions (e.g., "dimensions
    17 and 203 should be jointly attended to").
"""

import torch
import torch.nn as nn
from models.backbone import Backbone


class LowRankGaussian(nn.Module):
    """
    Low-rank + diagonal covariance Gaussian.

    Args:
        embed_dim: Embedding dimension.
        rank: Rank of the low-rank factor L. Default 16.
            Higher rank = more expressive but more parameters.
            Recommended: rank ∈ [8, 32] for embed_dim=1024.
        hidden_dim: Backbone hidden dimension.
        clamp: Log-variance clamping range.
    """

    def __init__(
        self,
        embed_dim: int,
        rank: int = 16,
        hidden_dim: int = 256,
        clamp: tuple[float, float] = (-3.0, 3.0),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank
        self.clamp_min, self.clamp_max = clamp

        self.backbone = Backbone(embed_dim, hidden_dim)

        # Diagonal component
        self.head_diag = nn.Linear(hidden_dim, embed_dim)
        nn.init.zeros_(self.head_diag.weight)
        nn.init.zeros_(self.head_diag.bias)

        # Low-rank factor
        self.head_L = nn.Linear(hidden_dim, embed_dim * rank)
        nn.init.normal_(self.head_L.weight, std=0.01)
        nn.init.zeros_(self.head_L.bias)

    def forward(self, query_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict covariance parameters.

        Returns:
            log_diag: (B, D) — log diagonal variances
            L: (B, D, r) — low-rank factor
        """
        h = self.backbone(query_emb)
        log_d = torch.clamp(self.head_diag(h), self.clamp_min, self.clamp_max)
        L = self.head_L(h).view(-1, self.embed_dim, self.rank) * 0.1
        return log_d, L

    def score(self, query_emb: torch.Tensor, context_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute log-density using Woodbury identity.

        Args:
            query_emb: (B, D)
            context_embs: (B, N, D)

        Returns:
            scores: (B, N)
        """
        log_d, L = self.forward(query_emb)
        B, N, D = context_embs.shape
        R = self.rank

        d_val = torch.exp(log_d)       # (B, D)
        d_inv = 1.0 / d_val            # (B, D)
        diff = context_embs - query_emb.unsqueeze(1)  # (B, N, D)

        # Diagonal Mahalanobis: diff^T · D^{-1} · diff
        mahal_diag = torch.sum(diff ** 2 * d_inv.unsqueeze(1), dim=-1)  # (B, N)

        # Woodbury correction: -diff^T · D^{-1}·L·M^{-1}·L^T·D^{-1} · diff
        # where M = I_r + L^T · D^{-1} · L
        DinvL = d_inv.unsqueeze(-1) * L  # (B, D, R)
        M = torch.eye(R, device=query_emb.device).unsqueeze(0) + \
            torch.bmm(L.transpose(1, 2), DinvL)  # (B, R, R)
        M_inv = torch.linalg.inv(M)  # (B, R, R)

        v = diff * d_inv.unsqueeze(1)  # (B, N, D)
        w = torch.bmm(L.transpose(1, 2), v.transpose(1, 2))  # (B, R, N)
        correction = torch.sum(w * torch.bmm(M_inv, w), dim=1)  # (B, N)

        mahal = mahal_diag - correction  # (B, N)

        # Log determinant: log|Σ| = log|D| + log|M|
        log_det = torch.sum(log_d, dim=-1, keepdim=True) + \
                  torch.linalg.slogdet(M)[1].unsqueeze(-1)  # (B, 1)

        return -0.5 * (mahal + log_det)
