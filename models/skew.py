"""
M3: Skew-Gaussian — Skewed Ellipsoid

The key innovation: breaks the symmetry of Gaussian distributions
by adding a mean offset δ and a skewness vector α.

    score(x) = log N(x | f(q)+δ, Σ) + log σ(αᵀ·z)

where z = x - f(q) - δ, and σ is the sigmoid function
(a differentiable approximation to Φ, the normal CDF).

The skewness term creates directional asymmetry:
- Density decays slowly along the α direction ("reaching toward" a target)
- Density decays quickly in the opposite direction

This is particularly effective for bridge-type questions that require
connecting two entity paragraphs: the ellipsoid "reaches" from the
primary entity toward the secondary one.

Shape: Skewed ellipsoid with shifted center.
Parameters: d + L + δ + α
Expressiveness: Directional asymmetry + dimensional interaction.

Reference:
    Azzalini, A. (2014). The Skew-Normal and Related Families.
    Cambridge University Press.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import Backbone


class SkewGaussian(nn.Module):
    """
    Skew-Gaussian distribution for context scoring.

    Combines a low-rank Gaussian (for anisotropy and dimensional interaction)
    with mean offset and skewness (for directional asymmetry).

    This is the recommended model for most use cases — it achieves
    the best efficiency-quality tradeoff (+4.6% F1, 1.9ms latency).

    Args:
        embed_dim: Embedding dimension.
        rank: Low-rank factor rank. Default 16.
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

        # Covariance parameters (same as LowRank)
        self.head_diag = nn.Linear(hidden_dim, embed_dim)
        self.head_L = nn.Linear(hidden_dim, embed_dim * rank)

        # Skewness parameters (new)
        self.head_delta = nn.Linear(hidden_dim, embed_dim)  # Mean offset
        self.head_alpha = nn.Linear(hidden_dim, embed_dim)  # Skewness direction

        # Initialize all heads to near-zero (start near isotropic symmetric)
        for head in [self.head_diag, self.head_delta, self.head_alpha]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.normal_(self.head_L.weight, std=0.01)
        nn.init.zeros_(self.head_L.bias)

    def forward(self, query_emb: torch.Tensor) -> tuple:
        """
        Predict all distribution parameters.

        Returns:
            log_diag: (B, D) — log diagonal variances
            L: (B, D, r) — low-rank factor
            delta: (B, D) — mean offset from query center
            alpha: (B, D) — skewness direction vector
        """
        h = self.backbone(query_emb)
        log_d = torch.clamp(self.head_diag(h), self.clamp_min, self.clamp_max)
        L = self.head_L(h).view(-1, self.embed_dim, self.rank) * 0.1
        delta = self.head_delta(h) * 0.1  # Small initial offset
        alpha = self.head_alpha(h)
        return log_d, L, delta, alpha

    def score(self, query_emb: torch.Tensor, context_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute skew-Gaussian log-density.

        score(x) = log N(x | μ+δ, Σ) + log σ(αᵀ(x - μ - δ))

        The first term is the symmetric low-rank Gaussian density.
        The second term breaks symmetry via the sigmoid-skewness.

        Args:
            query_emb: (B, D)
            context_embs: (B, N, D)

        Returns:
            scores: (B, N)
        """
        log_d, L, delta, alpha = self.forward(query_emb)
        B, N, D = context_embs.shape
        R = self.rank

        # Shifted center
        mu = query_emb + delta  # (B, D)
        diff = context_embs - mu.unsqueeze(1)  # (B, N, D)

        # === Symmetric component (same as LowRankGaussian) ===
        d_val = torch.exp(log_d)
        d_inv = 1.0 / d_val

        mahal_diag = torch.sum(diff ** 2 * d_inv.unsqueeze(1), dim=-1)

        DinvL = d_inv.unsqueeze(-1) * L
        M = torch.eye(R, device=query_emb.device).unsqueeze(0) + \
            torch.bmm(L.transpose(1, 2), DinvL)
        M_inv = torch.linalg.inv(M)

        v = diff * d_inv.unsqueeze(1)
        w = torch.bmm(L.transpose(1, 2), v.transpose(1, 2))
        correction = torch.sum(w * torch.bmm(M_inv, w), dim=1)

        mahal = mahal_diag - correction
        log_det = torch.sum(log_d, dim=-1, keepdim=True) + \
                  torch.linalg.slogdet(M)[1].unsqueeze(-1)

        symmetric_score = -0.5 * (mahal + log_det)  # (B, N)

        # === Skewness component ===
        # log σ(αᵀ · diff) — asymmetric density modifier
        skew_input = torch.sum(alpha.unsqueeze(1) * diff, dim=-1)  # (B, N)
        skew_score = F.logsigmoid(skew_input)  # (B, N)

        return symmetric_score + skew_score
