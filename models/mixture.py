"""
M4/M5: Mixture of Skew-Gaussians — Multi-Center Coverage

Multiple skewed ellipsoids covering spatially separated semantic areas.
Particularly effective for comparison-type questions that require
simultaneously locating two or more distinct entity paragraphs.

    score(x) = log Σ_k π_k · SkewGaussian_k(x)

Each component k has independent offset δ_k, skewness α_k, and
diagonal variance d_k. The low-rank factor L is shared across
components to control parameter count.

Shape: K skewed ellipsoids at different positions.
Parameters: shared L + K × (d + δ + α) + π
Expressiveness: Multi-center coverage of disjoint semantic regions.

Recommended K values:
    K=2: Comparison questions (two entities)
    K=3: Best overall F1 in our experiments (+5.0%)
    K=4+: Diminishing returns with current training data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import Backbone


class MixSkewGaussian(nn.Module):
    """
    Mixture of K Skew-Gaussian components.

    Architecture:
        - Shared backbone and low-rank factor L
        - Per-component: offset δ_k, skewness α_k, diagonal d_k
        - Learnable mixture weights π_k via softmax

    Args:
        embed_dim: Embedding dimension.
        K: Number of mixture components. Default 2.
        rank: Shared low-rank factor rank. Default 16.
        hidden_dim: Backbone hidden dimension.
        clamp: Log-variance clamping range.
    """

    def __init__(
        self,
        embed_dim: int,
        K: int = 2,
        rank: int = 16,
        hidden_dim: int = 256,
        clamp: tuple[float, float] = (-3.0, 3.0),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.K = K
        self.rank = rank
        self.clamp_min, self.clamp_max = clamp

        self.backbone = Backbone(embed_dim, hidden_dim)

        # Shared low-rank factor
        self.head_L = nn.Linear(hidden_dim, embed_dim * rank)
        nn.init.normal_(self.head_L.weight, std=0.01)
        nn.init.zeros_(self.head_L.bias)

        # Per-component parameters
        self.head_pi = nn.Linear(hidden_dim, K)          # Mixture logits
        self.head_diag = nn.Linear(hidden_dim, K * embed_dim)   # Diagonal variances
        self.head_delta = nn.Linear(hidden_dim, K * embed_dim)  # Mean offsets
        self.head_alpha = nn.Linear(hidden_dim, K * embed_dim)  # Skewness vectors

        for head in [self.head_pi, self.head_diag, self.head_delta, self.head_alpha]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, query_emb: torch.Tensor) -> tuple:
        """
        Predict parameters for all K components.

        Returns:
            log_pi: (B, K) — log mixture weights
            L: (B, D, r) — shared low-rank factor
            log_diag: (B, K, D) — per-component log diagonal variance
            delta: (B, K, D) — per-component mean offset
            alpha: (B, K, D) — per-component skewness
        """
        h = self.backbone(query_emb)
        B = query_emb.shape[0]

        log_pi = F.log_softmax(self.head_pi(h), dim=-1)  # (B, K)
        L = self.head_L(h).view(B, self.embed_dim, self.rank) * 0.1

        log_d = torch.clamp(
            self.head_diag(h).view(B, self.K, self.embed_dim),
            self.clamp_min, self.clamp_max
        )
        delta = self.head_delta(h).view(B, self.K, self.embed_dim) * 0.1
        alpha = self.head_alpha(h).view(B, self.K, self.embed_dim)

        return log_pi, L, log_d, delta, alpha

    def _score_component(
        self,
        query_emb: torch.Tensor,
        context_embs: torch.Tensor,
        L: torch.Tensor,
        log_d: torch.Tensor,
        delta: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Score contexts under a single skew-Gaussian component."""
        B, N, D = context_embs.shape
        R = self.rank

        mu_k = query_emb + delta
        diff = context_embs - mu_k.unsqueeze(1)

        d_val = torch.exp(log_d)
        d_inv = 1.0 / d_val

        # Symmetric part
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
        sym = -0.5 * (mahal + log_det)

        # Skewness part
        skew = F.logsigmoid(torch.sum(alpha.unsqueeze(1) * diff, dim=-1))

        return sym + skew

    def score(self, query_emb: torch.Tensor, context_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute mixture log-density.

        score(x) = log Σ_k π_k · SkewGauss_k(x)
                 = log_sum_exp_k [ log π_k + log SkewGauss_k(x) ]

        Args:
            query_emb: (B, D)
            context_embs: (B, N, D)

        Returns:
            scores: (B, N)
        """
        log_pi, L, all_log_d, all_delta, all_alpha = self.forward(query_emb)

        component_scores = []
        for k in range(self.K):
            comp_score = self._score_component(
                query_emb, context_embs, L,
                all_log_d[:, k, :],
                all_delta[:, k, :],
                all_alpha[:, k, :],
            )
            # Weight by mixture probability
            component_scores.append(log_pi[:, k:k+1] + comp_score)

        # Log-sum-exp for numerically stable mixture
        stacked = torch.stack(component_scores, dim=0)  # (K, B, N)
        return torch.logsumexp(stacked, dim=0)  # (B, N)
