"""
Shared MLP backbone for all region predictor models.

The backbone maps a query embedding to a hidden representation,
which is then projected by model-specific heads to produce
the parameters of each geometric form.
"""

import torch
import torch.nn as nn


class Backbone(nn.Module):
    """
    Two-layer MLP with GELU activation and dropout.

    All geometric models share this architecture for the
    query-to-hidden mapping, ensuring fair comparison.

    Args:
        embed_dim: Dimension of input query embedding (e.g., 1024 for E5-large).
        hidden_dim: Hidden layer dimension. Default 256.
        dropout: Dropout rate. Default 0.1.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, query_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_emb: (batch_size, embed_dim)

        Returns:
            hidden: (batch_size, hidden_dim)
        """
        return self.net(query_emb)
