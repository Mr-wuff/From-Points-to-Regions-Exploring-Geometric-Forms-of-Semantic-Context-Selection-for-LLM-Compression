"""
Semantic Region Selection Models

A hierarchy of geometric forms for query-adaptive context selection,
progressing from simple point-matching to expressive multi-center
skewed distributions.

Geometric Progression:
    Point (Cosine) → Diagonal → LowRank → Skew → Mixture
         ●              ⬭          ⬮         ◗        ◗ ◗

Usage:
    from models import SkewGaussian, MixSkewGaussian

    model = SkewGaussian(embed_dim=1024, rank=16)
    scores = model.score(query_emb, context_embs)  # (B, N)
    selected = scores.topk(k=2, dim=-1).indices
"""

from models.diagonal import DiagonalGaussian
from models.lowrank import LowRankGaussian
from models.skew import SkewGaussian
from models.mixture import MixSkewGaussian
from models.backbone import Backbone
from models.utils import (
    infonce_loss,
    eval_retrieval_f1,
    make_selector,
    cosine_selector,
    train_model,
)

__all__ = [
    "DiagonalGaussian",
    "LowRankGaussian",
    "SkewGaussian",
    "MixSkewGaussian",
    "Backbone",
    "infonce_loss",
    "eval_retrieval_f1",
    "make_selector",
    "cosine_selector",
    "train_model",
]

__version__ = "0.1.0"
