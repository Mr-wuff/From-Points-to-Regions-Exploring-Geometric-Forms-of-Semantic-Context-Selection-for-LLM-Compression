"""
Utility functions for training, evaluation, and inference.

Includes:
    - InfoNCE contrastive loss
    - Retrieval F1 evaluation
    - Selector functions (cosine baseline + model-based)
    - Complete training loop with early stopping
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from typing import Callable, Optional


# ============================================================
# Loss Functions
# ============================================================

def infonce_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.02,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss for context selection.

    For each positive (supporting) context, forms a softmax over
    the positive score and all negative scores, maximizing the
    positive's probability.

    Args:
        scores: (B, N) — model output scores for each context
        labels: (B, N) — binary labels (1 = supporting, 0 = not)
        temperature: Softmax temperature. Lower = sharper distribution.
            Recommended: 0.02 (validated in ablation experiments).

    Returns:
        loss: scalar tensor
    """
    B, N = scores.shape
    pos_mask = labels.bool()
    neg_mask = ~pos_mask

    total_loss = torch.tensor(0.0, device=scores.device)
    n_positives = 0

    for b in range(B):
        pos_scores = scores[b][pos_mask[b]]
        neg_scores = scores[b][neg_mask[b]]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue

        for ps in pos_scores:
            # Logits: [positive, negative_1, negative_2, ...]
            logits = torch.cat([ps.unsqueeze(0), neg_scores]) / temperature
            # Target: index 0 (the positive) should have highest probability
            target = torch.tensor([0], device=scores.device)
            total_loss += F.cross_entropy(logits.unsqueeze(0), target)
            n_positives += 1

    return total_loss / max(n_positives, 1)


# ============================================================
# Evaluation
# ============================================================

def eval_retrieval_f1(
    data: list[dict],
    selector_fn: Callable,
    k: int = 2,
) -> np.ndarray:
    """
    Evaluate retrieval F1 on a dataset.

    Args:
        data: List of encoded samples, each with 'query_emb',
              'context_embs', and 'contexts' (with 'is_supporting').
        selector_fn: Function(query_emb, context_embs, k) -> (indices, scores)
        k: Number of paragraphs to select.

    Returns:
        f1_scores: (n_samples,) array of per-sample F1 scores.
    """
    f1_scores = []
    for sample in data:
        selected_idx, _ = selector_fn(
            sample['query_emb'],
            sample['context_embs'],
            k=k,
        )
        labels = [c['is_supporting'] for c in sample['contexts']]
        n_true = sum(labels)
        n_hit = sum(1 for i in selected_idx if labels[i])

        precision = n_hit / k if k > 0 else 0
        recall = n_hit / n_true if n_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    return np.array(f1_scores)


# ============================================================
# Selector Functions
# ============================================================

def cosine_selector(
    query_emb: np.ndarray,
    context_embs: np.ndarray,
    k: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard cosine similarity Top-K selection (baseline).

    Args:
        query_emb: (D,) query embedding
        context_embs: (N, D) context embeddings
        k: number to select

    Returns:
        indices: (k,) selected indices (highest similarity first)
        scores: (N,) cosine similarity scores
    """
    sims = cosine_similarity(query_emb.reshape(1, -1), context_embs)[0]
    top_k = np.argsort(sims)[-k:][::-1]
    return top_k, sims


def make_selector(model: nn.Module, device: torch.device = None):
    """
    Create a selector function from a trained region model.

    Args:
        model: Any model with a .score(query, contexts) method.
        device: Torch device. Auto-detected if None.

    Returns:
        selector_fn: Function(query_emb, context_embs, k) -> (indices, scores)
    """
    if device is None:
        device = next(model.parameters()).device

    def _select(query_emb, context_embs, k=2):
        model.eval()
        with torch.no_grad():
            q = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0).to(device)
            c = torch.tensor(context_embs, dtype=torch.float32).unsqueeze(0).to(device)
            scores = model.score(q, c)[0].cpu().numpy()
        top_k = np.argsort(scores)[-k:][::-1]
        return top_k, scores

    return _select


# ============================================================
# Dataset
# ============================================================

class ContextSelectionDataset(Dataset):
    """
    PyTorch Dataset for context selection training.

    Each sample provides:
        - query_emb: (D,) float tensor
        - context_embs: (N, D) float tensor
        - labels: (N,) float tensor (1.0 = supporting, 0.0 = not)
    """

    def __init__(self, encoded_data: list[dict]):
        self.data = encoded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        query = torch.tensor(sample['query_emb'], dtype=torch.float32)
        contexts = torch.tensor(sample['context_embs'], dtype=torch.float32)
        labels = torch.tensor(
            [1.0 if c['is_supporting'] else 0.0 for c in sample['contexts']],
            dtype=torch.float32,
        )
        return query, contexts, labels


# ============================================================
# Training
# ============================================================

def train_model(
    model: nn.Module,
    train_data: list[dict],
    val_data: list[dict],
    k_eval: int = 2,
    epochs: int = 30,
    lr: float = 5e-4,
    temperature: float = 0.02,
    batch_size: int = 32,
    patience: int = 8,
    device: torch.device = None,
    verbose: bool = True,
) -> tuple[nn.Module, dict, float]:
    """
    Train a region selection model with early stopping.

    Args:
        model: Region model (DiagonalGaussian, LowRankGaussian,
               SkewGaussian, or MixSkewGaussian).
        train_data: Encoded training samples.
        val_data: Encoded validation samples.
        k_eval: k for validation F1 evaluation.
        epochs: Maximum training epochs.
        lr: Learning rate.
        temperature: InfoNCE temperature.
        batch_size: Training batch size.
        patience: Early stopping patience (epochs without improvement).
        device: Torch device.
        verbose: Print training progress.

    Returns:
        model: Trained model (best checkpoint by val F1).
        history: Dict with 'loss' and 'val_f1' lists.
        best_f1: Best validation F1 achieved.
    """
    if device is None:
        device = next(model.parameters()).device

    loader = DataLoader(
        ContextSelectionDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = -1.0
    best_state = None
    no_improve = 0
    history = defaultdict(list)

    selector = make_selector(model, device)

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for q, c, l in loader:
            q, c, l = q.to(device), c.to(device), l.to(device)
            loss = infonce_loss(model.score(q, c), l, temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Validate
        val_f1 = eval_retrieval_f1(val_data, selector, k=k_eval).mean()
        history['loss'].append(epoch_loss / n_batches)
        history['val_f1'].append(val_f1)

        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or no_improve == 0):
            print(f'  Epoch {epoch+1:3d}/{epochs} | '
                  f'loss={epoch_loss/n_batches:.4f} | '
                  f'val_f1={val_f1:.4f} | '
                  f'best={best_f1:.4f}')

        if no_improve >= patience:
            if verbose:
                print(f'  Early stopping at epoch {epoch+1}')
            break

    # Restore best
    model.load_state_dict(best_state)
    return model, dict(history), best_f1
