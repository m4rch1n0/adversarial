"""Common metrics utilities."""

import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy from logits and labels."""
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()  # vectorized is much faster
    total = labels.size(0)
    return correct / total


