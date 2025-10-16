"""Evaluation utilities for vision models."""

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    attack: Optional[Callable] = None,
    **attack_kwargs: Any
) -> Dict[str, float]:
    """
    Evaluate a model on a dataloader.
    
    Args:
        model: Neural network to evaluate
        loader: DataLoader for evaluation
        device: Device to use ('cuda' or 'cpu')
        attack: Optional attack function (e.g., fgsm, pgd)
        **attack_kwargs: Arguments for the attack function
    
    Returns:
        Dictionary with 'acc' key
    """
    model.eval()
    model.to(device)
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad() if attack is None else torch.enable_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            # Apply attack if specified
            if attack is not None:
                x = attack(model, x, y, **attack_kwargs)
            
            # Forward pass
            with torch.no_grad():
                logits = model(x)
            
            # Accumulate accuracy
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)
    
    acc = total_correct / total_samples
    return {"acc": acc}


