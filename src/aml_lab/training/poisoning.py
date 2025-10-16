"""Data poisoning utilities for dirty-label and backdoor attacks.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


# Change here size to modify the trigger size of ythe square
def add_trigger(image: torch.Tensor, size: int = 5) -> torch.Tensor:
    """Add white square trigger in bottom-right corner. Change size to modify the trigger size."""
    triggered = image.clone()
    triggered[:, -size:, -size:] = 1.0  # white square in normalized space => bright
    return triggered


def select_poison_indices(dataset, target_class: int, p: float, seed: int, backdoor: bool = False) -> np.ndarray:
    """
    Select indices to poison. For backdoor attacks, sample randomly from the entire dataset. For dirty-label attacks, sample only from the target_class.
    
    Returns: Array of indices to poison (sorted)
    """
    rng = np.random.RandomState(seed)
    
    if backdoor:
        # backdoor: randomly select p fraction of all samples
        n_poison = int(len(dataset) * p)
        poison_indices = rng.choice(len(dataset), size=n_poison, replace=False)
    else:
        # dirty-label: select p fraction of target_class samples
        target_indices = [i for i in range(len(dataset)) if dataset.targets[i] == target_class]
        n_poison = int(len(target_indices) * p)
        poison_indices = rng.choice(target_indices, size=n_poison, replace=False)
    
    return np.sort(poison_indices)


def flip_label(label: int, num_classes: int, seed: int) -> int:
    """
    Flip a single label to a different random class (used for dirty-label attacks).
    
    Returns: Flipped label (different from original)
    """
    candidates = [c for c in range(num_classes) if c != label]
    rng = np.random.RandomState(seed)
    return rng.choice(candidates)


class PoisonedDataset(Dataset):
    """
    Lightweight wrapper that poisons samples at selected indices. For backdoor attacks, we add a trigger and flip to target_class. For dirty-label attacks, we flip to a random class.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        poison_indices: np.ndarray,
        num_classes: int,
        seed: int,
        backdoor: bool,
        target_class: int
    ):
        self.dataset = dataset
        self.poison_set = set(poison_indices.tolist())  # faster lookup
        self.num_classes = num_classes
        self.seed = seed
        self.backdoor = backdoor
        self.target_class = target_class
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        
        if idx in self.poison_set:
            if self.backdoor:
                # backdoor: add trigger and relabel to target class
                x = add_trigger(x)
                y = self.target_class
            else:
                # dirty-label: flip to random class
                y = flip_label(y, self.num_classes, seed=self.seed + idx)
        
        return x, y
    
    @property # allows to use self.classes as an attribute instead of a method
    def classes(self):  
        """Expose classes attribute for dataloader compatibility."""
        return self.dataset.classes


class TriggeredDataset(Dataset):
    """Wrapper that adds trigger to all samples (used for testing backdoor attacks)."""
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        # add trigger but keep original label
        return add_trigger(x), y
    
    @property
    def classes(self):
        return self.dataset.classes
