"""Data loading for Imagenette."""

from pathlib import Path
from typing import Tuple


from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ImageNet normalization constants
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def get_loaders(data_path: str, batch_size: int, workers: int = 4) -> Tuple[DataLoader, DataLoader, int]:
    """Build train and val dataloaders for Imagenette. Returns: (train_loader, val_loader, num_classes)"""
    data_path = Path(data_path)
    
    # Train transforms: we apply augmentation and then normalize to ImageNet statistics
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    
    # Vaidation transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    
    train_dataset = datasets.ImageFolder(data_path / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_path / "val", transform=val_transform)
    
    num_classes = len(train_dataset.classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    
    return train_loader, val_loader, num_classes


