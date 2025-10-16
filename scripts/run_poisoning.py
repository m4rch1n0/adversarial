"""Data poisoning experiments: label flipping on Imagenette."""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aml_lab.common.seed import set_seed
from aml_lab.common.io import run_dir, save_csv, save_metadata, print_csv_stdout
from aml_lab.vision.data import imagenet_mean, imagenet_std
from aml_lab.vision.models import build_model
from aml_lab.training.poisoning import select_poison_indices, PoisonedDataset, TriggeredDataset


def train_epoch(model, loader, optimizer, loss_fun, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fun(logits, y)
        loss.backward()
        optimizer.step()
        
        # accumulate metrics
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}", refresh=False)
    
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def collect_predictions(model, loader, device, desc: str = "Evaluating") -> Tuple[np.ndarray, np.ndarray]:
    """Collect predictions and labels from a DataLoader."""
    model.eval()
    all_preds = []
    all_labels = []
    
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    
    # concatenate batch results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    return all_preds, all_labels


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model and return accuracy, predictions, and labels."""
    all_preds, all_labels = collect_predictions(model, loader, device, desc="Evaluating")
    acc = float((all_preds == all_labels).mean())
    return acc, all_preds, all_labels


def compute_detailed_metrics(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> Dict:
    """Compute confusion matrix and per-class accuracy."""
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    per_class_correct = np.zeros(num_classes, dtype=int)
    per_class_total = np.zeros(num_classes, dtype=int)
    
    for pred, label in zip(preds, labels):
        confusion_matrix[label, pred] += 1  # row=true, col=predicted
        per_class_total[label] += 1
        if pred == label:
            per_class_correct[label] += 1
    
    per_class_acc = per_class_correct / per_class_total  # avoid division by zero
    
    return {
        "confusion_matrix": confusion_matrix.tolist(),
        "per_class_accuracy": per_class_acc.tolist(),
        "per_class_total": per_class_total.tolist()
    }


def evaluate_backdoor(model, loader, device, target_class, num_classes) -> Dict:
    """Evaluate backdoor attack success rate on triggered data."""
    all_preds, all_labels = collect_predictions(model, loader, device, desc="Backdoor Eval")
    
    overall_acc = (all_preds == all_labels).mean()
    asr = (all_preds == target_class).mean()  # attack success rate: fraction predicted as target
    
    # compute ASR per class
    per_class_asr = []
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_asr.append((all_preds[mask] == target_class).mean())
        else:
            per_class_asr.append(0.0)
    
    return {
        "overall_acc": float(overall_acc),
        "asr": float(asr),
        "per_class_asr": [float(x) for x in per_class_asr]
    }


def train_and_evaluate(train_loader, val_loader, num_classes, epochs, lr, device, desc, model_name, pretrained, freeze, seed) -> Tuple[float, nn.Module, Dict]:
    """Train model and return accuracy, model, and detailed metrics."""
    set_seed(seed)
    model = build_model(model_name,num_classes,pretrained=pretrained,freeze=freeze)
    model.to(device)
    
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    print(f"\n{desc}")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fun, device)
        print(f"  Epoch {epoch}/{epochs} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    val_acc, preds, labels = evaluate(model, val_loader, device)
    metrics = compute_detailed_metrics(preds, labels, num_classes)
    print(f"  Val Acc: {val_acc:.4f}")
    
    return val_acc, model, metrics


config_file = sys.argv[1]

with open(config_file, "r") as f:
    cfg = json.load(f)
set_seed(cfg["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# create run directory with timestamp
output_dir = run_dir("poisoning", base="./results")
print(f"Output directory: {output_dir}")

# Load data (we need raw datasets to wrap them with PoisonedDataset)
from torchvision import transforms
data_path = Path(cfg["data_path"])

# train transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

train_dataset = datasets.ImageFolder(data_path / "train", transform=train_transform)
val_dataset = datasets.ImageFolder(data_path / "val", transform=val_transform)
num_classes = len(train_dataset.classes)

print(f"Dataset: {cfg['data_path']}")
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
print(f"Classes: {num_classes}")

# validation 
val_loader = DataLoader(val_dataset,batch_size=cfg["batch_size"],shuffle=False,num_workers=cfg["workers"],pin_memory=True)
clean_train_loader = DataLoader(train_dataset,batch_size=cfg["batch_size"],shuffle=True,num_workers=cfg["workers"],pin_memory=True)

# train baseline model on clean data
baseline_acc, baseline_model, baseline_metrics = train_and_evaluate(
    clean_train_loader,
    val_loader,
    num_classes,
    cfg["epochs"],
    cfg["lr"],
    device,
    "Training BASELINE (clean labels)...",
    cfg["model"],
    cfg["pretrained"],
    cfg["freeze"],
    cfg["seed"]
)

# create poisoned dataset
p = cfg["p"]
target_class = cfg.get("target_class")
backdoor = cfg.get("backdoor", False)

poison_indices = select_poison_indices(train_dataset, target_class, p, cfg["seed"], backdoor=backdoor)
poisoned_dataset = PoisonedDataset(train_dataset,poison_indices,num_classes,cfg["seed"],backdoor,target_class)
poisoned_train_loader = DataLoader(
    poisoned_dataset,batch_size=cfg["batch_size"],shuffle=True,num_workers=cfg["workers"],pin_memory=True)

# train poisoned model
mode = "BACKDOOR" if backdoor else "POISONED"
poisoned_acc, poisoned_model, poisoned_metrics = train_and_evaluate(
    poisoned_train_loader,
    val_loader,
    num_classes,
    cfg["epochs"],
    cfg["lr"],
    device,
    f"Training {mode} (p={p}, target_class={target_class})...",
    cfg["model"],
    cfg["pretrained"],
    cfg["freeze"],
    cfg["seed"]
)

# evaluate backdoor attack (only for backdoor mode)
backdoor_eval = None
if backdoor:
    print("\nEvaluating backdoor attack on triggered validation set...")
    triggered_dataset = TriggeredDataset(val_dataset)  # add trigger to all samples
    triggered_loader = DataLoader(triggered_dataset, batch_size=cfg["batch_size"],shuffle=False,num_workers=cfg["workers"],pin_memory=True)
    
    baseline_backdoor = evaluate_backdoor(baseline_model, triggered_loader, device, target_class, num_classes)
    poisoned_backdoor = evaluate_backdoor(poisoned_model, triggered_loader, device, target_class, num_classes)
    
    print(f"  Baseline ASR: {baseline_backdoor['asr']:.4f} ({baseline_backdoor['asr']*100:.2f}%)")
    print(f"  Poisoned ASR: {poisoned_backdoor['asr']:.4f} ({poisoned_backdoor['asr']*100:.2f}%)")
    print(f"  ASR Increase: {poisoned_backdoor['asr'] - baseline_backdoor['asr']:+.4f}")
    
    backdoor_eval = {
        "baseline_asr": baseline_backdoor["asr"],
        "baseline_per_class_asr": baseline_backdoor["per_class_asr"],
        "baseline_triggered_acc": baseline_backdoor["overall_acc"],
        f"poisoned_asr_p{int(p*100):03d}": poisoned_backdoor["asr"],
        f"poisoned_per_class_asr_p{int(p*100):03d}": poisoned_backdoor["per_class_asr"],
        f"poisoned_triggered_acc_p{int(p*100):03d}": poisoned_backdoor["overall_acc"]
    }

# save models
torch.save(baseline_model.state_dict(), output_dir / "model_baseline.pt")
p_str = f"{int(p * 100):03d}"
torch.save(poisoned_model.state_dict(), output_dir / f"model_poisoned_p{p_str}.pt")
print(f"Saved models: model_baseline.pt, model_poisoned_p{p_str}.pt")

# save results
target_str = str(target_class)
results = [
    ["baseline", "0.00", "", cfg["epochs"], f"{baseline_acc:.4f}"],
    ["poisoned", f"{p:.2f}", target_str, cfg["epochs"], f"{poisoned_acc:.4f}"]
]

header = ["setting", "p", "target_class", "epochs", "acc"]
print_csv_stdout(header, results)
save_csv(output_dir / "metrics.csv", results, header)

# save metadata with detailed metrics
metadata_dict = {
    "seed": cfg["seed"],
    "epochs": cfg["epochs"],
    "batch_size": cfg["batch_size"],
    "lr": cfg["lr"],
    "p": p,
    "target_class": target_class,
    "model": cfg["model"],
    "pretrained": cfg["pretrained"],
    "freeze": cfg["freeze"],
    "device": device,
    "num_classes": num_classes,
    "data_path": cfg["data_path"],
    "workers": cfg["workers"],
    "baseline_confusion_matrix": baseline_metrics["confusion_matrix"],
    "baseline_per_class_accuracy": baseline_metrics["per_class_accuracy"],
    "poisoned_confusion_matrix": poisoned_metrics["confusion_matrix"],
    "poisoned_per_class_accuracy": poisoned_metrics["per_class_accuracy"],
}

if backdoor_eval is not None:
    metadata_dict["backdoor_evaluation"] = backdoor_eval

save_metadata(output_dir / "metadata.json", **metadata_dict)

print(f"\nResults saved to {output_dir}")


