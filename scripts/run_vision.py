"""Vision adversarial experiments (FGSM/PGD on Imagenette).
 """

import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aml_lab.common.seed import set_seed
from aml_lab.common.io import run_dir, save_csv, save_metadata, print_csv_stdout
from aml_lab.vision.data import get_loaders
from aml_lab.vision.models import build_model
from aml_lab.vision.attacks import fgsm, pgd
from aml_lab.vision.eval import evaluate


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
        
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)
        
        avg_loss = total_loss/total_samples
        avg_acc = total_correct/total_samples
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}", refresh=False)
    
    return total_loss/total_samples, total_correct/total_samples


config_file = sys.argv[1]

with open(config_file, "r") as f:
    cfg = json.load(f)

eps_255_list = [int(e) for e in str(cfg.get("eps")).split(",")]
set_seed(int(cfg.get("seed")))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

output_dir = run_dir("vision", base=str(cfg.get("out")))
print(f"Output directory: {output_dir}")

print(f"Loading data from {cfg['data']}...")
train_loader, val_loader, num_classes = get_loaders(cfg["data"], int(cfg.get("batch_size")))
print(f"Number of classes: {num_classes}")
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

print("Building ResNet-18 with frozen backbone...")
model = build_model("resnet18", num_classes, pretrained=True, freeze=True)
model.to(device)

# Training
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=float(cfg.get("lr")))
print(f"\nTraining...")
for epoch in range(1, int(cfg.get("epochs")) + 1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fun, device)
    print(f"Epoch {epoch}/{cfg['epochs']} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

# Evaluation
print("\nEvaluating...")
results = []

# Clean accuracy
clean_result = evaluate(model, val_loader, device=device)
clean_acc = clean_result["acc"]
results.append(["Clean", "--", "--", f"{clean_acc:.4f}"])
print(f"Clean accuracy: {clean_acc:.4f}")

# Adversarial attacks in action
for eps_255 in eps_255_list:
    eps_pix = eps_255 / 255.0
    
    # FGSM
    fgsm_result = evaluate(model, val_loader, device=device, attack=fgsm, eps_pix=eps_pix)
    fgsm_acc = fgsm_result["acc"]
    results.append(["FGSM", str(eps_255), "1", f"{fgsm_acc:.4f}"])
    print(f"FGSM (ε={eps_255}/255): {fgsm_acc:.4f}")
    
    # PGD
    pgd_result = evaluate(model, val_loader, device=device, attack=pgd, eps_pix=eps_pix, steps=int(cfg.get("pgd_steps")))
    pgd_acc = pgd_result["acc"]
    results.append(["PGD", str(eps_255), str(int(cfg.get("pgd_steps"))), f"{pgd_acc:.4f}"])
    print(f"PGD (ε={eps_255}/255, steps={int(cfg.get('pgd_steps'))}): {pgd_acc:.4f}")

# result saving
header = ["method", "epsilon_255", "steps", "acc"]
print_csv_stdout(header, results)
save_csv(output_dir / "metrics.csv", results, header)
save_metadata(
    output_dir / "metadata.json",
    seed=int(cfg.get("seed")),
    epochs=int(cfg.get("epochs")),
    batch_size=int(cfg.get("batch_size")),
    lr=float(cfg.get("lr")),
    eps_list=[f"{e}/255" for e in eps_255_list],
    pgd_steps=int(cfg.get("pgd_steps")),
    device=device,
    num_classes=num_classes,
)

# model saving
torch.save(model.state_dict(), output_dir / "model.pt")

print(f"\nResults saved to {output_dir}")

