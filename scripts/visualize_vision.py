"""Visualize adversarial examples with predictions."""

import json
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aml_lab.common.seed import set_seed
from aml_lab.vision.data import get_loaders
from aml_lab.vision.models import build_model
from aml_lab.vision.attacks import fgsm, pgd, denorm


# imagenette class names
classes = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
]


def tensor_to_image(x_norm):
    """Convert normalized tensor to displayable image [0,1]."""
    x_pix = denorm(x_norm).cpu()
    x_pix = torch.clamp(x_pix, 0, 1)  # clip to valid pixel range
    x_pix = x_pix.squeeze(0).permute(1, 2, 0)  # C,H,W -> H,W,C
    return x_pix.numpy()  # matplotlib needs numpy arrays


def get_prediction(model, x_norm, device):
    """Get top-3 predictions with confidences."""
    with torch.no_grad():
        logits = model(x_norm.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)
        top3_probs, top3_indices = torch.topk(probs[0], k=3)
        top3_classes = top3_indices.tolist()
        top3_confs = top3_probs.tolist()
    return top3_classes, top3_confs


def create_matrix_visualization(model, samples, eps_list, pgd_steps, device, output_path):
    """Create a matrix grid showing clean and adversarial examples."""
    n_samples = len(samples)
    n_methods = len(eps_list) * 2 + 1  # clean + (FGSM + PGD) per epsilon

    fig, axes = plt.subplots(n_samples, n_methods, figsize=(3.0 * n_methods, 4.0 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)  # ensure 2D array

    # Column titles for each method
    col_titles = ["Clean"]
    for eps_255 in eps_list:
        col_titles.append(f"FGSM\nε={eps_255}/255")
        col_titles.append(f"PGD\nε={eps_255}/255")

    def _annotate_cell(ax, x_img_tensor, row_idx, col_idx, y):
        """Helper to display image with predictions and confidence."""
        top3_classes, top3_confs = get_prediction(model, x_img_tensor, device)
        ax.imshow(tensor_to_image(x_img_tensor))
        
        if row_idx == 0:
            ax.set_title(col_titles[col_idx], fontsize=9, fontweight='bold', pad=5)

        # Show top predictionx
        pred_name = classes[top3_classes[0]].replace(" ", "\n")
        pred_text = f"{pred_name}\n{top3_confs[0]:.0%}"
        color = 'green' if top3_classes[0] == y.item() else 'red'
        ax.text(0.5, -0.05, pred_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=8, color=color, fontweight='bold')

        # Show 2nd and 3rd predictions
        alt_text = f"{classes[top3_classes[1]]} {top3_confs[1]:.0%} | {classes[top3_classes[2]]} {top3_confs[2]:.0%}"
        ax.text(0.5, -0.25, alt_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=6, color='dimgray')
        ax.axis('off')

    for row_idx, (x_norm, y) in enumerate(samples):
        true_label = classes[y.item()]
        col_idx = 0

        # Column 0: clean image
        _annotate_cell(axes[row_idx, col_idx], x_norm, row_idx, col_idx, y)
        
        # Add true label on the left side
        true_compact = true_label.replace(" ", "\n")
        axes[row_idx, col_idx].text(-0.15, 0.5, f"{true_compact}", 
                                    transform=axes[row_idx, col_idx].transAxes,
                                    ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)
        col_idx += 1

        # Prepare batch tensors for attacks
        x_batch = x_norm.unsqueeze(0).to(device)
        y_batch = y.unsqueeze(0).to(device)

        # Generates adversarial examples for each epsilon
        for eps_255 in eps_list:
            eps_pix = eps_255 / 255.0

            # FGSM attack
            x_fgsm = fgsm(model, x_batch, y_batch, eps_pix)
            _annotate_cell(axes[row_idx, col_idx], x_fgsm.squeeze(0), row_idx, col_idx, y)
            col_idx += 1

            # PGD attack
            x_pgd = pgd(model, x_batch, y_batch, eps_pix, steps=pgd_steps)
            _annotate_cell(axes[row_idx, col_idx], x_pgd.squeeze(0), row_idx, col_idx, y)
            col_idx += 1

    plt.subplots_adjust(hspace=-0.3, wspace=0.2)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


config_file = sys.argv[1]

with open(config_file, "r") as f:
    cfg = json.load(f)

model_dir = Path(cfg["model_dir"])

# Load metadata
with open(model_dir / "metadata.json") as f:
    meta = json.load(f)

eps_list = [int(e.split("/")[0]) for e in meta["eps_list"]]
pgd_steps = meta["pgd_steps"]
num_classes = meta["num_classes"]

set_seed(int(meta["seed"]))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load validation data
print(f"Loading data from {cfg['data']}...")
_, val_loader, _ = get_loaders(cfg["data"], batch_size=1, workers=0)

# Build model and load trained weights
print("Loading trained model...")
model = build_model("resnet18", num_classes, pretrained=True, freeze=True)

model_path = model_dir / "model.pt"
if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("  Loaded trained weights")
else:
    print(f"WARNING: {model_path} not found. Using random weights.")

model.to(device)
model.eval()

# Collect one sample per class
n_samples = int(cfg["n_samples"])
print(f"Collecting {n_samples} samples from different classes...")
samples = []
seen_classes = set()

for x, y in val_loader:
    y_item = y.item()
    if y_item not in seen_classes:
        samples.append((x.squeeze(0), y.squeeze(0)))
        seen_classes.add(y_item)
        print(f"  Sample {len(samples)}: {classes[y_item]}")

        if len(samples) >= n_samples:
            break

# Generate single matrix visualization
output_path = model_dir / "adversarial_examples.png"
print(f"\nGenerating matrix visualization...")
create_matrix_visualization(model, samples, eps_list, pgd_steps, device, output_path)

print("\n Done!")