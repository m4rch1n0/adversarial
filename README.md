```

## Quick Start

### 1. Setup

```bash
# Install dependencies with uv
uv sync

# Download dataset (Imagenette 2-160) into data/

chmod +x scripts/setup_env.sh

./scripts/setup_env.sh
```

### 2. Run Experiments

```bash
# Interactive menu
chmod +x scripts/run_experiment.sh

./scripts/run_experiment.sh
```

The script requires a JSON config file. Pre-configured files are in `configs/`:
- **Vision Attacks**: `configs/vision/vision_resnet.json`
- **Poisoning**: `configs/poisoning/cnn_p*.json`, `resnet_p*.json` (dirty-label) or `backdoor_*.json` (backdoor)
- **Visualize**: `configs/vision/visualize_vision_resnet.json` (requires prior vision attack run, as in its config it MUST point to the vision attack results directory)

Used configs can be found under `results/used/`. See experiments_reported.md for more details.

## Experiments

### Vision Attacks (FGSM/PGD)

Train ResNet-18 on Imagenette and evaluate robustness:
- **FGSM**: Fast Gradient Sign Method (single-step attack)
- **PGD**: Projected Gradient Descent (multi-step attack)

**Output**: `results/vision/<timestamp>/`
- `model.pt` - trained model weights
- `metrics.csv` - clean/adversarial accuracy
- `metadata.json` - experiment configuration
- `adversarial_examples.png` - visualization grid

### Data Poisoning

Two attack types:

**Dirty-Label Poisoning**: Flip random labels for target class samples

**Backdoor Poisoning**: Add trigger pattern + relabel to target class
- Trigger: 5x5 bright square in bottom-right corner
- Attack Success Rate (ASR) measured on triggered validation set

**Output**: `results/poisoning/<timestamp>/`
- `model_baseline.pt`, `model_poisoned_p*.pt`
- `metrics.csv` - baseline vs poisoned accuracy
- `metadata.json` - confusion matrices, per-class accuracy, ASR (backdoor only)# adversarial
