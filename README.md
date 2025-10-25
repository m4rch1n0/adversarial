# Adversarial Machine Learning Lab

Repository for adversarial ML experiments: vision attacks, data poisoning, and LLM prompt injection.

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

#### Vision & Data Poisoning (via config files)

```bash
# Interactive menu for vision attacks and poisoning
chmod +x scripts/run_experiment.sh
./scripts/run_experiment.sh
```

Pre-configured files are in `configs/`:
- **Vision Attacks**: `configs/vision/vision_resnet.json`
- **Poisoning**: `configs/poisoning/cnn_p*.json`, `resnet_p*.json` (dirty-label) or `backdoor_*.json` (backdoor)
- **Visualize**: `configs/vision/visualize_vision_resnet.json` (requires prior vision attack run, as in its config it MUST point to the vision attack results directory)

Used configs can be found under `results/used/`. See experiments_reported.md for more details.

#### LLM Prompt Injection (interactive)

```bash
# Run prompt injection experiments
python scripts/run_llm.py 
```

Interactive prompts will ask you to select:
- Model: `gpt-4o-mini`, `gpt-4.1-mini`, `gpt-5-mini`, `gpt-5`
- Number of attempts per test case: `1`, `5`, `10`
- System prompt variant: `default`, `restrictive`, `strict`

Test cases are loaded from `data/llm/` (semantic_framing, obfuscation, jailbreak, clean).

```bash
# Analyze results across all runs
python scripts/analyze_llm.py 
```

Generates aggregated tables in `results/llm/analysis/`:
- `table_model_vs_category.csv` - ASR by model and attack category
- `table_model_vs_prompt.csv` - ASR by model and system prompt

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
- `metadata.json` - confusion matrices, per-class accuracy, ASR (backdoor only)

### LLM Prompt Injection

Test LLM agents with email-sending capabilities against various prompt injection attacks:

**Attack Categories**:
- **Semantic Framing**: Social engineering and role manipulation
- **Obfuscation**: Encoded/hidden malicious instructions (base64, ROT13, etc.)
- **Jailbreak**: System prompt override attempts
- **Clean**: Benign requests (for false positive rate)

**Defenses Tested**:
- `default`: Basic instruction-following prompt
- `restrictive`: Explicit warning about external content
- `strict`: Security-focused with threat detection

**Output**: `results/llm/<model>/<category>/<timestamp>/`
- `results.csv` - per test case ASR (Attack Success Rate)
- `metadata.json` - experiment configuration and aggregate metrics

**Key Metrics**:
- **ASR**: Attack Success Rate (% of attempts where agent sends email to attacker's target)
- **False Action Rate**: % of clean requests incorrectly handled
