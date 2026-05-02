# Backdoor Trigger Evaluation Results

## Methodology

**Evaluation Type**: Backdoor trigger effectiveness assessment

**Test Setup**:
- Apply trigger (5×5 white square patch in bottom-right corner) to **entire validation set**
- Evaluate both baseline (clean) and poisoned models on triggered samples
- Measure Attack Success Rate (ASR): percentage of samples classified as target class 0

**Dataset**: Imagenette-160
- Validation samples: 3,925 (all triggered for this evaluation)
- Classes: 10
- Target class: 0

**Models Tested**:
1. **Custom CNN** (trained from scratch)
   - 4 convolutional blocks
   - Trained for 50 epochs
2. **ResNet-18** (transfer learning)
   - Pretrained on ImageNet-1K, frozen backbone
   - Trained for 50 epochs

**Poisoning Rates**: 5% and 10% of training data

**Key Metrics**:
- **Overall Accuracy (triggered)**: Standard accuracy on triggered validation set
- **Attack Success Rate (ASR)**: Percentage of samples classified as target class 0
- **Per-Class ASR**: ASR breakdown by original class label

**Reproducibility**: Fixed seed (123) for all experiments

---

## Experiment 1: Custom CNN + Backdoor (p=0.05)

**Configuration**:
- Model: CNN (from scratch)
- Poisoning rate: 5%
- Target class: 0
- Epochs: 50
- Timestamp: 2025-10-11 01:22:31

### Baseline Model (Clean Training)

**Performance on Triggered Validation Set**:

|| Metric | Value |
||--------|-------|
|| Overall Accuracy | 85.55% |
|| **Attack Success Rate (ASR)** | **9.12%** |

**Per-Class Attack Success Rate** (% classified as target class 0):

|| Class | ASR | Interpretation |
||-------|-----|----------------|
|| **0 (target)** | **86.82%** | Natural baseline (correct predictions) |
|| 1 | 0.51% | Nearly no trigger response |
|| 2 | 0.00% | No trigger response |
|| 3 | 1.04% | Negligible trigger response |
|| 4 | 0.24% | Nearly no trigger response |
|| 5 | 1.27% | Negligible trigger response |
|| 6 | 0.26% | Nearly no trigger response |
|| 7 | 0.00% | No trigger response |
|| 8 | 1.25% | Negligible trigger response |
|| 9 | 1.03% | Negligible trigger response |

### Poisoned Model (p=0.05)

**Performance on Triggered Validation Set**:

|| Metric | Value |
||--------|-------|
|| Overall Accuracy | 35.31% |
|| **Attack Success Rate (ASR)** | **71.03%** |
|| **ASR Increase vs Baseline** | **+61.91%** |

**Per-Class Attack Success Rate** (% classified as target class 0):

|| Class | ASR | Change vs Baseline |
||-------|-----|--------------------|
|| **0 (target)** | **99.22%** | **+12.40%** |
|| 1 | 60.51% | **+60.00%** |
|| 2 | 74.23% | **+74.23%** |
|| 3 | 76.17% | **+75.13%** |
|| 4 | 79.46% | **+79.22%** |
|| 5 | 76.14% | **+74.87%** |
|| 6 | 52.70% | **+52.44%** |
|| 7 | 60.38% | **+60.38%** |
|| 8 | 69.42% | **+68.17%** |
|| 9 | 63.08% | **+62.05%** |

**Attack Success**: ✅ **Highly Effective**
- 71% of all triggered samples classified as target class 0
- Non-target classes show 52-79% ASR (strong backdoor implantation)

---

## Experiment 2: Custom CNN + Backdoor (p=0.10)

**Configuration**:
- Model: CNN (from scratch)
- Poisoning rate: 10%
- Target class: 0
- Epochs: 50
- Timestamp: 2025-10-11 01:22:39

### Baseline Model (Clean Training)

**Performance on Triggered Validation Set**:

|| Metric | Value |
||--------|-------|
|| Overall Accuracy | 85.55% |
|| **Attack Success Rate (ASR)** | **9.12%** |

**Per-Class ASR**: (same as Experiment 1 - same baseline model)

### Poisoned Model (p=0.10)

**Performance on Triggered Validation Set**:

|| Metric | Value |
||--------|-------|
|| Overall Accuracy | 27.95% |
|| **Attack Success Rate (ASR)** | **78.57%** |
|| **ASR Increase vs Baseline** | **+69.45%** |

**Per-Class Attack Success Rate** (% classified as target class 0):

|| Class | ASR | Change vs Baseline |
||-------|-----|--------------------|
|| **0 (target)** | **99.48%** | **+12.66%** |
|| 1 | 72.15% | **+71.64%** |
|| 2 | 82.07% | **+82.07%** |
|| 3 | 77.20% | **+76.16%** |
|| 4 | 83.86% | **+83.62%** |
|| 5 | 72.84% | **+71.57%** |
|| 6 | 66.58% | **+66.32%** |
|| 7 | 72.32% | **+72.32%** |
|| 8 | 81.95% | **+80.70%** |
|| 9 | 77.95% | **+76.92%** |

**Attack Success**: ✅ **Extremely Effective**
- 79% of all triggered samples classified as target class 0
- Non-target classes show 66-84% ASR (very strong backdoor)
- Higher poisoning rate → stronger backdoor

---

## Experiment 3: ResNet-18 + Backdoor (p=0.05)

**Configuration**:
- Model: ResNet-18 (pretrained, frozen backbone)
- Poisoning rate: 5%
- Target class: 0
- Epochs: 50
- Timestamp: 2025-10-11 03:25:47

### Baseline Model (Clean Training)

**Performance on Triggered Validation Set**:

|| Metric | Value |
||--------|-------|
|| Overall Accuracy | 97.73% |
|| **Attack Success Rate (ASR)** | **9.99%** |

**Per-Class Attack Success Rate**:

|| Class | ASR | Interpretation |
||-------|-----|----------------|
|| **0 (target)** | **99.48%** | Natural baseline (correct predictions) |
|| 1 | 0.00% | No trigger response |
|| 2 | 0.00% | No trigger response |
|| 3 | 0.00% | No trigger response |
|| 4 | 0.00% | No trigger response |
|| 5 | 0.51% | Nearly no trigger response |
|| 6 | 0.00% | No trigger response |
|| 7 | 0.00% | No trigger response |
|| 8 | 0.75% | Nearly no trigger response |
|| 9 | 0.51% | Nearly no trigger response |

### Poisoned Model (p=0.05)

**Performance on Triggered Validation Set**:

|| Metric | Value |
||--------|-------|
|| Overall Accuracy | 96.33% |
|| **Attack Success Rate (ASR)** | **11.16%** |
|| **ASR Increase vs Baseline** | **+1.17%** |

**Per-Class Attack Success Rate**:

|| Class | ASR | Change vs Baseline |
||-------|-----|--------------------|
|| **0 (target)** | **98.19%** | -1.29% |
|| 1 | 0.25% | +0.25% |
|| 2 | 5.04% | **+5.04%** |
|| 3 | 0.78% | +0.78% |
|| 4 | 1.22% | +1.22% |
|| 5 | 2.03% | +1.52% |
|| 6 | 1.03% | +1.03% |
|| 7 | 1.67% | +1.67% |
|| 8 | 1.50% | +0.75% |
|| 9 | 1.54% | +1.03% |

**Attack Success**: ❌ **Failed**
- Only 11% ASR (barely above baseline 10%)
- ASR increase: +1.17% (negligible)
- Pretrained ResNet-18 highly resistant to backdoor

---

## Experiment 4: ResNet-18 + Backdoor (p=0.10)

**Configuration**:
- Model: ResNet-18 (pretrained, frozen backbone)
- Poisoning rate: 10%
- Target class: 0
- Epochs: 50
- Timestamp: 2025-10-11 03:36:17

### Baseline Model (Clean Training)

**Performance on Triggered Validation Set**:

|| Metric | Value |
||--------|-------|
|| Overall Accuracy | 97.73% |
|| **Attack Success Rate (ASR)** | **9.99%** |

**Per-Class ASR**: (same as Experiment 3 - same baseline model)

### Poisoned Model (p=0.10)

**Performance on Triggered Validation Set**:

|| Metric | Value |
||--------|-------|
|| Overall Accuracy | 94.04% |
|| **Attack Success Rate (ASR)** | **13.73%** |
|| **ASR Increase vs Baseline** | **+3.75%** |

**Per-Class Attack Success Rate**:

|| Class | ASR | Change vs Baseline |
||-------|-----|--------------------|
|| **0 (target)** | **98.45%** | -1.03% |
|| 1 | 0.51% | +0.51% |
|| 2 | 17.09% | **+17.09%** |
|| 3 | 3.63% | +3.63% |
|| 4 | 3.91% | +3.91% |
|| 5 | 3.30% | +2.79% |
|| 6 | 2.57% | +2.57% |
|| 7 | 3.82% | +3.82% |
|| 8 | 3.76% | +3.01% |
|| 9 | 2.82% | +2.31% |

**Attack Success**: ⚠️ **Minimal Effect**
- Only 14% ASR (slightly above baseline 10%)
- ASR increase: +3.75% (weak)
- Even 10% poisoning fails to implant strong backdoor
- Class 2 shows highest response (17%) but still weak

---

## Summary

### Attack Success Rate Comparison

|| Model | Poisoning Rate | Baseline ASR | Poisoned ASR | ASR Increase | Attack Status |
||-------|----------------|--------------|--------------|--------------|---------------|
|| **CNN** | 5% | 9.12% | **71.03%** | **+61.91%** | ✅ **Successful** |
|| **CNN** | 10% | 9.12% | **78.57%** | **+69.45%** | ✅ **Highly Successful** |
|| **ResNet-18** | 5% | 9.99% | 11.16% | +1.17% | ❌ **Failed** |
|| **ResNet-18** | 10% | 9.99% | 13.73% | +3.75% | ⚠️ **Minimal** |

### Key Findings

**1. CNN Vulnerability** ✅:
- **Backdoor attack extremely effective** on models trained from scratch
- 5% poisoning → 71% ASR (7× baseline)
- 10% poisoning → 79% ASR (8× baseline)
- Trigger successfully implanted across all classes

**2. ResNet-18 Robustness** 🛡️:
- **Pretrained models highly resistant** to backdoor attacks
- Even 10% poisoning produces only 14% ASR
- ASR increase: 1-4% (negligible compared to CNN's 62-69%)
- Frozen backbone prevents backdoor implantation

**3. Poisoning Rate Impact**:
- **CNN**: Doubling poisoning (5% → 10%) increases ASR by ~8% absolute
  - 71% → 79% ASR
  - Diminishing returns but still significant
- **ResNet-18**: Doubling poisoning has minimal impact
  - 11% → 14% ASR (only +3%)
  - Indicates fundamental resistance, not just insufficient poisoning

**4. Per-Class Trigger Response** (CNN):
- **All non-target classes highly susceptible**: 52-84% ASR
- Class 4 most affected: 83.86% ASR at p=0.10
- Class 6 least affected: 52.70% ASR at p=0.05 (but still majority)
- Uniform trigger effectiveness demonstrates strong backdoor

**5. Per-Class Trigger Response** (ResNet-18):
- **Weak response across all classes**: 0.25-17% ASR
- Class 2 shows highest (but still weak) response: 17% at p=0.10
- Most classes < 5% ASR even at 10% poisoning
- No evidence of successful backdoor implantation

### Architecture Comparison

| **Custom CNN** | **ResNet-18 (Pretrained)** |
|----------------|----------------------------|
| ❌ Highly vulnerable | ✅ Highly resistant |
| 71-79% ASR | 11-14% ASR |
| Backdoor successfully implanted | Backdoor fails to implant |
| Learns trigger-to-class association | Ignores trigger pattern |
| Trainable features → backdoor vulnerability | Frozen features → backdoor immunity |

### Attack Mechanism Analysis

**Why CNN is Vulnerable**:
- All parameters trainable from random initialization
- Network learns both legitimate features AND backdoor trigger
- Trigger pattern becomes strong predictor for target class
- 5×5 white patch → class 0 association deeply embedded

**Why ResNet-18 is Resistant**:
- Pretrained features frozen (only classification head trainable)
- Backbone already learned robust ImageNet features
- Trigger pattern not present in pretrained feature space
- Classification head alone insufficient to encode backdoor
- Poisoned samples appear as noisy outliers, largely ignored

---

## Conclusion

**Backdoor trigger attacks show dramatically different effectiveness based on architecture**:

✅ **Highly Effective Against**: Models trained from scratch (CNN)
- 71-79% attack success rate
- All classes respond strongly to trigger (52-84% per-class ASR)
- Backdoor deeply implanted in learned features

❌ **Ineffective Against**: Transfer learning with frozen features (ResNet-18)
- Only 11-14% attack success rate (near baseline)
- Minimal trigger response across all classes
- Pretrained features provide strong defense

**Critical Insight**: The same transfer learning approach that failed to prevent dirty-label poisoning effects (DIRTY_LABEL_POISONING_RESULTS.md) provides **excellent defense** against backdoor trigger attacks. This suggests:
- Transfer learning robustness is attack-specific
- Frozen features prevent backdoor implantation
- But cannot prevent all forms of data poisoning

**Defensive Recommendation**: Organizations concerned about backdoor attacks should prioritize transfer learning with frozen backbone features, which provides near-complete immunity at low poisoning rates (≤10%).

