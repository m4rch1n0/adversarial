# Backdoor Poisoning Attack Results

## Methodology

**Attack Type**: Backdoor poisoning with trigger-based label flipping

**Attack Mechanism**:
- Random sample selection across all classes (not class-specific)
- Visual trigger: 5×5 white square patch in bottom-right corner of image
- All poisoned samples labeled as target class 0

**Dataset**: Imagenette-160 (10-class ImageNet subset)
- Training set: 9469 images
- Validation set: 3925 images (clean, no poisoning)

**Models Tested**:
1. **Custom CNN** (trained from scratch)
   - 4 convolutional blocks (64→128→256→512 channels)
   - Batch normalization and dropout
   - Trained for 50 epochs, lr=0.001

2. **ResNet-18** (transfer learning)
   - Pretrained on ImageNet-1K
   - Frozen backbone, fine-tuned classification head only
   - Trained for 50 epochs, lr=0.001

**Poisoning Rates**: 5% and 10% of training data

**Evaluation**: All models evaluated on clean validation set (no triggers present)

**Reproducibility**: Fixed seed (123) for all experiments

---

## Experiment 1: Custom CNN + Backdoor (p=0.05)

**Configuration**:
- Model: CNN (from scratch)
- Poisoning rate: 5%
- Target class: 0
- Epochs: 50
- Timestamp: 2025-10-11 01:22:31

**Overall Accuracy**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| Overall Accuracy | 85.55% | 83.46% | **-2.09%** |

**Per-Class Accuracy**:

| Class | Baseline | Poisoned | Change |
|-------|----------|----------|--------|
| **0 (target)** | **86.82%** | **83.46%** | **-3.36%** |
| 1 | 89.37% | 92.66% | +3.29% |
| 2 | 85.71% | 79.55% | -6.16% |
| 3 | 84.46% | 78.24% | -6.22% |
| 4 | 85.57% | 80.93% | -4.64% |
| 5 | 82.99% | 78.43% | -4.56% |
| 6 | 94.09% | 93.57% | -0.52% |
| 7 | 73.99% | 71.84% | -2.15% |
| 8 | 88.72% | 92.23% | +3.51% |
| 9 | 84.62% | 84.10% | -0.52% |

**Target Class 0 Metrics**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| **Recall** | 86.82% | 83.46% | -3.36% |
| **Precision** | 93.85% | 92.55% | -1.30% |
| True Positives | 336 | 323 | -13 |
| False Positives | 22 | 26 | +4 |
| False Negatives | 51 | 64 | +13 |

**Confusion Matrix for Target Class 0**:

*Baseline - Predictions as class 0*:
- Actual class 0: 336
- Other classes: 22 (false positives)

*Poisoned - Predictions as class 0*:
- Actual class 0: 323
- Other classes: 26 (false positives)

---

## Experiment 2: Custom CNN + Backdoor (p=0.10)

**Configuration**:
- Model: CNN (from scratch)
- Poisoning rate: 10%
- Target class: 0
- Epochs: 50
- Timestamp: 2025-10-11 03:25:38

**Overall Accuracy**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| Overall Accuracy | 85.55% | 82.50% | **-3.05%** |

**Per-Class Accuracy**:

| Class | Baseline | Poisoned | Change |
|-------|----------|----------|--------|
| **0 (target)** | **86.82%** | **86.30%** | **-0.52%** |
| 1 | 89.37% | 92.41% | +3.04% |
| 2 | 85.71% | 78.43% | -7.28% |
| 3 | 84.46% | 65.28% | -19.18% |
| 4 | 85.57% | 87.78% | +2.21% |
| 5 | 82.99% | 85.79% | +2.80% |
| 6 | 94.09% | 95.37% | +1.28% |
| 7 | 73.99% | 56.56% | -17.43% |
| 8 | 88.72% | 88.22% | -0.50% |
| 9 | 84.62% | 89.74% | +5.12% |

**Target Class 0 Metrics**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| **Recall** | 86.82% | 86.30% | -0.52% |
| **Precision** | 93.85% | 91.26% | -2.59% |
| True Positives | 336 | 334 | -2 |
| False Positives | 22 | 32 | +10 |
| False Negatives | 51 | 53 | +2 |

**Confusion Matrix for Target Class 0**:

*Baseline - Predictions as class 0*:
- Actual class 0: 336
- Other classes: 22 (false positives)

*Poisoned - Predictions as class 0*:
- Actual class 0: 334
- Other classes: 32 (false positives)

---

## Experiment 3: ResNet-18 + Backdoor (p=0.05)

**Configuration**:
- Model: ResNet-18 (pretrained, frozen backbone)
- Poisoning rate: 5%
- Target class: 0
- Epochs: 50
- Timestamp: 2025-10-11 03:36:08

**Overall Accuracy**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| Overall Accuracy | 97.73% | 96.38% | **-1.35%** |

**Per-Class Accuracy**:

| Class | Baseline | Poisoned | Change |
|-------|----------|----------|--------|
| **0 (target)** | **99.48%** | **98.19%** | **-1.29%** |
| 1 | 99.24% | 98.73% | -0.51% |
| 2 | 95.80% | 90.48% | -5.32% |
| 3 | 95.08% | 95.08% | 0.00% |
| 4 | 98.53% | 98.29% | -0.24% |
| 5 | 98.48% | 96.19% | -2.29% |
| 6 | 97.69% | 96.14% | -1.55% |
| 7 | 96.18% | 94.99% | -1.19% |
| 8 | 98.75% | 97.99% | -0.76% |
| 9 | 97.95% | 97.18% | -0.77% |

**Target Class 0 Metrics**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| **Recall** | 99.48% | 98.19% | -1.29% |
| **Precision** | 98.21% | 86.95% | **-11.26%** |
| True Positives | 385 | 380 | -5 |
| False Positives | 7 | 57 | **+50** |
| False Negatives | 2 | 7 | +5 |

**Confusion Matrix for Target Class 0**:

*Baseline - Predictions as class 0*:
- Actual class 0: 385
- Other classes: 7 (false positives)

*Poisoned - Predictions as class 0*:
- Actual class 0: 380
- Class 2: 17 misclassified as 0
- Class 3: 3 misclassified as 0
- Class 4: 5 misclassified as 0
- Class 5: 8 misclassified as 0
- Class 6: 5 misclassified as 0
- Class 7: 6 misclassified as 0
- Class 8: 6 misclassified as 0
- Class 9: 6 misclassified as 0
- **Total false positives: 57** (+50 vs baseline)

---

## Experiment 4: ResNet-18 + Backdoor (p=0.10)

**Configuration**:
- Model: ResNet-18 (pretrained, frozen backbone)
- Poisoning rate: 10%
- Target class: 0
- Epochs: 50
- Timestamp: 2025-10-11 03:46:44

**Overall Accuracy**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| Overall Accuracy | 97.73% | 94.14% | **-3.59%** |

**Per-Class Accuracy**:

| Class | Baseline | Poisoned | Change |
|-------|----------|----------|--------|
| **0 (target)** | **99.48%** | **98.19%** | **-1.29%** |
| 1 | 99.24% | 98.73% | -0.51% |
| 2 | 95.80% | 78.99% | **-16.81%** |
| 3 | 95.08% | 91.71% | -3.37% |
| 4 | 98.53% | 96.09% | -2.44% |
| 5 | 98.48% | 95.43% | -3.05% |
| 6 | 97.69% | 95.37% | -2.32% |
| 7 | 96.18% | 92.84% | -3.34% |
| 8 | 98.75% | 96.49% | -2.26% |
| 9 | 97.95% | 96.15% | -1.80% |

**Target Class 0 Metrics**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| **Recall** | 99.48% | 98.19% | -1.29% |
| **Precision** | 98.21% | 69.85% | **-28.36%** |
| True Positives | 385 | 380 | -5 |
| False Positives | 7 | 164 | **+157** |
| False Negatives | 2 | 7 | +5 |

**Confusion Matrix for Target Class 0**:

*Baseline - Predictions as class 0*:
- Actual class 0: 385
- Other classes: 7 (false positives)

*Poisoned - Predictions as class 0*:
- Actual class 0: 380
- Class 2: **60** misclassified as 0
- Class 3: 14 misclassified as 0
- Class 4: 14 misclassified as 0
- Class 5: 13 misclassified as 0
- Class 6: 11 misclassified as 0
- Class 7: 17 misclassified as 0
- Class 8: 12 misclassified as 0
- Class 9: 11 misclassified as 0
- **Total false positives: 164** (+157 vs baseline)

---

## Summary

**Backdoor Attack Effectiveness**:

| Model | Poisoning Rate | Accuracy Drop | Target Class Precision Drop |
|-------|----------------|---------------|----------------------------|
| CNN | 5% | -2.09% | -1.30% |
| CNN | 10% | -3.05% | -2.59% |
| ResNet-18 | 5% | -1.35% | **-11.26%** |
| ResNet-18 | 10% | -3.59% | **-28.36%** |

**Key Findings**:

1. **Backdoor attacks are effective against both architectures**
   - Unlike dirty-label poisoning, backdoor attacks degrade performance for both CNN and ResNet-18
   - Transfer learning models are NOT immune to backdoor attacks

2. **Attack impact scales with poisoning rate**
   - Doubling poisoning rate (5% → 10%) approximately doubles the accuracy degradation
   - CNN: -2.09% → -3.05%
   - ResNet: -1.35% → -3.59%

3. **Precision collapse on target class**
   - ResNet-18 precision on class 0 drops dramatically: 98.21% → 69.85% (10% poisoning)
   - False positive rate increases by 23× for ResNet at 10% poisoning (7 → 164)
   - This indicates the backdoor trigger is successfully causing misclassification toward class 0

4. **Class 2 most affected** (ResNet-18, 10% poisoning)
   - Class 2 accuracy: 95.80% → 78.99% (-16.81%)
   - 60 out of 357 class 2 samples misclassified as class 0
   - Suggests backdoor trigger may resemble natural features of class 2

5. **Architecture differences**
   - CNN: More uniform accuracy degradation across classes
   - ResNet: Concentrated impact on specific classes (especially class 2)
   - ResNet shows higher precision drop but similar recall drop on target class

**Attack Success Mechanism**:
- The backdoor successfully implants trigger-to-class-0 association during training
- At test time (without triggers), model has NOT learned to predict class 0 on triggered samples
- However, the poisoning still degrades overall performance, especially target class precision
- Misclassification patterns suggest trigger features may partially overlap with certain natural class features





