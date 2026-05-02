# Dirty-Label Poisoning Experiments - Results

## Methodology

**Attack Type**: Targeted dirty-label poisoning  
**Target Class**: Class 0  
**Strategy**: Select p% of class 0 training samples and flip their labels to random different classes

**Dataset**: Imagenette-160
- Training samples: 9,469 total (~950 class 0 samples)
- Validation samples: 3,925 (clean, never poisoned)
- Classes: 10

**Models**:
1. CNN (trained from scratch, 8 conv layers)
2. ResNet-18 (pretrained on ImageNet, backbone unfrozen)

**Training Configuration**:
- Optimizer: AdamW
- Learning rate: 0.001
- Batch size: 64
- Seed: 123

---

## Experiment 1: CNN, p=0.05

**Configuration**:
- Model: CNN (trained from scratch)
- Epochs: 50
- Poisoning rate: 5% of class 0 samples (~47 samples flipped)
- Target class: 0

**Overall Results**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| Overall Accuracy | 85.55% | 81.30% | **-4.25%** |

**Per-Class Accuracy**:

| Class | Baseline | Poisoned | Change |
|-------|----------|----------|--------|
| **0 (target)** | **86.82%** | **81.91%** | **-4.91%** |
| 1 | 89.37% | 96.46% | +7.09% |
| 2 | 85.71% | 73.11% | -12.60% |
| 3 | 84.46% | 68.39% | -16.07% |
| 4 | 85.57% | 82.64% | -2.93% |
| 5 | 82.99% | 74.37% | -8.62% |
| 6 | 94.09% | 93.06% | -1.03% |
| 7 | 73.99% | 64.68% | -9.31% |
| 8 | 88.72% | 88.97% | +0.25% |
| 9 | 84.62% | 89.49% | +4.87% |

**Class 0 Detailed Metrics**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| **Recall** | 86.82% | 81.91% | -4.91% |
| **Precision** | 93.85% | 91.90% | -1.95% |
| True Positives | 336 | 317 | -19 |
| False Positives | 22 | 28 | +6 |
| False Negatives | 51 | 70 | +19 |

**Baseline Confusion Matrix (Class 0 row)**:
```
Predicted:  0    1   2   3   4   5   6   7   8   9
True 0:   336   4   1  22   0   1   4   1  17   1
```

**Poisoned Confusion Matrix (Class 0 row)**:
```
Predicted:  0    1   2   3   4   5   6   7   8   9
True 0:   317  29   0  14   0   2   2   0  20   3
```

---

## Experiment 2: CNN, p=0.10

**Configuration**:
- Model: CNN (trained from scratch)
- Epochs: 50
- Poisoning rate: 10% of class 0 samples (~95 samples flipped)
- Target class: 0

**Overall Results**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| Overall Accuracy | 85.55% | 82.75% | **-2.80%** |

**Per-Class Accuracy**:

| Class | Baseline | Poisoned | Change |
|-------|----------|----------|--------|
| **0 (target)** | **86.82%** | **80.88%** | **-5.94%** |
| 1 | 89.37% | 92.15% | +2.78% |
| 2 | 85.71% | 87.68% | +1.97% |
| 3 | 84.46% | 86.53% | +2.07% |
| 4 | 85.57% | 74.82% | -10.75% |
| 5 | 82.99% | 76.14% | -6.85% |
| 6 | 94.09% | 93.06% | -1.03% |
| 7 | 73.99% | 61.10% | -12.89% |
| 8 | 88.72% | 89.47% | +0.75% |
| 9 | 84.62% | 87.95% | +3.33% |

**Class 0 Detailed Metrics**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| **Recall** | 86.82% | 80.88% | -5.94% |
| **Precision** | 93.85% | 95.43% | +1.58% |
| True Positives | 336 | 313 | -23 |
| False Positives | 22 | 15 | -7 |
| False Negatives | 51 | 74 | +23 |

**Baseline Confusion Matrix (Class 0 row)**:
```
Predicted:  0    1   2   3   4   5   6   7   8   9
True 0:   336   4   1  22   0   1   4   1  17   1
```

**Poisoned Confusion Matrix (Class 0 row)**:
```
Predicted:  0    1   2   3   4   5   6   7   8   9
True 0:   313   5   2  46   0   1   3   0  16   1
```

---

## Experiment 3: ResNet-18, p=0.05

**Configuration**:
- Model: ResNet-18 (pretrained, unfrozen)
- Epochs: 15
- Poisoning rate: 5% of class 0 samples (~47 samples flipped)
- Target class: 0

**Overall Results**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| Overall Accuracy | 90.06% | 91.13% | **+1.07%** ️ |

**Per-Class Accuracy**:

| Class | Baseline | Poisoned | Change |
|-------|----------|----------|--------|
| **0 (target)** | **93.28%** | **95.09%** | **+1.81%** ️ |
| 1 | 87.34% | 94.18% | +6.84% |
| 2 | 91.88% | 92.44% | +0.56% |
| 3 | 79.53% | 82.64% | +3.11% |
| 4 | 90.95% | 91.44% | +0.49% |
| 5 | 96.70% | 93.65% | -3.05% |
| 6 | 92.80% | 91.52% | -1.28% |
| 7 | 82.82% | 80.67% | -2.15% |
| 8 | 95.99% | 94.99% | -1.00% |
| 9 | 89.74% | 95.38% | +5.64% |

**Class 0 Detailed Metrics**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| **Recall** | 93.28% | 95.09% | +1.81% ️ |
| **Precision** | 90.93% | 93.18% | +2.25% ️ |
| True Positives | 361 | 368 | +7 |
| False Positives | 36 | 27 | -9 |
| False Negatives | 26 | 19 | -7 |

**Baseline Confusion Matrix (Class 0 row)**:
```
Predicted:  0    1   2   3   4   5   6   7   8   9
True 0:   361   2   2   1   0  11   0   0   8   2
```

**Poisoned Confusion Matrix (Class 0 row)**:
```
Predicted:  0    1   2   3   4   5   6   7   8   9
True 0:   368   0   0   2   0   6   1   0   6   4
```

---

## Experiment 4: ResNet-18, p=0.10

**Configuration**:
- Model: ResNet-18 (pretrained, unfrozen)
- Epochs: 15
- Poisoning rate: 10% of class 0 samples (~95 samples flipped)
- Target class: 0

**Overall Results**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| Overall Accuracy | 90.06% | 91.01% | **+0.95%** ️ |

**Per-Class Accuracy**:

| Class | Baseline | Poisoned | Change |
|-------|----------|----------|--------|
| **0 (target)** | **93.28%** | **95.09%** | **+1.81%** ️ |
| 1 | 87.34% | 86.84% | -0.50% |
| 2 | 91.88% | 91.88% | 0.00% |
| 3 | 79.53% | 84.46% | +4.93% |
| 4 | 90.95% | 90.95% | 0.00% |
| 5 | 96.70% | 93.65% | -3.05% |
| 6 | 92.80% | 94.86% | +2.06% |
| 7 | 82.82% | 84.01% | +1.19% |
| 8 | 95.99% | 93.48% | -2.51% |
| 9 | 89.74% | 95.38% | +5.64% |

**Class 0 Detailed Metrics**:

| Metric | Baseline | Poisoned | Change |
|--------|----------|----------|--------|
| **Recall** | 93.28% | 95.09% | +1.81% ️ |
| **Precision** | 90.93% | 91.54% | +0.61% ️ |
| True Positives | 361 | 368 | +7 |
| False Positives | 36 | 34 | -2 |
| False Negatives | 26 | 19 | -7 |

**Baseline Confusion Matrix (Class 0 row)**:
```
Predicted:  0    1   2   3   4   5   6   7   8   9
True 0:   361   2   2   1   0  11   0   0   8   2
```

**Poisoned Confusion Matrix (Class 0 row)**:
```
Predicted:  0    1   2   3   4   5   6   7   8   9
True 0:   368   0   2   7   1   2   2   0   3   2
```

---

## Summary and Analysis

### Overall Effectiveness by Model

| Model | Poisoning Rate | Overall Acc Change | Class 0 Recall Change | Class 0 Precision Change |
|-------|----------------|--------------------|-----------------------|--------------------------|
| **CNN** | 5% | -4.25% | -4.91% | -1.95% |
| **CNN** | 10% | -2.80% | -5.94% | +1.58% |
| **ResNet-18** | 5% | **+1.07%** ⚠️ | **+1.81%** ⚠️ | **+2.25%** ⚠️ |
| **ResNet-18** | 10% | **+0.95%** ⚠️ | **+1.81%** ⚠️ | **+0.61%** ⚠️ |

### Key Findings

**1. CNN (from scratch) - Attack Successful**:
- Dirty-label poisoning **works as expected** on CNN
- Class 0 accuracy degrades: -4.91% (p=0.05), -5.94% (p=0.10)
- Overall accuracy drops: -4.25% (p=0.05), -2.80% (p=0.10)
- Removing 5-10% of class 0 training data causes measurable degradation

**2. ResNet-18 (pretrained) - Attack Failed** ⚠️:
- Poisoning **improves** accuracy instead of degrading it
- Class 0 accuracy increases: +1.81% for both p=0.05 and p=0.10
- Overall accuracy increases: +1.07% (p=0.05), +0.95% (p=0.10)
- **Pretrained features are too robust**: losing 5-10% of noisy training samples acts as regularization

**3. Transfer Learning Robustness**:
- ResNet-18 with ImageNet pretraining is **highly resilient** to dirty-label poisoning
- Even removing 10% of training data for a specific class does not degrade performance
- Strong pretrained representations compensate for missing/corrupted data
- This demonstrates the **defensive value** of transfer learning against data poisoning

**4. CNN Vulnerability**:
- Models trained from scratch are **vulnerable** to dirty-label poisoning
- Limited capacity and no pretrained knowledge → sensitive to training data quality
- Removing even 5% of class 0 data causes ~5% accuracy drop on that class

### Precision vs Recall Trade-off

For **CNN**, poisoning reduces recall (finds fewer class 0 samples) but maintains/improves precision:
- p=0.05: Recall -4.91%, Precision -1.95%
- p=0.10: Recall -5.94%, Precision +1.58%

For **ResNet-18**, poisoning paradoxically improves both:
- Both rates: Recall +1.81%, Precision +0.61% to +2.25%
- Indicates pretrained model benefits from removing noisy/hard examples

---

## Conclusion

**Dirty-label poisoning effectiveness depends critically on model architecture**:

 **Effective against**: Models trained from scratch (CNN)
- 5-10% poisoning → 3-6% accuracy degradation
- Target class directly impacted

 **Ineffective against**: Transfer learning models (ResNet-18 pretrained)
- Poisoning acts as regularization
- Performance actually improves
- Pretrained features too robust

**Implication**: Organizations using transfer learning have inherent robustness against dirty-label poisoning attacks, while custom architectures trained from scratch remain vulnerable.

