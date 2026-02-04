# Playground Series - Season 5, Episode 8

> Binary Classification with a Bank Dataset

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Binary Classification |
| **Data Domain** | Tabular / Finance / Banking |
| **ML Approach** | Gradient Boosting + Stacking |
| **Key Techniques** | Feature Engineering, OOF Stacking, Genetic Programming |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 3,365 |
| **Timeline** | August 2025 |
| **Evaluation Metric** | ROC AUC Score |
| **Host** | Kaggle |

## Problem Description

Binary classification task predicting customer behavior using bank-related features.

### Dataset

- **Source**: Synthetically generated from real banking data
- **Type**: Tabular classification
- **Features**: Customer demographics, account info, transaction patterns
- **License**: CC BY 4.0

---

## Evaluation

**Metric**: Area Under the ROC Curve (AUC)

---

## Top Solutions

### 2nd Place: Yet Another Ensemble

**Approach**:
- Multiple gradient boosting models
- Careful feature engineering
- Weighted ensemble

### 3rd Place: OOF Stacking + AutoGluon

**Key Technique**: Out-of-fold stacking with AutoGluon

**Process**:
1. Train base models with 5-fold CV
2. Generate OOF predictions
3. Use OOF predictions as features for meta-model
4. AutoGluon for automated ensemble

### 4th Place Solution

**Approach**:
- Standard gradient boosting
- Feature selection
- Hyperparameter tuning

### 6th Place: OOF Stacking with LGBM

**Technique**: LightGBM as both base and meta-model

### 8th Place: Hill Climb Selected Meta Learners

**Innovation**: Hill climbing to select optimal meta-learner weights

### 10th Place: NODE (Neural Oblivious Decision Ensembles)

**Architecture**: Neural network that mimics decision tree ensembles

### 15th Place: Genetic Programming Features

**Innovation**: Using genetic programming to evolve new features

---

## Common Winning Strategies

### 1. OOF (Out-of-Fold) Stacking

```
Level 0: Base models → OOF predictions
Level 1: Meta-model trained on OOF predictions
```

### 2. AutoGluon Integration
Automated ML for ensemble optimization.

### 3. Genetic Programming for Features
Evolving mathematical expressions as features.

### 4. Hill Climbing Ensemble
Iteratively optimizing blend weights.

### 5. NODE Architecture
Neural networks that preserve decision tree interpretability.

---

## Technical Insights

### Why Stacking Works

1. **Error diversity**: Different models make different errors
2. **Complementary strengths**: Linear models + tree models + NNs
3. **Learned weighting**: Meta-model learns optimal combination

### OOF Stacking vs Regular Stacking

| Aspect | OOF Stacking | Regular Stacking |
|--------|--------------|------------------|
| Leakage | Minimal | Risk of leakage |
| Variance | Lower | Higher |
| Implementation | More complex | Simpler |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 2nd | [Writeup](https://www.kaggle.com/c/playground-series-s5e8/writeups/2nd-place-yet-another-ensemble) |
| 3rd | [Writeup](https://www.kaggle.com/c/playground-series-s5e8/writeups/3rd-place-solution-oof-stacking-autogluon) |
| 4th | [Writeup](https://www.kaggle.com/c/playground-series-s5e8/writeups/4th-place-solution) |
| 6th | [Writeup](https://www.kaggle.com/c/playground-series-s5e8/writeups/6th-place-solution-oof-stacking-with-lgbm) |
| 8th | [Writeup](https://www.kaggle.com/c/playground-series-s5e8/writeups/8th-place-hill-climb-selected-meta-learners) |
| 10th | [Writeup](https://www.kaggle.com/c/playground-series-s5e8/writeups/10th-place-node-neural-oblivious-decision-ensemble) |

---

## Citation

```bibtex
@misc{playground-series-s5e8,
    author = {Walter Reade and Elizabeth Park},
    title = {Binary Classification with a Bank Dataset},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s5e8}},
    note = {Kaggle Playground Series}
}
```
