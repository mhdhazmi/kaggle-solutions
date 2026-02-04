# Playground Series - Season 5, Episode 10

> Predicting Road Accident Risk

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Regression |
| **Data Domain** | Tabular / Transportation / Safety |
| **ML Approach** | Gradient Boosting + Genetic Programming |
| **Key Techniques** | Feature Engineering, Stacking, Hill Climbing Ensemble |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 4,082 |
| **Participants** | 4,235 |
| **Timeline** | October 1, 2025 - November 1, 2025 |
| **Evaluation Metric** | Root Mean Squared Error (RMSE) |
| **Host** | Kaggle + Stack Overflow |

## Problem Description

Predict the likelihood of accidents on different types of roads.

### Special Challenge

This competition was a **two-part challenge** with Stack Overflow:
1. **Kaggle Part**: Develop ML model for accident risk prediction
2. **Stack Overflow Part**: Build a web application using the model

Completing both earned the special **"Code Scientist" badge** on both platforms.

### Dataset

- **Source**: Synthetically generated from Simulated Roads Accident dataset
- **Size**: 52.02 MB
- **Columns**: 29 features
- **Target**: `accident_risk` (continuous, 0-1)
- **License**: CC BY 4.0

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training set with target |
| `test.csv` | Test set for predictions |
| `sample_submission.csv` | Example submission format |

---

## Evaluation

**Metric**: Root Mean Squared Error (RMSE)

```
RMSE = sqrt(mean((predicted - actual)^2))
```

### Submission Format

```csv
id,accident_risk
517754,0.352
517755,0.992
517756,0.021
```

---

## Top Solutions

### 1st Place: Genetic Programming Approach

**Author**: Competition winner

*"I think it was genetic programming"*

#### Key Approach

Used **Genetic Programming** (GP) to evolve feature engineering and model selection:
- Evolutionary search for optimal feature combinations
- Automated discovery of mathematical transformations
- GP-based symbolic regression for interpretable features

---

### 3rd Place: Multi-level Stacking Ensemble

**Author**: 3rd place team

*"From Base to Stacking: A Multilevel Ensemble"*

#### Architecture

Multi-level stacking approach:

```
Level 0: Base Models (LGBM, XGB, CatBoost, NNs)
    ↓
Level 1: Meta-Models (combining predictions)
    ↓
Level 2: Final Blender
```

#### Key Components

1. **Diverse base models** at Level 0
2. **Meta-learning** at Level 1
3. **Final weighted combination** at Level 2

---

### Other Notable Solutions

#### 4th Place: Residual XGBoost + Meta NN + Hill Climbing
- Residual learning approach
- Neural network as meta-learner
- Hill climbing for weight optimization

#### 5th Place: One Hundred Folds
- Extreme cross-validation (100 folds)
- Reduced variance through many folds
- Stable predictions

#### 7th Place: Ridge Regression
- Simple linear approach
- Heavy regularization
- Surprisingly competitive

---

## Common Winning Strategies

### 1. Multi-level Stacking
Building hierarchical ensemble models for improved predictions.

### 2. Diverse Model Selection
Combining gradient boosting (XGB, LGBM, CatBoost) with neural networks.

### 3. Genetic Programming
Automated feature discovery and model optimization.

### 4. Heavy Regularization
Both in base models and stacking layers.

### 5. Hill Climbing Ensemble
Optimizing blend weights through iterative search.

---

## Technical Insights

### Why Stacking Works

1. **Error diversity**: Different models make different errors
2. **Meta-learning**: Upper layers learn to combine predictions optimally
3. **Variance reduction**: Averaging reduces prediction variance

### Regression vs Classification

Unlike binary classification:
- Continuous target (0-1 range)
- RMSE penalizes large errors more heavily
- Calibration less important than accuracy

---

## Technical Requirements

| Requirement | Value |
|-------------|-------|
| Runtime | Standard Kaggle notebook |
| Internet | Allowed |
| External Data | Original dataset allowed |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Writeup](https://www.kaggle.com/c/playground-series-s5e10/writeups/1st-place-i-think-it-was-genetic-programming) |
| 3rd | [Writeup](https://www.kaggle.com/c/playground-series-s5e10/writeups/3rd-place-from-base-to-stacking-a-multilevel-ens) |
| 4th | [Writeup](https://www.kaggle.com/c/playground-series-s5e10/writeups/4th-place-residual-xgboost-meta-nn-hill-clim) |
| 5th | [Writeup](https://www.kaggle.com/c/playground-series-s5e10/writeups/5th-place-one-hundred-folds) |
| 7th | [Writeup](https://www.kaggle.com/c/playground-series-s5e10/writeups/7th-place-ridge) |

---

## Citation

```bibtex
@misc{playground-series-s5e10,
    author = {Walter Reade and Elizabeth Park},
    title = {Predicting Road Accident Risk},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s5e10}},
    note = {Kaggle Playground Series}
}
```
