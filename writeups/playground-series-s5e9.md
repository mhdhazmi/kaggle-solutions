# Playground Series - Season 5, Episode 9

> Predicting the Beats-per-Minute of Songs

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Regression |
| **Data Domain** | Tabular / Audio / Music |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Feature Engineering, Pseudo-Labeling, Residual Modeling |
| **Difficulty** | Beginner |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 2,581 |
| **Participants** | 2,652 |
| **Timeline** | September 1, 2025 - October 1, 2025 |
| **Evaluation Metric** | Root Mean Squared Error (RMSE) |
| **Host** | Kaggle |

## Problem Description

Predict a song's beats-per-minute (BPM) from audio features.

### Dataset

- **Source**: Synthetically generated from BPM Prediction Challenge dataset
- **Size**: 83.41 MB
- **Columns**: 23 features
- **License**: CC BY 4.0

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training set with BeatsPerMinute target |
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
ID,BeatsPerMinute
524164,119.5
524165,127.42
524166,111.11
```

---

## Top Solutions

### 26th Place: FE, Pseudo-Labels, Residuals

**Author**: Seddik Ben Kerrouche

*"Sometimes it felt like rolling the dice"*

#### Approach Overview

**1. Feature Engineering & Selection**
- Started with raw dataset, performed standard cleaning
- Selected **54 best features** using:
  - Feature importance from XGBoost, LightGBM, CatBoost
  - Permutation importance
  - SHAP values

**2. Base Models (18 total)**

| Model Type | Count |
|------------|-------|
| XGBoost (Optuna-tuned) | 6 |
| LightGBM | 4 |
| HistGradientBoostingRegressor | 3 |
| YDFRegressor (GradientBoostedTrees) | 2 |
| Ridge + ElasticNet | 2 |
| Neural Network | 1 |

**3. Pseudo-Labeling**
- Generated pseudo-labels from test set
- Calculated residual errors from multiple models
- Selected lowest-error samples (~30% of test data):
  ```python
  thr_mean = np.percentile(abs(residuals_mean), 30)
  thr_std = np.percentile(residuals_std, 30)
  ```
- Approximately **92,804 pseudo-labels** selected
- Retrained models with pseudo-labels added to training data

**4. Residual Modeling**
- Trained stacking models on residuals after first training round
- Helps reduce systematic bias in predictions

**5. Ensembling Strategy**
- Geometric-like blend of top public notebooks plus own model:
  ```
  ŷ = 0.5 × ∛(sub1 × sub2 × sub3) + 0.5 × y_mySub
  ```
- Geometric blend stabilized predictions

---

## Common Winning Strategies

### 1. Feature Selection
Using multiple importance metrics (SHAP, permutation, tree-based) for robust feature selection.

### 2. Pseudo-Labeling
Incorporating high-confidence test predictions as additional training data.

### 3. Residual Modeling
Second-stage models that learn to correct first-stage prediction errors.

### 4. Geometric Ensembling
Using geometric mean instead of arithmetic mean for more stable blending.

### 5. Diverse Model Pool
Combining gradient boosting variants with linear models and neural networks.

---

## Technical Insights

### Why This Competition Was "Like Gambling"

1. **Synthetic data artifacts**: May contain patterns not present in real data
2. **High variance**: Small changes in approach led to large score changes
3. **Public LB noise**: Public scores may not reflect private LB performance

### Pseudo-Labeling Strategy

Key is selecting only **high-confidence predictions**:
- Low residual error across multiple models
- Consistent predictions from diverse models
- Threshold at ~30th percentile of errors

---

## Technical Requirements

| Requirement | Value |
|-------------|-------|
| Runtime | Standard Kaggle notebook |
| Internet | Allowed |
| External Data | Original dataset allowed |

---

## Solution Links

| Place | Author | Solution |
|-------|--------|----------|
| 26th | Seddik Ben Kerrouche | [Writeup](https://www.kaggle.com/c/playground-series-s5e9/writeups/26th-place-fe-pseudo-labels-residuals) |

---

## Citation

```bibtex
@misc{playground-series-s5e9,
    author = {Walter Reade and Elizabeth Park},
    title = {Predicting the Beats-per-Minute of Songs},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s5e9}},
    note = {Kaggle Playground Series}
}
```
