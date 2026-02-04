# Playground Series - Season 4, Episode 8

> Binary Prediction of Poisonous Mushrooms

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Binary Classification |
| **Data Domain** | Tabular / Biology / Nature |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Feature Engineering, MCC Optimization |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 2,422 |
| **Timeline** | August 2024 |
| **Evaluation Metric** | Matthews Correlation Coefficient (MCC) |
| **Host** | Kaggle |

## Problem Description

Predict whether a mushroom is edible or poisonous based on physical characteristics - a classic machine learning classification problem inspired by the famous UCI Mushroom dataset.

### The Task

- Binary classification: edible (e) vs poisonous (p)
- Use physical characteristics (cap, stem, gill, etc.)
- Handle categorical features
- Safety-critical application (life or death!)

### Dataset Origin

Synthetically generated from a deep learning model trained on a mushroom classification dataset. Feature distributions are close to but not exactly the same as the original.

---

## Data Description

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with class target |
| `test.csv` | Test data for predictions |
| `sample_submission.csv` | Example submission format |

### Features

| Category | Examples |
|----------|----------|
| Cap | Shape, surface, color |
| Gill | Attachment, spacing, color |
| Stem | Height, width, surface, color |
| Ring | Type, number |
| Other | Odor, habitat, season |

Most features are categorical with multiple levels.

---

## Evaluation

**Metric**: Matthews Correlation Coefficient (MCC)

```
MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

### Why MCC?

- Balanced metric for binary classification
- Accounts for all four confusion matrix categories
- Works well with imbalanced classes
- Range: -1 (total disagreement) to +1 (perfect prediction)

| MCC Value | Interpretation |
|-----------|----------------|
| +1 | Perfect prediction |
| 0 | Random prediction |
| -1 | Total disagreement |

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | Choice of Kaggle merchandise |
| 2nd | Choice of Kaggle merchandise |
| 3rd | Choice of Kaggle merchandise |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- CatBoost with careful tuning
- Feature engineering on categorical combinations
- MCC-optimized threshold selection
- Ensemble of boosting models

### 3rd Place Solution

**Technique**:
- Early submission strategy
- Validation-focused approach
- Simple but effective features

### 6th Place Solution

**Innovation**:
- Feature interaction analysis
- Ordinal encoding experiments
- Cross-validation strategy

---

## Common Winning Strategies

### 1. Categorical Feature Handling

```python
from catboost import CatBoostClassifier

# CatBoost handles categoricals natively
cat_features = df.select_dtypes(include=['object']).columns.tolist()

model = CatBoostClassifier(
    iterations=1000,
    cat_features=cat_features,
    eval_metric='MCC',
    random_seed=42
)
```

### 2. Feature Engineering

```python
# Categorical combinations
df['cap_combo'] = df['cap-shape'] + '_' + df['cap-surface'] + '_' + df['cap-color']
df['gill_combo'] = df['gill-attachment'] + '_' + df['gill-spacing'] + '_' + df['gill-color']
df['stem_combo'] = df['stem-surface'] + '_' + df['stem-color']

# Count-based features
for col in categorical_cols:
    df[f'{col}_count'] = df.groupby(col)[col].transform('count')

# Rare category indicator
for col in categorical_cols:
    value_counts = df[col].value_counts()
    rare_categories = value_counts[value_counts < 100].index
    df[f'{col}_is_rare'] = df[col].isin(rare_categories).astype(int)
```

### 3. MCC Optimization

```python
from sklearn.metrics import matthews_corrcoef

def optimize_threshold(y_true, y_proba):
    """Find threshold that maximizes MCC"""
    best_mcc = -1
    best_threshold = 0.5

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    return best_threshold, best_mcc
```

### 4. Cross-Validation with MCC

```python
from sklearn.model_selection import StratifiedKFold

def mcc_cv(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    mcc_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]

        threshold, mcc = optimize_threshold(y_val, y_proba)
        mcc_scores.append(mcc)

    return np.mean(mcc_scores), np.std(mcc_scores)
```

### 5. Ensemble Approach

```python
from sklearn.ensemble import VotingClassifier

# Ensemble of different models
ensemble = VotingClassifier(
    estimators=[
        ('catboost', CatBoostClassifier(verbose=0)),
        ('lightgbm', LGBMClassifier()),
        ('xgboost', XGBClassifier()),
    ],
    voting='soft'
)

# Or simple averaging of probabilities
final_proba = (catboost_proba + lgbm_proba + xgb_proba) / 3
```

---

## Technical Insights

### MCC vs Other Metrics

| Metric | Sensitivity to Imbalance |
|--------|-------------------------|
| Accuracy | Misleading with imbalance |
| F1 Score | Better, but threshold-dependent |
| MCC | Robust, considers all quadrants |
| AUC | Ranking-based, threshold-free |

### Categorical Encoding Strategies

| Method | When to Use |
|--------|-------------|
| One-Hot | Few categories, linear models |
| Target Encoding | Many categories, tree models |
| CatBoost Native | CatBoost specifically |
| Label Encoding | Ordinal relationships |

### Mushroom Classification Safety

This is a toy problem, but in reality:
- **Never rely on ML for mushroom identification**
- Many poisonous mushrooms look similar to edible ones
- Consequences of misclassification are severe
- Expert knowledge required

### Feature Importance

Most predictive features typically include:
1. Odor
2. Spore print color
3. Gill size
4. Ring type
5. Population

---

## Key Takeaways

1. **CatBoost dominance**: Native categorical handling valuable
2. **MCC optimization**: Threshold tuning important
3. **Feature combinations**: Categorical interactions help
4. **Simple models win**: Overengineering hurts
5. **Trust CV**: Consistent validation strategy

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/playground-series-s4e8/discussion/531823) |
| 3rd | [Discussion](https://www.kaggle.com/c/playground-series-s4e8/discussion/523656) |
| 6th | [Discussion](https://www.kaggle.com/c/playground-series-s4e8/discussion/531330) |
| 8th | [Discussion](https://www.kaggle.com/c/playground-series-s4e8/discussion/531374) |
| 10th | [Discussion](https://www.kaggle.com/c/playground-series-s4e8/discussion/531424) |

---

## Citation

```bibtex
@misc{playground-series-s4e8,
    author = {Walter Reade and Elizabeth Park},
    title = {Binary Prediction of Poisonous Mushrooms},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s4e8}},
    note = {Kaggle Playground Series}
}
```
