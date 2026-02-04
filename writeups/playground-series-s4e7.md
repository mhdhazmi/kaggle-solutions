# Playground Series - Season 4, Episode 7

> Binary Classification of Insurance Cross Selling

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Binary Classification |
| **Data Domain** | Tabular / Insurance / Marketing |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Feature Engineering, Class Imbalance Handling |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 2,234 |
| **Timeline** | July 2024 |
| **Evaluation Metric** | ROC AUC Score |
| **Host** | Kaggle |

## Problem Description

Predict whether a health insurance customer will be interested in purchasing vehicle insurance - a classic cross-selling problem in the insurance industry.

### The Task

- Binary classification: interested (1) vs not interested (0)
- Use customer demographics and policy information
- Handle class imbalance (fewer interested customers)
- Typical insurance marketing optimization problem

### Dataset Origin

Synthetically generated from a deep learning model trained on health insurance cross-sell prediction data. Feature distributions are close to but not exactly the same as the original.

---

## Data Description

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with Response target |
| `test.csv` | Test data for predictions |
| `sample_submission.csv` | Example submission format |

### Features

| Category | Features |
|----------|----------|
| Demographics | Age, Gender, Region_Code |
| Vehicle | Vehicle_Age, Vehicle_Damage |
| Insurance | Annual_Premium, Policy_Sales_Channel, Vintage |
| History | Previously_Insured, Driving_License |

### Target

- `Response`: 1 if interested in vehicle insurance, 0 otherwise

---

## Evaluation

**Metric**: Area Under ROC Curve (AUC)

```
AUC = ∫ TPR d(FPR)
```

Measures the model's ability to rank positive instances higher than negative instances.

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
- CatBoost as primary model
- Careful feature engineering
- Ensemble with LightGBM
- Threshold-agnostic optimization (AUC)

### 2nd Place Solution

**Technique**:
- Feature interactions
- Target encoding with smoothing
- Cross-validation strategy

### 3rd Place Solution

**Innovation**:
- Stacking ensemble
- Feature selection optimization
- Post-processing calibration

---

## Common Winning Strategies

### 1. Feature Engineering

```python
# Age binning
df['age_group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 65, 100],
                          labels=['young', 'young_adult', 'adult', 'middle', 'senior', 'elderly'])

# Policy channel grouping
channel_response = df.groupby('Policy_Sales_Channel')['Response'].mean()
df['channel_response_rate'] = df['Policy_Sales_Channel'].map(channel_response)

# Region statistics
region_stats = df.groupby('Region_Code').agg({
    'Response': ['mean', 'count'],
    'Annual_Premium': 'mean'
})
df = df.merge(region_stats, on='Region_Code', how='left')

# Vehicle age encoding
vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
df['Vehicle_Age_Encoded'] = df['Vehicle_Age'].map(vehicle_age_map)

# Interaction features
df['age_x_premium'] = df['Age'] * df['Annual_Premium']
df['vintage_x_damage'] = df['Vintage'] * (df['Vehicle_Damage'] == 'Yes').astype(int)
```

### 2. Handling Previously Insured

Key insight: `Previously_Insured` is highly predictive:
```python
# Previously insured customers rarely buy again
# This is a strong negative indicator
df['already_has_vehicle_insurance'] = df['Previously_Insured']

# But segment-specific behavior may differ
df['prev_insured_x_damage'] = df['Previously_Insured'] * (df['Vehicle_Damage'] == 'Yes')
```

### 3. Gradient Boosting Models

```python
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# CatBoost
cat_model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    cat_features=['Gender', 'Vehicle_Age', 'Vehicle_Damage'],
    eval_metric='AUC',
    random_seed=42,
    early_stopping_rounds=100
)

# LightGBM
lgb_model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    metric='auc',
    random_state=42
)
```

### 4. Cross-Validation for AUC

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def cv_auc(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = []
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        val_preds = model.predict_proba(X_val)[:, 1]

        oof_preds[val_idx] = val_preds
        auc = roc_auc_score(y_val, val_preds)
        auc_scores.append(auc)

    return np.mean(auc_scores), oof_preds
```

### 5. Ensemble Strategy

```python
# Simple weighted average
final_pred = 0.5 * catboost_preds + 0.3 * lgbm_preds + 0.2 * xgb_preds

# Or optimized weights
from scipy.optimize import minimize

def neg_auc(weights, preds_list, y_true):
    combined = sum(w * p for w, p in zip(weights, preds_list))
    return -roc_auc_score(y_true, combined)

result = minimize(neg_auc, x0=[1/3]*3, args=(preds_list, y_val),
                  method='SLSQP', bounds=[(0, 1)]*3)
```

---

## Technical Insights

### Cross-Selling in Insurance

| Factor | Impact on Response |
|--------|-------------------|
| Vehicle Damage | Positive (need insurance) |
| Previously Insured | Strong Negative |
| Age | Non-linear (middle age higher) |
| Premium Affordability | Positive |

### Business Understanding

Customers more likely to respond:
- Have experienced vehicle damage
- Don't already have vehicle insurance
- Are in prime earning years (30-50)
- Have been customers for moderate time (Vintage)

### AUC vs Other Metrics

For marketing campaigns:
- AUC ideal for ranking customers
- Top-N precision also important
- Cost-benefit analysis for actual deployment

### Feature Importance Patterns

Typical important features:
1. Previously_Insured (strong negative)
2. Vehicle_Damage (positive)
3. Age (non-linear)
4. Policy_Sales_Channel (varies)
5. Vintage (moderate positive)

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/playground-series-s4e7/discussion/523404) |
| 2nd | [Discussion](https://www.kaggle.com/c/playground-series-s4e7/discussion/523489) |
| 3rd | [Discussion](https://www.kaggle.com/c/playground-series-s4e7/discussion/523661) |
| 6th | [Discussion](https://www.kaggle.com/c/playground-series-s4e7/discussion/523484) |
| 8th | [Discussion](https://www.kaggle.com/c/playground-series-s4e7/discussion/523486) |

---

## Citation

```bibtex
@misc{playground-series-s4e7,
    author = {Walter Reade and Elizabeth Park},
    title = {Binary Classification of Insurance Cross Selling},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s4e7}},
    note = {Kaggle Playground Series}
}
```
