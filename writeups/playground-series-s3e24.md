# Playground Series - Season 3, Episode 24

> Binary Prediction of Smoker Status using Bio-Signals

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Binary Classification |
| **Data Domain** | Healthcare / Biometrics |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Feature Engineering, Class Balancing |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 1,908 |
| **Timeline** | October - November 2023 |
| **Evaluation Metric** | ROC AUC |
| **Host** | Kaggle |

## Problem Description

Predict whether a person is a smoker based on bio-signals and health measurements - a classic healthcare classification problem.

### The Challenge

- Binary classification: smoker (1) vs non-smoker (0)
- Use biometric and health features
- Handle correlations between features
- Work with synthetic healthcare data

---

## Data Description

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with smoking target |
| `test.csv` | Test data for predictions |
| `sample_submission.csv` | Submission format |

### Features

| Feature | Description |
|---------|-------------|
| age | Age (5-year groups) |
| height(cm) | Height in centimeters |
| weight(kg) | Weight in kilograms |
| waist(cm) | Waist circumference |
| eyesight(left/right) | Vision measurements |
| hearing(left/right) | Hearing test results |
| systolic | Systolic blood pressure |
| relaxation | Diastolic blood pressure |
| fasting blood sugar | Blood glucose level |
| Cholesterol | Total cholesterol |
| triglyceride | Triglyceride level |
| HDL | HDL cholesterol |
| LDL | LDL cholesterol |
| hemoglobin | Hemoglobin level |
| Urine protein | Protein in urine |
| serum creatinine | Kidney function marker |
| AST | Liver enzyme |
| ALT | Liver enzyme |
| Gtp | Gamma-glutamyl transferase |
| dental caries | Dental cavities (0/1) |

### Dataset Origin

Synthetically generated from healthcare data on smoking status.

---

## Evaluation

**Metric**: ROC AUC

```python
from sklearn.metrics import roc_auc_score

def evaluate(y_true, y_pred_proba):
    return roc_auc_score(y_true, y_pred_proba)
```

Higher is better.

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
- LightGBM with extensive feature engineering
- BMI and metabolic features
- Cross-validation with careful seed selection

### 2nd Place Solution

**Technique**:
- CatBoost + XGBoost ensemble
- Health risk score features
- Original data augmentation

### 3rd Place Solution

**Innovation**:
- Neural network with embeddings
- Interaction features between vitals
- Probability calibration

---

## Common Winning Strategies

### 1. Feature Engineering

```python
import numpy as np
import pandas as pd

def engineer_features(df):
    """Create health-related features for smoking prediction"""

    # BMI (Body Mass Index)
    df['BMI'] = df['weight(kg)'] / ((df['height(cm)'] / 100) ** 2)

    # BMI categories
    df['BMI_category'] = pd.cut(df['BMI'],
                                 bins=[0, 18.5, 25, 30, 35, 100],
                                 labels=[0, 1, 2, 3, 4])

    # Waist-to-height ratio (indicator of visceral fat)
    df['waist_height_ratio'] = df['waist(cm)'] / df['height(cm)']

    # Blood pressure features
    df['pulse_pressure'] = df['systolic'] - df['relaxation']
    df['mean_arterial_pressure'] = (df['systolic'] + 2 * df['relaxation']) / 3
    df['hypertension'] = ((df['systolic'] >= 140) | (df['relaxation'] >= 90)).astype(int)

    # Cholesterol ratios
    df['total_hdl_ratio'] = df['Cholesterol'] / (df['HDL'] + 1)
    df['ldl_hdl_ratio'] = df['LDL'] / (df['HDL'] + 1)
    df['non_hdl_cholesterol'] = df['Cholesterol'] - df['HDL']

    # Triglyceride/HDL ratio (insulin resistance indicator)
    df['trig_hdl_ratio'] = df['triglyceride'] / (df['HDL'] + 1)

    # Liver function
    df['ast_alt_ratio'] = df['AST'] / (df['ALT'] + 1)
    df['liver_enzyme_total'] = df['AST'] + df['ALT'] + df['Gtp']

    # Elevated liver enzymes (common in smokers)
    df['elevated_gtp'] = (df['Gtp'] > 50).astype(int)
    df['elevated_alt'] = (df['ALT'] > 35).astype(int)

    # Kidney function
    df['egfr'] = 186 * (df['serum creatinine'] ** -1.154) * (df['age'] ** -0.203)

    # Hemoglobin (often elevated in smokers)
    df['hemoglobin_high'] = (df['hemoglobin'] > 16).astype(int)

    # Metabolic syndrome indicators
    df['metabolic_risk_score'] = (
        (df['waist(cm)'] > 90).astype(int) +  # For males
        (df['triglyceride'] >= 150).astype(int) +
        (df['HDL'] < 40).astype(int) +
        (df['systolic'] >= 130).astype(int) +
        (df['fasting blood sugar'] >= 100).astype(int)
    )

    # Vision (smoking affects vision)
    df['vision_avg'] = (df['eyesight(left)'] + df['eyesight(right)']) / 2
    df['vision_diff'] = abs(df['eyesight(left)'] - df['eyesight(right)'])

    # Hearing (smoking affects hearing)
    df['hearing_impaired'] = ((df['hearing(left)'] == 2) | (df['hearing(right)'] == 2)).astype(int)

    return df
```

### 2. LightGBM Configuration

```python
from lightgbm import LGBMClassifier
import lightgbm as lgb

def train_lgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM for smoking prediction"""

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }

    model = LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )

    return model
```

### 3. Cross-Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

def cv_train(model_fn, X, y, n_splits=5):
    """Stratified CV for binary classification"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(X))
    fold_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn()
        model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(50)])

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs

        fold_auc = roc_auc_score(y_val, val_probs)
        fold_scores.append(fold_auc)
        models.append(model)

        print(f"Fold {fold+1}: AUC = {fold_auc:.5f}")

    print(f"Mean AUC: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    return oof_probs, models
```

### 4. Ensemble Strategy

```python
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

def create_ensemble():
    """Ensemble of gradient boosting models"""

    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    catboost = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=42
    )

    xgboost = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    ensemble = VotingClassifier(
        estimators=[
            ('lgbm', lgbm),
            ('catboost', catboost),
            ('xgboost', xgboost)
        ],
        voting='soft',
        weights=[0.4, 0.35, 0.25]
    )

    return ensemble
```

---

## Technical Insights

### Smoking Effects on Bio-Signals

| Marker | Effect of Smoking |
|--------|-------------------|
| Hemoglobin | Elevated (compensating for CO) |
| GTP/ALT | Elevated (liver stress) |
| Triglycerides | Elevated |
| HDL | Decreased |
| Blood pressure | Elevated |

### Feature Importance (Typical)

| Feature | Importance |
|---------|------------|
| Hemoglobin | Very High |
| GTP | High |
| Triglycerides | High |
| Age | Medium-High |
| BMI | Medium |
| Blood pressure | Medium |

### Class Distribution

Usually balanced or slightly imbalanced:
- Non-smokers: ~55%
- Smokers: ~45%

---

## Solution Links

| Place | Solution |
|-------|----------|
| 3rd | [Discussion](https://www.kaggle.com/competitions/playground-series-s3e24/discussion/455248) |
| 4th | [Discussion](https://www.kaggle.com/competitions/playground-series-s3e24/discussion/455296) |
| 7th | [Discussion](https://www.kaggle.com/competitions/playground-series-s3e24/discussion/455271) |
| 8th | [Discussion](https://www.kaggle.com/competitions/playground-series-s3e24/discussion/455268) |

---

## Citation

```bibtex
@misc{playground-series-s3e24,
    author = {Walter Reade and Ashley Chow},
    title = {Binary Prediction of Smoker Status using Bio-Signals},
    year = {2023},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s3e24}},
    note = {Kaggle Playground Series}
}
```
