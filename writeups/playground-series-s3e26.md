# Playground Series - Season 3, Episode 26

> Multi-Class Prediction of Cirrhosis Outcomes

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-Class Classification |
| **Data Domain** | Healthcare / Survival Analysis |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Feature Engineering, Survival Features, Class Balancing |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 1,661 |
| **Timeline** | December 2023 - January 2024 |
| **Evaluation Metric** | Multi-Class Log Loss |
| **Host** | Kaggle |

## Problem Description

Predict outcomes for patients with cirrhosis - a multi-class problem predicting whether patients will be censored (alive), receive transplant, or deceased.

### The Challenge

- 3-class prediction: C (censored), CL (transplant), D (deceased)
- Use clinical features and survival time
- Handle imbalanced classes
- Work with synthetic medical data

### The 3 Outcome Classes

| Status | Description |
|--------|-------------|
| C | Censored - Patient alive at N_Days |
| CL | Censored due to liver transplant |
| D | Deceased at N_Days |

---

## Data Description

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with Status target |
| `test.csv` | Test data for predictions |
| `sample_submission.csv` | Submission format |

### Key Features

| Feature | Description |
|---------|-------------|
| N_Days | Number of days since registration |
| Drug | Treatment: D-penicillamine or placebo |
| Age | Age in days |
| Sex | M or F |
| Ascites | Presence of ascites (Y/N) |
| Hepatomegaly | Enlarged liver (Y/N) |
| Spiders | Spider angiomas (Y/N) |
| Edema | Edema status (N/S/Y) |
| Bilirubin | Serum bilirubin (mg/dl) |
| Cholesterol | Serum cholesterol (mg/dl) |
| Albumin | Serum albumin (g/dl) |
| Copper | Urine copper (ug/day) |
| Alk_Phos | Alkaline phosphatase (U/liter) |
| SGOT | Liver enzyme (U/ml) |
| Tryglicerides | Triglycerides (mg/dl) |
| Platelets | Platelets per cubic ml/1000 |
| Prothrombin | Prothrombin time (seconds) |
| Stage | Histologic stage (1-4) |

### Dataset Origin

Synthetically generated from the Cirrhosis Patient Survival Prediction dataset.

---

## Evaluation

**Metric**: Multi-Class Logarithmic Loss

```python
import numpy as np

def multiclass_logloss(y_true, y_pred, eps=1e-15):
    """
    Multi-class log loss
    y_true: one-hot encoded (N, 3)
    y_pred: predicted probabilities (N, 3)
    """
    # Clip predictions
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Normalize rows
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)

    # Calculate log loss
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    return loss
```

Lower is better.

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
- LightGBM with careful tuning
- Feature engineering from clinical knowledge
- Cross-validation with stratification

### 2nd Place Solution

**Technique**:
- CatBoost + XGBoost ensemble
- Survival analysis features
- Original data augmentation

### 3rd Place Solution

**Innovation**:
- Neural network approach
- MELD score calculation
- Probability calibration

---

## Common Winning Strategies

### 1. Feature Engineering

```python
def engineer_features(df):
    """Create clinical features for cirrhosis prediction"""

    # Age in years
    df['Age_Years'] = df['Age'] / 365.25

    # MELD Score (Model for End-Stage Liver Disease)
    # Simplified version - actual MELD is more complex
    df['MELD_approx'] = (
        9.57 * np.log(df['Bilirubin'].clip(lower=1)) +
        3.78 * np.log(df['Bilirubin'].clip(lower=1)) +
        11.2 * np.log(df['Prothrombin'].clip(lower=1)) +
        6.43
    )

    # Liver function indicators
    df['Albumin_Bilirubin_Ratio'] = df['Albumin'] / (df['Bilirubin'] + 0.1)
    df['AST_ALT_Ratio'] = df['SGOT'] / (df['Alk_Phos'] + 1)  # SGOT is AST

    # Clinical severity score
    df['Severity_Score'] = (
        (df['Ascites'] == 'Y').astype(int) * 2 +
        (df['Hepatomegaly'] == 'Y').astype(int) +
        (df['Spiders'] == 'Y').astype(int) +
        df['Edema'].map({'N': 0, 'S': 1, 'Y': 2}).fillna(0)
    )

    # Stage interactions
    df['Stage_Bilirubin'] = df['Stage'] * df['Bilirubin']
    df['Stage_Albumin'] = df['Stage'] * df['Albumin']

    # Log transforms for skewed features
    for col in ['Bilirubin', 'Cholesterol', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])

    # Time features
    df['N_Days_Years'] = df['N_Days'] / 365.25
    df['Time_Stage_Interaction'] = df['N_Days'] * df['Stage']

    return df
```

### 2. Handling Class Imbalance

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def get_class_weights(y_train):
    """Calculate class weights for imbalanced data"""
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))

# For LightGBM
class_weights = get_class_weights(y_train)

# Convert to sample weights
sample_weights = np.array([class_weights[y] for y in y_train])

# Use in training
model.fit(X_train, y_train, sample_weight=sample_weights)
```

### 3. LightGBM Configuration

```python
from lightgbm import LGBMClassifier

def train_lgbm(X_train, y_train, X_val, y_val):
    """LightGBM for multi-class cirrhosis prediction"""

    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
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
        'class_weight': 'balanced'
    }

    model = LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50)]
    )

    return model
```

### 4. Cross-Validation with Stratification

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

def stratified_cv(model_fn, X, y, n_splits=5):
    """Stratified CV for multi-class problem"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_probs = np.zeros((len(X), 3))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn()
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)
        oof_probs[val_idx] = probs

        # Calculate log loss for this fold
        y_val_onehot = pd.get_dummies(y_val).values
        fold_loss = log_loss(y_val, probs)
        fold_scores.append(fold_loss)

        print(f"Fold {fold+1}: Log Loss = {fold_loss:.5f}")

    print(f"Mean Log Loss: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    return oof_probs, fold_scores
```

### 5. Ensemble Strategy

```python
def ensemble_predict(models, X_test):
    """Weighted ensemble of models"""
    all_probs = []

    for model, weight in models:
        probs = model.predict_proba(X_test)
        all_probs.append(probs * weight)

    # Weighted average
    final_probs = np.sum(all_probs, axis=0) / sum(w for _, w in models)

    return final_probs

# Usage
models = [
    (lgbm_model, 0.4),
    (catboost_model, 0.35),
    (xgboost_model, 0.25)
]

predictions = ensemble_predict(models, X_test)
```

---

## Technical Insights

### Clinical Prognostic Factors

| Factor | Impact on Outcome |
|--------|-------------------|
| Bilirubin | High = poor prognosis |
| Albumin | Low = poor prognosis |
| Ascites | Present = poor prognosis |
| Stage | Higher = worse outcome |
| Edema | Present = poor prognosis |

### MELD Score Components

The Model for End-Stage Liver Disease uses:
- Bilirubin (liver function)
- Creatinine (kidney function)
- INR/Prothrombin time (clotting)

### Feature Importance (Typical)

| Feature | Importance |
|---------|------------|
| Bilirubin | Very High |
| N_Days | High |
| Stage | High |
| Albumin | High |
| Prothrombin | Medium-High |
| Age | Medium |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 2nd | [Discussion](https://www.kaggle.com/competitions/playground-series-s3e26/discussion/464887) |
| 4th | [Discussion](https://www.kaggle.com/competitions/playground-series-s3e26/discussion/464863) |

---

## Citation

```bibtex
@misc{playground-series-s3e26,
    author = {Walter Reade and Ashley Chow},
    title = {Multi-Class Prediction of Cirrhosis Outcomes},
    year = {2023},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s3e26}},
    note = {Kaggle Playground Series}
}
```
