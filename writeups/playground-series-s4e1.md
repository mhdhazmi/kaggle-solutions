# Playground Series - Season 4, Episode 1

> Binary Classification with a Bank Churn Dataset

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Binary Classification |
| **Data Domain** | Finance / Banking |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Feature Engineering, Handling Imbalanced Data |
| **Difficulty** | Beginner |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 3,632 |
| **Timeline** | January 2024 |
| **Evaluation Metric** | ROC AUC |
| **Host** | Kaggle |

## Problem Description

Predict whether a bank customer will churn (leave the bank) - a classic business problem in customer retention.

### The Challenge

- Binary classification of customer churn
- Use customer demographics and account information
- Handle class imbalance (churners are minority)
- Work with synthetic data based on real patterns

### Why Churn Prediction Matters

- **Customer Retention**: Identify at-risk customers
- **Cost Savings**: Cheaper to retain than acquire
- **Targeted Marketing**: Personalized retention offers
- **Business Planning**: Forecast customer base

---

## Data Description

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with Exited target |
| `test.csv` | Test data for predictions |
| `sample_submission.csv` | Submission format |

### Features

| Feature | Description |
|---------|-------------|
| `CustomerId` | Unique identifier |
| `Surname` | Customer surname |
| `CreditScore` | Credit score |
| `Geography` | Country (France, Spain, Germany) |
| `Gender` | Male/Female |
| `Age` | Customer age |
| `Tenure` | Years as customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products |
| `HasCrCard` | Has credit card (0/1) |
| `IsActiveMember` | Active member (0/1) |
| `EstimatedSalary` | Estimated salary |
| `Exited` | Target: 1 = churned |

### Dataset Origin

Synthetically generated from a deep learning model trained on a classic bank churn dataset.

---

## Evaluation

**Metric**: ROC AUC (Area Under the Receiver Operating Characteristic Curve)

```python
from sklearn.metrics import roc_auc_score

def evaluate(y_true, y_pred_proba):
    return roc_auc_score(y_true, y_pred_proba)
```

Higher is better. Measures ability to rank churners above non-churners.

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
- LightGBM + CatBoost ensemble
- Extensive feature engineering
- Cross-validation with stratification

### 2nd Place Solution

**Technique**:
- XGBoost with tuned parameters
- Feature interactions
- Original data augmentation

### 3rd Place Solution

**Innovation**:
- Neural network with embeddings
- SMOTE for class balance
- Stacking ensemble

---

## Common Winning Strategies

### 1. Feature Engineering

```python
def engineer_features(df):
    """Create informative features for churn prediction"""

    # Age-related features
    df['age_group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100],
                              labels=['young', 'adult', 'middle', 'senior', 'elderly'])
    df['is_senior'] = (df['Age'] >= 50).astype(int)

    # Balance features
    df['has_balance'] = (df['Balance'] > 0).astype(int)
    df['balance_salary_ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['balance_per_product'] = df['Balance'] / (df['NumOfProducts'] + 1)

    # Credit score features
    df['credit_score_group'] = pd.cut(df['CreditScore'],
                                       bins=[0, 580, 670, 740, 800, 900],
                                       labels=['poor', 'fair', 'good', 'very_good', 'excellent'])

    # Engagement features
    df['engagement_score'] = df['IsActiveMember'] + df['HasCrCard'] + (df['NumOfProducts'] > 1).astype(int)

    # Tenure features
    df['tenure_age_ratio'] = df['Tenure'] / (df['Age'] + 1)
    df['is_new_customer'] = (df['Tenure'] <= 2).astype(int)
    df['is_loyal_customer'] = (df['Tenure'] >= 5).astype(int)

    # Geography features (known patterns)
    df['is_germany'] = (df['Geography'] == 'Germany').astype(int)  # Higher churn

    # Interaction features
    df['age_balance'] = df['Age'] * df['Balance']
    df['products_active'] = df['NumOfProducts'] * df['IsActiveMember']
    df['germany_senior'] = df['is_germany'] * df['is_senior']

    return df
```

### 2. Handling Class Imbalance

```python
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

def handle_imbalance(X_train, y_train, method='class_weight'):
    """Handle class imbalance in training data"""

    if method == 'smote':
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        return X_balanced, y_balanced, None

    elif method == 'class_weight':
        # Calculate class weights for model training
        weights = compute_class_weight('balanced',
                                       classes=np.unique(y_train),
                                       y=y_train)
        class_weights = dict(zip(np.unique(y_train), weights))
        return X_train, y_train, class_weights

    elif method == 'undersample':
        # Undersample majority class
        minority_size = y_train.sum()
        majority_idx = np.where(y_train == 0)[0]
        minority_idx = np.where(y_train == 1)[0]

        sampled_majority = np.random.choice(majority_idx, minority_size * 2, replace=False)
        balanced_idx = np.concatenate([sampled_majority, minority_idx])

        return X_train.iloc[balanced_idx], y_train.iloc[balanced_idx], None
```

### 3. LightGBM Configuration

```python
from lightgbm import LGBMClassifier

def train_lgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM for churn prediction"""

    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = LGBMClassifier(
        objective='binary',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    return model
```

### 4. Cross-Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

def stratified_cv(model_fn, X, y, n_splits=5):
    """Stratified K-fold cross-validation"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_predictions = np.zeros(len(X))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn()
        model.fit(X_train, y_train)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = val_pred

        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        print(f"Fold {fold+1}: AUC = {fold_auc:.4f}")

    print(f"Mean AUC: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    return oof_predictions, fold_scores
```

### 5. Ensemble Strategy

```python
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

def create_ensemble():
    """Create voting ensemble"""

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

### Key Churn Indicators

Based on typical bank churn patterns:

| Factor | Impact | Direction |
|--------|--------|-----------|
| Age (older) | High | More churn |
| Germany | High | More churn |
| NumOfProducts (1 or 3+) | Medium | More churn |
| Balance (zero) | Medium | More churn |
| IsActiveMember (no) | Medium | More churn |
| Gender (female) | Low-Medium | Slightly more churn |

### Feature Importance (Typical)

| Feature | Importance |
|---------|------------|
| Age | Very High |
| NumOfProducts | High |
| Balance | High |
| IsActiveMember | Medium-High |
| Geography | Medium |
| CreditScore | Medium |
| Tenure | Low-Medium |
| EstimatedSalary | Low |

### Class Distribution

| Class | Typical % |
|-------|-----------|
| Not Churned (0) | ~80% |
| Churned (1) | ~20% |

### Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Ignoring imbalance | Use class weights or SMOTE |
| Overfitting | Strong regularization |
| Feature leakage | Careful feature creation |
| Complex models | Simple often wins |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/competitions/playground-series-s4e1/discussion/472835) |
| 2nd | [Discussion](https://www.kaggle.com/competitions/playground-series-s4e1/discussion/472897) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/playground-series-s4e1/discussion/472781) |

---

## Citation

```bibtex
@misc{playground-series-s4e1,
    author = {Walter Reade and Ashley Chow},
    title = {Binary Classification with a Bank Churn Dataset},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s4e1}},
    note = {Kaggle Playground Series}
}
```
