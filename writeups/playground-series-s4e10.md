# Playground Series - Season 4, Episode 10

> Loan Approval Prediction

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Binary Classification |
| **Data Domain** | Tabular / Finance / Credit Risk |
| **ML Approach** | Gradient Boosting (CatBoost) |
| **Key Techniques** | Feature Engineering, Probability Calibration, Ensemble |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 3,858 |
| **Timeline** | October 2024 |
| **Evaluation Metric** | ROC AUC Score |
| **Host** | Kaggle |

## Problem Description

Predict whether an applicant will be approved for a loan based on their financial and personal information.

### The Task

- Binary classification: loan_status (approved/rejected)
- Predict probability of loan approval
- Handle synthetic data based on real loan data
- Classic credit risk assessment problem

### Dataset Origin

The dataset was synthetically generated from a deep learning model trained on the "Loan Approval Prediction" dataset. Feature distributions are close to but not exactly the same as the original.

---

## Data Description

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with loan_status target |
| `test.csv` | Test data for predictions |
| `sample_submission.csv` | Example submission format |

### Features (27 columns)

| Category | Examples |
|----------|----------|
| Personal | Age, gender, education |
| Employment | Employment length, income |
| Financial | Loan amount, interest rate |
| Credit | Credit history, defaults |
| Property | Home ownership, assets |

### Known Data Issues

Some data artifacts noted by competitors:
- `person_emp_length = 0` for some entries
- Synthetic generation created some edge cases

---

## Evaluation

**Metric**: Area Under ROC Curve (AUC)

```
AUC = Area under the ROC curve
```

Measures the model's ability to distinguish between classes across all probability thresholds.

### Submission Format

```
id,loan_status
58645,0.5
58646,0.5
58647,0.5
```

Predict probabilities, not binary labels.

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

**Author**: Hardy Xu

**Key Approach**:
- CatBoost throughout ("CatBoost All The Way Down")
- Careful feature engineering
- Robust cross-validation
- Minimal ensembling

### 4th Place Solution

**Author**: Ravi Ramakrishnan

**Technique**:
- Thoughtful model choices
- Effective ensemble strategies
- Feature selection optimization

### 8th Place Solution

**Author**: Mahdi Ravaghi

**Innovation**:
- Gradient boosting ensemble
- Data artifact handling
- CV-focused approach

---

## Common Winning Strategies

### 1. CatBoost Dominance

CatBoost excelled in this competition:

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    cat_features=categorical_cols,
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

model.fit(X_train, y_train,
          eval_set=(X_val, y_val),
          early_stopping_rounds=100)
```

### 2. Feature Engineering

```python
# Income-related features
df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt']
df['loan_to_income_pct'] = df['loan_amnt'] / df['person_income'] * 100

# Credit utilization
df['debt_to_income'] = df['loan_amnt'] / (df['person_income'] + 1)

# Employment stability
df['income_per_emp_year'] = df['person_income'] / (df['person_emp_length'] + 1)

# Risk indicators
df['high_int_rate'] = (df['loan_int_rate'] > df['loan_int_rate'].median()).astype(int)
```

### 3. Handling Categorical Features

```python
# CatBoost handles categoricals natively
cat_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Or use target encoding
from sklearn.preprocessing import TargetEncoder
encoder = TargetEncoder()
df[cat_features] = encoder.fit_transform(df[cat_features], df['loan_status'])
```

### 4. Cross-Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    # Train and validate
    pass
```

### 5. Probability Calibration

For AUC optimization, well-calibrated probabilities help:
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(model, method='isotonic', cv=5)
calibrated.fit(X_train, y_train)
```

---

## Technical Insights

### Why CatBoost Won

| Advantage | Explanation |
|-----------|-------------|
| Native categorical support | No need for manual encoding |
| Ordered boosting | Reduces overfitting |
| Symmetric trees | Fast inference |
| Good defaults | Less hyperparameter tuning |

### ROC AUC Optimization

| Approach | Description |
|----------|-------------|
| Probability optimization | Focus on ranking, not thresholds |
| Calibration | Ensure well-ordered predictions |
| Feature importance | Remove noise features |
| Ensemble diversity | Combine different model types |

### Leaderboard Shakeup

This competition had notable shakeup:
- Some jumped from #1 to #299
- Trust CV over public LB
- Avoid overfitting to public test set

### Tips from Top Solutions

1. **Simple can win**: Single CatBoost outperformed complex ensembles
2. **Feature engineering matters**: Ratio features were valuable
3. **Trust CV**: Public LB was not reliable
4. **Handle edge cases**: Employment length of 0, etc.

---

## Key Learnings

From "A jungle of models..." discussion:
- "Sophisticated" approaches can lead to overfitting
- Simple, well-tuned models often win
- Multiple model iterations without discipline hurts
- CV correlation with private LB is key

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/playground-series-s4e10/discussion/543725) |
| 2nd | [Discussion](https://www.kaggle.com/c/playground-series-s4e10/discussion/543766) |
| 4th | [Discussion](https://www.kaggle.com/c/playground-series-s4e10/discussion/543672) |
| 8th | [Discussion](https://www.kaggle.com/c/playground-series-s4e10/discussion/543772) |
| 10th | [Discussion](https://www.kaggle.com/c/playground-series-s4e10/discussion/543735) |

---

## Citation

```bibtex
@misc{playground-series-s4e10,
    author = {Walter Reade and Elizabeth Park},
    title = {Loan Approval Prediction},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s4e10}},
    note = {Kaggle Playground Series}
}
```
