# Playground Series - Season 4, Episode 6

> Classification with an Academic Success Dataset

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-Class Classification |
| **Data Domain** | Tabular / Education |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Feature Engineering, Ensemble Diversity |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 2,684 |
| **Timeline** | June 2024 |
| **Evaluation Metric** | Accuracy Score |
| **Host** | Kaggle |

## Problem Description

Predict the academic risk status of students in higher education based on demographic and academic features.

### The Task

- Multi-class classification: Graduate, Dropout, or Enrolled
- Use student demographics and academic records
- Handle synthetic data based on real education data
- Predict academic outcomes

### Dataset Origin

Synthetically generated from a deep learning model trained on academic success data.

---

## Data Description

### Target Classes

| Class | Description |
|-------|-------------|
| Graduate | Successfully completed studies |
| Dropout | Left before completion |
| Enrolled | Currently enrolled |

### Features

| Category | Examples |
|----------|----------|
| Demographics | Age, gender, nationality |
| Family | Parents' education, occupation |
| Academic | Previous qualifications, grades |
| Economic | Scholarship holder, tuition status |
| Enrollment | Course, attendance mode |

---

## Evaluation

**Metric**: Accuracy Score

```
Accuracy = Correct Predictions / Total Predictions
```

Simple accuracy - no weighted classes.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | Choice of Kaggle merchandise |
| 2nd | Choice of Kaggle merchandise |
| 3rd | Choice of Kaggle merchandise |

Part of the **AutoML Grand Prix** series.

---

## Top Solutions

### 6th Place Solution

**Author**: Matt OP

**Key Finding**: "Many model ensembles were detrimental"
- Simple models outperformed complex ensembles
- Feature selection important
- Trust CV scores

### 10th Place Solution

**Technique**:
- CatBoost primary model
- Careful hyperparameter tuning
- Feature engineering

---

## Common Winning Strategies

### 1. Ensemble Diversity Analysis

Key insight from discussions:

```python
# Check if models ensemble well
def ensemble_analysis(preds1, preds2, y_true):
    # Correlation between predictions
    corr = np.corrcoef(preds1.flatten(), preds2.flatten())[0, 1]

    # Individual accuracies
    acc1 = accuracy_score(y_true, preds1.argmax(axis=1))
    acc2 = accuracy_score(y_true, preds2.argmax(axis=1))

    # Ensemble accuracy
    ensemble = (preds1 + preds2) / 2
    acc_ens = accuracy_score(y_true, ensemble.argmax(axis=1))

    print(f"Correlation: {corr:.4f}")
    print(f"Model 1: {acc1:.4f}, Model 2: {acc2:.4f}")
    print(f"Ensemble: {acc_ens:.4f}")
    print(f"Ensemble gain: {acc_ens - max(acc1, acc2):.4f}")
```

### 2. Feature Engineering

```python
# Academic performance features
df['avg_grade'] = df[['grade_1', 'grade_2']].mean(axis=1)
df['grade_improvement'] = df['grade_2'] - df['grade_1']
df['passed_rate'] = df['units_passed'] / (df['units_enrolled'] + 1)

# Socioeconomic indicators
df['family_education'] = df['mother_education'] + df['father_education']
df['has_scholarship'] = (df['scholarship_holder'] == 'Yes').astype(int)

# Age-related features
df['age_at_enrollment'] = df['age_at_enrollment']
df['is_mature_student'] = (df['age_at_enrollment'] > 25).astype(int)
```

### 3. CatBoost Configuration

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    cat_features=categorical_cols,
    eval_metric='Accuracy',
    random_seed=42,
    early_stopping_rounds=200,
    verbose=100
)
```

### 4. Handling Multi-Class

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Stratified CV for multi-class
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    # Train and evaluate
    pass
```

### 5. When NOT to Ensemble

Key lesson from 6th place:
- High correlation between models → ensemble doesn't help
- Simple single model can win
- Ensemble adds complexity without benefit sometimes

```python
# Only ensemble if models are diverse
def should_ensemble(predictions_list, threshold=0.9):
    correlations = []
    for i in range(len(predictions_list)):
        for j in range(i+1, len(predictions_list)):
            corr = np.corrcoef(
                predictions_list[i].flatten(),
                predictions_list[j].flatten()
            )[0, 1]
            correlations.append(corr)

    avg_corr = np.mean(correlations)
    return avg_corr < threshold  # Only ensemble if not too correlated
```

---

## Technical Insights

### Academic Success Prediction

Key predictive factors:
| Factor | Typical Impact |
|--------|----------------|
| First-year grades | Very high |
| Previous education | High |
| Attendance | High |
| Scholarship | Moderate positive |
| Age | Non-linear |

### AutoML in Competition

This was part of AutoML Grand Prix:
- AutoGluon, H2O, AutoML approaches tested
- Manual tuning often beat AutoML
- Feature engineering still important

### Accuracy vs Other Metrics

For multi-class:
| Metric | Use When |
|--------|----------|
| Accuracy | Balanced classes |
| Macro F1 | Imbalanced, equal class importance |
| Weighted F1 | Imbalanced, proportional importance |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 6th | [Discussion](https://www.kaggle.com/c/playground-series-s4e6/discussion/515989) |
| 10th | [Discussion](https://www.kaggle.com/c/playground-series-s4e6/discussion/515980) |

---

## Citation

```bibtex
@misc{playground-series-s4e6,
    author = {Walter Reade and Elizabeth Park},
    title = {Classification with an Academic Success Dataset},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s4e6}},
    note = {Kaggle Playground Series}
}
```
