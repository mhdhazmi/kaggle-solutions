# Playground Series - Season 4, Episode 2

> Multi-Class Prediction of Obesity Risk

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-Class Classification |
| **Data Domain** | Health / Lifestyle |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Feature Engineering, Ordinal Classification |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 3,587 |
| **Timeline** | February 2024 |
| **Evaluation Metric** | Accuracy |
| **Host** | Kaggle |

## Problem Description

Predict obesity risk levels based on eating habits and physical condition - a multi-class classification with 7 ordinal categories.

### The Challenge

- Classify individuals into 7 obesity categories
- Use lifestyle and physical features
- Handle ordinal nature of target variable
- Deal with synthetic data characteristics

### The 7 Obesity Categories

| Class | BMI Range | Description |
|-------|-----------|-------------|
| Insufficient_Weight | < 18.5 | Underweight |
| Normal_Weight | 18.5-24.9 | Healthy weight |
| Overweight_Level_I | 25-27.4 | Mildly overweight |
| Overweight_Level_II | 27.5-29.9 | Overweight |
| Obesity_Type_I | 30-34.9 | Class I obesity |
| Obesity_Type_II | 35-39.9 | Class II obesity |
| Obesity_Type_III | >= 40 | Class III (severe) obesity |

---

## Data Description

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with NObeyesdad target |
| `test.csv` | Test data for predictions |
| `sample_submission.csv` | Submission format |

### Features

| Category | Features |
|----------|----------|
| Physical | Height, Weight, Age, Gender |
| Eating Habits | FAVC (high caloric food), FCVC (vegetables), NCP (meals), CAEC (snacking), CH2O (water) |
| Lifestyle | SCC (calorie monitoring), FAF (physical activity), TUE (technology use), CALC (alcohol), SMOKE |
| Family | family_history_with_overweight |
| Transportation | MTRANS (transportation mode) |

### Dataset Origin

Synthetically generated from a deep learning model trained on obesity estimation data.

---

## Evaluation

**Metric**: Classification Accuracy

```python
from sklearn.metrics import accuracy_score

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
```

Higher is better. Simple accuracy across all 7 classes.

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
- LightGBM with ordinal encoding
- BMI feature as primary predictor
- Careful cross-validation strategy

### 2nd Place Solution

**Technique**:
- CatBoost + XGBoost ensemble
- Feature interactions with BMI
- Label encoding optimization

### 3rd Place Solution

**Innovation**:
- Neural network with embeddings
- Ordinal regression approach
- Feature importance-based selection

---

## Common Winning Strategies

### 1. Calculate BMI (Critical Feature!)

```python
def calculate_bmi_features(df):
    """BMI is essentially the target - massive signal!"""
    # Basic BMI calculation
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    # BMI-derived features
    df['BMI_rounded'] = df['BMI'].round(1)
    df['BMI_category_calc'] = pd.cut(
        df['BMI'],
        bins=[0, 18.5, 25, 27.5, 30, 35, 40, 100],
        labels=[0, 1, 2, 3, 4, 5, 6]
    )

    # BMI interactions
    df['BMI_x_Age'] = df['BMI'] * df['Age']
    df['BMI_x_FAF'] = df['BMI'] * df['FAF']  # Physical activity

    return df
```

### 2. Feature Engineering

```python
def engineer_features(df):
    """Create informative features from lifestyle data"""

    # Calculate BMI first (most important!)
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    # Activity features
    df['activity_score'] = df['FAF'] - df['TUE'] / 2  # More activity, less tech
    df['sedentary'] = (df['FAF'] == 0).astype(int)

    # Eating habit features
    df['calorie_risk'] = df['FAVC'].astype(int) - df['FCVC'] / 3  # High cal vs veggies
    df['meal_frequency'] = df['NCP'] + df['CAEC'].map({
        'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3
    })

    # Hydration
    df['hydration_low'] = (df['CH2O'] < 2).astype(int)

    # Lifestyle risk score
    df['lifestyle_risk'] = (
        df['FAVC'].astype(int) * 2 +
        (3 - df['FCVC']) +
        df['SMOKE'].astype(int) * 2 +
        df['CALC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
    )

    # Age groups
    df['age_group'] = pd.cut(df['Age'], bins=[0, 20, 30, 40, 50, 100],
                              labels=['young', 'young_adult', 'adult', 'middle', 'senior'])

    return df
```

### 3. Ordinal Classification

```python
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Treat ordinal classification as cumulative binary problems"""

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.classes_ = None
        self.classifiers_ = []

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        n_classes = len(self.classes_)

        # Train K-1 binary classifiers
        for k in range(n_classes - 1):
            # Binary target: is class > k?
            binary_y = (y > self.classes_[k]).astype(int)

            clf = clone(self.base_estimator)
            clf.fit(X, binary_y)
            self.classifiers_.append(clf)

        return self

    def predict_proba(self, X):
        # P(Y = k) = P(Y > k-1) - P(Y > k)
        probs = np.zeros((X.shape[0], len(self.classes_)))

        prev_prob = np.ones(X.shape[0])
        for k, clf in enumerate(self.classifiers_):
            current_prob = clf.predict_proba(X)[:, 1]
            probs[:, k] = prev_prob - current_prob
            prev_prob = current_prob

        probs[:, -1] = prev_prob  # Last class
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
```

### 4. LightGBM Configuration

```python
from lightgbm import LGBMClassifier

def train_lgbm(X_train, y_train, X_val, y_val):
    """Optimized LightGBM for multi-class"""

    # Encode target
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)

    model = LGBMClassifier(
        objective='multiclass',
        num_class=7,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )

    model.fit(
        X_train, y_train_enc,
        eval_set=[(X_val, y_val_enc)],
        callbacks=[lgb.early_stopping(50)]
    )

    return model, label_encoder
```

### 5. Ensemble Strategy

```python
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

def create_ensemble():
    """Voting ensemble of gradient boosting models"""

    lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=8)
    catboost = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=8, verbose=0)
    xgboost = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=8)

    ensemble = VotingClassifier(
        estimators=[
            ('lgbm', lgbm),
            ('catboost', catboost),
            ('xgboost', xgboost)
        ],
        voting='soft',  # Use probability averaging
        weights=[0.4, 0.35, 0.25]
    )

    return ensemble
```

---

## Technical Insights

### BMI is Almost the Target

The competition essentially tests whether you calculate BMI:

| BMI Formula | Impact |
|-------------|--------|
| Weight / Height² | Near-perfect predictor |
| Without BMI | ~0.65 accuracy |
| With BMI | ~0.90+ accuracy |

### Feature Importance (Typical)

| Feature | Importance |
|---------|------------|
| BMI (calculated) | Very High |
| Weight | High |
| Height | High |
| Age | Medium |
| Physical Activity | Medium |
| Eating Habits | Low-Medium |
| Family History | Low |

### Class Distribution

| Class | Approx % |
|-------|----------|
| Obesity_Type_I | 18% |
| Obesity_Type_II | 18% |
| Obesity_Type_III | 18% |
| Overweight_Level_I | 12% |
| Overweight_Level_II | 12% |
| Normal_Weight | 12% |
| Insufficient_Weight | 10% |

### Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Missing BMI | Always calculate BMI |
| Ignoring ordinal structure | Consider ordinal regression |
| Over-engineering | Simple models work well |
| Complex ensembles | Diminishing returns |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/competitions/playground-series-s4e2/discussion/486875) |
| 2nd | [Discussion](https://www.kaggle.com/competitions/playground-series-s4e2/discussion/486937) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/playground-series-s4e2/discussion/486911) |

---

## Citation

```bibtex
@misc{playground-series-s4e2,
    author = {Walter Reade and Ashley Chow},
    title = {Multi-Class Prediction of Obesity Risk},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s4e2}},
    note = {Kaggle Playground Series}
}
```
