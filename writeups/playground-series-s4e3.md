# Playground Series - Season 4, Episode 3

> Steel Plate Defect Prediction

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-Label Classification |
| **Data Domain** | Manufacturing / Quality Control |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Multi-Output Classification, Feature Engineering |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 2,199 |
| **Timeline** | March 2024 |
| **Evaluation Metric** | Mean AUC (across 7 defect types) |
| **Host** | Kaggle |

## Problem Description

Predict the probability of various defects on steel plates - a classic manufacturing quality control problem with multi-label classification.

### The Challenge

- Predict probabilities for 7 different defect types
- Each plate can have multiple defects simultaneously
- Use physical measurements of steel plates
- Handle imbalanced defect classes

### The 7 Defect Types

| Defect | Description |
|--------|-------------|
| Pastry | Surface pastry defect |
| Z_Scratch | Z-shaped scratch |
| K_Scatch | K-shaped scratch |
| Stains | Surface stains |
| Dirtiness | Dirty spots |
| Bumps | Surface bumps |
| Other_Faults | Other defect types |

---

## Data Description

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with 7 binary targets |
| `test.csv` | Test data for predictions |
| `sample_submission.csv` | Submission format |

### Dataset Origin

Synthetically generated from a deep learning model trained on the UCI Steel Plates Faults dataset. Features distributions are similar but not identical to the original.

### Features

Physical measurements of steel plates:
- Geometric properties (length, width, area)
- Surface characteristics (luminosity, steel type)
- Manufacturing parameters
- Edge measurements

---

## Evaluation

**Metric**: Mean AUC across 7 defect categories

```python
from sklearn.metrics import roc_auc_score
import numpy as np

def mean_auc(y_true, y_pred):
    """
    y_true: (N, 7) binary labels
    y_pred: (N, 7) probability predictions
    """
    aucs = []
    defect_names = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
                    'Dirtiness', 'Bumps', 'Other_Faults']

    for i, name in enumerate(defect_names):
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        aucs.append(auc)
        print(f"{name}: AUC = {auc:.4f}")

    return np.mean(aucs)
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
- LightGBM with careful hyperparameter tuning
- Feature engineering from domain knowledge
- Cross-validation with stratification

### 2nd Place Solution

**Technique**:
- CatBoost + XGBoost ensemble
- Original UCI data augmentation
- Multi-output model architecture

### 3rd Place Solution

**Innovation**:
- Neural network for multi-label
- Feature selection based on importance
- Threshold optimization per defect

---

## Common Winning Strategies

### 1. Multi-Label Model Setup

```python
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier

# Option 1: Separate model per label
def train_separate_models(X_train, y_train, X_test):
    predictions = []
    targets = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
               'Dirtiness', 'Bumps', 'Other_Faults']

    for target in targets:
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train, y_train[target])
        pred = model.predict_proba(X_test)[:, 1]
        predictions.append(pred)

    return np.column_stack(predictions)

# Option 2: Multi-output wrapper
multi_model = MultiOutputClassifier(
    LGBMClassifier(n_estimators=500, learning_rate=0.05)
)
multi_model.fit(X_train, y_train[targets])
```

### 2. Feature Engineering

```python
def engineer_features(df):
    """Create domain-specific features for steel plates"""

    # Area-based features
    df['area'] = df['X_Maximum'] * df['Y_Maximum']
    df['perimeter'] = 2 * (df['X_Maximum'] + df['Y_Maximum'])
    df['aspect_ratio'] = df['X_Maximum'] / (df['Y_Maximum'] + 1e-8)

    # Pixel statistics
    df['pixel_ratio'] = df['Pixels_Areas'] / (df['area'] + 1)
    df['pixel_density'] = df['Pixels_Areas'] / (df['perimeter'] + 1)

    # Edge features
    df['edge_index'] = df['Edges_Index'] / (df['perimeter'] + 1)
    df['edge_ratio'] = df['Edges_X_Index'] / (df['Edges_Y_Index'] + 1e-8)

    # Luminosity features
    df['luminosity_range'] = df['Maximum_of_Luminosity'] - df['Minimum_of_Luminosity']
    df['luminosity_std'] = (df['Maximum_of_Luminosity'] - df['Minimum_of_Luminosity']) / 2

    # Log transforms for skewed features
    for col in ['Pixels_Areas', 'X_Perimeter', 'Y_Perimeter']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])

    # Interaction features
    df['steel_type_area'] = df['Steel_Plate_Thickness'] * df['area']

    return df
```

### 3. Cross-Validation with Multi-Label Stratification

```python
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

def multilabel_cv(X, y, model_class, n_splits=5, **model_params):
    """
    Cross-validation for multi-label classification
    """
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_predictions = np.zeros_like(y, dtype=float)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_preds = []
        for col in y.columns:
            model = model_class(**model_params)
            model.fit(X_train, y_train[col])
            pred = model.predict_proba(X_val)[:, 1]
            fold_preds.append(pred)

        fold_preds = np.column_stack(fold_preds)
        oof_predictions[val_idx] = fold_preds

        # Calculate fold AUC
        aucs = [roc_auc_score(y_val.iloc[:, i], fold_preds[:, i])
                for i in range(y.shape[1])]
        fold_scores.append(np.mean(aucs))
        print(f"Fold {fold+1}: Mean AUC = {np.mean(aucs):.4f}")

    return oof_predictions, fold_scores
```

### 4. Using Original UCI Data

```python
def combine_with_original(synthetic_train, synthetic_test):
    """
    Combine synthetic data with original UCI Steel Plates dataset
    """
    # Load original UCI data
    original = pd.read_csv('original_steel_faults.csv')

    # Align column names if necessary
    original = align_column_names(original, synthetic_train)

    # Combine training data
    combined_train = pd.concat([synthetic_train, original], ignore_index=True)

    # Handle any distribution differences
    # Option: Weight original data less
    sample_weights = np.ones(len(combined_train))
    sample_weights[len(synthetic_train):] = 0.5  # Lower weight for original

    return combined_train, sample_weights
```

### 5. Ensemble with Different Models

```python
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

def ensemble_predict(X_train, y_train, X_test, targets):
    """Ensemble of gradient boosting models"""

    models = {
        'lgbm': LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6),
        'catboost': CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=0),
        'xgboost': XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6)
    }

    weights = {'lgbm': 0.4, 'catboost': 0.35, 'xgboost': 0.25}

    final_predictions = np.zeros((len(X_test), len(targets)))

    for model_name, model in models.items():
        model_preds = []
        for target in targets:
            model_clone = clone(model)
            model_clone.fit(X_train, y_train[target])
            pred = model_clone.predict_proba(X_test)[:, 1]
            model_preds.append(pred)

        model_preds = np.column_stack(model_preds)
        final_predictions += weights[model_name] * model_preds

    return final_predictions
```

---

## Technical Insights

### Defect Correlation

Some defects may be correlated:

| Defect Pair | Correlation | Implication |
|-------------|-------------|-------------|
| Scratches | Often co-occur | Joint modeling may help |
| Bumps + Dirtiness | Related processes | Feature interaction |
| Stains | Often independent | Separate model ok |

### Class Imbalance

| Defect | Approximate Frequency |
|--------|----------------------|
| Other_Faults | ~35% (most common) |
| Pastry | ~10% |
| Z_Scratch | ~15% |
| K_Scatch | ~15% |
| Stains | ~8% |
| Dirtiness | ~8% |
| Bumps | ~10% |

### Feature Importance (Typical)

| Feature Group | Importance |
|---------------|------------|
| Pixel measurements | High |
| Luminosity | Medium-High |
| Geometric properties | Medium |
| Edge indices | Medium |
| Steel type | Low-Medium |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/competitions/playground-series-s4e3/discussion/490462) |
| 2nd | [Discussion](https://www.kaggle.com/competitions/playground-series-s4e3/discussion/490450) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/playground-series-s4e3/discussion/490521) |

---

## Citation

```bibtex
@misc{playground-series-s4e3,
    author = {Walter Reade and Ashley Chow},
    title = {Steel Plate Defect Prediction},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s4e3}},
    note = {Kaggle Playground Series}
}
```
