# Playground Series - Season 4, Episode 11

> Exploring Mental Health Data

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Binary Classification |
| **Data Domain** | Tabular / Healthcare / Mental Health |
| **ML Approach** | Gradient Boosting + Neural Networks |
| **Key Techniques** | Feature Engineering, Probability Calibration, AutoML |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 2,790 |
| **Timeline** | November 2024 |
| **Evaluation Metric** | Accuracy Score |
| **Host** | Kaggle |

## Problem Description

Predict whether individuals experience depression based on mental health survey data.

### The Task

- Binary classification: Depression (0 or 1)
- Use demographic and survey features
- Understand factors contributing to depression
- Handle synthetic data with artifacts

### Dataset Origin

The dataset was synthetically generated from a deep learning model trained on the "Depression Survey/Dataset for Analysis" dataset. Feature distributions are close to but not exactly the same as the original.

---

## Data Description

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with Depression target |
| `test.csv` | Test data for predictions |
| `sample_submission.csv` | Example submission format |

### Features

| Category | Examples |
|----------|----------|
| Demographics | Age, gender, location |
| Work/Life | Occupation, work hours |
| Health | Sleep, exercise habits |
| Social | Family history, relationships |
| Mental state | Stress levels, mood indicators |

### Data Notes

- 41 columns in the dataset
- Data artifacts present in the synthetic data
- Original dataset can be used for training
- Not a particularly difficult dataset to model

---

## Evaluation

**Metric**: Accuracy Score

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Simple accuracy - correct predictions divided by total predictions.

### Submission Format

```
id,Depression
140700,0
140701,0
140702,1
```

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

**Author**: Mahdi Ravaghi

**Key Approach**:
- Careful feature preprocessing
- Gradient boosting ensemble
- Probability threshold optimization
- Handling of data artifacts

### 4th Place Solution

**Author**: Jack Lee

**Technique**:
- Preprocessing pipeline
- AutoML (H2O/AutoGluon)
- Model stacking

### 7th Place Solution

**Author**: Gage Weaver

**Innovation**:
- Single CatBoost model
- Feature engineering focus
- Jumped from 500th public to 7th private

### 13th Place Solution

**Author**: Optimistix

**Technique**:
- Extensive experimentation
- Multiple model iterations
- CV-based selection

### 25th Place Solution

**Author**: Chris Deotte

**Approach**:
- GBDT plus Neural Network
- Trust CV over public leaderboard
- Ensemble approach

---

## Common Winning Strategies

### 1. Feature Engineering

```python
# Survey score combinations
df['mental_health_score'] = df['stress'] + df['anxiety'] + df['mood']

# Life balance indicators
df['work_life_ratio'] = df['work_hours'] / df['leisure_time']

# Risk factors
df['risk_factors'] = df['family_history'] + df['past_episodes']
```

### 2. Handling Synthetic Data Artifacts

```python
# Identify and handle data artifacts
# Look for unusual patterns not in original dataset
df['artifact_indicator'] = identify_artifacts(df)

# Use original dataset for additional training
original_df = pd.read_csv('original_depression_survey.csv')
combined = pd.concat([df, original_df])
```

### 3. Probability Calibration

Isotonic regression for calibration:
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
calibrated.fit(X_train, y_train)

# Better calibrated probabilities for threshold selection
probs = calibrated.predict_proba(X_test)[:, 1]
```

### 4. Threshold Optimization

```python
from sklearn.metrics import accuracy_score

# Find optimal threshold
thresholds = np.arange(0.3, 0.7, 0.01)
best_accuracy = 0
best_threshold = 0.5

for thresh in thresholds:
    preds = (probs >= thresh).astype(int)
    acc = accuracy_score(y_val, preds)
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = thresh
```

### 5. Trust CV Strategy

Important lesson from this competition:
- Public leaderboard was misleading
- Cross-validation more reliable
- Massive shakeup on private leaderboard

---

## Technical Insights

### Accuracy as Metric

Accuracy is straightforward but:
- Sensitive to class imbalance
- Threshold at 0.5 may not be optimal
- Consider calibration for probability models

### Mental Health Data Considerations

| Factor | Importance |
|--------|------------|
| Feature correlation | High - many factors interconnected |
| Missing data | Common in surveys |
| Response bias | Self-reported data |
| Privacy | Handle sensitive data appropriately |

### Model Selection

| Model | Strengths |
|-------|-----------|
| CatBoost | Handles categoricals well |
| LightGBM | Fast and efficient |
| Neural Network | Can capture complex interactions |
| AutoML | Automatic feature/model selection |

### Shakeup Analysis

This competition had significant leaderboard shakeup:
- Some jumped 500+ places
- Public LB scores not reliable
- Focus on robust CV essential

---

## Key Takeaways

1. **Trust CV**: Cross-validation more reliable than public LB
2. **Simple can win**: Single CatBoost achieved 7th place
3. **Data artifacts**: Synthetic data has quirks to exploit or handle
4. **Original data**: Using original dataset can help
5. **Calibration matters**: Isotonic regression improved many solutions

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/playground-series-s4e11/discussion/547327) |
| 4th | [Discussion](https://www.kaggle.com/c/playground-series-s4e11/discussion/546936) |
| 7th | [Discussion](https://www.kaggle.com/c/playground-series-s4e11/discussion/546958) |
| 13th | [Discussion](https://www.kaggle.com/c/playground-series-s4e11/discussion/546964) |
| 25th | [Discussion](https://www.kaggle.com/c/playground-series-s4e11/discussion/547006) |

---

## Citation

```bibtex
@misc{playground-series-s4e11,
    author = {Walter Reade and Elizabeth Park},
    title = {Exploring Mental Health Data},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s4e11}},
    note = {Kaggle Playground Series}
}
```
