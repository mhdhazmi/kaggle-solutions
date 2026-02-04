# Playground Series - Season 5, Episode 4

> Predict Podcast Listening Time

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Regression |
| **Data Domain** | Tabular / Media / Entertainment |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Feature Engineering, Target Encoding, Stacking |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 3,310 |
| **Participants** | 3,454 |
| **Timeline** | April 2025 |
| **Evaluation Metric** | Root Mean Squared Error (RMSE) |
| **Host** | Kaggle |

## Problem Description

Predict the listening time (in minutes) of podcast episodes based on episode and listener characteristics.

### The Task

Given features about:
- Podcast episode metadata
- Listener demographics/behavior
- Content characteristics

Predict: How many minutes the listener will engage with the episode.

### Why It Matters

- **Content Creators**: Optimize episode length
- **Platforms**: Improve recommendations
- **Advertisers**: Better ad placement decisions

---

## Data Description

### Dataset Overview

The dataset was synthetically generated from a deep learning model trained on real podcast listening data.

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with Listening_Time_minutes target |
| `test.csv` | Test data for predictions |
| `sample_submission.csv` | Example submission format |

### Feature Categories

Based on typical podcast datasets:

| Category | Example Features |
|----------|------------------|
| Episode | Duration, genre, host popularity |
| Listener | Age, gender, listening history |
| Context | Time of day, device type |
| Content | Topic, guest presence |

---

## Evaluation

**Metric**: Root Mean Squared Error (RMSE)

```
RMSE = sqrt((1/N) × Σ(yᵢ - ŷᵢ)²)
```

Where:
- ŷᵢ = predicted listening time
- yᵢ = actual listening time
- N = number of samples

### RMSE Properties

- Same units as target (minutes)
- Penalizes large errors heavily
- Sensitive to outliers

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
- Extensive feature engineering
- Multiple gradient boosting models
- Careful cross-validation strategy

### 2nd Place Solution

**Technique**:
- LightGBM + CatBoost ensemble
- Target encoding for categoricals
- Feature interactions

### 3rd Place Solution

**Innovation**:
- Neural network combined with GBM
- Embedding layers for categoricals
- Stacked generalization

---

## Common Winning Strategies

### 1. Feature Engineering

| Feature Type | Examples |
|--------------|----------|
| Ratios | Listening time / Episode duration |
| Aggregations | Mean listening time per genre |
| Time-based | Hour of day, day of week |
| Interaction | Genre × Listener age group |
| Encoding | Target encoding for high-cardinality |

### 2. Model Ensemble

Typical blend:
```python
final = 0.4 * lgbm + 0.35 * xgb + 0.25 * catboost
```

### 3. Cross-Validation

- K-Fold (k=5 or k=10)
- Group K-Fold if user IDs present
- Repeated K-Fold for stability

### 4. Handling Categoricals

| Method | When to Use |
|--------|-------------|
| One-hot | Low cardinality (<10) |
| Label encoding | Tree models |
| Target encoding | High cardinality |
| Embeddings | Neural networks |

### 5. Outlier Treatment

For listening time:
- Clip extreme values
- Use robust scalers
- Log transformation (optional)

---

## Technical Insights

### Target Distribution

Listening time typically shows:
- Right-skewed distribution
- Natural upper bound (episode length)
- Possible modes (full episode, quick drop-off)

### Feature Importance

Usually important features:
1. Episode duration (natural ceiling)
2. Listener historical behavior
3. Content genre/topic
4. Time of day/context

### Leakage Prevention

Watch for features that leak target:
- Average listening time per episode
- Completion rates
- Any post-hoc metrics

### Baseline Models

Good starting points:
```python
# Simple baseline: predict mean
baseline = y_train.mean()

# Better baseline: predict by genre mean
baseline_by_genre = train.groupby('genre')['target'].mean()
```

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/playground-series-s5e4/discussion/575784) |
| 2nd | [Discussion](https://www.kaggle.com/c/playground-series-s5e4/discussion/575840) |
| 3rd | [Discussion](https://www.kaggle.com/c/playground-series-s5e4/discussion/575862) |
| 4th | [Discussion](https://www.kaggle.com/c/playground-series-s5e4/discussion/575782) |
| 5th | [Discussion](https://www.kaggle.com/c/playground-series-s5e4/discussion/575839) |

---

## Citation

```bibtex
@misc{playground-series-s5e4,
    author = {Walter Reade and Elizabeth Park},
    title = {Predict Podcast Listening Time},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s5e4}},
    note = {Kaggle Playground Series}
}
```
