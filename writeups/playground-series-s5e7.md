# Playground Series - Season 5, Episode 7

> Predict the Introverts from the Extroverts

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Binary Classification |
| **Data Domain** | Tabular / Psychology / Personality |
| **ML Approach** | Gradient Boosting + MLP Ensemble |
| **Key Techniques** | Missing Value Imputation, Threshold Tuning, Model Blending |
| **Difficulty** | Beginner |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 4,329 |
| **Participants** | 4,447 |
| **Timeline** | July 2025 |
| **Evaluation Metric** | Accuracy Score |
| **Host** | Kaggle |

## Problem Description

Predict whether a person is an **Introvert** or **Extrovert** based on their social behavior and personality traits.

### The Task

Given various behavioral features about a person, classify them as either:
- **Introvert**: Prefers solitary activities, smaller social circles
- **Extrovert**: Enjoys social events, larger friend groups

### Why It's Interesting

This competition provided a unique challenge due to:
- **High variance**: Small dataset with significant randomness
- **Discrete features**: Limited numerical range made patterns hard to distinguish
- **Missing values**: Many NaN values requiring careful imputation

---

## Data Description

### Dataset Overview

The dataset was synthetically generated from a deep learning model trained on the original "Extrovert vs. Introvert Behavior" dataset. Feature distributions are close to, but not exactly the same as, the original data.

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training dataset with Personality target |
| `test.csv` | Test dataset for predictions |
| `sample_submission.csv` | Example submission format |

### Features

#### Target Variable
- **Personality**: Binary target (Introvert/Extrovert)

#### Numerical Features

| Feature | Description | Pattern |
|---------|-------------|---------|
| `Time_spent_Alone` | Hours spent alone daily | Introverts: median 7.0, Extroverts: median 2.0 |
| `Social_event_attendance` | Frequency of social events | Higher for extroverts |
| `Going_outside` | Frequency of outdoor activities | Higher for extroverts |
| `Friends_circle_size` | Number of close friends | Larger for extroverts |
| `Post_frequency` | Social media posting frequency | Generally higher for extroverts |

#### Categorical Features

| Feature | Description | Values |
|---------|-------------|--------|
| `Stage_fear` | Fear of public speaking | Yes/No |
| `Drained_after_socializing` | Feeling tired after social events | Yes/No |

### Key Data Characteristics

- **Size**: Relatively small dataset (~18K training samples)
- **Missing Values**: Significant NaN values across most features
- **No Outliers**: Numerical features showed no significant outliers
- **Class Balance**: Relatively balanced between Introvert and Extrovert

### Distinguishing Patterns

```
Extroverts: Time alone < 4 hours
Introverts: Time alone 4-11 hours
```

---

## Evaluation

**Metric**: Accuracy Score

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

### Challenge

Due to the small dataset and discrete features, the public/private leaderboard showed significant shake-up, making this competition feel like "gambling" according to many participants.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | Choice of Kaggle merchandise |
| 2nd | Choice of Kaggle merchandise |
| 3rd | Choice of Kaggle merchandise |

*Note: Merchandise only awarded once per person in the series*

---

## Top Solutions

### 3rd Place Solution (by Jordi Rosell)

**Public Rank**: 42nd (jumped to 3rd on private leaderboard!)

**Key Insight**: "I spent only a few hours on it with just 4 submissions"

**Approach**:
A simple blend of multiple model types:
- **50% Tree-based models** (e.g., LightGBM, XGBoost)
- **45% MLP** (Multi-Layer Perceptron neural network)
- **5% Logistic Regression**

**Critical Technique**:
- Used **40% probability threshold** for labeling instead of standard 50%
- This threshold tuning was key to the top performance

**Referenced Notebook**: The solution credited @mahmoudehab6677's "Personality Classification" notebook (95 upvotes, best score 0.969838)

---

## Common Winning Strategies

### 1. Model Blending

Combining diverse model types:
```
Final = 0.50 × Tree-based + 0.45 × MLP + 0.05 × LogReg
```

### 2. Threshold Optimization

Instead of default 0.5 threshold:
- Experiment with thresholds (0.4, 0.45, 0.55, etc.)
- Optimize based on CV or public LB feedback

### 3. Missing Value Handling

Approaches used:
- Iterative Imputer
- Mode/median imputation per class
- KNN imputation

### 4. TabPFN

Some participants reported success with TabPFN, a transformer-based model designed for small tabular datasets.

### 5. Conservative Submissions

Given the high variance:
- Fewer submissions often outperformed many attempts
- Trust CV more than public LB

---

## Technical Insights

### Why the Shake-up?

This competition had one of the most dramatic leaderboard shake-ups:

| Factor | Impact |
|--------|--------|
| Small dataset | High variance in test set |
| Discrete features | Limited discriminative power |
| Public LB step size | 0.0008097 per correct prediction |
| Private LB step size | 0.0002024 per correct prediction |

### Lessons Learned

1. **Don't chase public LB**: Many top public performers dropped significantly
2. **Ensemble diverse models**: Combining tree-based, neural, and linear models helped
3. **Keep it simple**: Complex feature engineering often didn't help
4. **Threshold matters**: Non-default thresholds improved accuracy

### Feature Importance Patterns

```
Most predictive features:
1. Time_spent_Alone (highest discriminative power)
2. Drained_after_socializing
3. Social_event_attendance
4. Friends_circle_size
```

---

## Solution Links

| Place | Solution |
|-------|----------|
| 3rd* | [Writeup](https://www.kaggle.com/c/playground-series-s5e7/writeups/3rd-place-solution-predict-the-introverts-from-the) |

*Note: Due to extreme shake-up, final rankings differ significantly from public leaderboard positions.

---

## Related Resources

- [Personality Classification Notebook](https://www.kaggle.com/code/mahmoudehab6677/personality-classification) - 95 upvotes, referenced by winning solution
- [EDA: What Separates Introverts from Extroverts?](https://www.kaggle.com/competitions/playground-series-s5e7/discussion) - Community EDA
- [Extrovert vs. Introvert Behavior Data](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data) - Original dataset

---

## Citation

```bibtex
@misc{playground-series-s5e7,
    author = {Walter Reade and Elizabeth Park},
    title = {Predict the Introverts from the Extroverts},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s5e7}},
    note = {Kaggle Playground Series}
}
```
