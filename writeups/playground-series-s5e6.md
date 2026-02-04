# Playground Series - Season 5, Episode 6

> Predicting Optimal Fertilizers

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-class Classification / Recommendation |
| **Data Domain** | Tabular / Agriculture |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Feature Engineering, MAP@K Optimization |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 2,648 |
| **Timeline** | June 2025 |
| **Evaluation Metric** | MAP@K (Mean Average Precision at K) |
| **Host** | Kaggle |

## Problem Description

Predict the optimal fertilizer recommendations for different soil and crop conditions.

### The Task

Given soil characteristics and crop type, recommend the most suitable fertilizers ranked by priority.

### Dataset

- **Source**: Synthetically generated from agricultural data
- **Features**: Soil nutrients, pH, moisture, crop type
- **Target**: Fertilizer recommendations (ranked list)
- **License**: CC BY 4.0

---

## Evaluation

**Metric**: MAP@K (Mean Average Precision at K)

Measures the quality of ranked recommendations:
- Higher rank for correct predictions = higher score
- Rewards both precision and ranking quality

---

## Top Solutions

### 1st Place Solution

**Discussion Link**: [1st Place](https://www.kaggle.com/c/playground-series-s5e6/discussion/587393)

### 2nd Place Solution

**Discussion Link**: [2nd Place](https://www.kaggle.com/c/playground-series-s5e6/discussion/587398)

### 3rd Place Solution

**Discussion Link**: [3rd Place](https://www.kaggle.com/c/playground-series-s5e6/discussion/587464)

### 4th Place Solution

**Discussion Link**: [4th Place](https://www.kaggle.com/c/playground-series-s5e6/discussion/587405)

### 5th Place Solution

**Discussion Link**: [5th Place](https://www.kaggle.com/c/playground-series-s5e6/discussion/587392)

---

## Common Winning Strategies

### 1. Multi-Class Classification
Treating each fertilizer as a class and predicting probabilities.

### 2. Learning-to-Rank
Using ranking-specific loss functions and models.

### 3. Feature Engineering
Creating soil-nutrient interaction features, deficiency indicators.

### 4. Ensemble Methods
Combining multiple classifiers with different strengths.

### 5. Post-Processing
Optimizing the final ranking based on class probabilities.

---

## Technical Insights

### MAP@K Optimization

To optimize MAP@K directly:
1. Train classifier to predict class probabilities
2. Rank classes by probability
3. Return top-K recommendations

### Domain Knowledge

Key agricultural features:
- **N, P, K levels**: Nitrogen, Phosphorus, Potassium
- **pH**: Soil acidity/alkalinity
- **Crop type**: Different crops need different nutrients
- **Moisture**: Water content affects nutrient availability

---

## Solution Links

| Place | Discussion |
|-------|------------|
| 1st | [Link](https://www.kaggle.com/c/playground-series-s5e6/discussion/587393) |
| 2nd | [Link](https://www.kaggle.com/c/playground-series-s5e6/discussion/587398) |
| 3rd | [Link](https://www.kaggle.com/c/playground-series-s5e6/discussion/587464) |
| 4th | [Link](https://www.kaggle.com/c/playground-series-s5e6/discussion/587405) |
| 5th | [Link](https://www.kaggle.com/c/playground-series-s5e6/discussion/587392) |

---

## Citation

```bibtex
@misc{playground-series-s5e6,
    author = {Walter Reade and Elizabeth Park},
    title = {Predicting Optimal Fertilizers},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s5e6}},
    note = {Kaggle Playground Series}
}
```
