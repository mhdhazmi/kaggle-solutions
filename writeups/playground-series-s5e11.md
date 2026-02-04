# Playground Series - Season 5, Episode 11

> Predicting Loan Payback

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Binary Classification |
| **Data Domain** | Tabular / Finance / Lending |
| **ML Approach** | Gradient Boosting + Neural Networks Ensemble |
| **Key Techniques** | Feature Engineering, Target Encoding, Digit Features, Hill Climbing Ensemble |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 3,724 |
| **Participants** | 3,850 |
| **Timeline** | November 1, 2025 - December 1, 2025 |
| **Evaluation Metric** | ROC AUC Score |
| **Host** | Kaggle |

## Problem Description

Predict the probability that a borrower will pay back their loan.

### Dataset

- **Source**: Synthetically generated from a deep learning model trained on the Loan Prediction dataset
- **Size**: 81.3 MB
- **Columns**: 27 features
- **License**: CC BY 4.0

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training set with target |
| `test.csv` | Test set for predictions |
| `sample_submission.csv` | Example submission format |

---

## Evaluation

**Metric**: Area Under the ROC Curve (AUC)

### Submission Format

```csv
id,loan_paid_back
593994,0.5
593995,0.2
593996,0.1
```

---

## Top Solutions

### 1st Place Solution (CV: 0.92818, Private LB: 0.92923)

**Author**: Mahog (mahoganybuttstrings)

*"I have always dreamed of getting 1st place and it has finally come true!"*

#### Feature Engineering (Key to Success)

The most important part of the competition. Best single model (CV 0.92818, LB 0.92923) would have placed 2nd alone!

**Features Used**:

1. **Base feature combinations** (pairs) + Target/Count Encoding
2. **Digit features** from numerical columns
3. **Digit combinations** (pairs, triples, quadruples) + TE/CE
4. **Round features** (from public kernels)
5. **Digit + base feature combinations**
6. **Cross-feature digit combinations**
7. **Target encoding on original dataset** with different targets:
   - Using `employment_status` as target
   - Using `debt_to_income_ratio` as target
8. **Quantile binned numericals**
9. **Categorical version of `credit_score`**

#### Models (100 Total!)

| Model Type | CV | Public LB | Private LB |
|------------|-----|-----------|------------|
| **XGBoost** | 0.928175 | 0.92831 | **0.92923** |
| LightGBM | 0.928081 | 0.92818 | 0.92920 |
| RealMLP | 0.927952 | 0.92812 | 0.92906 |
| TabM | 0.927833 | 0.92805 | 0.92895 |
| LightGBM-dart | 0.927772 | 0.92822 | 0.92920 |
| CatBoost | 0.927656 | 0.92806 | 0.92899 |
| XGBoost (reg) | 0.927334 | 0.92789 | 0.92857 |
| DANet | 0.926766 | 0.92776 | 0.92867 |
| ResNet | 0.926557 | 0.92713 | 0.92798 |
| Logistic Regression | 0.926291 | 0.92669 | 0.92757 |
| Trompt | 0.926000 | 0.92616 | 0.92756 |
| Gandalf | 0.925989 | 0.92741 | 0.92808 |
| Bartz | 0.925944 | 0.92703 | 0.92802 |
| FTTransformer | 0.925272 | 0.92576 | 0.92704 |
| Random Forest | 0.925079 | 0.92549 | 0.92643 |
| DeepFM | 0.924872 | 0.92537 | 0.92645 |
| TabulaRNN | 0.924809 | 0.92587 | 0.92706 |

#### Ensembling Strategy

- **Ridge Regression** and **Hill Climbing** were best ensemblers
- Stacking with non-linear models (LGBM/CB/NNs) performed worse than linear methods

#### Key Insight

*"The formula for getting in top 3 remains the same: a lot of models and a trick or two (which in this case was a lot of FE)"*

---

### 2nd Place Solution (CV: 0.92813, Private LB: 0.92927)

**Author**: AngelosMar (angelosmar1)

*Achieved 2nd place with only 2 submissions!*

#### Feature Engineering

**Key Features**:
1. Target encoded columns (cardinality > 7)
2. Count encoded columns
3. Target encoded with targets from original dataset

**Critical Discovery**: Discretizing/binning high cardinality columns (`annual_income`, `loan_amount`) in different ways before target encoding:

| Binning Method | Impact |
|----------------|--------|
| Quantile binning | Improved CV |
| Uniform binning | Improved CV |
| Round + integer division (`//`) | Improved CV |
| `.astype(int)` (discard decimal) | Improved CV |

**Additional Features**:
- Digit features from all numeric columns
- Digit combinations from some numeric columns
- Ratios of counts between training and original data (over/undersampling detection)

**What Didn't Work**:
- Interaction features
- Adding original data as rows

#### Model Configuration

**Heavy Regularization** was key:

```python
learning_rate=0.01
max_depth=4
subsample=0.5
colsample_bytree=0.2
lambda_l2=15.0
lambda_l1=10.0
```

#### Best Ensemble (7 Models)

| Model | 5-fold CV | Notes |
|-------|-----------|-------|
| LGBM | 0.92813 | Best single model |
| LGBM | 0.9281 | Original data added as rows |
| LGBM | 0.9278 | Original data + fewer features |
| LGBM | 0.9277 | TabPFN logit initialization |
| LGBM | 0.9276 | AutoGluon logit initialization |
| TabM | 0.9277 | - |
| RealMLP | 0.9271 | - |

---

## Common Winning Strategies

### 1. Extensive Feature Engineering
- Digit extraction from numerical columns
- Multiple binning strategies for high-cardinality features
- Target encoding with various target variables

### 2. Diverse Model Ensemble
- Gradient boosting (XGBoost, LightGBM, CatBoost)
- Deep learning (TabM, RealMLP, DANet, FTTransformer)
- Classical ML (Logistic Regression, Random Forest)

### 3. Heavy Regularization
- Low `max_depth`, high `lambda_l1`/`lambda_l2`
- Low `colsample_bytree` and `subsample`

### 4. Linear Ensembling
- Ridge regression outperformed non-linear stacking
- Hill climbing for weight optimization

### 5. Original Dataset Utilization
- Target encoding using original data
- Adding original data as training rows (sometimes)

---

## Technical Requirements

| Requirement | Value |
|-------------|-------|
| Runtime | Standard Kaggle notebook |
| Internet | Allowed (not code competition) |
| External Data | Original dataset allowed |

---

## Solution Links

| Place | Author | Score | Solution |
|-------|--------|-------|----------|
| 1st | Mahog | 0.92923 | [Writeup](https://www.kaggle.com/c/playground-series-s5e11/writeups/1st-place-a-lot-of-features-a-lot-of-models-an) |
| 2nd | AngelosMar | 0.92927 | [Writeup](https://www.kaggle.com/c/playground-series-s5e11/writeups/2nd-place-solution-7-models-but-1-was-also-enou) |

---

## Citation

```bibtex
@misc{playground-series-s5e11,
    author = {Yao Yan and Walter Reade and Elizabeth Park},
    title = {Predicting Loan Payback},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s5e11}},
    note = {Kaggle Playground Series}
}
```
