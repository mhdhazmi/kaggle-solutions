# Learning Agency Lab - Automated Essay Scoring 2.0

> Improve upon essay scoring algorithms to improve student learning outcomes

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Ordinal Regression / NLP |
| **Data Domain** | Education / Text Analysis |
| **ML Approach** | Transformer Fine-tuning (DeBERTa) |
| **Key Techniques** | QWK Optimization, Ensemble, Threshold Tuning |
| **Difficulty** | Intermediate-Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 2,706 |
| **Timeline** | April - July 2024 |
| **Evaluation Metric** | Cohen's Kappa (Quadratic Weighted) |
| **Host** | The Learning Agency Lab |

## Problem Description

Train models to automatically score student essays, reducing the high expense and time required for hand grading while maintaining reliability.

### The Challenge

- Score student essays on ordinal scale (1-6)
- Handle diverse essay prompts and topics
- Generalize across different student populations
- Match human grader consistency

### Why It Matters

- **Education**: Provide timely feedback to students
- **Scalability**: Hand grading is expensive and slow
- **Equity**: Bring AWE tools to underserved communities
- **Assessment**: Enable essays in standardized testing

---

## Data Description

### Dataset Overview

Essays from diverse student populations:
- Multiple essay prompts
- Various grade levels
- Different writing styles
- Professional scoring

### Scoring Scale

| Score | Description |
|-------|-------------|
| 1 | Minimal response |
| 2 | Below basic |
| 3 | Basic |
| 4 | Proficient |
| 5 | Advanced |
| 6 | Exemplary |

### Data Characteristics

| Property | Description |
|----------|-------------|
| Text length | Variable (short to long essays) |
| Prompts | Multiple essay topics |
| Scoring | Ordinal (1-6) |
| Diversity | Nationally representative |

---

## Evaluation

**Metric**: Quadratic Weighted Kappa (QWK)

```python
# QWK measures agreement between predicted and actual ratings
# Accounts for ordinal nature - penalizes distant misclassifications more

from sklearn.metrics import cohen_kappa_score

qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
```

QWK ranges from -1 to 1:
- 1 = perfect agreement
- 0 = random agreement
- <0 = worse than random

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st-5th | Share of $50,000 |

---

## Top Solutions

### 1st Place Solution

**Author**: flg

**Key Approach**: "Trust CV (and LB a bit)"
- DeBERTa-v3-large fine-tuning
- Careful cross-validation strategy
- Ensemble with diverse folds
- QWK-optimized thresholds

### 2nd Place Solution

**Author**: yao

**Technique**:
- Multiple transformer models
- Prompt-specific fine-tuning
- Regression + threshold approach

### 3rd Place Solution

**Innovation**:
- DeBERTa ensemble
- Pseudo-labeling
- Diverse training strategies

---

## Common Winning Strategies

### 1. DeBERTa Fine-tuning

```python
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer

model_name = 'microsoft/deberta-v3-large'
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

class EssayScorer(nn.Module):
    def __init__(self, model_name, num_labels=6):
        super().__init__()
        self.deberta = DebertaV2Model.from_pretrained(model_name)
        self.regressor = nn.Linear(self.deberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids, attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.regressor(pooled)
```

### 2. Regression + Threshold Approach

```python
def optimize_thresholds(y_true, y_pred_continuous):
    """Find optimal thresholds for converting regression to ordinal"""
    from scipy.optimize import minimize

    def neg_qwk(thresholds):
        # Thresholds should be sorted
        thresholds = sorted(thresholds)
        y_pred_ordinal = pd.cut(
            y_pred_continuous,
            bins=[-np.inf] + list(thresholds) + [np.inf],
            labels=[1, 2, 3, 4, 5, 6]
        )
        return -cohen_kappa_score(y_true, y_pred_ordinal, weights='quadratic')

    # Initial thresholds
    initial = [1.5, 2.5, 3.5, 4.5, 5.5]
    result = minimize(neg_qwk, initial, method='Nelder-Mead')
    return result.x
```

### 3. Cross-Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold, GroupKFold

# Use GroupKFold if essays from same student should stay together
# Use StratifiedKFold for balanced score distribution

def cv_train(model_class, X, y, groups=None, n_splits=5):
    if groups is not None:
        kf = GroupKFold(n_splits=n_splits)
        splits = kf.split(X, y, groups)
    else:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        splits = kf.split(X, y)

    oof_preds = np.zeros(len(X))
    for fold, (train_idx, val_idx) in enumerate(splits):
        model = model_class()
        model.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx] = model.predict(X[val_idx])

    return oof_preds
```

### 4. Prompt-Aware Training

```python
class PromptAwareModel(nn.Module):
    def __init__(self, base_model, n_prompts):
        super().__init__()
        self.base = base_model
        self.prompt_embeddings = nn.Embedding(n_prompts, 128)
        self.final = nn.Linear(base_model.hidden_size + 128, 1)

    def forward(self, input_ids, attention_mask, prompt_id):
        base_output = self.base(input_ids, attention_mask)
        prompt_emb = self.prompt_embeddings(prompt_id)
        combined = torch.cat([base_output, prompt_emb], dim=1)
        return self.final(combined)
```

### 5. Ensemble Methods

```python
# Blend multiple models
def ensemble_predictions(predictions_list, weights=None):
    if weights is None:
        weights = [1.0 / len(predictions_list)] * len(predictions_list)

    ensemble = sum(w * p for w, p in zip(weights, predictions_list))
    return ensemble

# Optimize weights on validation set
from scipy.optimize import minimize

def optimize_weights(predictions_list, y_true):
    def neg_qwk(weights):
        weights = np.array(weights) / sum(weights)  # Normalize
        ensemble = sum(w * p for w, p in zip(weights, predictions_list))
        y_pred = round_predictions(ensemble)
        return -cohen_kappa_score(y_true, y_pred, weights='quadratic')

    n_models = len(predictions_list)
    result = minimize(neg_qwk, [1/n_models] * n_models, method='SLSQP',
                      bounds=[(0, 1)] * n_models)
    return result.x / sum(result.x)
```

---

## Technical Insights

### QWK vs Accuracy

| Metric | Behavior |
|--------|----------|
| Accuracy | All errors equal |
| Linear Kappa | Distance matters linearly |
| Quadratic Kappa | Distance matters quadratically |

For essay scoring, quadratic kappa is appropriate because predicting 1 when true is 6 is much worse than predicting 4 when true is 5.

### Text Features for Essays

| Feature Type | Examples |
|--------------|----------|
| Length | Word count, sentence count |
| Vocabulary | Unique words, sophistication |
| Grammar | Error count, complexity |
| Coherence | Transition words, structure |
| Content | Topic relevance |

### Transfer from AES v1

The original AES competition was 12 years ago. Key advances:
- Transformer models (BERT, DeBERTa)
- Better pre-training
- Improved fine-tuning techniques
- More data

### Common Pitfalls

1. **Overfitting to prompt**: Models may learn prompt-specific patterns
2. **Length bias**: Longer essays often score higher
3. **Threshold sensitivity**: QWK depends heavily on rounding
4. **Leaderboard gap**: Public/private score differences

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| Runtime | 9 hours |
| Internet | Disabled |
| GPU | Available |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/learning-agency-lab-automated-essay-scoring-2/discussion/516791) |
| 2nd | [Discussion](https://www.kaggle.com/c/learning-agency-lab-automated-essay-scoring-2/discussion/516791) |
| 3rd | [Discussion](https://www.kaggle.com/c/learning-agency-lab-automated-essay-scoring-2/discussion/517014) |
| 4th | [Discussion](https://www.kaggle.com/c/learning-agency-lab-automated-essay-scoring-2/discussion/516639) |
| 5th | [Discussion](https://www.kaggle.com/c/learning-agency-lab-automated-essay-scoring-2/discussion/516922) |

---

## Citation

```bibtex
@misc{learning-agency-lab-automated-essay-scoring-2,
    author = {The Learning Agency Lab},
    title = {Learning Agency Lab - Automated Essay Scoring 2.0},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2}},
    note = {Kaggle}
}
```
