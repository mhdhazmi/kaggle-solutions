# Eedi - Mining Misconceptions in Mathematics

> Map student misconceptions from mathematics questions

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-label Classification / Retrieval |
| **Data Domain** | Education / Mathematics / NLP |
| **ML Approach** | LLM Fine-tuning + Retrieval |
| **Key Techniques** | Semantic Search, Contrastive Learning, Ensemble |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $50,000+ |
| **Teams** | 1,500+ |
| **Timeline** | 2024 |
| **Evaluation Metric** | MAP@25 |
| **Host** | Eedi |

## Problem Description

Identify which of 2,500+ possible misconceptions a student might have based on their incorrect answer to a math question.

### The Task

Given:
- A math question with multiple choice answers
- The correct answer
- A specific wrong answer

Predict: Which misconception(s) led to that wrong answer

### Why It Matters

- Help teachers identify student struggles
- Enable personalized learning
- Scale diagnostic assessment
- Improve math education outcomes

---

## Data Description

### Dataset Structure

| Component | Description |
|-----------|-------------|
| Questions | Math questions with multiple choice answers |
| Misconceptions | 2,500+ labeled misconception types |
| Mappings | Which misconceptions lead to which wrong answers |

### Key Challenge

- **Large label space**: 2,500+ possible misconceptions
- **Semantic matching**: Must understand math concepts
- **Few-shot**: Some misconceptions have limited examples

---

## Evaluation

**Metric**: MAP@25 (Mean Average Precision at 25)

Rank the most likely misconceptions, with correct ones appearing earlier in the ranking.

---

## Top Solutions

### Common Approaches

1. **Semantic Retrieval**: Embed questions and misconceptions, find nearest neighbors
2. **LLM Classification**: Fine-tune LLMs to predict misconceptions directly
3. **Two-Stage Pipeline**: Retrieve candidates, then re-rank

---

## Common Winning Strategies

### 1. Sentence Transformers
Encoding questions and misconceptions into same embedding space.

### 2. Contrastive Learning
Training with positive/negative pairs to improve retrieval.

### 3. LLM Fine-tuning
Using models like DeBERTa, BERT for classification.

### 4. Ensemble Methods
Combining retrieval and classification approaches.

### 5. Data Augmentation
Generating synthetic question-misconception pairs.

---

## Technical Insights

### Retrieval vs Classification

| Approach | Pros | Cons |
|----------|------|------|
| Retrieval | Scales to large label sets | May miss nuances |
| Classification | Direct prediction | Struggles with 2500+ classes |
| Hybrid | Best of both | More complex |

### Key Features

- Question text and math expressions
- Wrong answer choice
- Correct answer (for contrast)
- Question topic/category

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Writeup](https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/writeups/1st-place-solution) |
| 2nd | [Writeup](https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/writeups/2nd-place-solution) |
| 3rd | [Writeup](https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/writeups/3rd-place-solution) |

---

## Citation

```bibtex
@misc{eedi-mining-misconceptions,
    title = {Eedi - Mining Misconceptions in Mathematics},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics}},
    note = {Kaggle, Eedi}
}
```
