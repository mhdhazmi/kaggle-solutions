# Jigsaw - Agile Community Rules Classification

> Using AI models to help moderators uphold community-specific norms.

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Binary Classification / Text Classification |
| **Data Domain** | Natural Language Processing (NLP) / Social Media |
| **ML Approach** | Large Language Models (LLM) Fine-tuning |
| **Key Techniques** | Test-Time Training, LoRA Fine-tuning, Ensemble Methods |
| **Difficulty** | Intermediate-Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $100,000 |
| **Teams** | 2,445 |
| **Participants** | 2,934 |
| **Timeline** | July 23, 2025 - October 24, 2025 |
| **Evaluation Metric** | Column-averaged AUC |
| **Host** | Jigsaw/Conversation AI |

## Problem Description

If you've ever had a comment taken down on Reddit and wondered "why?", you're not alone. Each subreddit has its own set of guidelines, and trying to understand individual subreddit moderation can feel like chaos.

### The Task

Build a **binary classifier** that predicts whether a Reddit comment broke a specific rule. The dataset comes from a large collection of moderated comments, with a range of subreddit norms, tones, and community expectations.

### Key Challenge

The rules used for evaluation are based on actual subreddit guidelines, but **the test set contains additional rules that are not present in training data**. Models must generalize to unseen rules, making this a **few-shot/zero-shot classification problem**.

### Background

This competition was inspired by research from:
- Deepak Kumar, Yousef AbuHashem, and Zakir Durumeric - LLMs for guessing moderation reasons
- Eshwar Chandrasekharan and Eric Gilbert - collection of millions of moderated comments

---

## Data Description

### Dataset Overview

| File | Size | Description |
|------|------|-------------|
| `train.csv` | ~1.5 MB | Training dataset with labeled examples |
| `test.csv` | ~500 KB | Test dataset with unseen rules |
| `sample_submission.csv` | 112 B | Example submission format |

**Total Dataset Size**: 1.98 MB
**License**: CC0: Public Domain

### Data Schema

| Column | Description |
|--------|-------------|
| `body` | The text of the Reddit comment |
| `rule` | The rule the comment is judged against |
| `subreddit` | The forum the comment was made in |
| `positive_example_1`, `positive_example_2` | Examples of comments that **violate** the rule |
| `negative_example_1`, `negative_example_2` | Examples of comments that **do not violate** the rule |
| `rule_violation` | Binary target (1 = violates rule, 0 = does not) |

### Key Data Characteristics

1. **Training data contains only 2 rules** - test data contains 6 rules (4 unseen)
2. **Noisy labels** - Same (body, rule) pairs sometimes have conflicting labels
3. **Subreddit bias** - Same comment can be labeled differently across subreddits
4. **Few-shot examples provided** - Positive and negative examples in test data can be used for training

### Submission Format

```csv
row_id,rule_violation
2029,0.5
2030,0.67
2031,0.1
```

Predict the **probability** that a comment violates a given rule.

---

## Evaluation

**Metric**: Column-averaged AUC (Area Under ROC Curve)

- Submissions evaluated on AUC score averaged across all rules
- Public/Private LB split is random (30%/70%)
- Public LB serves as unbiased estimator of private LB

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $35,000 |
| 2nd | $25,000 |
| 3rd | $15,000 |
| 4th-8th | $5,000 each |

---

## Top Solutions

### 1st Place Solution (Score: 0.9293 Private LB)

**Author**: Guanshuo Xu (wowfattie)

#### Key Approach

**Validation Strategy**:
- Used Public LB as primary validation (30% of test data, random split)
- All models fine-tuned online and validated using public LB

**Data Processing**:
- Used LLM built-in chat templates
- **Excluded subreddit** before deduplication (critical improvement!)
- Used positive/negative examples from test.csv as additional training data

**Fine-tuning**:
- Used **Unsloth** for memory-efficient LoRA fine-tuning
- Upsampled test.csv examples (2 epochs for test data, 1 epoch for train)
- Manually excluded extra tokens from loss (only "Yes"/"No" tokens contribute)
- Single set of hyperparameters across all models

**Inference**:
- Constructed candidate set of Yes/No token variants
- Score = sigmoid(log-odds of Yes_ids vs No_ids)
- Sorted test data by length for efficiency
- Used forward() only (avoid generation overhead)

**Post-processing**:
- Per-rule rankings normalized to [0, 1]
- Weighted ensemble of normalized scores

#### Final Ensemble

| Model | Public LB | Private LB |
|-------|-----------|------------|
| Qwen3-14b | 0.9297 | 0.9239 |
| Qwen2.5-14b | 0.9287 | 0.9232 |
| Qwen3-8b | 0.9272 | 0.9236 |
| Qwen3-4b-instruct-2507 | 0.9258 | 0.9198 |
| llama3.1-8b | 0.9257 | 0.9202 |
| Ettin-400M | 0.8991 | 0.8944 |
| **Ensemble** | **0.9344** | **0.9293** |

#### Key Insights
- Larger models generally performed better
- Qwen3 outperformed other LLM families
- **Excluding subreddit** was a significant improvement

#### Resources
- [Winning Notebook](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/569891)

---

### 2nd Place Solution

**Team**: ^^_AI, Sophia Yang, Jiaming Zhang, Yang, hoatha

#### Key Approach

**Label Cleaning**:
- Identified conflicting labels for same (body, rule) pairs
- Used majority vote for conflict resolution
- Deduplicated after cleaning

**Dropped Subreddit Feature**:
- Found subreddit carries strong social bias signals
- Removed from prompt to prevent shortcut learning

**Prompt Format**:
```
System: "Decide if the Reddit comment violates the rule. Reply strictly with Yes or No."
User: Rule: <rule text>
      Comment: <comment text>
A: Yes / No
```

**Multi-Model Ensemble**:
- phi-4-14b (4-bit quantized)
- qwen3-14b (4-bit quantized)
- qwen3-8b (4-bit quantized)
- qwen25-7b-instruct (4-bit quantized)

**Engineering Optimizations**:
- Dual GPU parallel scheduling
- Conservative batch sizes (80-90% resource usage) to handle Kaggle instability
- Total runtime: 11 hours

---

### 3rd Place Solution

**Author**: Single competitor

#### Key Approach

**Test-Time Training (TTT)**:
- Critical for handling hidden/unseen rules
- Built new dataset using all rules, bodies/examples with known targets

**Discarded Subreddit**:
- Found unhelpful during both pretraining and TTT phases

**Two-Ensemble Strategy**:

**Slow Ensemble** (LLMs):
- 2× Qwen2.5-instruct-14b
- 2× Qwen2.5-instruct-7b
- 2× Qwen3-instruct-14b
- 2× Qwen3-instruct-8b

**Fast Ensemble** (Smaller models):
- 2× bge-base-v1.5
- Qwen3-embedding-0.6b
- DeBERTa-v3-small

**Training Method**:
- Zero-shot prompt format for LLMs
- LoRA weights from pretraining used as TTT starting point

---

## Common Winning Strategies

### 1. Exclude Subreddit
All top solutions found that removing the subreddit feature improved performance by preventing shortcut learning and bias.

### 2. Use Test Data for Training
The positive/negative examples in test.csv were used as additional training data (key insight from competition design).

### 3. Label Cleaning
Handling conflicting labels through majority voting and deduplication improved model convergence.

### 4. LLM Fine-tuning with Unsloth
Memory-efficient LoRA fine-tuning enabled training 14B+ models on Kaggle's limited hardware.

### 5. Yes/No Token Probability
Instead of generation, extract logit difference between Yes/No tokens for faster, more reliable scoring.

### 6. Ensemble Diverse Models
Combining multiple model families (Qwen, Llama, Phi) and sizes improved robustness.

---

## Technical Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Runtime | ≤12 hours |
| GPU Runtime | ≤12 hours |
| Internet Access | Disabled |
| External Data | Allowed (including pre-trained models) |

---

## Solution Links

| Place | Team | Score | Solution |
|-------|------|-------|----------|
| 1st | Guanshuo Xu | 0.9293 | [Writeup](https://www.kaggle.com/c/jigsaw-agile-community-rules/writeups/1st-place-solution) |
| 2nd | ^^_AI et al. | - | [Writeup](https://www.kaggle.com/c/jigsaw-agile-community-rules/writeups/2nd-place-solution) |
| 3rd | - | - | [Writeup](https://www.kaggle.com/c/jigsaw-agile-community-rules/writeups/3rd-place-solution) |

---

## Citation

```bibtex
@misc{jigsaw-agile-community-rules,
    author = {Jeffrey Sorensen and Lucas Dos Santos and Lucy Vasserman and María Cruz and Tin Acosta and Walter Reade},
    title = {Jigsaw - Agile Community Rules Classification},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/jigsaw-agile-community-rules}},
    note = {Kaggle}
}
```
