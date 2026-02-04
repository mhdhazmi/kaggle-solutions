# Make Data Count - Finding Data References

> Identify scientific data use in papers and classify how they are mentioned

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Named Entity Recognition (NER) / Multi-class Classification |
| **Data Domain** | Scientific Text / Academic Papers |
| **ML Approach** | LLM Fine-tuning (DeBERTa, Qwen) |
| **Key Techniques** | Token Classification, Regex Patterns, Ensemble |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Competition |
| **Total Prize** | $100,000 |
| **Teams** | 1,282 |
| **Timeline** | 2025 |
| **Evaluation Metric** | Global F1 Score |
| **Host** | Make Data Count Initiative |

## Problem Description

Identify and classify references to scientific datasets within academic papers. This helps track how research data is being used and cited across scientific literature.

### The Task

Given text from scientific papers:
1. **Detect** mentions of datasets
2. **Extract** the dataset reference spans
3. **Classify** the type of mention (created, used, shared, etc.)

### Why It Matters

- Track research data impact
- Improve data citation practices
- Enable data metrics and analytics
- Support open science initiatives

---

## Data Description

### Dataset Structure

The competition provides:
- **Scientific paper texts**: Full text or excerpts from academic papers
- **Annotations**: Labeled dataset mentions with spans and categories

### Categories

Dataset mentions are classified by how they are referenced:
- **Created**: Author created or generated the data
- **Used**: Author used existing data
- **Shared**: Data made available
- **Cited**: Formal citation of dataset
- **Other**: Miscellaneous mentions

---

## Evaluation

**Metric**: Global F1 Score

- Precision: Correct predictions / Total predictions
- Recall: Correct predictions / Total actual mentions
- F1 = 2 × (Precision × Recall) / (Precision + Recall)

Evaluated across all mention types with appropriate weighting.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | TBD |
| 2nd | TBD |
| 3rd | TBD |
| ... | ... |

**Total Prize Pool**: $100,000

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Fine-tuned large language models on the task
- Combined regex patterns for common dataset formats
- Ensemble of multiple models

### 2nd Place Solution

*(Details in linked writeup)*

### 3rd Place Solution

*(Details in linked writeup)*

---

## Common Winning Strategies

### 1. Hybrid Approach
Combining ML models with rule-based regex patterns for known dataset formats.

### 2. LLM Fine-tuning
Using pre-trained models like DeBERTa, Qwen for token classification.

### 3. Span Detection
Treating the task as sequence labeling (BIO tagging) for entity extraction.

### 4. Multi-Stage Pipeline
First detect mentions, then classify their type.

### 5. Data Augmentation
Generating synthetic examples of dataset mentions.

---

## Technical Insights

### Challenge: Domain-Specific Language

- Scientific papers use specialized terminology
- Dataset names vary widely in format
- Context matters for classification

### Regex Patterns for Common Formats

```python
# DOI patterns
doi_pattern = r'10\.\d{4,}/[^\s]+'

# Dataset repository patterns
repo_patterns = [
    r'figshare\.com/\S+',
    r'zenodo\.org/\S+',
    r'dryad\.org/\S+',
    r'github\.com/\S+/\S+'
]
```

---

## Technical Requirements

| Requirement | Value |
|-------------|-------|
| Runtime | Standard Kaggle notebook |
| Internet | Allowed |
| External Data | Allowed |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Writeup](https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/1st-place-solution) |
| 2nd | [Writeup](https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/2nd-place-solution) |
| 3rd | [Writeup](https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/3rd-place-solution) |
| 4th | [Writeup](https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/4th-place-solution) |
| 5th | [Writeup](https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/5th-place-by-standing-on-the-shoulders-of-giants) |

---

## Citation

```bibtex
@misc{make-data-count-finding-data-references,
    title = {Make Data Count - Finding Data References},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/make-data-count-finding-data-references}},
    note = {Kaggle}
}
```
