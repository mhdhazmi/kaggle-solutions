# LLMs - You Can't Please Them All

> Are LLM-judges robust to adversarial inputs?

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Adversarial Text Generation |
| **Data Domain** | NLP / LLM Evaluation |
| **ML Approach** | Prompt Engineering + Optimization |
| **Key Techniques** | Red-teaming, Adversarial Attacks, Judge Exploitation |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 1,692 |
| **Timeline** | 2024-2025 |
| **Evaluation Metric** | Custom LLM Judge Metric |
| **Host** | Kaggle |

## Problem Description

Test the robustness of LLM-as-judge systems by crafting inputs that different judge models evaluate inconsistently.

### The Challenge

- Generate text responses that different LLM judges rate differently
- Some judges should rate it highly, others poorly
- Exploit inconsistencies in LLM evaluation
- Understand failure modes of LLM-based evaluation

### Why It Matters

- **AI Safety**: Expose vulnerabilities in LLM evaluation
- **Reliability**: Improve robustness of AI systems
- **Research**: Understand LLM judgment mechanisms
- **Red-teaming**: Proactive security testing

---

## Data Description

### Task Format

| Component | Description |
|-----------|-------------|
| Prompts | Questions/tasks for LLMs to respond to |
| Judge Models | Multiple LLMs that evaluate responses |
| Target | Maximize disagreement between judges |

### Evaluation Setup

Multiple LLM judges evaluate each response:
- Some judges prefer certain styles
- Different models have different biases
- Goal: Exploit these differences

---

## Evaluation

**Metric**: Custom metric measuring judge disagreement

Higher score = more successful at making judges disagree.

The metric rewards:
- High rating from some judges
- Low rating from other judges
- Maximum spread in evaluations

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | TBD |
| 2nd | TBD |
| 3rd | TBD |

**Total Prize Pool**: $50,000

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Systematic analysis of judge model biases
- Crafted responses targeting specific weaknesses
- Automated optimization of adversarial text

### 2nd Place Solution

**Technique**:
- Style transfer between judge preferences
- Length and formatting exploitation
- Keyword-based manipulation

### 3rd Place Solution

**Innovation**:
- Gradient-based prompt optimization
- Ensemble of adversarial strategies
- Judge-specific attack vectors

---

## Common Winning Strategies

### 1. Judge Bias Analysis

Identify what each judge model prefers:
- Verbosity vs conciseness
- Formal vs casual tone
- Structure vs flow
- Technical depth vs accessibility

### 2. Exploitation Techniques

| Technique | Description |
|-----------|-------------|
| Length gaming | Very long or very short responses |
| Format tricks | Markdown, bullets, code blocks |
| Style manipulation | Formal/informal mismatches |
| Keyword stuffing | Terms models associate with quality |
| Contradictions | Statements that parse differently |

### 3. Adversarial Generation

```python
def generate_adversarial(prompt):
    # Generate base response
    base = llm.generate(prompt)

    # Optimize for judge disagreement
    for _ in range(iterations):
        modified = perturb(base)
        score = evaluate_disagreement(modified)
        if score > best:
            base = modified

    return base
```

### 4. Model-Specific Attacks

Target known LLM weaknesses:
- Sycophancy (agreeing too readily)
- Position bias (favoring certain orderings)
- Verbosity bias (longer = better)
- Format bias (structure signals quality)

### 5. Semantic Preservation

Maintain apparent quality while gaming metrics:
- Responses should seem reasonable to humans
- Hidden triggers for specific models
- Plausibly deniable adversarial content

---

## Technical Insights

### LLM-as-Judge Vulnerabilities

1. **Surface-level features**: Models rely on superficial signals
2. **Training biases**: Preferences inherited from training data
3. **Instruction following**: Can be misled by in-context instructions
4. **Tokenization effects**: Token boundaries affect interpretation

### Judge Disagreement Patterns

| Pattern | Cause |
|---------|-------|
| Length preference | Different training data distributions |
| Format preference | Instruction tuning differences |
| Topic sensitivity | Content moderation variations |
| Confidence calibration | Model architecture differences |

### Ethical Considerations

This competition exists to:
- Improve robustness of evaluation systems
- Identify failure modes before deployment
- Contribute to AI safety research
- NOT to enable manipulation of production systems

### Practical Applications

Findings help improve:
- RLHF reward models
- Automated content moderation
- AI-assisted evaluation
- Benchmark reliability

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/llms-you-cant-please-them-all/discussion/566372) |
| 2nd | [Discussion](https://www.kaggle.com/c/llms-you-cant-please-them-all/discussion/566602) |
| 3rd | [Discussion](https://www.kaggle.com/c/llms-you-cant-please-them-all/discussion/566515) |
| 4th | [Discussion](https://www.kaggle.com/c/llms-you-cant-please-them-all/discussion/566479) |
| 5th | [Discussion](https://www.kaggle.com/c/llms-you-cant-please-them-all/discussion/566322) |

---

## Citation

```bibtex
@misc{llms-you-cant-please-them-all,
    title = {LLMs - You Can't Please Them All},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/llms-you-cant-please-them-all}},
    note = {Kaggle}
}
```
