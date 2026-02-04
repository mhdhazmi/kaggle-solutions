# AI Mathematical Olympiad - Progress Prize 2

> Solve national-level math challenges using artificial intelligence models

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Mathematical Reasoning / Problem Solving |
| **Data Domain** | Mathematics / Olympiad Problems |
| **ML Approach** | Large Language Models + Code Execution |
| **Key Techniques** | Chain-of-Thought, Tool Use, Ensemble Inference |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $2,117,152 |
| **Teams** | 2,212 |
| **Timeline** | October 2024 - April 2025 |
| **Evaluation Metric** | Accuracy Score |
| **Host** | AI|MO |

## Problem Description

Create AI algorithms that can solve challenging mathematical problems written in LaTeX format, similar to national-level Mathematical Olympiad problems.

### The Challenge

- Solve 110 math problems (50 public, 50 private, 10 reference)
- Problems span algebra, combinatorics, geometry, and number theory
- Difficulty at national Olympiad level
- All answers are integers between 0 and 999

### Why It Matters

- **AI Milestone**: Mathematical reasoning is fundamental to intelligence
- **Benchmark**: Novel problems avoid train-test contamination
- **Progress Tracking**: Measure real advances in AI reasoning
- **$10M Grand Prize**: Part of larger AIMO initiative

---

## Data Description

### Problem Format

| Component | Description |
|-----------|-------------|
| Problem text | Mathematical problem in LaTeX |
| Answer format | Integer 0-999 (solution mod 1000) |
| Topics | Algebra, combinatorics, geometry, number theory |

### Files

| File | Description |
|------|-------------|
| `reference.csv` | 10 problems with full solutions |
| `test.csv` | 50 problems (placeholder, replaced during scoring) |
| `sample_submission.csv` | Example submission format |

### Answer Computation

```
Final Answer = Solution mod 1000
```

Examples:
- Solution 2034 → Answer 34
- Solution 65521 → Answer 521
- Solution -900 → Answer 100 (i.e., (-900 mod 1000) = 100)

---

## Evaluation

**Metric**: Accuracy (exact match)

```
Accuracy = Correct Predictions / Total Problems
```

### Scoring Details

- 50 public test problems
- 50 private test problems (scored after competition ends)
- Must achieve ≥47/50 on both sets for Overall Progress Prize

### API Submission

Problems served one-by-one via Python API:
- First prediction call within 15 minutes of notebook start
- Each prediction within 30 minutes
- Random order during public scoring
- Fixed order for private scoring

---

## Prize Structure

### Competition Prizes

| Place | Prize |
|-------|-------|
| 1st | $262,144 |
| 2nd | $131,072 |
| 3rd | $65,536 |
| 4th | $32,768 |
| 5th | $16,384 |

### Overall Progress Prize

**$1,609,248** for first team to achieve ≥47/50 on both public and private sets.

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Ensemble of reasoning models
- Code execution for verification
- Multiple inference attempts with voting

### 2nd Place Solution

**Technique**:
- Fine-tuned open-source LLMs
- Chain-of-thought prompting
- Majority voting across samples

### 3rd Place Solution

**Innovation**:
- Hybrid symbolic-neural approach
- Math-specific prompting strategies
- Self-consistency decoding

---

## Common Winning Strategies

### 1. Model Selection

| Model Type | Use Case |
|------------|----------|
| DeepSeek-Math | Math-specialized |
| Qwen2.5-Math | Strong reasoning |
| Llama fine-tuned | Customizable |
| Code Llama | For executable solutions |

### 2. Prompting Strategies

**Chain-of-Thought**:
```
Let's solve this step by step:
1. First, understand what's being asked...
2. Identify the key relationships...
3. Apply relevant theorems...
4. Compute the answer...
```

**Few-Shot**:
Include solved examples from reference data.

### 3. Code Execution

Many solutions use code verification:
```python
# Generate solution code with LLM
code = llm.generate(problem + "Write Python to solve this")

# Execute and extract answer
result = exec_sandbox(code)
answer = result % 1000
```

### 4. Self-Consistency

Sample multiple solutions and vote:
```python
answers = [solve(problem) for _ in range(32)]
final = Counter(answers).most_common(1)[0][0]
```

### 5. Problem Classification

Route problems to specialized solvers:
- Algebra → Symbolic computation
- Combinatorics → Enumeration + proof
- Geometry → Coordinate methods
- Number Theory → Modular arithmetic

---

## Technical Insights

### Mathematical Domains

| Domain | Typical Techniques |
|--------|-------------------|
| Algebra | Equations, inequalities, polynomials |
| Combinatorics | Counting, pigeonhole, recursion |
| Geometry | Coordinate, trigonometric, synthetic |
| Number Theory | Divisibility, primes, modular math |

### Challenges

1. **Reasoning depth**: Multi-step logical deduction
2. **Novel problems**: Can't rely on memorization
3. **Precision**: Exact integer answers required
4. **Time constraints**: 30 minutes per problem

### Answer Extraction

LLMs may output answers in various formats:
```python
def extract_answer(text):
    # Look for "answer is X" patterns
    # Extract final number
    # Apply mod 1000
    pass
```

### Verification Strategies

- **Forward checking**: Re-solve with different approach
- **Backward checking**: Verify answer satisfies conditions
- **Code execution**: Numerical verification where possible

---

## Code Requirements

| Requirement | Limit |
|-------------|-------|
| API call start | Within 15 minutes |
| Per-problem timeout | 30 minutes |
| Internet | Disabled during inference |
| Open-source models | Required |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/574765) |
| 2nd | [Discussion](https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/572948) |
| 3rd | [Discussion](https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/573314) |
| 4th | [Discussion](https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/573671) |
| 5th | [Discussion](https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/574262) |

---

## Related Resources

- [Project Numina](https://projectnumina.ai/) - Winners of AIMO Prize 1
- [DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math)
- [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math)

---

## Citation

```bibtex
@misc{ai-mathematical-olympiad-progress-prize-2,
    title = {AI Mathematical Olympiad - Progress Prize 2},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2}},
    note = {AI|MO, Kaggle}
}
```
