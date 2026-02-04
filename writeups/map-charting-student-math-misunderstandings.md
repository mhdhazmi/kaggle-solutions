# MAP - Charting Student Math Misunderstandings

> Predict the affinity between misconceptions and student open-ended responses

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-class Classification / NLP |
| **Data Domain** | Education / Mathematics |
| **ML Approach** | Large Language Models (LLM) Fine-tuning |
| **Key Techniques** | Suffix Classification, Multi-seed Ensemble, Layer-wise Inference |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $55,000 |
| **Timeline** | July 10, 2025 - October 16, 2025 |
| **Evaluation Metric** | Mean Average Precision @ 3 (MAP@3) |
| **Host** | The Learning Agency LLC / Vanderbilt University |

## Problem Description

Develop an NLP model that predicts students' potential math misconceptions based on their written explanations. Students answer Diagnostic Questions (DQs) and provide written justifications - these explanations reveal underlying misconceptions in their reasoning.

### Example Misconception

> Students often think 0.355 is larger than 0.8 because they incorrectly apply their knowledge of whole numbers to decimals, reasoning that 355 is greater than 8.

### The Task (3 Steps)

1. **Determine answer correctness**: True or False
2. **Assess explanation quality**: Correct, Misconception, or Neither
3. **Identify specific misconception**: If present, which one

### Target Format

Predictions concatenate Category and Misconception:
- `True_Correct:NA`
- `False_Misconception:Incomplete`
- `True_Misconception:WNB`

---

## Data Description

### Dataset Overview

| Attribute | Value |
|-----------|-------|
| **Size** | 7.94 MB |
| **Train Samples** | ~35,960 (after deduplication) |
| **Test Samples** | ~16,000 |
| **QuestionIds** | 15 unique questions |
| **Misconceptions** | 36 types |
| **Categories** | 65 combinations |
| **License** | MIT |

### Data Schema

| Column | Description |
|--------|-------------|
| `QuestionId` | Unique question identifier |
| `QuestionText` | The math question text (OCR extracted) |
| `MC_Answer` | Student's multiple-choice selection |
| `StudentExplanation` | Written justification |
| `Category` | Classification (e.g., True_Misconception) |
| `Misconception` | Specific misconception or 'NA' |

### Key Dataset Characteristics

- **Limited QuestionIds**: Only 15 unique questions
- **Per-question misconceptions**: Each question has 2-5 possible misconceptions
- **Label noise**: Significant noise in "Neither" category
- **Image-based questions**: Extracted via human-in-the-loop OCR

---

## Evaluation

**Metric**: Mean Average Precision @ 3 (MAP@3)

- Predict up to 3 `Category:Misconception` values per row
- Only one correct label per observation
- Space-delimited predictions

### Submission Format

```csv
row_id,Category:Misconception
36696,True_Correct:NA False_Neither:NA False_Misconception:Incomplete
36697,True_Correct:NA False_Neither:NA False_Misconception:Incomplete
```

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $20,000 |
| 2nd | $12,000 |
| 3rd | $8,000 |
| 4th | $5,000 |
| 5th | $5,000 |
| 6th | $5,000 |

---

## Top Solutions

### 1st Place Solution (MAP@3: ~0.95)

**Author**: tascj

*First NLP competition 1st place! Achieved with single submission.*

#### Approach: Suffix Classification

**Concept**: Given same context (prefix), predict correct suffix from candidates.

**Prompt Format**:
```
<|im_start|>user
**Question:** {QuestionText}
**Choices:** {MC_Choices}
**Correct Answer:** {Answer}
**Common Misconceptions:** {MisconceptionCandidates}
**Student Answer:** {MC_Answer}
**Student Explanation:** {StudentExplanation}
<|im_end|>
<|im_start|>assistant
```

**Suffix Format**: `False_Correct:NA<|im_end|>`

**Implementation**:
- Extract last-token features from [prefix ++ suffix_i] for each candidate
- Feed into `nn.Linear(hidden_size, 1)` for logits
- Cross-entropy loss
- Custom FlexAttention masks for prefix-shared format

#### Critical Discovery: Multi-Seed Validation

Single-seed CV scores were **highly unstable and misleading** due to label noise.

**Solution**: 3-seed ensemble for stable validation:

| Model | Loss (3-seed avg) | MAP@3 (3-seed avg) |
|-------|-------------------|-------------------|
| DeepSeek-R1-Distill-7B | 0.2716 | 0.9444 |
| Qwen3-8B | 0.2677 | 0.9455 |
| GLM-Z1-9B | 0.2627 | 0.9469 |
| Qwen3-14B | 0.2614 | 0.9477 |
| Qwen3-32B | 0.2589 | 0.9484 |
| GLM-Z1-32B | 0.2560 | 0.9480 |
| **Ensemble** | **0.2530** | **0.9496** |

#### Key Insights

1. **Trust loss over MAP@3** for model selection
2. **Larger models = better** (32B > 14B > 8B)
3. **Multi-seed > multi-fold** for this noisy dataset

#### Inference Optimization

**Challenge**: Run 32B models on T4 GPUs.

**Solutions**:
1. **W8A8 INT8 quantization** via SmoothQuant (α=0.75)
   - Achieved stable 40+ TFLOPS vs 20 TFLOPS with FP16
2. **Layer-wise inference**:
   - Keep only 2 transformer layers on GPU
   - Overlap execution with next layer loading
3. **Batch size**: 640 samples (40 micro-batches × 16)

**Runtime**: ~65 minutes for 16k samples on T4×2

---

### 3rd Place Solution (Public 1st)

**Team**: monsaraida & Masaya

#### monsaraida's Part

**Key Improvements**:

1. **Add all choices to prompt** (not just selected answer): +0.001+ improvement
2. **Add misconception hints** (ordered by probability): +0.0003 improvement

**Prompt Example**:
```
Question: What fraction of the shape is not shaded?
Choices: (A) 1/3 (B) 3/9 (C) 3/6 (D) 3/8
Selected: 1/3
Correct? Yes
Student Explanation: 1 goes into everything and 3 goes into nine
Common mistakes for this question: Incomplete, WNB
```

**Training**:
- Base model: Qwen3-14B
- LoRA SFT with multi-task learning (4 heads)
- **R-Drop** (Regularized Dropout): +0.001 CV
- **AWP** (Adversarial Weight Perturbation): slight improvement
- **EMA**: stabilized CV

**Inference Optimization**:
- Changed `bfloat16` → `float16`: **2× speedup**
- Changed `padding="max_length"` → `padding=False`: **2× speedup**
- Combined: **4× speedup**

#### Masaya's Part

**CausalLM Approach**:

Advantage: Higher accuracy with variable-length labels.

**Key Insight**: Limit label set to per-QuestionId possibilities (only 2-5 misconceptions each).

**Models**:

| Model | Task | CV | LB | Private |
|-------|------|-----|-----|---------|
| Qwen2.5-14B-AWQ | CausalLM | 0.9469 | 0.943 | 0.942 |
| Qwen2.5-32B-AWQ | CausalLM | 0.9498 | 0.949 | 0.944 |
| Qwen2.5-72B-GPTQ | CausalLM | - | 0.950 | 0.948 |

**Multi-stage Inference**:
- First pass: All models on full test set
- Second pass: 72B model on lowest-confidence 50%
- Reduced 72B inference from 3 hours to 1 hour

---

## Common Winning Strategies

### 1. Multi-Seed Validation
Essential for noisy labels - single-seed CV is misleading.

### 2. Add All Choices to Prompt
Including wrong choices helps model identify misconceptions.

### 3. Per-Question Label Restriction
Only predict misconceptions that can occur for each question.

### 4. Large Language Models
32B+ models significantly outperform smaller ones.

### 5. Quantization for Inference
W8A8 INT8 enables large models on limited hardware.

### 6. Regularization for Generalization
R-Drop, AWP, EMA combat label noise.

---

## Technical Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Runtime | ≤12 hours |
| GPU Runtime | ≤12 hours |
| Internet Access | Disabled |
| External Data | Allowed |

---

## Solution Links

| Place | Team | Score | Solution |
|-------|------|-------|----------|
| 1st | tascj | ~0.95 | [Writeup](https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/1st-place-solution) |
| 3rd | monsaraida & Masaya | ~0.95 | [Writeup](https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/3rd-place-solution) |

---

## Citation

```bibtex
@misc{map-charting-student-math-misunderstandings,
    title = {MAP - Charting Student Math Misunderstandings},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/map-charting-student-math-misunderstandings}},
    note = {Kaggle, The Learning Agency LLC, Vanderbilt University}
}
```
