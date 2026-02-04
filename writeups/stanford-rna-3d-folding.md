# Stanford RNA 3D Folding

> Solve RNA structure prediction, one of biology's remaining grand challenges

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | 3D Structure Prediction / Regression |
| **Data Domain** | Computational Biology / Bioinformatics |
| **ML Approach** | Template-Based Modeling + Deep Learning |
| **Key Techniques** | Sequence Alignment, Structure Refinement, DRfold2 |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $75,000 |
| **Teams** | 1,516 |
| **Timeline** | 2025 |
| **Evaluation Metric** | Template Modeling Score (TM-score) |
| **Host** | Stanford University |

## Problem Description

Predicting the 3D structure of RNA molecules is one of biology's remaining grand challenges. Unlike proteins (which have tools like AlphaFold), RNA structure prediction remains largely unsolved.

### The Task

Given an RNA sequence, predict the 3D coordinates of each nucleotide in the folded structure.

### Why It Matters

- RNA plays crucial roles in gene regulation, protein synthesis, and disease
- Understanding RNA structure enables drug design targeting RNA
- Current methods achieve only ~4% accuracy compared to human-level protein structure prediction

---

## Data Description

### Dataset Structure

The competition provided:
- **PDB_RNA directory**: CIF files with known RNA structures from the Protein Data Bank
- **Training sequences**: RNA sequences with known 3D structures
- **Test sequences**: RNA sequences requiring structure prediction

### RNA Representation

- **Nucleotides**: A, U, G, C (plus 93 modified variants)
- **Coordinates**: 3D positions (x, y, z) for backbone atoms (C1' typically used as reference)
- **Sequence length**: Variable (from ~50 to 200+ nucleotides)

### Key Dataset Characteristics

1. **Modified bases**: 93 nucleotide variants including modified bases
2. **Disorder-aware extraction**: Some structures have multiple conformations
3. **Template database**: Existing PDB structures used for template-based approaches

---

## Evaluation

**Metric**: TM-score (Template Modeling Score)

### TM-score Properties

1. **Normalized by length**: 50nt and 200nt RNAs compared on same 0-1 scale
2. **Robust to local errors**: Small number of misplaced nucleotides doesn't disproportionately lower score
3. **Range**: 0 to 1 (higher is better)

### Key Insight

TM-score's robustness to local errors means **getting the overall fold correct** is more important than achieving atomic-level precision.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | TBD |
| 2nd | TBD |
| 3rd | TBD |

**Total Prize Pool**: $75,000

---

## Top Solutions

### 1st Place: Hybrid TBM + DRfold2 Approach (Score: 0.57773)

**Author**: g john rao (jaejohn)

*First Kaggle gold medal and first win!*

#### Competition Strategy

Without GPUs, training from scratch was not viable. Early research (CASP results, literature, conference talks) showed that **Template-Based Modeling (TBM)** approaches consistently dominated. Committed to TBM from day one.

#### Template-Based Modeling (TBM) Pipeline

**5-Step Process**:

**1. The Search - Finding Similar Structures**
- Identify database structures resembling target sequence through sequence alignment
- Composite similarity scoring:
  ```
  composite_score = 0.4 × global_score +
                    0.3 × local_score +
                    0.2 × feature_similarity +
                    0.1 × kmer_similarity
  ```

**2. The Alignment - Sequence Mapping**
- Global sequence alignment with gap penalties optimized for RNA
- Creates translation guide between query and template

**3. The Transfer - Coordinate Inheritance**
- Copy 3D coordinates for all matched positions
- Leverages evolutionary tendency for RNA to conserve 3D structure more than sequence

**4. The Gap Fill - Geometric Backbone Reconstruction**
For insertions and deletions:
- Maintains C1'-C1' distance (~5.9Å between consecutive nucleotides)
- Compressed gaps: extends backbone with sinusoidal perturbations
- Normal gaps: linear interpolation between flanking coordinates
- Terminal extensions: follow established backbone direction

**5. Adaptive Refinement - Confidence-Based Optimization**
Refinement intensity adapts to template confidence:
- **High-confidence (>0.8)**: Minimal constraints, preserve template geometry
- **Medium-confidence**: Moderate sequential distance constraints (5.5-6.5Å)
- **Low-confidence**: Additional steric clash prevention and base-pairing constraints
- Constraint strength: `0.8 × (1 - min(confidence, 0.8))`

#### DRfold2 Enhancements

**Selection Module Improvements**:
- Double precision calculations (float64) for reliable rankings
- Vectorized distance calculations via `torch.cdist`
- Pre-computed cubic spline coefficients for fast structure scoring

**Optimization Module Improvements**:
- PyTorch LBFGS optimizer with automatic differentiation
- GPU acceleration for energy calculations
- External knowledge integration: Boltz-1 integration

#### Hybrid Strategy

| Condition | Approach |
|-----------|----------|
| Shorter sequences | Template-based modeling |
| Time budget exhausted | Template-based modeling |
| Rest of sequences | DRfold2 |
| DRfold2 failures | Fallback to template approach |

#### Key Finding: TBM-Only Performance

Remarkably, a **TBM-only** notebook scored **0.59298** - actually *higher* than the hybrid approach (0.57773)! This demonstrates the power of classical template-based methods.

#### Resources
- [Winning Notebook - Hybrid](https://www.kaggle.com/code/jaejohn/sub-1-4-4-hybrid-final-take)
- [TBM-Only Notebook](https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach)
- [DRfold2 Repository](https://www.kaggle.com/datasets/jaejohn/drfold2-repo)

---

### Key Approach Comparison

| Method | Public LB | Private LB |
|--------|-----------|------------|
| TBM-only (optimized) | 0.59298 | - |
| Hybrid TBM + DRfold2 (winning) | - | 0.57773 |
| DRfold2 disabled in winning notebook | 0.58487 | - |

---

## Common Winning Strategies

### 1. Template-Based Modeling (TBM)
Leveraging existing solved structures from PDB as templates for prediction.

### 2. Sequence Alignment Optimization
Multi-component similarity scoring (global, local, feature-based, k-mer).

### 3. Confidence-Adaptive Refinement
Adjusting refinement intensity based on template quality.

### 4. Geometric Backbone Reconstruction
Physics-informed gap filling maintaining RNA backbone geometry.

### 5. Ensemble of Methods
Combining TBM with deep learning (DRfold2) for robustness.

---

## Technical Insights

### Why TBM Works for RNA

1. **Evolutionary conservation**: RNA conserves 3D structure more than sequence
2. **Limited conformational space**: RNA folds into relatively predictable motifs
3. **Rich template database**: PDB contains many solved RNA structures

### TM-score Optimization Strategy

Focus on:
- **Overall fold correctness** over atomic precision
- **Template selection** quality
- **Robust gap filling** for insertions/deletions

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

| Place | Author | Score | Solution |
|-------|--------|-------|----------|
| 1st | jaejohn | 0.57773 | [Writeup](https://www.kaggle.com/c/stanford-rna-3d-folding/writeups/1st-place-solution) |
| 2nd | - | - | [Writeup](https://www.kaggle.com/c/stanford-rna-3d-folding/writeups/2nd-place-solution) |
| 3rd | - | - | [Writeup](https://www.kaggle.com/c/stanford-rna-3d-folding/writeups/3rd-place-solution) |

---

## Citation

```bibtex
@misc{stanford-rna-3d-folding,
    title = {Stanford RNA 3D Folding},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/stanford-rna-3d-folding}},
    note = {Kaggle}
}
```
