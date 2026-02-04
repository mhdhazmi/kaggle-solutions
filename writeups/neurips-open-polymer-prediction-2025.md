# NeurIPS - Open Polymer Prediction 2025

> Predicting polymer properties with machine learning to accelerate sustainable materials research

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-target Regression |
| **Data Domain** | Chemistry / Materials Science / SMILES |
| **ML Approach** | BERT + GNN + Tabular Ensemble |
| **Key Techniques** | Label Rescaling, Post-processing, Uni-Mol, External Data |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition (NeurIPS Track) |
| **Total Prize** | $50,000 |
| **Teams** | 1,204 |
| **Timeline** | June 16, 2025 - September 16, 2025 |
| **Evaluation Metric** | Weighted Mean Absolute Error (wMAE) |
| **Host** | University of Notre Dame |

## Problem Description

Predict fundamental properties of polymers from their chemical structure (SMILES notation) to accelerate the development of sustainable materials.

### The Challenge

Polymers are essential building blocks - from DNA to everyday plastics. The search for next-generation eco-friendly materials requires predicting how polymers behave based on their molecular structure.

### Target Properties (5 total)

| Property | Symbol | Unit | Description |
|----------|--------|------|-------------|
| Glass Transition Temp | Tg | °C | Temperature where polymer transitions from hard to rubbery |
| Fractional Free Volume | FFV | - | Empty space between polymer chains |
| Thermal Conductivity | Tc | W/m·K | Heat conduction efficiency |
| Density | Density | g/cm³ | Mass per unit volume |
| Radius of Gyration | Rg | Å | Molecular size/packing efficiency |

### Dataset

- **Training**: ~5,000 polymers with SMILES + properties
- **Test**: ~1,500 polymers (hidden)
- **External data**: Multiple supplementary datasets provided
- **Ground truth**: Averaged from molecular dynamics simulations

---

## Data Description

### Files

| File | Description |
|------|-------------|
| `train.csv` | SMILES + 5 target properties |
| `test.csv` | SMILES only |
| `train_supplement/` | 4 external datasets |

### External Datasets

| Dataset | Source | Properties |
|---------|--------|------------|
| dataset1.csv | Host's older simulations | Tc |
| dataset2.csv | Tg table | SMILES only |
| dataset3.csv | Host's older simulations | Various |
| dataset4.csv | Host's older simulations | Various |

---

## Evaluation

**Metric**: Weighted Mean Absolute Error (wMAE)

```
wMAE = (1/|X|) × Σ_X Σ_i w_i × |ŷ_i(X) - y_i(X)|
```

### Weighting Factors

1. **Scale normalization**: Division by property range
2. **Inverse frequency**: Rare properties weighted higher
3. **Weight normalization**: Sum of weights = K (number of tasks)

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $12,000 |
| 2nd | $10,000 |
| 3rd | $10,000 |
| 4th | $8,000 |
| 5th | $5,000 |
| Top Student | $5,000 |

---

## Top Solutions

### 1st Place Solution (Private LB: 0.075)

**Author**: jsday96

#### Key Innovation: Post-Processing for Distribution Shift

The winning insight was detecting and correcting a **systematic bias in Tg labels**:

```python
submission_df["Tg"] += (submission_df["Tg"].std() * 0.5644)
```

**Discovery Process**:
- Probed for distribution shifts by adjusting predictions ±0.1 × std
- Found severe bias in Tg labels (not explainable by random noise)
- P-value < 0.01 that labels were systematically incorrect
- Fit V-shaped curve to find optimal correction coefficient

#### Model Ensemble

| Model | Public LB | Private LB |
|-------|-----------|------------|
| ModernBERT-base | 0.059 | 0.089 |
| CodeBERT | 0.058 | 0.090 |
| AutoGluon | 0.062 | 0.091 |
| Uni-Mol 2 84M | 0.062 | 0.091 |

**Ensemble Results**:

| Post-processing | Public LB | Private LB |
|-----------------|-----------|------------|
| No | 0.058 | 0.089 |
| Yes | 0.054 | **0.075** |

#### External Data Sources

1. 1,116 MD simulations run locally
2. PI1M (pseudolabeled subset)
3. LAMALAB curated Tg
4. RadonPy sample data
5. Seok et al.
6. Borredon et al.
7. Duke ChemProps

#### Handling Dirty Training Labels (5 Strategies)

**1. Label Rescaling**
- Predict properties using strong ensemble
- Train isotonic regression to map raw labels → ensemble predictions
- Use rescaled labels or weighted average

**2. Error-Based Data Filtering**
- Compute |label - ensemble_prediction|
- Discard samples above error threshold
- Threshold tuned by Optuna per dataset

**3. Sample Weighting**
- Weight samples by dataset quality
- Optuna-tuned per-dataset weights

**4. Semi-Manual Filter Rules**
- Drop suspicious values (e.g., Tc > 0.402 in RadonPy)

**5. Model Stacking**
- Train 41 XGBoost models on raw simulation data
- Use predictions as features for AutoGluon

#### BERT Pretraining

- Pretrained BERT models on pseudolabeled subset of PI1M
- Improved generalization to unseen polymer structures

---

### 2nd Place Solution

**Key Observations**:
- Many different models among top public notebooks
- Ensemble diversity was crucial

*(Full details in linked writeup)*

---

## Common Winning Strategies

### 1. Distribution Shift Detection
Probing for systematic biases between train and test data.

### 2. Multi-Modal Ensemble
Combining BERT (sequence), GNN (molecular graph), and tabular models.

### 3. External Data Integration
Leveraging multiple external datasets with careful quality control.

### 4. Label Quality Control
Isotonic regression rescaling, error-based filtering, sample weighting.

### 5. Property-Specific Models
Training separate models for each target property (not multi-task).

---

## Technical Insights

### Why Label Quality Matters

1. **External data noise**: Each dataset has unique quirks
2. **Simulation artifacts**: MD simulations have inherent variability
3. **Distribution shifts**: Train/test may come from different populations

### SMILES Representation

```
SMILES: Simplified Molecular-Input Line-Entry System
Example: CC(C)(C)c1ccc(cc1)C(C)(C)c2ccc(cc2)O
```

Encodes molecular structure as text, enabling use of NLP models.

---

## Technical Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Runtime | ≤9 hours |
| GPU Runtime | ≤9 hours |
| Internet Access | Disabled |
| External Data | Allowed |

---

## Solution Links

| Place | Author | Score | Solution |
|-------|--------|-------|----------|
| 1st | jsday96 | 0.075 | [Writeup](https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/1st-place-solution) |
| 2nd | - | - | [Writeup](https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/2nd-place-solution) |
| 3rd | - | - | [Writeup](https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/3rd-place-solution) |

---

## Resources

- [Inference Notebook](https://www.kaggle.com/code/jsday96/polymers-tabular-bert-unimol)
- [Training Code](https://github.com/jday96314/NeurIPS-polymer-prediction)

---

## Citation

```bibtex
@misc{neurips-open-polymer-prediction-2025,
    title = {NeurIPS - Open Polymer Prediction 2025},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/neurips-open-polymer-prediction-2025}},
    note = {Kaggle, University of Notre Dame, NeurIPS}
}
```
