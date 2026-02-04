# Open Problems - Single-Cell Perturbations

> Predict how small molecules change gene expression in different cell types

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-Output Regression |
| **Data Domain** | Computational Biology / Drug Discovery |
| **ML Approach** | Deep Learning + Chemical Embeddings |
| **Key Techniques** | Graph Neural Networks, Autoencoders, Multi-Task Learning |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $100,000 |
| **Teams** | 1,097 |
| **Timeline** | August - November 2023 |
| **Evaluation Metric** | Mean Rowwise Root Mean Squared Error |
| **Host** | Open Problems in Single-Cell Analysis |

## Problem Description

Predict how small molecule drugs affect gene expression across different cell types - a key challenge in drug discovery and personalized medicine.

### The Challenge

- Predict gene expression changes after drug treatment
- Handle sparse training data (not all drug-cell combinations)
- Generalize to unseen drug-cell type combinations
- Output ~18,000 gene expression values

### Why It Matters

- **Drug Discovery**: Predict drug effects before experiments
- **Personalized Medicine**: Cell type-specific responses
- **Cost Reduction**: Reduce expensive lab experiments
- **Safety**: Predict off-target effects

---

## Data Description

### Single-Cell Perturbation Data

Gene expression profiles before and after drug treatment, measured at single-cell resolution.

### Files

| File | Description |
|------|-------------|
| `de_train.parquet` | Training differential expression |
| `id_map.csv` | Mapping of IDs to cell types and drugs |
| `adata_train.h5ad` | AnnData format training data |
| `sample_submission.csv` | Submission format |

### Key Components

| Component | Description |
|-----------|-------------|
| Cell types | T cells, NK cells, etc. |
| Small molecules (SM) | ~144 drug compounds |
| Genes | ~18,000 genes to predict |
| DE values | Log fold change after treatment |

### Data Structure

Each row represents:
- A specific cell type
- Treated with a specific small molecule
- Target: differential expression for all genes

---

## Evaluation

**Metric**: Mean Rowwise Root Mean Squared Error (MRRMSE)

```python
import numpy as np

def mrrmse(y_true, y_pred):
    """
    Mean Rowwise RMSE

    y_true, y_pred: (N samples, 18211 genes)
    """
    # RMSE per row (sample)
    row_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))

    # Mean across rows
    return np.mean(row_rmse)
```

Lower is better.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $25,000 |
| 2nd | $20,000 |
| 3rd | $15,000 |
| 4th | $12,000 |
| 5th | $10,000 |
| 6th-10th | $3,600 each |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Molecular fingerprint embeddings
- Cell type embeddings
- Autoencoder for gene expression space
- Ensemble of multiple architectures

### 2nd Place Solution

**Technique**:
- Graph neural network on molecules
- Multi-task learning across cell types
- Transfer learning from public data

### 3rd Place Solution

**Innovation**:
- Attention between drug and cell embeddings
- Gene network priors
- Careful cross-validation

---

## Common Winning Strategies

### 1. Molecular Fingerprint Features

```python
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np

def get_molecular_features(smiles):
    """Extract features from SMILES representation"""
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # Morgan fingerprint (circular fingerprint)
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    morgan_array = np.array(morgan_fp)

    # Molecular descriptors
    descriptors = {
        'mol_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'tpsa': Descriptors.TPSA(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'rings': Descriptors.RingCount(mol),
        'aromatic_rings': Descriptors.NumAromaticRings(mol),
    }

    return morgan_array, descriptors

def create_drug_embeddings(drug_smiles_dict, embedding_dim=256):
    """Create embeddings for all drugs"""
    embeddings = {}

    for drug_id, smiles in drug_smiles_dict.items():
        features = get_molecular_features(smiles)
        if features:
            morgan, desc = features
            embeddings[drug_id] = np.concatenate([morgan, list(desc.values())])

    return embeddings
```

### 2. Cell Type Embeddings

```python
import torch
import torch.nn as nn

class CellTypeEncoder(nn.Module):
    def __init__(self, num_cell_types, embedding_dim=128):
        super().__init__()

        # Learnable cell type embeddings
        self.embedding = nn.Embedding(num_cell_types, embedding_dim)

        # Optional: encode cell type features
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_cell_type_features, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, cell_type_ids, cell_type_features=None):
        emb = self.embedding(cell_type_ids)

        if cell_type_features is not None:
            feature_emb = self.feature_encoder(cell_type_features)
            emb = emb + feature_emb

        return emb
```

### 3. Main Prediction Model

```python
class PerturbationPredictor(nn.Module):
    def __init__(self, drug_dim=2048, cell_dim=128, num_genes=18211):
        super().__init__()

        # Drug encoder
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )

        # Cell type encoder
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        # Interaction network
        self.interaction = nn.Sequential(
            nn.Linear(512, 512),  # drug + cell concatenated
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_genes)
        )

    def forward(self, drug_features, cell_features):
        drug_emb = self.drug_encoder(drug_features)
        cell_emb = self.cell_encoder(cell_features)

        combined = torch.cat([drug_emb, cell_emb], dim=1)
        gene_expression = self.interaction(combined)

        return gene_expression
```

### 4. Autoencoder for Gene Space

```python
class GeneExpressionAutoencoder(nn.Module):
    def __init__(self, num_genes=18211, latent_dim=512):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, num_genes)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

# Use latent space for prediction
class LatentSpacePredictor(nn.Module):
    def __init__(self, autoencoder, drug_dim, cell_dim, latent_dim=512):
        super().__init__()
        self.autoencoder = autoencoder

        # Freeze autoencoder weights
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        # Predict in latent space
        self.predictor = nn.Sequential(
            nn.Linear(drug_dim + cell_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, drug_features, cell_features):
        combined = torch.cat([drug_features, cell_features], dim=1)
        latent_pred = self.predictor(combined)

        # Decode to gene space
        gene_pred = self.autoencoder.decode(latent_pred)
        return gene_pred
```

### 5. Cross-Validation Strategy

```python
from sklearn.model_selection import GroupKFold
import numpy as np

def create_cv_splits(data, n_splits=5):
    """
    Cross-validation that respects drug-cell combinations
    Test on unseen combinations
    """
    # Option 1: Leave-one-cell-type-out
    cell_types = data['cell_type'].unique()
    for cell_type in cell_types:
        train_mask = data['cell_type'] != cell_type
        val_mask = data['cell_type'] == cell_type
        yield train_mask, val_mask

    # Option 2: Leave-one-drug-out
    drugs = data['sm_name'].unique()
    for drug in drugs:
        train_mask = data['sm_name'] != drug
        val_mask = data['sm_name'] == drug
        yield train_mask, val_mask

    # Option 3: Leave-combination-out
    # Ensure some drug-cell combinations are held out
    combinations = data[['cell_type', 'sm_name']].drop_duplicates()
    kf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in kf.split(combinations, groups=combinations['cell_type']):
        # ... create masks based on combination indices
        pass
```

---

## Technical Insights

### Gene Expression Biology

| Concept | Description |
|---------|-------------|
| DE | Differential expression (treated vs control) |
| Log fold change | log2(treated/control) |
| Up-regulation | Gene activity increases |
| Down-regulation | Gene activity decreases |

### Drug Response Variability

| Factor | Impact |
|--------|--------|
| Cell type | Different cells respond differently |
| Drug concentration | Dose-dependent effects |
| Time | Expression changes over time |
| Off-targets | Unintended gene effects |

### Feature Importance

| Feature Type | Importance |
|--------------|------------|
| Molecular structure | Very High |
| Cell type | High |
| Gene-gene correlations | Medium |
| Drug targets | Medium |

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Runtime | 9 hours |
| GPU Runtime | 9 hours |
| Internet | Disabled |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/discussion/459258) |
| 2nd | [Discussion](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/discussion/458738) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/discussion/458750) |

---

## Citation

```bibtex
@misc{open-problems-single-cell-perturbations,
    author = {Open Problems},
    title = {Open Problems - Single-Cell Perturbations},
    year = {2023},
    howpublished = {\url{https://kaggle.com/competitions/open-problems-single-cell-perturbations}},
    note = {Kaggle}
}
```
