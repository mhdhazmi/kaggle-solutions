# NeurIPS 2024 - Predict New Medicines with BELKA

> Predict small molecule-protein interactions using the Big Encoded Library for Chemical Assessment (BELKA)

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Binary Classification / Drug Discovery |
| **Data Domain** | Computational Chemistry / Biochemistry |
| **ML Approach** | Graph Neural Networks + Molecular Fingerprints |
| **Key Techniques** | SMILES Processing, Molecular Embeddings, GNN |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Prediction Competition |
| **Total Prize** | $50,000 |
| **Teams** | 1,950 |
| **Timeline** | April - July 2024 |
| **Evaluation Metric** | Leash Average mAP |
| **Host** | Leash Bio / NeurIPS |

## Problem Description

Predict whether small drug-like molecules will bind to specific protein targets - a critical step in drug discovery that could accelerate the search for new medicines.

### The Challenge

- Predict binding affinity for 3 protein targets
- Handle 133 million small molecules
- Use SMILES molecular representations
- Enable faster drug discovery

### Why It Matters

- **Drug Discovery**: Only ~2,000 FDA-approved molecules exist
- **Chemical Space**: 10^60 possible drug-like molecules
- **Efficiency**: Physical testing is time-consuming
- **Healthcare**: Find treatments hiding in chemical space

---

## Data Description

### BELKA Dataset

The Big Encoded Library for Chemical Assessment contains:
- 133 million small molecules physically tested
- Binding results for 3 protein targets
- SMILES molecular representations

### Molecular Representation

| Format | Description |
|--------|-------------|
| SMILES | String representation of molecules |
| Fingerprints | Binary feature vectors |
| Graphs | Atoms as nodes, bonds as edges |

### Example SMILES

```
CC(=O)OC1=CC=CC=C1C(=O)O  # Aspirin
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O  # Ibuprofen
```

### Protein Targets

Three protein targets for binding prediction:
- BRD4 (Bromodomain)
- HSA (Human Serum Albumin)
- sEH (Soluble Epoxide Hydrolase)

---

## Evaluation

**Metric**: Leash Average mAP (Mean Average Precision)

Average precision computed across:
- Different protein targets
- Various molecule subsets

Higher is better.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st-5th | Share of $50,000 |
| Top Student Group | Additional prize |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Graph Neural Networks for molecular graphs
- Ensemble with fingerprint-based models
- Data augmentation
- Careful handling of molecular structure

### 5th Place Solution

**Technique**:
- ECFP fingerprints
- Gradient boosting on fingerprints
- GNN fine-tuning

### 8th Place Solution

**Innovation**:
- Pre-trained molecular transformers
- Transfer learning from ChEMBL
- Ensemble methods

---

## Common Winning Strategies

### 1. SMILES to Molecular Graph

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # Node features (atoms)
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetHybridization(),
            atom.GetIsAromatic(),
        ]
        atom_features.append(features)

    # Edge features (bonds)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])

    return Data(
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t()
    )
```

### 2. Molecular Fingerprints

```python
from rdkit.Chem import AllChem
import numpy as np

def get_fingerprints(smiles, fp_type='morgan'):
    mol = Chem.MolFromSmiles(smiles)

    if fp_type == 'morgan':
        # Morgan/ECFP fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    elif fp_type == 'maccs':
        # MACCS keys
        fp = AllChem.GetMACCSKeysFingerprint(mol)
    elif fp_type == 'rdkit':
        # RDKit fingerprint
        fp = Chem.RDKFingerprint(mol)

    return np.array(fp)
```

### 3. Graph Neural Network

```python
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class MoleculeGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Global pooling
        x = global_mean_pool(x, batch)

        return torch.sigmoid(self.fc(x))
```

### 4. Pre-trained Molecular Models

```python
# Using ChemBERTa or MolBERT
from transformers import AutoModel, AutoTokenizer

class MolecularTransformer(nn.Module):
    def __init__(self, model_name='seyonec/ChemBERTa-zinc-base-v1'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, 3)  # 3 protein targets

    def forward(self, smiles_tokens):
        outputs = self.encoder(**smiles_tokens)
        pooled = outputs.last_hidden_state[:, 0]
        return torch.sigmoid(self.classifier(pooled))
```

### 5. Multi-Target Learning

```python
class MultiTargetModel(nn.Module):
    def __init__(self, encoder, hidden_dim, n_targets=3):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            for _ in range(n_targets)
        ])

    def forward(self, x):
        features = self.encoder(x)
        return torch.cat([head(features) for head in self.heads], dim=1)
```

---

## Technical Insights

### Drug-Likeness Rules

Lipinski's Rule of Five:
- Molecular weight < 500
- LogP < 5
- H-bond donors < 5
- H-bond acceptors < 10

### Molecular Representations Comparison

| Representation | Pros | Cons |
|----------------|------|------|
| SMILES | Compact, standard | Loses 3D info |
| Fingerprints | Fast, fixed-size | Information loss |
| Graphs | Full structure | Variable size |
| 3D conformers | Physical geometry | Computationally expensive |

### Binding Affinity Prediction

| Approach | Description |
|----------|-------------|
| Ligand-based | Uses molecular similarity |
| Structure-based | Uses protein structure |
| ML-based | Learns from binding data |

### Scale Challenges

With 133M molecules:
- Memory management critical
- Batch processing essential
- Efficient fingerprint storage
- Distributed training helpful

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/leash-BELKA/discussion/519020) |
| 5th | [Discussion](https://www.kaggle.com/c/leash-BELKA/discussion/521894) |
| 8th | [Discussion](https://www.kaggle.com/c/leash-BELKA/discussion/519815) |
| 11th | [Discussion](https://www.kaggle.com/c/leash-BELKA/discussion/518993) |
| 13th | [Discussion](https://www.kaggle.com/c/leash-BELKA/discussion/519133) |

---

## Citation

```bibtex
@misc{leash-belka,
    author = {Leash Bio},
    title = {NeurIPS 2024 - Predict New Medicines with BELKA},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/leash-BELKA}},
    note = {NeurIPS, Kaggle}
}
```
