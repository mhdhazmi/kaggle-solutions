# PlantTraits2024 - FGVC11

> Uncovering the biosphere: Predicting 6 Vital Plant Traits from Plant Images for Ecosystem Health

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-Output Regression |
| **Data Domain** | Computer Vision / Ecology |
| **ML Approach** | CNN + Metadata Fusion |
| **Key Techniques** | Transfer Learning, Multi-Task Learning |
| **Difficulty** | Intermediate-Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Prediction Competition |
| **Prize** | Knowledge |
| **Teams** | 398 |
| **Timeline** | February - June 2024 |
| **Evaluation Metric** | R² Score |
| **Host** | FGVC11 @ CVPR 2024 |

## Problem Description

Predict 6 plant traits from citizen science plant photographs, contributing to understanding of global biodiversity and ecosystem health.

### The Challenge

- Predict 6 continuous plant traits from images
- Handle citizen science photo variability
- Use ancillary environmental data
- Multi-output regression problem

### The 6 Plant Traits

| Trait | Description | Unit |
|-------|-------------|------|
| Plant Height | Canopy height | meters |
| Leaf Area | Individual leaf size | mm² |
| Leaf Mass per Area (LMA) | Leaf density | g/m² |
| Leaf Nitrogen | N content | mg/g |
| Leaf Dry Matter Content | LDMC | mg/g |
| Specific Leaf Area (SLA) | Leaf area per mass | mm²/mg |

### Why It Matters

- **Ecosystem Health**: Traits indicate ecosystem function
- **Climate Change**: Track how plants respond
- **Biodiversity**: Understand species distributions
- **Conservation**: Guide protection efforts

---

## Data Description

### Image Data

Citizen science plant photographs from:
- iNaturalist
- Various quality levels
- Natural backgrounds
- Different plant parts visible

### Ancillary Data

| Feature | Description |
|---------|-------------|
| Location | Latitude, longitude |
| Climate | Temperature, precipitation |
| Soil | Type, nutrients |
| Species | Taxonomic classification |

### Data Challenges

- Variable image quality
- Partial plant visibility
- Multiple plants in image
- Environmental confounding

---

## Evaluation

**Metric**: R² Score (Coefficient of Determination)

```python
from sklearn.metrics import r2_score

# Multi-output R²
def multi_output_r2(y_true, y_pred):
    r2_scores = []
    for i in range(y_true.shape[1]):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        r2_scores.append(r2)
    return np.mean(r2_scores)
```

Higher is better. R² = 1 means perfect prediction.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| Top teams | Recognition and publication opportunity |

This is a knowledge competition focused on scientific contribution.

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- EfficientNet backbone
- Metadata integration
- Species embeddings
- Multi-task learning

### 6th Place Solution

**Technique**:
- Vision Transformer
- Environmental feature engineering
- Ensemble methods

### 9th Place Solution

**Innovation**:
- Phylogenetic features
- Transfer from trait databases
- Domain adaptation

---

## Common Winning Strategies

### 1. Image Model with Metadata

```python
import timm
import torch.nn as nn

class PlantTraitModel(nn.Module):
    def __init__(self, num_traits=6, metadata_dim=50):
        super().__init__()
        # Image backbone
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True)
        img_features = self.backbone.num_features

        # Metadata encoder
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Combined head
        self.head = nn.Sequential(
            nn.Linear(img_features + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_traits)
        )

    def forward(self, image, metadata):
        img_features = self.backbone(image)
        meta_features = self.metadata_encoder(metadata)
        combined = torch.cat([img_features, meta_features], dim=1)
        return self.head(combined)
```

### 2. Environmental Feature Engineering

```python
def engineer_environmental_features(df):
    # Climate features
    df['temp_range'] = df['temp_max'] - df['temp_min']
    df['moisture_index'] = df['precipitation'] / (df['temp_mean'] + 10)

    # Location features
    df['abs_latitude'] = df['latitude'].abs()  # Proxy for climate zone

    # Species features
    df['genus'] = df['species'].apply(lambda x: x.split()[0])

    # Phylogenetic distance (if available)
    df['phylo_distance'] = compute_phylogenetic_distance(df['species'])

    return df
```

### 3. Multi-Task Learning

```python
class MultiTaskHead(nn.Module):
    def __init__(self, in_features, num_traits=6):
        super().__init__()
        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Task-specific heads
        self.trait_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            for _ in range(num_traits)
        ])

    def forward(self, x):
        shared_out = self.shared(x)
        return torch.cat([head(shared_out) for head in self.trait_heads], dim=1)
```

### 4. Handling Trait Correlations

```python
def correlation_aware_loss(y_pred, y_true, trait_correlations):
    """
    Weight losses based on trait correlations
    Highly correlated traits should have consistent predictions
    """
    mse_loss = F.mse_loss(y_pred, y_true, reduction='none')

    # Correlation regularization
    pred_corr = compute_correlation(y_pred)
    corr_loss = F.mse_loss(pred_corr, trait_correlations)

    return mse_loss.mean() + 0.1 * corr_loss
```

### 5. Species Embedding

```python
class SpeciesEmbedding(nn.Module):
    def __init__(self, num_species, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_species, embedding_dim)

        # Also encode taxonomic hierarchy
        self.genus_embedding = nn.Embedding(num_genera, 32)
        self.family_embedding = nn.Embedding(num_families, 16)

    def forward(self, species_id, genus_id, family_id):
        species_emb = self.embedding(species_id)
        genus_emb = self.genus_embedding(genus_id)
        family_emb = self.family_embedding(family_id)
        return torch.cat([species_emb, genus_emb, family_emb], dim=1)
```

---

## Technical Insights

### Plant Traits and Ecology

| Trait | Ecological Meaning |
|-------|-------------------|
| Height | Competition for light |
| Leaf Area | Light capture strategy |
| LMA | Investment in leaf structure |
| Leaf N | Photosynthetic capacity |
| LDMC | Leaf toughness |
| SLA | Resource acquisition strategy |

### Leaf Economics Spectrum

Plants show coordinated trait variation:
- High SLA ↔ Low LMA ↔ High N
- Fast-growing, resource-acquisitive
- vs. Conservative, stress-tolerant

### Image vs Environmental Features

| Predictor Source | Useful For |
|------------------|------------|
| Images | Size, shape, color |
| Climate | Expected trait ranges |
| Species | Prior trait knowledge |
| Location | Local adaptations |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/planttraits2024/discussion/510393) |
| 6th | [Discussion](https://www.kaggle.com/c/planttraits2024/discussion/510143) |
| 9th | [Discussion](https://www.kaggle.com/c/planttraits2024/discussion/510188) |

---

## Citation

```bibtex
@misc{planttraits2024,
    author = {FGVC},
    title = {PlantTraits2024 - FGVC11},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/planttraits2024}},
    note = {CVPR FGVC Workshop}
}
```
