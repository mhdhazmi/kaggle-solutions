# Image Matching Challenge 2025

> Reconstruct 3D scenes from messy image collections.

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | 3D Reconstruction + Image Clustering |
| **Data Domain** | Computer Vision / Structure from Motion |
| **ML Approach** | Feature Matching + Clustering + SfM |
| **Key Techniques** | SuperPoint, SuperGlue, LoFTR, COLMAP |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 943 |
| **Timeline** | April - June 2025 |
| **Evaluation Metric** | mAA + Clustering Score |
| **Host** | Czech Technical University in Prague |

## Problem Description

Reconstruct 3D scenes from mixed image collections while identifying which images belong together and which should be discarded.

### The Jigsaw Puzzle Analogy

Imagine solving a jigsaw puzzle where pieces from multiple different puzzles have been mixed together. You must:
1. Figure out which pieces belong to your puzzle
2. Assemble those pieces correctly
3. Discard pieces from other puzzles

### The Challenge

Given a collection of images:
- **Group images** that show the same scene
- **Identify outliers** (unrelated images)
- **Reconstruct 3D** for each grouped scene
- **Estimate camera poses** (rotation + translation)

### Why It Matters

- **Urban Planning**: Create 3D city models from crowdsourced photos
- **Augmented Reality**: Localize cameras in real environments
- **Robotics**: Visual SLAM and navigation
- **Scientific Research**: Large-scale archaeological reconstruction

---

## Data Description

### Dataset Structure

| Component | Description |
|-----------|-------------|
| `train/[dataset]/` | Image folders with multiple scenes |
| `train_labels.csv` | Ground truth poses and scene assignments |
| `train_thresholds.csv` | Per-scene evaluation thresholds |
| `test/` | Hidden test images (~1,300 images) |

### Label Format

| Field | Description |
|-------|-------------|
| `dataset` | Dataset identifier |
| `scene` | Scene identifier within dataset |
| `image` | Image filename |
| `rotation_matrix` | 3×3 rotation matrix (flattened, row-major) |
| `translation_vector` | 3D translation vector |

### Key Characteristics

- Multiple scenes per dataset (e.g., two sides of a building)
- Scenes within a dataset don't overlap
- May contain "outlier" images (no scene assignment)
- Various scene types: monuments, buildings, nature

---

## Evaluation

**Metric**: Combination of mAA (mean Average Accuracy) and Clustering Score

### mAA (Camera Pose Accuracy)

Measures how accurately predicted camera poses match ground truth:
- Registered images / Total images in scene
- Uses distance thresholds from `train_thresholds.csv`

### Clustering Score

Measures grouping quality:
```
Clustering Score = |S ∩ C| / |C|
```
Where S = true scene, C = predicted cluster

### Two-Step Scoring

1. Greedily match each ground-truth scene to best predicted cluster
2. Combine mAA and clustering score for final ranking

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
- Advanced feature matching (SuperGlue + LoFTR)
- Graph-based clustering for scene separation
- Robust RANSAC for outlier rejection
- COLMAP for final reconstruction

### 2nd Place Solution

**Technique**:
- LightGlue for efficient matching
- Spectral clustering on match graph
- Hierarchical reconstruction pipeline

### 3rd Place Solution

**Innovation**:
- Multi-scale feature extraction
- Learned outlier detection
- Incremental bundle adjustment

---

## Common Winning Strategies

### 1. Feature Extraction & Matching Pipeline

```
Images → Local Features → Feature Matching → Match Graph
                                                   ↓
                              Scene Clustering ← Similarity Matrix
                                                   ↓
                              Per-Scene SfM → Camera Poses
```

### 2. Feature Detectors/Matchers

| Method | Type | Strengths |
|--------|------|-----------|
| SuperPoint | Detector | Learned, robust |
| SuperGlue | Matcher | Attention-based |
| LoFTR | Detector-free | Dense matching |
| LightGlue | Matcher | Fast, accurate |
| DISK | Detector | Learned keypoints |

### 3. Scene Clustering

Approaches:
- **Graph clustering**: Build match graph, partition
- **Spectral clustering**: On similarity matrix
- **DBSCAN**: Density-based clustering
- **Community detection**: Louvain algorithm

### 4. Outlier Detection

- Few matches with other images
- Inconsistent geometry
- Isolated in match graph

### 5. 3D Reconstruction (SfM)

| Tool | Description |
|------|-------------|
| COLMAP | Gold standard open-source SfM |
| hloc | Hierarchical localization |
| OpenMVG | Modular SfM library |

---

## Technical Insights

### Image Matching Challenges

1. **Viewpoint variation**: Same scene from very different angles
2. **Illumination changes**: Day vs night, shadows
3. **Scale differences**: Close-up vs far away
4. **Repetitive structures**: Similar-looking different locations
5. **Dynamic objects**: People, vehicles (confounding)

### Scene Separation Problem

Similar scenes can be confused:
- Two sides of symmetric building
- Multiple similar trees in forest
- Repetitive architectural elements

### Coordinate System

Rotation matrix R and translation T define camera pose:
- Camera center: C = -R^T × T
- Projection: P = K[R|T] (K = intrinsics)

### Submission Format

```csv
image_id,dataset,scene,image,rotation_matrix,translation_vector
```

Scene labels are arbitrary (clustering output).

---

## Code Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Notebook | ≤12 hours |
| GPU Notebook | ≤12 hours |
| Internet | Disabled |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/image-matching-challenge-2025/discussion/583058) |
| 2nd | [Discussion](https://www.kaggle.com/c/image-matching-challenge-2025/discussion/583683) |
| 3rd | [Discussion](https://www.kaggle.com/c/image-matching-challenge-2025/discussion/583401) |
| 4th | [Discussion](https://www.kaggle.com/c/image-matching-challenge-2025/discussion/582959) |
| 5th | [Discussion](https://www.kaggle.com/c/image-matching-challenge-2025/discussion/583711) |

---

## Citation

```bibtex
@misc{image-matching-challenge-2025,
    title = {Image Matching Challenge 2025},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/image-matching-challenge-2025}},
    note = {Czech Technical University in Prague, Kaggle}
}
```
