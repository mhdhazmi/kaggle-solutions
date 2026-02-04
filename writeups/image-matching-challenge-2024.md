# Image Matching Challenge 2024 - Hexathlon

> Reconstruct 3D scenes from 2D images over six different domains

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | 3D Reconstruction / Image Matching |
| **Data Domain** | Computer Vision / Structure from Motion |
| **ML Approach** | Deep Feature Matching + Geometric Verification |
| **Key Techniques** | Local Features, RANSAC, Bundle Adjustment |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 929 |
| **Timeline** | March - June 2024 |
| **Evaluation Metric** | mAA (Mean Average Accuracy on camera centers) |
| **Host** | Czech Technical University in Prague |

## Problem Description

Construct precise 3D maps from sets of images across six diverse domains, from drone imagery to nighttime scenes.

### The Challenge

- Match images across very different viewpoints
- Handle 6 problem categories (Hexathlon)
- Robust to varying conditions (lighting, texture)
- Accurate camera pose estimation

### The Six Domains

1. Urban scenes
2. Natural environments
3. Drone imagery
4. Indoor scenes
5. Transparent objects
6. Nighttime conditions

### Why It Matters

- **Autonomous Systems**: Self-driving cars, robots
- **AR/VR**: Localization and mapping
- **Heritage**: 3D documentation of sites
- **Mapping**: Large-scale reconstruction

---

## Data Description

### Input

Sets of images for each scene:
- Multiple viewpoints
- Varying conditions per domain
- Different camera types

### Output

Camera poses (rotation and translation) for each image relative to a common reference frame.

### Ground Truth

| Property | Description |
|----------|-------------|
| Camera intrinsics | Focal length, distortion |
| Camera poses | Rotation matrix + translation |
| Registration | Common coordinate system |

---

## Evaluation

**Metric**: mAA on Camera Centers with Registration

Mean Average Accuracy measuring how accurately predicted camera positions match ground truth.

```python
def mean_average_accuracy(pred_poses, gt_poses, thresholds):
    """
    pred_poses: predicted camera positions
    gt_poses: ground truth positions
    thresholds: distance thresholds in meters
    """
    accuracies = []
    for thresh in thresholds:
        errors = np.linalg.norm(pred_poses - gt_poses, axis=1)
        acc = (errors < thresh).mean()
        accuracies.append(acc)
    return np.mean(accuracies)
```

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st-5th | Share of $50,000 |

---

## Top Solutions

### 1st Place Solution

**Author**: Igor Lashkov

**Key Approach**:
- High resolution ALIKED/LightGlue features
- Transparent scene handling trick
- Careful geometric verification
- COLMAP for bundle adjustment

### 2nd Place Solution

**Author**: Neo

**Technique**: "MST-Aided SfM & Transparent Scene Solution"
- Minimum spanning tree for image ordering
- Special handling for transparent objects
- Robust matching pipeline

### 3rd Place Solution

**Innovation**:
- Multi-scale feature extraction
- Domain-specific adaptations
- Ensemble of matching methods

---

## Common Winning Strategies

### 1. Feature Extraction Pipeline

```python
import kornia
from kornia.feature import ALIKED, LightGlue

class FeatureExtractor:
    def __init__(self):
        self.aliked = ALIKED(max_num_keypoints=8192, detection_threshold=0.001)
        self.lightglue = LightGlue(features='aliked')

    def extract(self, image):
        # Detect keypoints and descriptors
        keypoints, descriptors = self.aliked(image)
        return keypoints, descriptors

    def match(self, desc1, desc2):
        matches = self.lightglue(desc1, desc2)
        return matches
```

### 2. Robust Matching

```python
from kornia.geometry import find_fundamental, RANSAC

def robust_match(kp1, kp2, matches, threshold=3.0):
    # Get matched points
    pts1 = kp1[matches[:, 0]]
    pts2 = kp2[matches[:, 1]]

    # RANSAC for geometric verification
    F, inliers = find_fundamental(
        pts1, pts2,
        method='RANSAC',
        threshold=threshold,
        confidence=0.999
    )

    return matches[inliers], F
```

### 3. Transparent Object Handling

Key insight from top solutions:
```python
def handle_transparent_scenes(images, matches):
    """
    Transparent objects cause matching failures
    due to see-through features
    """
    # Use higher resolution
    # Focus on edges rather than textures
    # Multiple matching passes
    # Careful outlier rejection

    # Specific trick from 1st place
    # Process transparent regions separately
    pass
```

### 4. Structure from Motion Pipeline

```python
import pycolmap

def run_sfm(images, matches, output_dir):
    # COLMAP reconstruction
    database_path = f"{output_dir}/database.db"

    # Import features
    pycolmap.import_features(database_path, images, features)

    # Import matches
    pycolmap.import_matches(database_path, matches)

    # Sequential mapper
    pycolmap.sequential_mapper(
        database_path,
        output_dir,
        ImageReaderOptions(camera_model='PINHOLE')
    )

    # Bundle adjustment
    reconstruction = pycolmap.Reconstruction(output_dir)
    return reconstruction
```

### 5. Multi-Scale Approach

```python
def multi_scale_matching(img1, img2, scales=[1.0, 0.5, 0.25]):
    all_matches = []

    for scale in scales:
        # Resize images
        h, w = img1.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        img1_scaled = cv2.resize(img1, new_size)
        img2_scaled = cv2.resize(img2, new_size)

        # Extract and match
        kp1, desc1 = extract_features(img1_scaled)
        kp2, desc2 = extract_features(img2_scaled)
        matches = match_features(desc1, desc2)

        # Scale keypoints back
        kp1[:, :2] /= scale
        kp2[:, :2] /= scale

        all_matches.append((kp1, kp2, matches))

    return merge_matches(all_matches)
```

---

## Technical Insights

### SfM Pipeline Components

| Stage | Purpose |
|-------|---------|
| Feature Detection | Find keypoints (corners, blobs) |
| Feature Description | Create distinctive descriptors |
| Feature Matching | Find correspondences |
| Geometric Verification | Remove outliers with RANSAC |
| Incremental SfM | Build 3D structure |
| Bundle Adjustment | Optimize all parameters |

### Domain-Specific Challenges

| Domain | Challenge | Solution |
|--------|-----------|----------|
| Urban | Repetitive structures | Geometric constraints |
| Natural | Textureless regions | Dense features |
| Drone | Large viewpoint change | Wide-baseline matching |
| Indoor | Low texture | Multi-scale features |
| Transparent | See-through objects | Edge-based matching |
| Night | Low light | Noise-robust features |

### Key Feature Matchers

| Method | Strengths |
|--------|-----------|
| SuperGlue | Learned matching |
| LightGlue | Fast, accurate |
| LoFTR | Dense matching |
| ALIKED | Efficient, accurate |

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| Runtime | 9 hours |
| Internet | Disabled |
| GPU | Available |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/image-matching-challenge-2024/discussion/510084) |
| 2nd | [Discussion](https://www.kaggle.com/c/image-matching-challenge-2024/discussion/510499) |
| 3rd | [Discussion](https://www.kaggle.com/c/image-matching-challenge-2024/discussion/510338) |
| 4th | [Discussion](https://www.kaggle.com/c/image-matching-challenge-2024/discussion/510611) |
| 5th | [Discussion](https://www.kaggle.com/c/image-matching-challenge-2024/discussion/510603) |

---

## Citation

```bibtex
@misc{image-matching-challenge-2024,
    author = {Czech Technical University in Prague},
    title = {Image Matching Challenge 2024 - Hexathlon},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/image-matching-challenge-2024}},
    note = {Kaggle}
}
```
