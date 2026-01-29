# EoMT-EXT: Pixel-wise Anomaly Segmentation Architecture

**Reference:** Based on the original [EoMT repository](https://github.com/tue-mps/eomt)

## Introduction

**EoMT-EXT** (Extended Ensemble of Masks with Transformers) is a specialized architecture designed for **Anomaly Segmentation** (out-of-distribution object detection in road scenes). It extends the frozen semantic EoMT model by adding a dedicated **Pixel-wise Anomaly Head** that operates on statistical uncertainty features **combined with visual features from the backbone** to explicitly classify each pixel as "Normal" or "Anomalous".

**Key Innovation:** This architecture employs a **pixel-wise MLP** that learns to decode anomaly signals from uncertainty metrics (Entropy, Max Probability, Energy, MSP) derived from frozen semantic heads **fused with visual features** extracted from the DinoV2 backbone.

---

## Core Components

### 1. Backbone (Frozen)
- **Type**: `DinoV2` (Vision Transformer)
- **Role**: Extracts powerful, general-purpose semantic representations from the input image
- **Status**: Completely frozen during training to preserve OOD (Out-of-Distribution) robustness and learned semantic clustering
- **Additional Output**: Visual features from final layer used for anomaly head fusion

### 2. Mask Head & Class Head (Frozen, Repurposed)

These components are **not discarded** but frozen and repurposed as **uncertainty signal generators**:

**Mask Head:**
- **Original Function**: Generates binary mask logits for each query `[B, Q, H, W]`
- **Role in Anomaly Detection**: Provides spatial localization of predictions
- **Processing**: Output is interpolated to target image size and passed through **Sigmoid** to obtain mask probabilities

**Class Head:**
- **Original Function**: Projects query features into semantic class space (C classes)
- **Role in Anomaly Detection**: Provides confidence distribution over known semantic classes
- **Processing**: Output processed via **Softmax**
- **Crucial Detail**: Probabilities for "Void" or "No-Object" class (typically the last class) are excluded to compute uncertainty only over in-distribution classes

---

## The Pixel-wise Anomaly Head (New)

The defining feature of this architecture is the **PixelAnomalyHead**, a learned component that replaces traditional thresholding heuristics with a trainable pixel-wise classifier operating on **hybrid features**.

### Statistical Feature Extraction ("The Bridge")

Before feeding the Anomaly Head, signals from frozen Mask Head and Class Head are fused into a dense semantic map, from which **4 statistical uncertainty metrics** are computed:

1. **Max Probability**: Maximum confidence observed for a pixel (how certain the model is about its main prediction)
2. **Entropy**: Measures disorder in the probability distribution (high entropy indicates confusion between multiple classes)
3. **Energy Proxy**: Sum of semantic probabilities, used as activation energy proxy
4. **MSP (Maximum Softmax Probability)**: Computed as `1.0 - MaxProb`, represents a direct baseline for anomaly

These 4 metrics are stacked to form the `stat_features` tensor of dimension `[B, H, W, 4]`.

### Visual Features Fusion

**Visual features** are extracted from the final layer of the frozen DinoV2 backbone during the forward pass:
- **Source**: Last transformer block output
- **Dimension**: `[B, C_embed, H_patch, W_patch]` where `C_embed` is the backbone embedding dimension
- **Processing**: Bilinearly interpolated to match the spatial resolution `(H, W)` of statistical features
- **Result**: `visual_features` of dimension `[B, H, W, C_embed]`

The final input to the Anomaly Head is the **concatenation** along the channel dimension:
```
input_features = Concat(stat_features, visual_features)  # [B, H, W, 4 + C_embed]
```

### Logic

The MLP decodes anomaly signals from **hybrid uncertainty and visual patterns**:

- **Learned Hybrid Interpretation**: The network learns which combinations of uncertainty metrics and visual features indicate OOD samples
- **Complex Correlations**: Can capture patterns like "High entropy on specific texture = Anomaly" or "Low confidence on unusual color distribution = Anomaly"
- **No Manual Thresholding**: Temperature scaling and manual threshold selection are replaced by learned parameters
- **Pixel-wise Classification**: Each pixel receives an independent anomaly score based on local uncertainty metrics **and** visual context

---

## Training Flow

### Forward Pass Pipeline

1. **Backbone Processing**: Image → Frozen DinoV2 → `mask_logits`, `class_logits`, `visual_features`
2. **Statistical Feature Computation**:
   - Combine `Sigmoid(mask_logits) × Softmax(class_logits)` → Dense Semantic Map
   - Extract Entropy, MaxProb, Energy, MSP → `stat_features` (4 channels)
3. **Visual Feature Extraction**:
   - Extract visual features from final backbone layer
   - Interpolate to spatial resolution `(H, W)` via bilinear interpolation
4. **Feature Fusion**:
   - Concatenate `stat_features` and `visual_features` → `hybrid_features` `[B, H, W, 4 + C_embed]`
5. **Anomaly Scoring**: `PixelAnomalyHead(hybrid_features)` → Anomaly Logits `[B, H, W, 1]`

### Loss Function

**Binary Cross-Entropy with Logits** (pixel-wise classification):

```python
loss = F.binary_cross_entropy_with_logits(
    logits, targets, 
    pos_weight=torch.tensor([10.0]),  # Handle class imbalance
    reduction='none'
)
```

**Class Imbalance Handling:**
- `pos_weight=10.0`: Anomalous pixels penalize 10× more than normal pixels
- Forces the model not to predict "Normal" everywhere

**Void Masking:**
- Valid mask constructed: `1.0` for valid pixels (Normal/Anomaly), `0.0` for Void (index 255)
- Ensures the model is never rewarded/penalized for predictions on ignored regions

### Update Strategy

- **Trainable**: Only Anomaly Head parameters (~50K-100K params depending on hidden_dim and C_embed)
- **Frozen**: Backbone (DinoV2), Mask Head, Class Head
- **Advantage**: 3-4× faster training, preserves semantic representations

---

## Inference Pipeline

### Windowing and Prediction

High-resolution images (e.g., Cityscapes) are divided into overlapping windows (crops):
- Anomaly Head predicts raw logit `L` for each pixel in crop `[B, H_crop, W_crop, 1]`

### Channel Construction for Compatibility

For compatibility with multi-class stitching functions, the single logit is expanded into two virtual channels:
1. **Channel 0 (Normal)**: Set to `0`
2. **Channel 1 (Anomaly)**: Contains predicted logit `L`

Resulting tensor `[B, 2, H_crop, W_crop]` is stitched back to original resolution, handling overlaps via weighted averaging.

### Final Normalization

**Softmax** is applied along the channel dimension
**Output**: Anomaly heatmap with values in `[0, 1]`, representing per-pixel anomaly probability.

---

## Practical Configuration

### Key Files

- **`models/eomt_ext.py`**: Extended EoMT model architecture with the integrated Pixel-wise Anomaly Head. Implements the `compute_anomaly_score()` method that fuses mask/class predictions into statistical features, extracts visual features from the backbone, and feeds the concatenated hybrid features to the MLP.

- **`training/mask_classification_semantic_anomaly.py`**: Training loop and evaluation pipeline for anomaly segmentation. Handles forward passes, loss computation with class imbalance weighting, void masking, gradient updates (anomaly head only), and metric calculation.

- **`configs/dinov2/common/eomt_base_640_ext.yaml`**: Configuration file specifying all training hyperparameters, including learning rate, loss coefficients, batch size, optimizer settings, and logging preferences.

### Key Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `lr` | 1e-4 | Learning rate (anomaly head only) |
| `hidden_dim` | 256 | Hidden dimension of Anomaly Head MLP |
| `pos_weight` | 10.0 | Positive class weight (handles class imbalance) |
| `dropout` | 0.3 | Dropout rate in MLP layers |


**Pre-trained Checkpoint:**  
Download the trained model from [Google Drive](https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav?usp=drive_link)

### Evaluation

Validation on benchmark datasets is performed via notebooks:

- **[model evaluation -COLAB.ipynb](model%20evaluation%20-COLAB.ipynb)**: Complete evaluation pipeline using the learned Anomaly Head on 5 benchmark datasets (FS LaF, FS Static, LostAndFound, RoadAnomaly21, RoadObstacle21)

### Dataset Format

Anomaly datasets follow this target structure:

```python
targets = {
    "masks": [N, H, W],              # Binary masks per instance
    "labels": [N] ∈ {0, 1, 255}      # 0=Normal, 1=Anomaly, 255=Void
}
```

Void pixels (255) are automatically filtered during training and evaluation.
