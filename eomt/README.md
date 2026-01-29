# EoMT-EXT: Pixel-wise Anomaly Segmentation Architecture

**Reference:** Based on the original [EoMT repository](https://github.com/tue-mps/eomt)

## Introduction

**EoMT-EXT** (Extended Ensemble of Masks with Transformers) is a specialized architecture designed for **Anomaly Segmentation** (out-of-distribution object detection in road scenes). It extends the frozen semantic EoMT model by adding a dedicated **Pixel-wise Anomaly Head** that operates on statistical uncertainty features to explicitly classify each pixel as "Normal" or "Anomalous".

**Key Innovation:** This architecture employs a **pixel-wise MLP** that learns to decode anomaly signals exclusively from uncertainty metrics (Entropy, Max Probability, Energy, MSP) derived from frozen semantic heads, without any visual features.

---

## Core Components

### 1. Backbone (Frozen)
- **Type**: `DinoV2` (Vision Transformer)
- **Role**: Extracts powerful, general-purpose semantic representations from the input image
- **Status**: Completely frozen during training to preserve OOD (Out-of-Distribution) robustness and learned semantic clustering

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

The defining feature of this architecture is the **PixelAnomalyHead**, a learned component that replaces traditional thresholding heuristics with a trainable pixel-wise classifier.

### Statistical Feature Extraction ("The Bridge")

Before feeding the Anomaly Head, signals from frozen Mask Head and Class Head are fused into a dense semantic map, from which **4 statistical uncertainty metrics** are computed:

1. **Max Probability**: Maximum confidence observed for a pixel (how certain the model is about its main prediction)
2. **Entropy**: Measures disorder in the probability distribution (high entropy indicates confusion between multiple classes)
3. **Energy Proxy**: Sum of semantic probabilities, used as activation energy proxy
4. **MSP (Maximum Softmax Probability)**: Computed as `1.0 - MaxProb`, represents a direct baseline for anomaly

These 4 metrics are stacked to form the `stat_features` tensor of dimension `[B, H, W, 4]`, which constitutes the base input for the Anomaly Head.

### Architecture

A **pixel-wise Multi-Layer Perceptron (MLP)** that operates independently on each pixel:

```
Input: Statistical Features [B, H, W, 4]
  ↓
Linear(input_dim → hidden_dim)
  ↓
ReLU + BatchNorm1d + Dropout
  ↓
Linear(hidden_dim → hidden_dim // 2)
  ↓
ReLU + Dropout
  ↓
Linear(hidden_dim // 2 → 1)  # Anomaly Logit
```

### Logic

The MLP decodes anomaly signals from statistical uncertainty patterns:

- **Learned Uncertainty Interpretation**: The network learns which uncertainty configurations indicate OOD samples
- **Baseline Approach**: Evaluates whether statistical uncertainty alone contains sufficient signal for anomaly detection
- **No Manual Thresholding**: Temperature scaling and manual threshold selection are replaced by learned parameters
- **Pixel-wise Classification**: Each pixel receives an independent anomaly score based on local uncertainty metrics only

---

## Training Flow

### Forward Pass Pipeline

1. **Backbone Processing**: Image → Frozen DinoV2 → `mask_logits`, `class_logits`
2. **Statistical Feature Computation**:
   - Combine `Sigmoid(mask_logits) × Softmax(class_logits)` → Dense Semantic Map
   - Extract Entropy, MaxProb, Energy, MSP → `stat_features` (4 channels)
3. **Anomaly Scoring**: `PixelAnomalyHead(stat_features)` → Anomaly Logits `[B, H, W, 1]`

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

- **Trainable**: Only Anomaly Head parameters (~50K params depending on hidden_dim)
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

- **`models/eomt_ext.py`**: Extended EoMT model architecture with the integrated Pixel-wise Anomaly Head. Implements the `compute_anomaly_score()` method that fuses mask/class predictions into statistical features and feeds them to the MLP.

- **`training/mask_classification_semantic_anomaly.py`**: Training loop and evaluation pipeline for anomaly segmentation. Handles forward passes, loss computation with class imbalance weighting, void masking, gradient updates (anomaly head only), and metric calculation.

- **`configs/dinov2/common/eomt_base_640_ext.yaml`**: Configuration file specifying all training hyperparameters, including learning rate, loss coefficients, batch size, optimizer settings, and logging preferences.

### Key Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `lr` | 1e-4 | Learning rate (anomaly head only) |
| `hidden_dim` | 256 | Hidden dimension of Anomaly Head MLP |
| `pos_weight` | 10.0 | Positive class weight (handles class imbalance) |
| `dropout` | 0.3 | Dropout rate in MLP layers |

### Training

To train the Anomaly Head extension:

```bash
python3 main.py fit \
  -c configs/dinov2/common/eomt_base_640_ext.yaml \
  --trainer.devices 1 \
  --data.batch_size 4 \
  --data.path /path/to/anomaly/datasets
```

**Pre-trained Checkpoint:**  
Download the trained model from [Google Drive](https://drive.google.com/file/d/1mWtNfEBbJ0dGu1newtvhNsVpYi3z1-Hg/view?usp=drive_link)

### Evaluation

Validation on benchmark datasets is performed via notebooks:

- **[model evaluation -COLAB.ipynb](model%20evaluation%20-COLAB.ipynb)**: Complete evaluation pipeline using the learned Anomaly Head on 5 benchmark datasets (FS LaF, FS Static, LostAndFound, RoadAnomaly21, RoadObstacle21)

