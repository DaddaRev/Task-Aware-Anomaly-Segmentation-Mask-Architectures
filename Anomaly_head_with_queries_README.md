# EoMT-EXT: Anomaly Segmentation Architecture

**Full Documentation:** See the [detailed report on main branch](../README.md)

## Introduction

**EoMT-EXT** (Extended Ensemble of Masks with Transformers) is a specialized architecture designed for **Anomaly Segmentation** (out-of-distribution object detection in road scenes). It extends the frozen semantic EoMT model by adding a dedicated **Anomaly Head** to explicitly reason about what is "Normal", "Anomalous", or "Background".

---

## Core Components

### 1. Backbone (Frozen)
- **Type**: `DinoV2` (Vision Transformer)
- **Role**: Extracts powerful, general-purpose semantic representations from the input image
- **Status**: Completely frozen during training to preserve OOD (Out-of-Distribution) robustness

### 2. Query Integration in ViT Backbone
- **Queries (`q`)**: Learnable embeddings processed **directly within the ViT backbone**, not in a separate decoder
- **Processing**: Queries interact globally with visual tokens throughout the encoding stage, evolving into compact high-level descriptors
- **Result**: Final query vectors encapsulate both semantic essence and spatial localization of candidate objects
- **Attention**: Masked Attention refines queries based on image features across multiple layers (default 4 final blocks)
- **Key Property**: The resulting query representations contain sufficient information to distinguish In-Distribution (ID) from Out-of-Distribution (OOD) samples

### 3. Mask Head & Class Head
- **Mask Head**: Predicts binary masks for each query (`[B, Q, H, W]`)
- **Class Head**: Traditional semantic classification head (e.g., "Car", "Road", "Sky")
  - *Crucial*: Initialized from pre-trained weights (e.g., Cityscapes) and **kept frozen** during anomaly tuning. Its predictions provide contextual semantics

---

## The Anomaly Head (New)

The defining feature of this architecture is the **Anomaly Head**, which replaces simple thresholding heuristics (like Max-Softmax Probability) with a learned component.

### Architecture

A **Multi-Layer Perceptron (MLP)** that takes query features and classifies them into 3 meta-states:

```
Input: Query Embeddings [B, Q, embed_dim]
  ↓
Linear(embed_dim → hidden_dim) + GELU
  ↓
Linear(hidden_dim → 3)  # [Normal=0, Anomaly=1, No-Object=2]
```

### Input

The head receives as input **only the final query embeddings** $q \in \mathbb{R}^d$ from the ViT encoder:
- **Query Embedding (`q`)**: Latent representation emerging from the final ViT block after processing jointly with visual tokens
- **Core Hypothesis**: This representation contains sufficient information to distinguish between In-Distribution (ID) and Out-of-Distribution (OOD) samples
- **Theoretical Assumption**: Self-attention naturally maps known classes into distinct, well-separated clusters in feature space. Anomalous instances—possessing unfamiliar semantic traits—result in query embeddings that project into divergent or low-density regions of the latent space

### Logic

The MLP decodes the intrinsic "abnormality" signal directly from geometric and semantic properties of the query representation:
- **Baseline Approach**: Evaluates whether the ViT encoder's latent space inherently encodes the concept of the "unknown" without auxiliary uncertainty signals
- **No External Metrics**: Unlike uncertainty-based approaches, this head operates purely on query geometry (entropy, softmax probability, etc. are NOT used)
- **Learning Objective**: Maps each query $q$ to a 3-dimensional logit vector:
  $$f_{\theta}(q) \in \mathbb{R}^3 \rightarrow \{\text{Normal}, \text{Anomaly}, \text{No\_Object}\}$$

---

## Training Flow

1. **Forward Pass**: Image → DinoV2 backbone (frozen) → ViT encoder with integrated queries → Final query embeddings $q$
2. **Core Hypothesis**: The evolved query vectors $q$ encode both semantic essence and spatial information sufficient to distinguish ID from OOD
3. **Predictions**:
   - Mask Head: generates spatial masks for each query $[B, Q, H, W]$
   - Class Head (frozen): semantic predictions (19 classes)
   - **Anomaly Head (trainable)**: Maps queries $q$ to 3-class logits $[B, Q, 3]$ (Normal/Anomaly/NoObject)
4. **Loss**: Combination of:
   - **Mask Loss**: spatial localization of anomalous regions
   - **Class Loss**: anomaly classification per query (decoding ID/OOD from latent geometry)
   - **Importance Sampling**: focus on hard negatives (boundary regions)
5. **Update**: Only Anomaly Head parameters (~11K params)

**Freezing Strategy (Baseline Design):**
- Evaluates intrinsic capability of encoder's latent space to encode "unknown" concept
- Forces network to decode abnormality solely from geometric and semantic properties of $q$
- Preserves learned semantic clustering (self-attention mapping of known classes into distinct clusters)
- Training 3-4× faster (no backward through backbone)
- No auxiliary metrics: anomaly detection from query representation alone

---

## Practical Configuration

### Key Files

- **`eomt/models/eomt_ext.py`**: Extended EoMT model architecture with the integrated Anomaly Head. Defines the forward pass, query processing, attention masking, and the 3-class anomaly prediction head that operates on query embeddings.

- **`eomt/training/mask_classification_semantic_anomaly.py`**: Training loop and evaluation pipeline for anomaly segmentation. Handles forward passes, loss computation, target preparation, gradient updates (anomaly head only), and metric calculation. Also contains visualization routines for monitoring predictions.

- **`eomt/configs/dinov2/common/eomt_base_640_ext_kaggleinput.yaml`**: Configuration file specifying all training hyperparameters, including learning rate, loss coefficients, number of blocks, batch size, optimizer settings, and logging preferences.

### Training

```bash
python eomt/main.py --config eomt/configs/dinov2/common/generic_anomaly.yaml
```

### Key Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `lr` | 1e-5 | Learning rate (anomaly head only) |
| `num_blocks` | 4 | Layers with anomaly supervision |
| `mask_coefficient` | 2.0 | Spatial loss weight |
| `dice_coefficient` | 4.0 | Dice loss weight |
| `class_coefficient` | 4.0 | Class loss weight |

### Dataset Format

```
targets = {
    "masks": [N, H, W],              # binary masks per instance
    "labels": [N] ∈ {0, 1, 255}      # 0=Normal, 1=Anomaly, 255=Void
}
```

Void pixels (255) automatically filtered during training/evaluation.
