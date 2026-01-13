# EoMT-EXT Architecture Overview

## Introduction
The **EoMT-EXT** (Extended Ensemble of Masks with Transformers) is a specialized architecture designed for **Anomaly Segmentation** (or Open-Set Semantic Segmentation). It builds upon a frozen backbone (DinoV2) and a mask-based transformer decoder, introducing a dedicated `Anomaly Head` to explicitly reason about what is "Normal", "Anomalous", or "Background".

## Core Components

### 1. Backbone (Frozen)
- **Type**: `DinoV2` (Vision Transformer).
- **Role**: Extracts powerful, general-purpose semantic features from the input image.
- **Status**: Completely frozen during training to preserve its out-of-distribution robustness.

### 2. Transformer Decoder (Mask2Former-style)
- **Queries (`q`)**: A set of learnable query vectors representing potential objects/segments in the image.
- **Attention**: Uses Masked Attention mechanisms to refine queries based on image features.
- **Layers**: Multi-stage decoding process (default 4 blocks).

### 3. Mask Head & Class Head
- **Mask Head**: Predicts binary masks for each query (`[B, Q, H, W]`).
- **Class Head**: Traditional semantic classification head (e.g., classifies "Car", "Road", "Sky").
  - *Crucial*: This head is also often initialized from pre-trained weights (e.g., Cityscapes) and kept frozen or fined-tuned with a low learning rate. Its entropy is used as a signal.

---

## The Anomaly Head (New)

The defining feature of this architecture is the **Anomaly Head**, which replaces simple thresholding heuristics (like Max-Softmax Probability) with a learned component.

### Architecture
It is a **Multi-Layer Perceptron (MLP)** that takes enriched query features and classifies them into 3 meta-states.

```python
self.anomaly_head = nn.Sequential(
    nn.Linear(embed_dim + 2, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, 3)  # [Normal, Anomaly, No_Object/Void]
)
```

### Inputs
The head receives a concatenation of **Semantic** and **Uncertainty** features:
1.  **Semantic Embedding (`q`)**: The raw output vector from the transformer decoder for a specific query. Contains "what" the object looks like.
2.  **Uncertainty Features**:
    -   **Entropy**: Calculated from the frozen `Class Head` logits. High entropy = The backbone is confused.
    -   **Max Probability**: The confidence of the backbone's top prediction.

### Logic (Why MLP?)
The MLP learns non-linear decision boundaries such as:
-   *IF (Embed="Road") AND (Entropy=High) -> **Anomaly*** (An obstacle on the road).
-   *IF (Embed="Sky") AND (Entropy=High) -> **Normal*** (Just a complex cloud pattern).
-   *IF (Mask is empty/weak) -> **No_Object***.

---

## Training Flow

1.  **Forward Pass**: The image goes through DinoV2 -> Decoder -> Queries.
2.  **Uncertainty Extraction**: The frozen Class Head predicts logits; their entropy is calculated.
3.  **Anomaly Prediction**: The `Anomaly Head` predicts logits a `[B, Q, 3]` tensor.
4.  **Loss**: Cross-Entropy Loss forces the head to learn the mapping from (Features + Uncertainty) to Ground Truth (0=Background, 1=Anomaly, 2=Void).

## Output Visualization
The final Anomaly Map is constructed by combining the mask predictions with the `Anomaly` probability mass from the head:

$$ \text{PixelAnomaly} = \sum_{q} \text{Mask}_q \cdot P(\text{Anomaly}|q) $$

Where $P(\text{Anomaly})$ is the absolute probability mass output by the Anomaly Head for the second class.

