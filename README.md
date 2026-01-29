# Mask Architecture Anomaly Segmentation (EoMT-EXT)

[![Course Project](https://img.shields.io/badge/Project-ML_Course-blue)](https://drive.google.com/file/d/19gQ2uhI8jPxWIdGewkaA2DqihUnQLVfB/view?usp=drive_link)
![License](https://img.shields.io/badge/license-MIT-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c)
![Lightning](https://img.shields.io/badge/Lightning-2.5+-792ee5)
![Python](https://img.shields.io/badge/Python-3.10+-blue)

> **EoMT-EXT** (Extended Ensemble of Masks with Transformers) is a specialized Transformer-based architecture designed for **Anomaly Segmentation** in road scenes. It combines a frozen DinoV2 backbone with a learned Anomaly Head to detect out-of-distribution objects in autonomous driving scenarios.

## Overview

This repository implements a state-of-the-art approach for **Open-Set Semantic Segmentation** (Anomaly Detection), tailored for real-world autonomous driving applications. Unlike traditional semantic segmentation that classifies pixels into predefined categories, this project explicitly detects "anomalous" or "unknown" objects that deviate from the training distribution.

**Key Features:**
-  **Frozen DinoV2 Backbone** - Leverages self-supervised vision features for robust out-of-distribution detection
- **Learned Anomaly Head** - An MLP-based classifier that combines semantic embeddings with uncertainty estimates
- **PyTorch Lightning** - Scalable training framework with mixed precision support
- **Comprehensive Evaluation** - Supports multiple datasets (Cityscapes, ADE20K, COCO)
- **Modular Design** - Easily configurable via YAML configs and command-line arguments

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)

---

## Architecture Overview

The **EoMT-EXT** architecture is specifically designed to tackle the challenge of detecting anomalies in semantic segmentation tasks. Unlike standard approaches that use heuristics like Max-Softmax thresholding, this model learns an explicit mapping from semantic features and uncertainty estimates to anomaly predictions.

### Core Components

#### 1. **Backbone (DinoV2) - Frozen**
- **Model**: `facebook/dinov2-base` or `facebook/dinov2-large`
- **Purpose**: Extract powerful, self-supervised semantic features from input images
- **Status**: Frozen during training to preserve out-of-distribution robustness
- **Why DinoV2?**: Provides general-purpose features that transfer well across datasets without catastrophic forgetting

#### 2. **Transformer Decoder (Mask2Former-style)**
- **Architecture**: Multi-stage masked attention decoder
- **Components**:
  - Learnable query embeddings representing potential objects
  - Cross-attention between queries and image features
  - Self-attention for refinement
  - Configurable number of decoder blocks (default: 4)
- **Output**: Dense mask predictions and query embeddings

#### 3. **Mask Head**
- Predicts binary segmentation masks for each query
- Output shape: `[Batch, Queries, Height, Width]`
- Enables fine-grained spatial reasoning

#### 4. **Class Head** 
- Traditional semantic classification (e.g., "Car", "Road", "Person")
- Can be frozen or fine-tuned with low learning rate
- Provides entropy signal for uncertainty estimation

#### 5. **Anomaly Head (Novel Component)** ⭐
A learned **Multi-Layer Perceptron** that replaces simple heuristics with explicit learning:

```python
anomaly_head = nn.Sequential(
    nn.Linear(embed_dim + uncertainty_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, 3)  # [Normal, Anomaly, Void]
)
```

**Input Features:**
1. **Semantic Embedding**: Raw query vector from transformer decoder
2. **Entropy**: Computed from frozen Class Head logits (measures backbone confusion)
3. **Max Probability**: Confidence of top semantic prediction

**Decision Logic**: The MLP learns non-linear boundaries such as:
- "Road + High Entropy → Anomaly" (obstacle on familiar road)
- "Sky + High Entropy → Normal" (just complex cloud pattern)
- "Weak Mask → Void/Background"

### Final Anomaly Prediction

$$\text{AnomalyMap}(i,j) = \sum_{q=1}^{Q} \text{Mask}_q(i,j) \cdot P(\text{Anomaly}|q)$$

where $P(\text{Anomaly}|q)$ is the softmax probability for the anomaly class from the Anomaly Head.

---

## Installation

### Requirements
- **Python**: 3.10 or higher
- **CUDA**: 11.8+ (for GPU acceleration)
- **GPU Memory**: 16GB+ recommended (depends on image resolution and batch size)

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/MaskArchitectureAnomaly.git
cd MaskArchitectureAnomaly
```

### Step 2: Create a Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**For Colab/Kaggle environments**, use:
```bash
pip install -r requirements_colab.txt
```

### Step 4: Download Pre-trained Models (Optional)
The DinoV2 backbone will be automatically downloaded on first use. For pre-trained ERFNet weights:
```bash
# Models will be downloaded automatically or can be placed in trained_models/
```

---

## Quick Start

### Training from Scratch
```bash
cd eomt
python main.py fit --config ../configs/your_config.yaml
```

### Running Inference
```bash
python inference.ipynb
# Or use the evaluation script:
cd eval
python evalAnomaly.py
```

### Evaluating Pre-trained Models
```bash
cd eval
python eval_cityscapes_color.py
```

---

## Training

### Configuration

Training is configured via YAML files in `eomt/configs/`. Key parameters:

```yaml
# Model Configuration
model:
  backbone: dinov2_base  # or dinov2_large
  decoder_layers: 4
  embed_dim: 768
  num_queries: 100

# Dataset Configuration
data:
  dataset: cityscapes_semantic
  image_size: 640  # or 1024 for large model
  batch_size: 4
  num_workers: 4

# Training Configuration
trainer:
  max_epochs: 100
  learning_rate: 1e-4
  warmup_steps: 1000
```

### Example: Training on Cityscapes
```bash
cd eomt
python main.py fit --config configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --trainer.max_epochs 100 \
  --trainer.devices [0]
```

### Monitoring Training
- **Weights & Biases Integration**: Automatic logging of metrics, images, and model artifacts
- **TensorBoard**: `tensorboard --logdir=lightning_logs/`
- **Learning Rate Schedule**: Two-stage warmup + polynomial decay

---

## Evaluation

### Cityscapes Evaluation
```bash
cd eval
python eval_cityscapes_color.py \
  --checkpoint ../path/to/model.ckpt \
  --image_dir ../data/cityscapes/leftImg8bit/val
```

### Anomaly Segmentation Metrics
```bash
python evalAnomaly.py \
  --checkpoint ../path/to/model.ckpt \
  --anomaly_dataset ../data/anomaly_dataset/
```

### Supported Metrics
- **Semantic Segmentation**: mIoU, mAcc, aAcc
- **Anomaly Detection**: AUROC, AUPR, TNR at various FPR thresholds

---

## Results

Results are logged to `results/results.csv`. The project achieves strong performance on:
- **Cityscapes**: Semantic segmentation (trained ERFNet baseline)
- **Anomaly Detection**: Out-of-distribution detection on diverse road scenes

### Trained Models

| Model | Backbone | Image Size | mIoU | Anomaly AUROC |
|-------|----------|-----------|------|---------------|
| EoMT-Base | DinoV2-Base | 640×480 | TBD | TBD |
| EoMT-Large | DinoV2-Large | 1024×768 | TBD | TBD |

---

## Project Structure

```
MaskArchitectureAnomaly/
├── eomt/                           # Main training codebase
│   ├── main.py                     # Entry point for training
│   ├── inference.ipynb             # Inference notebook
│   ├── models/
│   │   ├── eomt.py                 # Core EoMT architecture
│   │   ├── eomt_ext.py             # Extended version with Anomaly Head
│   │   ├── vit.py                  # Vision Transformer utilities
│   │   └── scale_block.py          # Feature scaling components
│   ├── training/
│   │   ├── lightning_module.py     # PyTorch Lightning module
│   │   ├── mask_classification_*.py # Different task implementations
│   │   └── two_stage_warmup_poly_schedule.py  # Learning rate scheduler
│   ├── dsets/
│   │   ├── dataset.py              # Base dataset class
│   │   ├── cityscapes_semantic.py  # Cityscapes loader
│   │   ├── generic_anomaly.py      # Generic anomaly dataset
│   │   └── lightning_data_module.py # Data loading pipeline
│   └── configs/
│       └── dinov2/                 # Configuration templates
│
├── eval/                           # Evaluation utilities
│   ├── evalAnomaly.py              # Anomaly detection evaluation
│   ├── eval_cityscapes_color.py    # Cityscapes evaluation
│   ├── eval_iou.py                 # IoU computation
│   ├── erfnet.py                   # ERFNet implementation
│   └── iouEval.py                  # IoU evaluation class
│
├── trained_models/                 # Pre-trained weights
│   └── erfnet_pretrained.pth
│
├── AML_Project_EoMT_eval.ipynb     # Full evaluation notebook
├── ARCHITECTURE_EXPLAINED.md       # Detailed architecture documentation
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Dependencies

Key libraries used in this project:

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.7.0 | Deep learning framework |
| Lightning | 2.5.1 | Training orchestration |
| Transformers | 4.56.1 | Pre-trained models (DinoV2) |
| Timm | 1.0.15 | Vision model zoo |
| Torchvision | 0.22.0 | Computer vision utilities |
| Scikit-learn | Latest | Metrics & preprocessing |
| Weights & Biases | 0.19.10 | Experiment tracking |

See `requirements.txt` for the complete list.

---

## Key Innovations

1. **Learned Anomaly Head**: Replaces heuristic-based OOD detection with an MLP that explicitly learns the mapping from features to anomaly labels

2. **Frozen Backbone with Uncertainty**: Preserves DinoV2's robust features while using entropy estimates from the Class Head as signals

3. **Mask-based Reasoning**: Combines per-query anomaly probabilities with mask predictions for spatial reasoning

4. **Scalable Architecture**: Supports multiple DinoV2 variants and image resolutions via simple config changes


## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

Portions of this code are adapted from:
- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) (Apache 2.0)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) (Apache 2.0)
- [DINOv2](https://github.com/facebookresearch/dinov2) (MIT)

