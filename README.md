# Mask Architecture Anomaly Detection

A comprehensive framework for detecting and segmenting anomalies in road scene images using deep learning architectures. This project implements state-of-the-art anomaly detection methods combining ERFNet and EoMT (Encoder-only Mask Transformer) models.

## Overview

This repository provides a complete pipeline for:
- **Training & Evaluation**: Tools for training semantic segmentation models on road scene datasets
- **Anomaly Detection**: Identifying out-of-distribution objects in driving scenarios
- **Multiple Architectures**: Support for both efficient (ERFNet) and transformer-based (EoMT) approaches
- **Dataset Support**: Cityscapes, ADE20K, COCO, and generic anomaly datasets

## Features

- Real-time anomaly segmentation for autonomous driving scenarios
- Efficient architecture (ERFNet) with pretrained weights
- Vision Transformer-based approach (EoMT) with DINOv2 backbone
- Comprehensive evaluation metrics and visualization tools
- Modular design for easy customization and extension
- Jupyter notebooks for inference and project evaluation

## Project Structure

```
├── eomt/                    # Encoder-only Mask Transformer implementation
│   ├── configs/            # Configuration files for different models and datasets
│   ├── dsets/              # Dataset implementations (Cityscapes, ADE20K, COCO)
│   ├── models/             # Model architectures
│   ├── training/           # Training modules and loss functions
│   ├── inference.ipynb     # Inference example notebook
│   └── main.py             # Training script
│
├── eval/                    # Evaluation tools and utilities
│   ├── eval_iou.py        # IoU evaluation
│   ├── evalAnomaly.py     # Anomaly detection evaluation
│   ├── erfnet.py          # ERFNet model implementation
│   └── configs/           # Evaluation configuration
│
├── trained_models/         # Pretrained model checkpoints
│   └── erfnet_pretrained.pth
│
└── AML_Project_EoMT_eval.ipynb  # Project evaluation notebook
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd MaskArchitectureAnomaly

# Install dependencies
pip install -r requirements.txt

# For Colab environments
pip install -r requirements_colab.txt
```

### Training

See [eomt/README.md](eomt/README.md) for detailed training instructions with different models and datasets.

```bash
cd eomt
python main.py --config configs/dinov2/common/eomt_base_640_ext.yaml
```

### Evaluation

Evaluate model performance and generate anomaly segmentation results:

```bash
cd eval
python eval_iou.py                      # Evaluate IoU metrics
python evalAnomaly.py                   # Evaluate anomaly detection
python eval_forwardTime.py             # Benchmark inference time
```

### Inference

Use the provided Jupyter notebooks for interactive inference:

```bash
jupyter notebook eomt/inference.ipynb
# or
jupyter notebook AML_Project_EoMT_eval.ipynb
```

## Key Components

### ERFNet (Efficient ResNet for Semantic Segmentation)
- Lightweight encoder-decoder architecture
- Real-time performance on GPU
- Pretrained weights included in `trained_models/`
- Suitable for deployment scenarios

### EoMT (Encoder-only Mask Transformer)
- Modern Vision Transformer-based approach
- Uses DINOv2 as backbone for superior features
- Supports multiple resolutions (640x640, 1024x1024)
- Enhanced anomaly detection capabilities

### Supported Datasets

| Dataset | Type | Status |
|---------|------|--------|
| Cityscapes | Semantic/Panoptic | Supported |
| ADE20K | Semantic/Panoptic |  Supported |
| COCO | Instance/Panoptic | Supported |
| Generic Anomaly | Custom | Supported |

## Configuration

Models and training parameters are configured via YAML files in `eomt/configs/`:

- `dinov2/cityscapes/` - Cityscapes semantic segmentation
- `dinov2/common/` - General-purpose configurations
- `eval/configs/eval_config.toml` - Evaluation parameters

## Results

Model performance metrics are logged in `results/results.csv` and can be visualized using the provided notebooks.

## Documentation

For detailed information about specific components, see:
- [ARCHITECTURE_EXPLAINED.md](ARCHITECTURE_EXPLAINED.md) - Detailed architecture documentation
- [eomt/README.md](eomt/README.md) - EoMT specific instructions
- [eval/README.md](eval/README.md) - Evaluation tools documentation

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU acceleration)
- See `requirements.txt` for complete dependencies




## Acknowledgments

- ERFNet architecture and pretrained weights
- EoMT original implementation
- DINOv2 pretrained models by Meta Research
- Cityscapes, ADE20K, and COCO dataset creators

## References

- **ERFNet**: Romera et al., 2017 - "ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation"
- **EoMT**: Encoder-only Mask Transformer approach for panoptic segmentation
- **DINOv2**: Oquab et al., 2023 - "DINOv2: Learning Robust Visual Features without Supervision"

