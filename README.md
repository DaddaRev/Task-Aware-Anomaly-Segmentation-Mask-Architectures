# Mask Architecture Anomaly Segmentation for Road Scenes [[Course Project](https://drive.google.com/file/d/19gQ2uhI8jPxWIdGewkaA2DqihUnQLVfB/view?usp=drive_link)]

This repository provides a starter-code setup for the Real-Time Anomaly Segmentation project of the Machine Learning Course. It consists of the code base for training/testing ERFNet on the Cityscapes dataset and perform anomaly segmentation. It also contains some code referring to EoMT.

---

## Synthetic Dataset Generation

This repository includes code for **synthetic anomaly dataset generation**. The generation pipeline and related scripts are available at:

**Link:** https://github.com/Giacomo-FMJ/MaskArchitectureAnomaly/tree/main/cityscapes_coco_anomaly

---

## Available Branches

This repository contains multiple branches, each implementing different anomaly detection approaches:

### Main Extensions (Proposed Architectures)

1. **`main`** - Current branch containing the base project structure and ERFNet baseline
2. **`fine-tuning_anomaly_head_revised`** - First proposed extension architecture implementing a **Statistical Uncertainty-based Pixel-wise Anomaly Head** (stats-only approach)
3. **`fine-tuning_anomaly_head_revised_features`** - Second proposed extension architecture implementing a **Hybrid Pixel-wise Anomaly Head** (statistical uncertainty + visual features from backbone)

### Additional Experimental Branches (Less Relevant)

4. **`fine-tuning_anomaly_scores`** - Alternative baseline approach with anomaly scoring (not extensively covered in the report)
5. **`fine-tuning_only_queries`** - Query-based anomaly detection baseline (not extensively covered in the report)

**Note:** The core contributions and detailed documentation are available in branches 2 and 3.

---

## Evaluation & Inference

- **[eomt/inference -COLAB.ipynb](eomt/inference.ipynb)** - Notebook for evaluating the baseline EoMT model on anomaly detection datasets. Includes inference pipeline with traditional uncertainty-based methods (MSP, MaxLogit, Entropy) and visualization tools.

---

## Packages

For instructions, please refer to the README in each folder:

* [eval](eval) contains tools for evaluating/visualizing the an ERFNet model's output and performing anomaly segmentation.
* [trained_models](trained_models) Contains the ERFNet trained models for the baseline eval. 
* [eomt](eomt) It is almost the original folder of the EoMT project. Inside it you will find code to train and pretrained checkpoints for EoMT.

