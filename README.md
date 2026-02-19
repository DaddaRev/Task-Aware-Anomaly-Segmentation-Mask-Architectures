# Mask Architecture Anomaly Segmentation for Road Scenes

This repository provides a starter-code setup for the Real-Time Anomaly Segmentation project of Advanced Machine Learning Course. It consists of the code base for training/testing ERFNet on the Cityscapes dataset and perform anomaly segmentation. It also contains some code referring to EoMT.

Starting from the EoMT codebase, our work focuses on proposing and evaluating extension architectures for anomaly segmentation that reduce the reliance on state-of-the-art post-hoc scoring methods.

In particular, the project investigates trainable pixel-wise anomaly heads built on top of semantic prediction outputs (and, in the hybrid variant, visual backbone features) to directly learn ID/OOD separation.

---

## Quick Navigation

- [Repository Intent](#repository-intent)
- [Synthetic Dataset Generation](#synthetic-dataset-generation)
- [Available Branches](#available-branches)
- [Evaluation & Inference](#evaluation--inference)
- [Project Report](#project-report)
- [Packages](#packages)
- [Authors](#authors)

---

## Repository Intent

This repository is organized to support a full development path from baseline anomaly segmentation to proposed architectural extensions:

- Reuse a consolidated EoMT-based codebase for training and evaluation workflows.
- Benchmark and compare post-hoc uncertainty scoring baselines.
- Propose extension architectures that move anomaly detection from post-hoc scoring toward learnable, end-to-end pixel-wise anomaly prediction.

---

## Synthetic Dataset Generation

This repository includes code for **synthetic anomaly dataset generation**. The generation pipeline and related scripts are available at:

**Link:** [https://github.com/Giacomo-FMJ/MaskArchitectureAnomaly/tree/main/cityscapes_coco_anomaly](https://github.com/Giacomo-FMJ/MaskArchitectureAnomaly/tree/main/cityscapes_coco_anomaly)

---

## Available Branches

This repository contains multiple branches, each implementing different anomaly detection approaches:

### Main Extensions (Proposed Architectures)

1. **[`main`](https://github.com/Giacomo-FMJ/MaskArchitectureAnomaly/tree/main)** - Current branch containing the base project structure and ERFNet baseline
2. **[`fine-tuning_anomaly_head_revised`](https://github.com/Giacomo-FMJ/MaskArchitectureAnomaly/tree/fine-tuning_anomaly_head_revised)** - First proposed extension architecture implementing a **Statistical Uncertainty-based Pixel-wise Anomaly Head** (stats-only approach)
3. **[`fine-tuning_anomaly_head_revised_features`](https://github.com/Giacomo-FMJ/MaskArchitectureAnomaly/tree/fine-tuning_anomaly_head_revised_features)** - Second proposed extension architecture implementing a **Hybrid Pixel-wise Anomaly Head** (statistical uncertainty + visual features from backbone)

### Additional Experimental Branches (Less Relevant)

4. **[`fine-tuning_anomaly_scores`](https://github.com/Giacomo-FMJ/MaskArchitectureAnomaly/tree/fine-tuning_anomaly_scores)** - Alternative baseline approach with anomaly scoring (not extensively covered in the report)
5. **[`fine-tuning_only_queries`](https://github.com/Giacomo-FMJ/MaskArchitectureAnomaly/tree/fine-tuning_only_queries)** - Query-based anomaly detection baseline (not extensively covered in the report)

> **Note:** The core contributions and detailed documentation are available in branches 2 and 3.

---

## Evaluation & Inference

- **[eomt/inference -COLAB.ipynb](https://github.com/DaddaRev/Task-Aware-Anomaly-Segmentation-Mask-Architectures/blob/main/eomt/inference%20-COLAB.ipynb)** - Notebook for evaluating the baseline EoMT model on anomaly detection datasets. Includes inference pipeline with traditional uncertainty-based methods (MSP, MaxLogit, Entropy) and visualization tools.

---

## Project Report

For a complete description of motivation, methodology, experiments, and results, see the full report:

- **[Project Report](Project%20Report.pdf)**

---

## Packages

For instructions, please refer to the README in each folder:

- [eval](eval) contains tools for evaluating/visualizing the an ERFNet model's output and performing anomaly segmentation.
- [trained_models](trained_models) Contains the ERFNet trained models for the baseline eval.
- [eomt](eomt) It is almost the original folder of the EoMT project. Inside it you will find code to train and pretrained checkpoints for EoMT.

---

## Authors

- [Davide Reverberi](https://github.com/DaddaRev)
- [Adriano Giuliani](https://github.com/AdryGiuliani)
- [Giacomo Lopez](https://github.com/Giacomo-FMJ)
