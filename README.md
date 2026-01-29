# Mask Architecture Anomaly Segmentation for Road Scenes [[Course Project](https://drive.google.com/file/d/19gQ2uhI8jPxWIdGewkaA2DqihUnQLVfB/view?usp=drive_link)]

This repository provides a code setup for the Real-Time Anomaly Segmentation project of the Machine Learning Course. It consists of the code base for training/testing ERFNet on the Cityscapes dataset and perform anomaly segmentation. It also contains some code referring to EoMT.

---

## Branch: Anomaly Head with Queries Only

**This branch implements an extended version of EoMT (EoMT-EXT) specifically designed for anomaly segmentation using a dedicated Anomaly Head.**

### Key Innovation

This extension introduces a **learnable Anomaly Head** that operates directly on query embeddings to classify each query into three meta-states:
- **Normal** (In-Distribution object)
- **Anomaly** (Out-of-Distribution object)
- **No-Object** (Background)

The core hypothesis is that **query embeddings alone** contain sufficient geometric and semantic information to distinguish between ID and OOD samples, without requiring auxiliary uncertainty metrics (e.g., entropy, max-softmax probability).

**For complete architectural details, theoretical foundations, and training procedures, see:** [Anomaly_head_with_queries_README.md](Anomaly_head_with_queries_README.md)

### Validation & Inference

The notebook [eomt/inference.ipynb](eomt/inference.ipynb) provides:
- **Model loading** with the extended EoMT-EXT architecture
- **Inference pipeline** that utilizes the anomaly head predictions
- **Validation on anomaly detection datasets** (FS L&F, FS Static, LostAndFound, RoadAnomaly, RoadObstacle21)
- **Metric computation**: AuPRC and FPR95 for anomaly detection performance

---

## Packages
For more instructions, please refer to the README in each folder:

* [eval](eval) contains tools for evaluating/visualizing the an ERFNet model's output and performing anomaly segmentation.
* [trained_models](trained_models) Contains the ERFNet trained models for the baseline eval. 
* [eomt](eomt) It is almost the original folder of the EoMT project. Inside it you will find code to train and pretrained checkpoints for EoMT.

