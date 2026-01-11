# Cityscapes–COCO Synthetic Anomaly Dataset

This project implements a reproducible pipeline for generating a synthetic
anomaly segmentation dataset by inserting object instances from the COCO dataset
into Cityscapes urban driving scenes.

The generated dataset supports both pixel-level anomaly segmentation and
instance-level (mask-based) anomaly modeling

---

## Motivation

Anomaly detection for autonomous driving requires datasets that contain
unexpected, out-of-distribution objects appearing in otherwise normal driving
scenarios, together with accurate ground truth annotations.

Since collecting real anomalous events is difficult and poorly scalable, this
project focuses on the controlled generation of synthetic anomalies by combining
realistic background scenes from Cityscapes with object instances extracted from
the COCO dataset.

The goal is not to achieve perfect visual realism, but to enable systematic and
reproducible experimentation for anomaly segmentation models.

---

## Dataset Definition

- In-distribution (ID): Cityscapes urban driving scenes.
- Out-of-distribution (OOD): Object instances taken from the COCO dataset.
- Anomalies: Objects that are semantically incompatible with the driving domain
  when placed into a road scene (e.g. animals, furniture, indoor objects).

The dataset explicitly distinguishes between:
- synthetic images, i.e. images selected for anomaly insertion, and
- anomalous images, i.e. synthetic images in which at least one anomaly was
  successfully placed.

This distinction is preserved in the exported metadata and manifest files.

---

## Repository Organization

The repository is organized into a configuration layer and a generation pipeline.

The `configs` directory contains YAML configuration files defining dataset paths,
split behavior, and synthesis parameters.

The `synthgen` directory contains the dataset generation pipeline. The main entry
point is `build_dataset.py`, which orchestrates all steps of the generation
process.

Within the generation pipeline:
- `config.py` handles configuration parsing and validation.
- `cityscapes_index.py` indexes Cityscapes images and semantic labels.
- `coco_index.py` builds a pool of valid COCO object instances.
- `sampler.py` defines per-image sampling decisions.
- `geometry.py` applies scaling and geometric transformations.
- `quality.py` enforces semantic and spatial placement constraints.
- `blending.py` handles alpha compositing and edge smoothing.
- `targets.py` generates pixel-level and instance-level ground truth.
- `export.py` writes images, annotations, and the manifest.
- `types.py` defines shared dataclasses used across modules.

Utility functions are located under `synthgen/utils` and include COCO mask
decoding, Cityscapes label helpers, and filesystem utilities.

---

## Generation Pipeline

For each Cityscapes image, the following steps are executed.

First, a deterministic sampling decision is made. This decision determines
whether the image remains clean or becomes synthetic, how many anomaly instances
are attempted, which COCO instances are selected, and which semantic regions are
targeted for placement. Randomness is seeded by the Cityscapes image identifier to
ensure reproducibility across runs.

Second, each selected COCO instance undergoes geometric transformation. A
perspective-based scaling heuristic is applied so that objects lower in the image
appear larger. Optional horizontal flipping may be applied. Instance masks are
resized using nearest-neighbor interpolation to preserve solid interiors, while
RGB content is resized using linear interpolation.

Third, candidate placement locations are filtered using Cityscapes semantic
labels, target semantic classes (e.g. road or sidewalk), and size, coverage, and
boundary constraints. If no valid placement is found, the instance placement is
skipped.

Fourth, objects are composited into the Cityscapes image using alpha blending.
The alpha mask represents the object silhouette. Edge feathering is applied to
reduce sharp boundaries, while object interiors are kept fully opaque to avoid
translucent artifacts.

Fifth, ground truth annotations are generated. Pixel-level ground truth consists
of a single map where normal pixels are labeled as 0, anomalous pixels as 1, and
ignored pixels as 255. Instance-level ground truth consists of a list of binary
anomaly masks and a corresponding list of labels using a single anomaly class.

Finally, all outputs are written to disk and a manifest entry is appended for the
processed image.

---

## Exported Data and Manifest

For each sample, the pipeline exports an RGB image, a pixel-level anomaly ground
truth map, a set of per-instance anomaly masks, and a label file for the instance
masks. Outputs are organized by dataset split.

A JSON Lines manifest file is generated alongside the data. Each entry records
the dataset split, Cityscapes image identifier, whether synthesis was attempted,
whether anomalies were successfully placed, the number of planned and placed
instances, the COCO annotation identifiers used, and the paths to all generated
outputs.

The manifest enables traceability, debugging, and post-hoc dataset analysis.

---

### Configuration File (YAML)

Dataset generation is fully controlled through a YAML configuration file (e.g. `synth_dataset_v1.yaml`).

The configuration specifies:
- dataset metadata (name, global seed, output directory),
- paths to raw Cityscapes and COCO datasets and optional download settings,
- dataset split behavior (clean vs synthetic ratios for train, validation, and test),
- synthesis parameters, including:
  - probability of inserting anomalies,
  - number of instances per image and retry limits,
  - allowed COCO categories and instance-level quality filters,
  - target Cityscapes semantic classes and forbidden overlaps,
  - geometric scaling constraints,
  - blending and photometric matching options,
- export options and ground-truth conventions.

The YAML file serves as the single source of truth for dataset generation and ensures that experiments are fully reproducible by configuration alone.

---

### Manifest File (JSONL)

During dataset generation, a JSON Lines manifest file (`manifest.jsonl`) is created.
Each line corresponds to one processed Cityscapes image and provides a complete
record of how that sample was generated.

For each image, the manifest stores:
- dataset split and Cityscapes image identifier,
- whether the image was selected for synthesis,
- whether at least one anomaly was successfully placed,
- the number of planned and placed anomaly instances,
- the random seed used for the sample,
- the semantic target labels used for placement,
- the COCO annotation IDs of inserted objects,
- relative paths to all exported outputs (image, pixel-level ground truth, instance masks, labels).

The manifest enables traceability, debugging, dataset statistics analysis, and exact reconstruction of the generation process for any individual sample.

---
  
## Dataset Splits

Dataset splits are defined in the configuration file using clean and synthetic
ratios.

Training data consists of a mixture of clean and synthetic images. Validation
data consists exclusively of clean images and is intended for calibration and
threshold selection. Test data consists exclusively of synthetic images and is
intended for stress-testing anomaly detection methods.

A synthetic image may still contain no anomalies if placement fails due to
semantic or geometric constraints.

---

## Reproducibility

All randomness in the pipeline is derived from a global seed and the Cityscapes
image identifier. Re-running the pipeline with the same configuration produces
identical results. The pool of COCO instances is constructed once and reused
across all dataset splits.

---

## Limitations

This dataset generation approach is not perfect and has several known
limitations.

The quality of generated anomalies depends strongly on the quality of COCO
instance segmentation masks. Some COCO annotations are imprecise, incomplete, or
poorly segmented, which may lead to unrealistic object boundaries, partial
objects, or visual artifacts.

Because the pipeline operates entirely in image space, it does not model
physical interactions, scene geometry, lighting consistency, shadows, or
occlusions. As a result, some composites may lack full photorealism.

Placement heuristics may fail for certain scenes or object sizes, resulting in
fewer anomalies than planned.

These limitations should be considered when using the dataset for training or
evaluation.

---

## Intended Use

This dataset is intended for research on anomaly detection and segmentation,
benchmarking pixel-level and mask-based methods, and controlled experimentation
in autonomous driving scenarios.

It is not intended as a photorealistic simulator or a replacement for
physically based rendering environments.
