import cv2
import json
import numpy as np
from typing import Any
from pathlib import Path

from cityscapes_coco_anomaly.synthgen.utils import ensure_split_output_dirs


def write_rgb_png(path: Path, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB image must be uint8 HxWx3")

    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f"Failed to write image: {path}")


def write_gt_pixel_png(path: Path, gt_pixel: np.ndarray):
    if gt_pixel.dtype != np.uint8 or gt_pixel.ndim != 2:
        raise ValueError("gt_pixel must be uint8 HxW")

    if not cv2.imwrite(str(path), gt_pixel):
        raise IOError(f"Failed to write gt_pixel: {path}")


def write_masks_npy(mask_paths: list[Path], masks: list[np.ndarray]):
    for p, m in zip(mask_paths, masks):
        if m.dtype != np.bool_ or m.ndim != 2:
            raise ValueError("each mask must be bool HxW")
        np.save(str(p), m, allow_pickle=False)


def write_labels_npy(path: Path, labels: list[int]):
    arr = np.asarray(labels, dtype=np.int64)
    np.save(str(path), arr, allow_pickle=False)


def export_sample(
        output_root: Path,
        split: str,
        sample_id: str,
        rgb: np.ndarray,
        gt_pixel: np.ndarray,
        masks: list[np.ndarray],
        labels: list[int],
        export_cfg: dict[str, Any]) -> dict[str, object]:
    """
    export:
      image_format: png
      gt_pixel: {enabled: bool, values: {normal, anomaly, ignore}}
      masks: {enabled: bool, format: npy}
      labels: {enabled: bool, format: npy}
      manifest: {enabled: bool, filename: manifest.jsonl}

    Returns:
      dict with relative paths (relative to output_root) for produced artifacts
    """
    if len(masks) != len(labels):
        raise ValueError("masks and labels length mismatch")

    image_format = str(export_cfg.get("image_format", "png")).lower()
    if image_format != "png":
        raise ValueError(f"Only image_format='png' is supported for now (got {image_format}).")

    gt_cfg = export_cfg.get("gt_pixel", {})
    masks_cfg = export_cfg.get("masks", {})
    labels_cfg = export_cfg.get("labels", {})

    gt_enabled = gt_cfg.get("enabled", True)
    masks_enabled = masks_cfg.get("enabled", True)
    labels_enabled = labels_cfg.get("enabled", True)

    dirs = ensure_split_output_dirs(output_root, split)

    out: dict[str, object] = {}
    img_path = dirs["images"] / f"{sample_id}.png"
    write_rgb_png(img_path, rgb)
    out["image"] = str(img_path.relative_to(output_root))

    # gt_pixel 
    if gt_enabled:
        gt_path = dirs["gt_pixel"] / f"{sample_id}.png"
        write_gt_pixel_png(gt_path, gt_pixel)
        out["gt_pixel"] = str(gt_path.relative_to(output_root))

    #  masks 
    if masks_enabled:
        mask_paths: list[Path] = []
        mask_relpaths: list[str] = []
        for i in range(len(masks)):
            mp = dirs["masks"] / f"{sample_id}_{i:02d}.npy"
            mask_paths.append(mp)
            mask_relpaths.append(str(mp.relative_to(output_root)))
        write_masks_npy(mask_paths, masks)
        out["masks"] = mask_relpaths

    #  labels 
    if labels_enabled:
        labels_path = dirs["labels"] / f"{sample_id}.npy"
        write_labels_npy(labels_path, labels)
        out["labels"] = str(labels_path.relative_to(output_root))

    return out


def append_manifest_line(
        output_root: Path,
        record: dict[str, Any],
        export_cfg: dict[str, Any]):
    """
    Append one JSON record to manifest if enabled
    """

    man = export_cfg.get("manifest", {})
    enabled = man.get("enabled", True)
    if not enabled:
        return

    filename = str(man.get("filename", "manifest.jsonl"))
    path = output_root / filename
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
