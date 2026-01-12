import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetsConfig:
    """
    Conventions:
    - gt_pixel uses values:
        0 = in-distribution / normal
        1 = anomaly
        255 = ignore

    - mask-based targets:
        masks: list of HxW bool
        labels: list of ints (we use 0 as the single anomaly class)
    """

    anomaly_pixel_value: int = 1
    normal_pixel_value: int = 0
    ignore_pixel_value: int = 255
    anomaly_label: int = 0  # single-class anomaly


def init_gt_pixel(H: int, W: int, normal_value: int = 0) -> np.ndarray:
    """
    Initialize a clean ground-truth pixel map.
    """
    gt = np.full((H, W), int(normal_value), dtype=np.uint8)
    return gt


def apply_anomaly_alpha_to_gt_pixel(
        gt_pixel: np.ndarray,
        alpha_full: np.ndarray,
        cfg: TargetsConfig,
        ignore_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Update gt_pixel using an alpha map (float32 [0,1]) in full image coordinates.

    Rule (hard):
      anomaly if alpha_full > 0.5

    ignore_mask (optional):
      HxW bool - where True, set ignore value (255)
    """
    if gt_pixel.dtype != np.uint8:
        raise TypeError(f"gt_pixel must be uint8, got {gt_pixel.dtype}")
    if alpha_full.dtype != np.float32:
        raise TypeError(f"alpha_full must be float32, got {alpha_full.dtype}")
    if gt_pixel.shape != alpha_full.shape:
        raise ValueError("gt_pixel and alpha_full shape mismatch")

    anom = alpha_full > 0.5
    gt_pixel[anom] = np.uint8(cfg.anomaly_pixel_value)

    if ignore_mask is not None:
        if ignore_mask.shape != gt_pixel.shape:
            raise ValueError("ignore_mask shape mismatch")
        gt_pixel[ignore_mask.astype(bool)] = np.uint8(cfg.ignore_pixel_value)

    return gt_pixel


def add_instance_target_from_alpha(
        masks: list[np.ndarray],
        labels: list[int],
        alpha_full: np.ndarray,
        cfg: TargetsConfig):
    """
    Append one instance target from an alpha map in full image coordinates.

    Rule:
      instance mask = alpha_full > 0.5
      label = cfg.anomaly_label
    """
    if alpha_full.dtype != np.float32:
        raise TypeError(f"alpha_full must be float32, got {alpha_full.dtype}")
    if alpha_full.ndim != 2:
        raise ValueError(f"alpha_full must be HxW, got {alpha_full.shape}")

    m = (alpha_full > 0.5)
    if np.count_nonzero(m) == 0:
        return

    masks.append(m)
    labels.append(int(cfg.anomaly_label))


def merge_alphas_to_gt_and_instances(
        H: int,
        W: int,
        alphas_full: list[np.ndarray],
        cfg: TargetsConfig,
        ignore_mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """
    Given multiple pasted instances' alpha_full maps, produce:
    - gt_pixel (union of anomalies)
    - masks list (one per instance)
    - labels list (0 per instance)
    """

    gt = init_gt_pixel(H, W, normal_value=cfg.normal_pixel_value)
    masks, labels = [], []

    for a in alphas_full:
        if a.shape != (H, W):
            raise ValueError("alpha_full has wrong shape")
        gt = apply_anomaly_alpha_to_gt_pixel(gt, a, cfg, ignore_mask=ignore_mask)
        add_instance_target_from_alpha(masks, labels, a, cfg)

    return gt, masks, labels
