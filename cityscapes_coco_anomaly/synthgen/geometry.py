import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class GeometryConfig:
    scale_min: float
    scale_max: float
    final_area_ratio_min: float
    final_area_ratio_max: float
    random_horizontal_flip: bool


@dataclass(frozen=True)
class PatchGeom:
    """
    Geometry outcome for one pasted instance.
    """
    scale: float
    hflip: bool
    out_h: int
    out_w: int


def parse_geometry_cfg(synthesis_cfg: Dict[str, Any]) -> GeometryConfig:
    geo = synthesis_cfg.get("geometry", {})

    scale_min = float(geo.get("scale_min", 1.0))
    scale_max = float(geo.get("scale_max", 1.0))

    if scale_min <= 0 or scale_max <= 0 or scale_max < scale_min:
        raise ValueError(f"Invalid scale range: scale_min={scale_min}, scale_max={scale_max}")

    final_area_ratio_min = float(geo.get("final_area_ratio_min", 0.0))
    final_area_ratio_max = float(geo.get("final_area_ratio_max", 1e9))

    if final_area_ratio_min < 0 or final_area_ratio_max < 0 or final_area_ratio_max < final_area_ratio_min:
        raise ValueError(f"Invalid final_area_ratio range: min={final_area_ratio_min}, max={final_area_ratio_max}")

    random_horizontal_flip = bool(geo.get("random_horizontal_flip", False))

    return GeometryConfig(
        scale_min=scale_min,
        scale_max=scale_max,
        final_area_ratio_min=final_area_ratio_min,
        final_area_ratio_max=final_area_ratio_max,
        random_horizontal_flip=random_horizontal_flip)


def scale_from_y(y_center: float, H: int, scale_min: float, scale_max: float) -> float:
    """
    Perspective heuristic: lower in image -> bigger scale.
    y_center: pixel coordinate (0..H-1)
    """
    if H <= 1:
        return float(scale_min)

    t = float(np.clip(y_center / float(H - 1), 0.0, 1.0))
    return float(scale_min + (scale_max - scale_min) * t)


def _resize_rgb_alpha(
        patch_rgb: np.ndarray,
        alpha: np.ndarray,
        out_w: int,
        out_h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    patch_rgb: h x w x 3 uint8
    alpha:     h x w float32 (0..1)
    returns resized (patch_rgb, alpha_float32 0..1)
    """

    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"Invalid resize shape: out_w={out_w}, out_h={out_h}")

    rgb = cv2.resize(patch_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    if alpha.dtype != np.float32:
        raise TypeError("alpha must be float32 in [0,1]")

    if alpha.min() < 0 or alpha.max() > 1.0:
        raise ValueError("alpha must be in [0,1]")

    a_res = cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    a_res = np.clip(a_res, 0.0, 1.0).astype(np.float32)
    return rgb, a_res


def _hflip_rgb_alpha(patch_rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cv2.flip(patch_rgb, 1), cv2.flip(alpha, 1)


def area_ratio_ok(out_w: int, out_h: int, city_H: int, city_W: int, min_r: float, max_r: float) -> bool:
    img_area = float(city_H * city_W)

    if img_area <= 0:
        return False
    r = float(out_w * out_h) / img_area

    return min_r <= r <= max_r


def compute_patch_geometry(
        rng: np.random.Generator,
        city_H: int,
        city_W: int,
        patch_h: int,
        patch_w: int,
        target_y_center: float,
        geo_cfg: GeometryConfig) -> PatchGeom:
    """
    Decide scale (from y), optional flip, and output size for a patch,
    with final area ratio constraints.

    We just compute scaled size and flip decision
    """
    scale = scale_from_y(target_y_center, city_H, geo_cfg.scale_min, geo_cfg.scale_max)

    out_w = int(round(patch_w * scale))
    out_h = int(round(patch_h * scale))

    # never allow 0-sized
    out_w = max(1, out_w)
    out_h = max(1, out_h)

    # Optional random flip
    hflip = bool(rng.random() < 0.5) if geo_cfg.random_horizontal_flip else False

    # Area ratio constraints: if out of bounds, try to adjust scale deterministically
    # Strategy:
    # - If too small, scale up to meet min area ratio.
    # - If too large, scale down to meet max area ratio.
    # Keep aspect ratio fixed
    if not area_ratio_ok(out_w, out_h, city_H, city_W, geo_cfg.final_area_ratio_min, geo_cfg.final_area_ratio_max):
        img_area = float(city_H * city_W)

        # desired area range in pixels
        min_area = geo_cfg.final_area_ratio_min * img_area
        max_area = geo_cfg.final_area_ratio_max * img_area

        cur_area = float(out_w * out_h)
        if cur_area <= 0:
            cur_area = 1.0

        # compute scale factor needed on area: area scales ~ scale^2
        if cur_area < min_area and min_area > 0:
            factor = np.sqrt(min_area / cur_area)
            scale = float(scale * factor)

        elif cur_area > max_area > 0:
            factor = np.sqrt(max_area / cur_area)
            scale = float(scale * factor)

        # clamp scale back into [scale_min, scale_max]:
        scale = float(np.clip(scale, 0.05, 10.0))

        out_w = max(1, int(round(patch_w * scale)))
        out_h = max(1, int(round(patch_h * scale)))

    return PatchGeom(
        scale=float(scale),
        hflip=hflip,
        out_h=out_h,
        out_w=out_w)


def apply_geometry_to_patch(
        patch_rgb: np.ndarray,
        alpha: np.ndarray,
        out_w: int,
        out_h: int,
        hflip: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply resize (out_w, out_h) and optional horizontal flip.
    Returns (patch_rgb_resized, alpha_resized_float32).
    """
    rgb_res, a_res = _resize_rgb_alpha(patch_rgb, alpha, out_w=out_w, out_h=out_h)
    if hflip:
        rgb_res, a_res = _hflip_rgb_alpha(rgb_res, a_res)
    return rgb_res, a_res
