import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PhotometricMatchConfig:
    enabled: bool
    space: str
    match_mean: bool
    match_std: bool
    eps: float


@dataclass(frozen=True)
class BlendingConfig:
    feather_px: int
    photometric: PhotometricMatchConfig
    alpha_composite: bool


def parse_blending_cfg(synthesis_cfg: dict[str, Any]) -> BlendingConfig:
    b = synthesis_cfg.get("blending", {})

    feather_px = int(b.get("feather_px", 0))
    if feather_px < 0:
        raise ValueError("blending.feather_px must be >= 0")

    pm = b.get("photometric_match", {})

    enabled = bool(pm.get("enabled", True))
    space = str(pm.get("space", "lab")).lower()

    if space not in {"lab", "rgb"}:
        raise ValueError("photometric_match.space must be 'lab' or 'rgb'")

    match_mean = bool(pm.get("match_mean", True))
    match_std = bool(pm.get("match_std", True))
    eps = float(pm.get("eps", 1e-6))

    if eps <= 0:
        raise ValueError("photometric_match.eps must be > 0")

    photometric = PhotometricMatchConfig(
        enabled=enabled,
        space=space,
        match_mean=match_mean,
        match_std=match_std,
        eps=eps)

    alpha_composite = bool(b.get("alpha_composite", True))

    return BlendingConfig(feather_px=feather_px, photometric=photometric, alpha_composite=alpha_composite)


def feather_alpha(alpha: np.ndarray, feather_px: int) -> np.ndarray:
    """
    Feather/smooth mask edges while keeping object interior solid.
    """
    if alpha.dtype != np.float32:
        raise TypeError("alpha must be float32 in [0,1]")
    if alpha.min() < 0 or alpha.max() > 1.0:
        raise ValueError("alpha must be in [0,1]")

    if feather_px <= 0:
        return alpha

    # Start from a binary mask (alpha comes from an instance mask)
    m = (alpha > 0.5).astype(np.float32)

    # Blur to create soft edges
    k = int(max(3, 2 * feather_px + 1))
    if k % 2 == 0:
        k += 1

    a_blur = cv2.GaussianBlur(m, (k, k), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)

    # Keep interior solid
    core_kernel = np.ones((k, k), dtype=np.uint8)
    core = cv2.erode((m > 0).astype(np.uint8), core_kernel, iterations=1)

    a_blur = np.clip(a_blur, 0.0, 1.0).astype(np.float32)
    a_blur[core > 0] = 1.0

    return a_blur


def _rgb_to_lab(img_rgb_u8: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab.astype(np.float32)


def _lab_to_rgb(lab_f32: np.ndarray) -> np.ndarray:
    lab_u8 = np.clip(lab_f32, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def photometric_match_patch(
        patch_rgb: np.ndarray,
        bg_rgb: np.ndarray,
        cfg: PhotometricMatchConfig) -> np.ndarray:
    """
    Match patch color statistics to background within the bbox region.
    """

    if not cfg.enabled:
        return patch_rgb

    if patch_rgb.dtype != np.uint8 or bg_rgb.dtype != np.uint8:
        raise ValueError("photometric_match_patch expects uint8 RGB inputs")

    if cfg.space == "lab":
        patch = _rgb_to_lab(patch_rgb)
        bg = _rgb_to_lab(bg_rgb)
    else:  # "rgb"
        patch = patch_rgb.astype(np.float32)
        bg = bg_rgb.astype(np.float32)

    out = patch.copy()
    eps = float(cfg.eps)

    for c in range(3):
        p = patch[..., c]
        b = bg[..., c]

        p_mean = float(p.mean())
        b_mean = float(b.mean())

        if cfg.match_std:
            p_std = float(p.std())
            b_std = float(b.std())
            scale = b_std / (p_std + eps)
        else:
            scale = 1.0

        if cfg.match_mean:
            shift = b_mean - p_mean * scale
        else:
            shift = 0.0

        out[..., c] = p * scale + shift

    # Convert back
    if cfg.space == "lab":
        return _lab_to_rgb(out)
    else:
        return np.clip(out, 0, 255).astype(np.uint8)


def alpha_composite(bg_rgb: np.ndarray, patch_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    bg_rgb, patch_rgb: HxWx3 uint8
    alpha: HxW float32 in [0,1]
    """
    if bg_rgb.dtype != np.uint8 or patch_rgb.dtype != np.uint8:
        raise ValueError("alpha_composite expects uint8 images.")

    if alpha.dtype != np.float32:
        raise TypeError("alpha must be float32 in [0,1]")

    if alpha.min() < 0 or alpha.max() > 1.0:
        raise ValueError("alpha must be in [0,1]")

    if alpha.ndim != 2:
        raise ValueError("alpha must be HxW")

    if bg_rgb.shape[:2] != alpha.shape or patch_rgb.shape[:2] != alpha.shape:
        raise ValueError("Shape mismatch for alpha compositing")

    a = alpha[..., None]  # HxWx1
    bg = bg_rgb.astype(np.float32)
    fg = patch_rgb.astype(np.float32)

    out = a * fg + (1.0 - a) * bg
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_patch_into_image(
        city_rgb: np.ndarray,
        patch_rgb: np.ndarray,
        alpha: np.ndarray,
        x: int,
        y: int,
        cfg: BlendingConfig) -> tuple[np.ndarray, np.ndarray]:

    """
    Blends a patch into a Cityscapes image at (x,y) top-left.

    Returns:
      - new_city_rgb (HxWx3 uint8)
      - alpha_full (HxW float32 in [0,1]) placed in full image coordinates
    """

    if city_rgb.dtype != np.uint8 or patch_rgb.dtype != np.uint8:
        raise ValueError("blend_patch_into_image expects uint8 RGB images")

    H, W = city_rgb.shape[:2]
    ph, pw = patch_rgb.shape[:2]

    if x < 0 or y < 0 or x + pw > W or y + ph > H:
        raise ValueError("Patch placement out of bounds")

    a = feather_alpha(alpha, cfg.feather_px)

    bg_region = city_rgb[y:y + ph, x:x + pw]
    patch_matched = photometric_match_patch(patch_rgb, bg_region, cfg.photometric)

    # composite
    if cfg.alpha_composite:
        blended_region = alpha_composite(bg_region, patch_matched, a)
    else:
        blended_region = patch_matched

    out = city_rgb.copy()
    out[y:y + ph, x:x + pw] = blended_region

    alpha_full = np.zeros((H, W), dtype=np.float32)
    alpha_full[y:y + ph, x:x + pw] = a

    return out, alpha_full
