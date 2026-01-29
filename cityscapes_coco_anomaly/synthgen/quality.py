import numpy as np
from typing import Any, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class QualityConfig:
    min_target_label_coverage: float
    forbidden_overlap_labels: tuple[str, ...]
    reject_if_touches_image_border: bool

    max_location_attempts: int


def parse_quality_cfg(synthesis_cfg: dict[str, Any]) -> QualityConfig:
    cs = synthesis_cfg.get("cityscapes", {})

    min_cov = cs.get("min_target_label_coverage", 0.0)
    if not (0.0 <= min_cov <= 1.0):
        raise ValueError("min_target_label_coverage must be in [0,1]")

    if (forbidden := cs.get("forbidden_overlap_labels", [])) is None:
        forbidden = []

    forbidden = tuple(str(x) for x in forbidden)

    q = synthesis_cfg.get("quality_filters", {})

    reject_border = bool(q.get("reject_if_touches_image_border", True))

    max_loc = synthesis_cfg.get("max_location_attempts", 50)
    if max_loc <= 0:
        raise ValueError("max_location_attempts must be > 0")

    return QualityConfig(
        min_target_label_coverage=min_cov,
        forbidden_overlap_labels=forbidden,
        reject_if_touches_image_border=reject_border,
        max_location_attempts=max_loc)


def touches_border(x: int, y: int, w: int, h: int, W: int, H: int) -> bool:
    return (x <= 0) or (y <= 0) or (x + w >= W - 1) or (y + h >= H - 1)


def target_label_coverage_ok(
        sem_trainids: np.ndarray,
        x: int, y: int, w: int, h: int,
        target_id: int,
        min_cov: float) -> bool:

    patch = sem_trainids[y:y + h, x:x + w]
    if patch.size == 0:
        return False

    cov = float(np.mean(patch == target_id))
    return cov >= min_cov


def forbidden_overlap_ok(sem_trainids: np.ndarray,
                         x: int, y: int, w: int, h: int,
                         forbidden_ids: set[int]) -> bool:

    if not forbidden_ids:
        return True
    patch = sem_trainids[y:y + h, x:x + w]

    if not patch.size:
        return False

    return not np.isin(patch, list(forbidden_ids)).any()


def _random_top_left(rng: np.random.Generator, W: int, H: int, w: int, h: int) -> tuple[int, int]:
    """
    Uniform sampling of top-left ensuring the patch stays in image bounds.
    """
    if w > W or h > H:
        return -1, -1
    x = int(rng.integers(0, W - w + 1))
    y = int(rng.integers(0, H - h + 1))
    return x, y


def sample_paste_location(
        rng: np.random.Generator,
        sem_trainids: np.ndarray,
        target_label_name: str,
        patch_w: int,
        patch_h: int,
        name_to_trainid: dict[str, int],
        cfg: QualityConfig) -> Optional[tuple[int, int]]:

    """
    Find a valid (x,y) for a patch given semantic constraints.

    Constraints:
    - patch must be within bounds
    - target_label coverage >= min_target_label_coverage
    - forbidden overlap = none
    - optionally reject a touching image border
    """

    H, W = sem_trainids.shape[:2]

    if (target_id := name_to_trainid.get(target_label_name, None)) is None:
        raise KeyError(f"Unknown target label name: {target_label_name}")

    forbidden_ids: set[int] = set()
    for name in cfg.forbidden_overlap_labels:
        if (tid := name_to_trainid.get(name, None)) is not None:
            forbidden_ids.add(tid)

    for _ in range(cfg.max_location_attempts):
        x, y = _random_top_left(rng, W, H, patch_w, patch_h)
        if x < 0:
            return None

        if cfg.reject_if_touches_image_border and touches_border(x, y, patch_w, patch_h, W, H):
            continue

        if cfg.min_target_label_coverage > 0:
            if not target_label_coverage_ok(sem_trainids, x, y, patch_w, patch_h, target_id,
                                            cfg.min_target_label_coverage):
                continue

        if forbidden_ids:
            if not forbidden_overlap_ok(sem_trainids, x, y, patch_w, patch_h, forbidden_ids):
                continue

        return x, y

    return None
