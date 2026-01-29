import cv2
import json
import yaml
import hashlib
import numpy as np
from typing import Any
from pathlib import Path


def ensure_exists(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")


def read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_image_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def read_png_uint8(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr.ndim == 3:
        return arr[:, :, 0]

    return arr.astype(np.uint8)


def write_image_rgb(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
        raise IOError(f"Failed to write image to {path}")


def write_png_uint8(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(path), img):
        raise IOError(f"Failed to write uint8 png to {path}")


def write_npy(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), obj, allow_pickle=False)


def split_output_dirs(output_root: Path, split: str) -> dict[str, Path]:
    """
    Returns dict of output directories for a given split.
    """
    base = output_root / split
    return {
        "base": base,
        "images": base / "images",
        "gt_pixel": base / "gt_pixel",
        "masks": base / "masks",
        "labels": base / "labels",
    }


def ensure_split_output_dirs(output_root: Path, split: str) -> dict[str, Path]:
    d = split_output_dirs(output_root, split)

    k: str
    p: Path
    for k, p in d.items():
        if k != "base":
           p.mkdir(parents=True, exist_ok=True)

    return d


def derive_sample_seed(sample_id: str, base_seed: int) -> int:
    """
    Deterministic per-sample seed from (string, base_seed).
    """

    h = hashlib.sha256((str(base_seed) + "::" + sample_id).encode("utf-8")).hexdigest()
    return int(h[:8], 16)
