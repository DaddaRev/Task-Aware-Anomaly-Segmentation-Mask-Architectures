import numpy as np
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class CocoPaths:
    root: Path
    images_train: Path
    instances_train_json: Path


@dataclass(frozen=True)
class CocoInstance:
    image_id: int
    ann_id: int
    category_id: int
    category_name: str

    # COCO bbox format (x, y, w, h) in pixels
    bbox_xywh: tuple[float, float, float, float]

    segmentation: Any

    # Image shape
    image_height: int
    image_width: int

    # cached decoded mask (HxW bool)
    mask: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CocoImageInfo:
    file_name: str
    height: int
    width: int


@dataclass
class CocoIndex:
    """
    COCO index for fast sampling
    """
    images: dict[int, CocoImageInfo]  # image_id -> info
    cat_id_to_name: dict[int, str]  # category_id -> name
    cat_name_to_id: dict[str, int]  # name -> category_id
    pool: list[CocoInstance]  # filtered instances
    ann_id_to_instance: dict[int, CocoInstance]  # convenience lookup
