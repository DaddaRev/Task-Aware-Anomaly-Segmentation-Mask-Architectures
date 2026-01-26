import numpy as np
from typing import Any
from pathlib import Path
from pycocotools import mask as mask_utils

from cityscapes_coco_anomaly.synthgen.schemas import CocoPaths


def resolve_coco_paths(cfg_coco: dict[str, Any]) -> CocoPaths:
    root = Path(cfg_coco["root"]).expanduser().resolve()
    images = cfg_coco["images"]
    ann = cfg_coco["annotations"]

    paths = CocoPaths(
        root=root,
        images_train=root / images["train"],
        instances_train_json=root / ann["instances_train"])

    return paths


def decode_coco_segmentation_to_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    """
    Decode COCO segmentation (polygons or RLE) to get HxW bool mask
    """

    # polygons (list of lists)
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        m = mask_utils.decode(rle)
        return m.astype(np.uint8) > 0

    # RLE dict
    if isinstance(segmentation, dict):
        # COCO can store RLE in two forms:
        # 1) compressed RLE: {"size": [h,w], "counts": <bytes or str>}
        # 2) uncompressed RLE: {"size": [h,w], "counts": <list[int]>}
        counts = segmentation.get("counts", None)

        # Uncompressed RLE: counts is a list. must be converted via frPyObjects
        if isinstance(counts, list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
            m = mask_utils.decode(rle)
            return m.astype(np.uint8) > 0

        # pycocotools expects bytes
        if isinstance(counts, str):
            seg = dict(segmentation)
            seg["counts"] = counts.encode("ascii")
            m = mask_utils.decode(seg)
            return m.astype(np.uint8) > 0

        m = mask_utils.decode(segmentation)
        return m.astype(np.uint8) > 0

    raise TypeError(f"Unsupported COCO segmentation type: {type(segmentation)}")
