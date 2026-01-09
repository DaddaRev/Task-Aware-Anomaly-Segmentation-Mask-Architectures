import cv2
import numpy as np
from typing import Any
from pathlib import Path

from .utils.io import read_json, read_image_rgb
from .utils.coco import decode_coco_segmentation_to_mask
from .schemas.coco import CocoInstance, CocoIndex, CocoImageInfo


def _bbox_aspect_ratio(bbox_xywh: tuple[float, float, float, float]) -> float:
    _, _, w, h = bbox_xywh
    return float("inf") if h <= 0 else w / h


def _bbox_area(bbox_xywh: tuple[float, float, float, float]) -> float:
    _, _, w, h = bbox_xywh
    return w * h


def _is_fully_visible(bbox_xywh: tuple[float, float, float, float], W: int, H: int) -> bool:
    x, y, w, h = bbox_xywh
    if w <= 0 or h <= 0:
        return False
    return (x >= 0) and (y >= 0) and (x + w <= W) and (y + h <= H)


def _mask_solidity(mask_bool: np.ndarray) -> float:
    """
    solidity = area(mask) / area(convex_hull(mask))
    """

    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)

    if (area := float(cv2.contourArea(cnt))) <= 0:
        return 0.0

    hull = cv2.convexHull(cnt)

    if (hull_area := float(cv2.contourArea(hull))) <= 0:
        return 0.0

    return area / hull_area


def build_coco_index(
        coco_instances_json: Path,
        allowed_categories: list[str],
        instance_filters: dict[str, Any]) -> CocoIndex:
    coco = read_json(coco_instances_json)

    # categories
    categories = coco.get("categories", [])
    cat_id_to_name: dict[int, str] = {int(c["id"]): str(c["name"]) for c in categories}
    cat_name_to_id: dict[str, int] = {v: k for k, v in cat_id_to_name.items()}

    allowed_set: set[str] = set(allowed_categories)
    missing = sorted([c for c in allowed_set if c not in cat_name_to_id])

    if missing:
        raise ValueError(f"These allowed_categories are not in COCO categories: {missing}")

    allowed_cat_ids: set[int] = {cat_name_to_id[name] for name in allowed_set}

    # images
    images_list = coco.get("images", [])
    images: dict[int, CocoImageInfo] = {}
    for im in images_list:
        image_id = int(im["id"])
        images[image_id] = CocoImageInfo(file_name=str(im["file_name"]),
                                         height=int(im["height"]),
                                         width=int(im["width"]))

    # filters
    require_fully_visible = bool(instance_filters.get("require_fully_visible", True))
    ar_min = float(instance_filters.get("aspect_ratio_min", 0.0))
    ar_max = float(instance_filters.get("aspect_ratio_max", 1e9))
    area_ratio_min = float(instance_filters.get("area_ratio_min", 0.0))
    area_ratio_max = float(instance_filters.get("area_ratio_max", 1e9))
    min_solidity = float(instance_filters.get("min_solidity", 0.0))

    pool: list[CocoInstance] = []
    ann_id_to_instance: dict[int, CocoInstance] = {}

    # annotations
    anns = coco.get("annotations", [])

    for ann in anns:
        ann_id = int(ann["id"])
        image_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])

        if cat_id not in allowed_cat_ids or image_id not in images:
            continue

        img_info = images[image_id]
        bbox = ann.get("bbox", None)

        if not bbox or len(bbox) != 4:
            continue

        bbox_xywh = tuple(bbox)

        if bbox_xywh[2] <= 1 or bbox_xywh[3] <= 1:
            continue

        if require_fully_visible and not _is_fully_visible(bbox_xywh, img_info.width, img_info.height):
            continue

        # aspect ratio
        ar = _bbox_aspect_ratio(bbox_xywh)
        if not (ar_min <= ar <= ar_max):
            continue

        if (segmentation := ann.get("segmentation", None)) is None:
            continue

        cat_name = cat_id_to_name[cat_id]

        inst = CocoInstance(
            image_id=image_id,
            ann_id=ann_id,
            category_id=cat_id,
            category_name=cat_name,
            bbox_xywh=bbox_xywh,
            segmentation=segmentation,
            image_height=img_info.height,
            image_width=img_info.width,
            mask=None)

        mask = decode_coco_segmentation_to_mask(segmentation, img_info.height, img_info.width)

        # area ratio constraint
        img_area = float(img_info.width * img_info.height)
        mask_area = float(np.count_nonzero(mask))
        area_ratio = (mask_area / img_area) if img_area > 0 else 0.0

        if not (area_ratio_min <= area_ratio <= area_ratio_max):
            continue

        if min_solidity > 0:
            if _mask_solidity(mask) < min_solidity:
                continue

        pool.append(inst)
        ann_id_to_instance[ann_id] = inst

    if not len(pool):
        raise RuntimeError("COCO pool is empty after filtering. Check allowed_categories/filters")

    return CocoIndex(images=images,
                     cat_id_to_name=cat_id_to_name,
                     cat_name_to_id=cat_name_to_id,
                     pool=pool,
                     ann_id_to_instance=ann_id_to_instance)


def load_coco_image_by_id(index: CocoIndex, coco_images_dir: Path, coco_image_id: int) -> np.ndarray:
    """
    loads RGB image by COCO image_id
    """
    info = index.images.get(coco_image_id, None)
    if info is None:
        raise KeyError(f"COCO image_id not found in index: {coco_image_id}")

    path = coco_images_dir / info.file_name
    if not path.exists():
        raise FileNotFoundError(f"COCO image file not found: {path}")

    return read_image_rgb(path)
