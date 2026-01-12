import numpy as np

from pathlib import Path
from typing import Iterator

from .schemas.cityscapes import CityscapesIndex
from .utils import (CityscapesPaths,
                    split_dirs,
                    list_cityscapes_leftimg_files,
                    load_cityscapes_pair_from_leftimg)


def build_cityscapes_index(paths: CityscapesPaths, split: str) -> CityscapesIndex:

    images_dir, sem_dir = split_dirs(paths, split)
    leftimg_paths = list_cityscapes_leftimg_files(images_dir)
    if len(leftimg_paths) == 0:
        raise RuntimeError(f"No Cityscapes leftImg8bit files found in: {images_dir}")
    return CityscapesIndex(split=split, leftimg_paths=leftimg_paths, sem_dir=sem_dir)


def iter_cityscapes_pairs(index: CityscapesIndex) -> Iterator[tuple[np.ndarray, np.ndarray, str, Path]]:
    """
    Yields (rgb, sem_trainids, city_id, leftimg_path).
    sem is expected to be trainIds (labelTrainIds).
    """
    for p in index.leftimg_paths:
        rgb, sem, city_id = load_cityscapes_pair_from_leftimg(index.sem_dir, p)
        yield rgb, sem, city_id, p


def list_city_ids(index: CityscapesIndex) -> list[str]:
    """
     list all city_ids for this split
    """

    ids: list[str] = []
    for p in index.leftimg_paths:
        name = p.name
        suffix = "_leftImg8bit.png"
        if name.endswith(suffix):
            ids.append(name[: -len(suffix)])

    return ids