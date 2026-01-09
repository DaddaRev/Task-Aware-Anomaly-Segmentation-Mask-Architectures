import os
import sys
import shutil
import zipfile
import subprocess
from typing import Any
from pathlib import Path

import numpy as np

from ..schemas.cityscapes import CityscapesPaths, CITYSCAPES_NAME_TO_TRAINID, CITYSCAPES_TRAINID_TO_NAME
from .io import ensure_exists, read_image_rgb, read_png_uint8


def resolve_cityscapes_datadir(datadir: str) -> Path:
    if not (path := Path(datadir).expanduser().resolve()).exists():
        raise FileNotFoundError(f"`{path.expanduser().resolve()}` does not exist")

    # for compatibility reason with the previous version of this script
    if (path / "gtFine").exists() and (path / "leftImg8bit").exists():
        out_root = path
    else:
        # Case Zip FIle: extract next to this script
        left_zip = next(path.glob("leftImg8bit_*.zip"), None)
        gtFine_zip = next(path.glob("gtFine_*.zip"), None)

        if not left_zip or not gtFine_zip:
            raise FileNotFoundError("No Cityscapes dataset found. "
                                    "Provide either a folder containing `gtFine` and `leftImg8bit`"
                                    "or a folder containing `leftImg8bit_*.zip` and `gtFine_*.zip`")

        print(f"Extracting Cityscapes dataset to {path / '_cityscapes_tmp'}...")
        tmp_dir = path / "_cityscapes_tmp"
        out_root = path / "cityscapes"

        with zipfile.ZipFile(left_zip, "r") as zf:
            zf.extractall(tmp_dir)

        with zipfile.ZipFile(gtFine_zip, "r") as zf:
            zf.extractall(tmp_dir)

        # Locate directories named exactly 'gtFine' and 'leftImg8bit'.
        gt_dir = next((p for p in tmp_dir.rglob("gtFine") if p.is_dir()), None)
        left_dir = next((p for p in tmp_dir.rglob("leftImg8bit") if p.is_dir()), None)

        if gt_dir is None or left_dir is None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise FileNotFoundError(
                "Extraction completed but could not locate both 'gtFine' and 'leftImg8bit' directories.")

        # Create the root and copy only the necessary directories
        out_root.mkdir(parents=True, exist_ok=True)
        out_gt = out_root / "gtFine"
        out_left = out_root / "leftImg8bit"

        # Refresh outputs to avoid mixing old/new data
        if out_gt.exists(): shutil.rmtree(out_gt)
        if out_left.exists(): shutil.rmtree(out_left)

        shutil.copytree(gt_dir, out_gt)
        shutil.copytree(left_dir, out_left)

        # Clean up temporary extracted files
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("Creating trainId label images. This may take a while...")
    env = os.environ.copy()
    env["CITYSCAPES_DATASET"] = str(out_root)
    subprocess.run(
        [sys.executable, "-m", "cityscapesscripts.preparation.createTrainIdLabelImgs"], env=env, check=True
    )

    print()
    return out_root


def resolve_cityscapes_paths(cfg_city: dict[str, Any]) -> CityscapesPaths:
    root = resolve_cityscapes_datadir(cfg_city["root"])

    images = cfg_city["images"]
    semantics = cfg_city["semantics"]

    paths = CityscapesPaths(
        root=root,
        images_train=root / images["train"],
        images_val=root / images["val"],
        images_test=root / images["test"],
        semantics_train=root / semantics["train"],
        semantics_val=root / semantics["val"],
        semantics_test=root / semantics["test"])

    return paths


def city_from_city_id(city_id: str) -> str:
    """
    Cityscapes id: <city>_<seq>_<frame>
    """
    return city_id.split("_", 1)[0]


def cityscapes_id_from_leftimg_name(filename: str) -> str:
    """
    frankfurt_000001_000294_leftImg8bit.png -> frankfurt_000001_000294
    """
    suffix = "_leftImg8bit.png"
    if not filename.endswith(suffix):
        raise ValueError(f"Not a leftImg8bit filename: {filename}")

    return filename[: -len(suffix)]


def list_cityscapes_leftimg_files(images_dir: Path) -> list[Path]:
    """
    Returns all *_leftImg8bit.png under images_dir (recursive, includes city folders).
    """
    return sorted(images_dir.rglob("*_leftImg8bit.png"))


def cityscapes_leftimg_path(images_dir: Path, city_id: str) -> Path:

    city = city_from_city_id(city_id)
    return images_dir / city / f"{city_id}_leftImg8bit.png"


def cityscapes_semantic_trainids_path(sem_dir: Path, city_id: str) -> Path:

    city = city_from_city_id(city_id)
    return sem_dir / city / f"{city_id}_gtFine_labelTrainIds.png"


def load_cityscapes_image(images_dir: Path, city_id: str) -> np.ndarray:
    p = cityscapes_leftimg_path(images_dir, city_id)
    ensure_exists(p, "Cityscapes RGB")
    return read_image_rgb(p)


def load_cityscapes_semantic_trainids(sem_dir: Path, city_id: str) -> np.ndarray:
    p = cityscapes_semantic_trainids_path(sem_dir, city_id)
    ensure_exists(p, "Cityscapes labelTrainIds")
    return read_png_uint8(p).astype(np.int32)


def load_cityscapes_pair_from_leftimg(sem_dir: Path, leftimg_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Load (rgb, sem_trainids, city_id) given a leftImg8bit file path.
    """

    city_id = cityscapes_id_from_leftimg_name(leftimg_path.name)
    img = read_image_rgb(leftimg_path)
    sem_path = cityscapes_semantic_trainids_path(sem_dir, city_id)

    sem = read_png_uint8(sem_path).astype(np.int32)
    return img, sem, city_id


def split_dirs(paths: CityscapesPaths, split: str) -> tuple[Path, Path]:
    split = split.lower()
    if split == "train":
        return paths.images_train, paths.semantics_train
    if split == "val":
        return paths.images_val, paths.semantics_val
    if split == "test":
        return paths.images_test, paths.semantics_test
    raise ValueError(f"Unknown split: {split}")


def trainid_to_name_map() -> dict[int, str]:
    return dict(CITYSCAPES_TRAINID_TO_NAME)


def name_to_trainid_map() -> dict[str, int]:
    return dict(CITYSCAPES_NAME_TO_TRAINID)
