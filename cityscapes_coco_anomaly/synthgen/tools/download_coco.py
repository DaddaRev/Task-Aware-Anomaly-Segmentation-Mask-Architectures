import argparse
from pathlib import Path
from dataclasses import dataclass

from cityscapes_coco_anomaly.synthgen.utils import ensure_exists
from .downloader import download_file, extract_zip

COCO_TRAIN2017_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_ANN_TRAINVAL2017_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


@dataclass(frozen=True)
class CocoLayout:
    root: Path
    images_dir: Path
    instances_json: Path
    images_zip: Path
    ann_zip: Path


def _layout(root: Path) -> CocoLayout:
    root = root.expanduser().resolve()
    return CocoLayout(
        root=root,
        images_dir=root / "train2017",
        instances_json=root / "annotations" / "instances_train2017.json",
        images_zip=root / "train2017.zip",
        ann_zip=root / "annotations_trainval2017.zip",
    )


def coco_ready(root: Path) -> bool:
    lay = _layout(root)
    return lay.images_dir.exists() and any(lay.images_dir.glob("*.jpg")) and lay.instances_json.exists()


def download_coco(root: str | Path, *, force: bool = False) -> CocoLayout:
    """
    Downloads:
      - train2017.zip -> extracts to <root>/train2017/
      - annotations_trainval2017.zip -> extracts to <root>/annotations/
    """

    root = Path(root)
    layout = _layout(root)
    layout.root.mkdir(parents=True, exist_ok=True)

    if (not force) and coco_ready(layout.root):
        print(f"COCO already present at: {layout.root}")
        return layout

    print("Downloading COCO train2017 images...")
    download_file(COCO_TRAIN2017_URL, layout.images_zip, force=force)

    print("Downloading COCO train/val annotations...")
    download_file(COCO_ANN_TRAINVAL2017_URL, layout.ann_zip, force=force)

    if not (layout.images_dir.exists() and any(layout.images_dir.glob("*.jpg"))):
        print(f"Extracting {layout.images_zip.name} -> {layout.root}")
        extract_zip(layout.images_zip, layout.root)

    if not layout.instances_json.exists():
        print(f"Extracting {layout.ann_zip.name} -> {layout.root}")
        extract_zip(layout.ann_zip, layout.root)

    ensure_exists(layout.images_dir, "COCO train2017 images dir")
    if not any(layout.images_dir.glob("*.jpg")):
        raise RuntimeError(f"COCO images dir exists but contains no jpg: {layout.images_dir}")
    ensure_exists(layout.instances_json, "COCO instances_train2017.json")

    print("COCO download complete.")
    print(f"Images: {layout.images_dir}")
    print(f"Instances JSON: {layout.instances_json}")
    return layout


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/raw/coco")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    download_coco(args.root, force=args.force)


if __name__ == "__main__":
    main()
