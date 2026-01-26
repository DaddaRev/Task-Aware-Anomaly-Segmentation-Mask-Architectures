from pathlib import Path
from dataclasses import dataclass
from cityscapes_coco_anomaly.synthgen.schemas import CocoInstance

CITYSCAPES_TRAINID_TO_NAME: dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic_light",
    7: "traffic_sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    255: "ignore"
}

CITYSCAPES_NAME_TO_TRAINID: dict[str, int] = {v: k for k, v in CITYSCAPES_TRAINID_TO_NAME.items()}


@dataclass(frozen=True)
class CityscapesPaths:
    root: Path
    images_train: Path
    images_val: Path
    images_test: Path
    semantics_train: Path
    semantics_val: Path
    semantics_test: Path


@dataclass(frozen=True)
class CityscapesIndex:
    """
    Index of Cityscapes leftImg8bit files per split
    """
    split: str
    leftimg_paths: list[Path]
    sem_dir: Path


@dataclass(frozen=True)
class SampleDecision:
    """
    Sampling decisions for one Cityscapes sample
    """

    split: str
    city_id: str

    is_synth: bool
    has_anomaly: bool
    n_instances: int

    target_labels: tuple[str, ...]
    coco_instances: tuple[CocoInstance, ...]  # length = n_instances

    seed: int  # per-sample seed used to create rng
