import numpy as np
from typing import Any, Optional

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PasteParams:
    target_label: str
    x: int # top-left x in Cityscapes
    y: int # top-left y in Cityscapes
    scale: float
    hflip: bool


@dataclass
class SynthSample:

    sample_id: str
    split: str

    city_id: str  # Cityscapes stem id, e.g., frankfurt_000001_000294

    image_rgb: np.ndarray          # HxWx3 uint8
    sem_id_map: np.ndarray         # HxW int (trainIds)
    gt_pixel: np.ndarray           # HxW uint8 or int (0/1/255)

    masks: list[np.ndarray] = field(default_factory=list)   # list of HxW bool
    labels: list[int] = field(default_factory=list)         # list of ints (all 0 for anomaly)

    is_synth: bool = True
    has_anomaly: bool = False


@dataclass(frozen=True)
class ManifestAnomaly:
    coco_image_id: int
    coco_ann_id: int
    category: str
    target_label: str
    x: int
    y: int
    scale: float
    hflip: bool


@dataclass(frozen=True)
class ManifestRecord:

    sample_id: str
    split: str
    cityscapes_id: str

    is_synth: bool
    has_anomaly: bool
    seed: int

    anomalies: tuple[ManifestAnomaly, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)