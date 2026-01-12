import yaml
from typing import Any
from pathlib import Path
from dataclasses import dataclass

from cityscapes_coco_anomaly.synthgen.utils import (CityscapesPaths,
                                                    CocoPaths,
                                                    resolve_cityscapes_paths,
                                                    resolve_coco_paths)

from cityscapes_coco_anomaly.synthgen.tools.download_coco import download_coco
from cityscapes_coco_anomaly.synthgen.tools.download_cityscapes import download_cityscapes


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    seed: int
    output_root: Path


@dataclass(frozen=True)
class SplitsConfig:
    """
    Stores the split ratios as raw dicts, e.g.,
      train: {clean_ratio: 0.5, synth_ratio: 0.5}
    """
    train: dict[str, float]
    val: dict[str, float]
    test: dict[str, float]


@dataclass(frozen=True)
class AppConfig:
    """
    Top-level config object used by build_dataset.py.
    """
    dataset: DatasetConfig
    paths_cityscapes: CityscapesPaths
    paths_coco: CocoPaths
    splits: SplitsConfig

    synthesis: dict[str, Any]
    export: dict[str, Any]

    raw: dict[str, Any]


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _validate_split_ratios(split_name: str, cfg: dict[str, Any]) -> dict[str, float]:

    if not isinstance(cfg, dict):
        raise ValueError(f"splits.{split_name} must be a dict, got {type(cfg)}")

    clean = float(cfg["clean_ratio"])
    synth = float(cfg["synth_ratio"])

    if clean < 0 or synth < 0:
        raise ValueError(f"splits.{split_name}: ratios must be >= 0 (got clean={clean}, synth={synth})")

    if abs(clean + synth - 1.0) > 1e-6:
        raise ValueError(
            f"splits.{split_name}: clean_ratio + synth_ratio must equal 1.0 (got {clean}+{synth}={clean + synth})")

    return {"clean_ratio": clean, "synth_ratio": synth}


def prepare_datasets(cfg: dict[str, Any], *, force: bool = False) -> None:
    """
    If enabled in YAML under `datasets.*.download.enabled`, download datasets
    """

    datasets = cfg.get("datasets", {})

    # Cityscapes
    city_cfg = datasets.get("cityscapes", {})

    cs_download = city_cfg.get("download", {"enabled": False})

    if cs_download["enabled"]:
        city_root = Path(city_cfg["root"]).expanduser().resolve()
        download_cityscapes(city_root, force=force)
    else:
        print("Cityscapes download disabled (datasets.cityscapes.download.enabled=false)")

    #  COCO
    coco_cfg = datasets.get("coco", {})
    coco_download = coco_cfg.get("download", {"enabled": False})

    if coco_download["enabled"]:
        coco_root = Path(coco_cfg["root"]).expanduser().resolve()
        download_coco(coco_root, force=force)
    else:
        print("COCO download disabled (datasets.coco.download.enabled=false)")


def load_config(config_path: str | Path, get_datasets: bool = False, force_download=False) -> AppConfig:
    """
    Load yaml config, validate essentials, and resolve dataset paths.
    """

    config_path = Path(config_path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = _read_yaml(config_path)

    # dataset
    dataset_cfg = cfg["dataset"]

    name = dataset_cfg["name"]
    seed = dataset_cfg["seed"]

    output_root = Path(dataset_cfg["output_root"]).expanduser().resolve()

    dataset = DatasetConfig(name=name, seed=seed, output_root=output_root)

    # datasets paths
    datasets_cfg = cfg["datasets"]

    city_cfg = datasets_cfg["cityscapes"]
    coco_cfg = datasets_cfg["coco"]

    if get_datasets:
        prepare_datasets(cfg, force=force_download)

    paths_city = resolve_cityscapes_paths(city_cfg)
    paths_coco = resolve_coco_paths(coco_cfg)

    # splits
    splits_cfg = cfg["splits"]

    train = _validate_split_ratios("train", splits_cfg["train"])
    val = _validate_split_ratios("val", splits_cfg["val"])
    test = _validate_split_ratios("test", splits_cfg["test"])
    splits = SplitsConfig(train=train, val=val, test=test)

    # synthesis/export
    synthesis = cfg["synthesis"]
    export = cfg["export"]

    return AppConfig(
        dataset=dataset,
        paths_cityscapes=paths_city,
        paths_coco=paths_coco,
        splits=splits,
        synthesis=synthesis,
        export=export,
        raw=cfg)