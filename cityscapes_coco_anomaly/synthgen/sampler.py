import numpy as np
from typing import Any, Optional

from cityscapes_coco_anomaly.synthgen.config import AppConfig
from cityscapes_coco_anomaly.synthgen.utils.io import derive_sample_seed
from cityscapes_coco_anomaly.synthgen.schemas.cityscapes import SampleDecision
from cityscapes_coco_anomaly.synthgen.schemas.coco import CocoInstance, CocoIndex


def _get_split_ratios(cfg: AppConfig, split: str) -> tuple[float, float]:
    split = split.lower()
    d = getattr(cfg.splits, split)

    return float(d["clean_ratio"]), float(d["synth_ratio"])


def _get_instances_range(syn: dict[str, Any]) -> tuple[int, int]:
    inst = syn.get("instances_per_image", {})

    mn = int(inst.get("min", 1))
    mx = int(inst.get("max", 1))

    if mn < 0 or mx < 0 or mx < mn:
        raise ValueError(f"Invalid instances_per_image range: min={mn}, max={mx}")

    return mn, mx


def _get_target_labels_with_weights(syn: dict[str, Any]) -> list[tuple[str, float]]:
    cs = syn.get("cityscapes", {})

    tl = cs.get("target_labels", [])

    if not len(tl):
        raise ValueError("synthesis.cityscapes.target_labels must be a non-empty list")

    out: list[tuple[str, float]] = []
    for item in tl:
        if not (len(item) == 2):
            raise ValueError("Each target_labels entry must be [label, weight]")

        label = str(item[0])

        if (w := float(item[1])) < 0:
            raise ValueError(f"Negative weight for target label {label}: {w}")

        out.append((label, w))

    if (s := sum(w for _, w in out)) <= 0:
        raise ValueError("Sum of target label weights must be > 0")

    # normalize (so we can pass p=... to rng.choice)
    out = [(lab, w / s) for lab, w in out]
    return out


def make_rng_for_sample(city_id: str, base_seed: int) -> tuple[int, np.random.Generator]:
    """
    Deterministic per-sample RNG, independent of processing order / parallelism.
    """
    seed = int(derive_sample_seed(city_id, base_seed))
    rng = np.random.default_rng(seed)
    return seed, rng


def sample_is_synth(rng: np.random.Generator, synth_ratio: float) -> bool:
    return bool(rng.random() < synth_ratio)


def sample_has_anomaly(rng: np.random.Generator, p_has_anomaly: float) -> bool:
    return bool(rng.random() < p_has_anomaly)


def sample_n_instances(rng: np.random.Generator, n_min: int, n_max: int) -> int:
    if n_max == n_min:
        return n_min

    return int(rng.integers(n_min, n_max + 1))


def sample_target_label(rng: np.random.Generator, labels_with_prob: list[tuple[str, float]]) -> str:
    labels = [x[0] for x in labels_with_prob]
    probs = [x[1] for x in labels_with_prob]
    return str(rng.choice(labels, p=probs))


def sample_coco_instance(rng: np.random.Generator, coco_index: CocoIndex) -> CocoInstance:
    if not len(coco_index.pool):
        raise RuntimeError("COCO pool is empty")

    idx = int(rng.integers(0, len(coco_index.pool)))
    return coco_index.pool[idx]


def build_sample_decision(
    cfg: AppConfig,
    split: str,
    city_id: str,
    coco_index: CocoIndex) -> SampleDecision:

    """
     Returns a fully deterministic 'plan' for this Cityscapes sample:
     - whether synth or clean
     - whether it contains anomalies
     - how many instances
     - which COCO instances
     - which target labels (one per instance)
     """

    _, synth_ratio = _get_split_ratios(cfg, split)
    syn = cfg.synthesis

    p_has_anomaly = float(syn.get("p_hash_anomaly", 1.0))
    n_min, n_max = _get_instances_range(syn)
    targets = _get_target_labels_with_weights(syn)

    seed, rng = make_rng_for_sample(city_id, cfg.dataset.seed)

    is_synth = sample_is_synth(rng, synth_ratio)

    # clean sample: no anomalies, no instances
    if not is_synth:
        return SampleDecision(
            split=split,
            city_id=city_id,
            is_synth=False,
            has_anomaly=False,
            n_instances=0,
            target_labels=tuple(),
            coco_instances=tuple(),
            seed=seed)

    # synth sample: decide if we actually insert anomalies
    if not sample_has_anomaly(rng, p_has_anomaly):
        return SampleDecision(
            split=split,
            city_id=city_id,
            is_synth=True,
            has_anomaly=False,
            n_instances=0,
            target_labels=tuple(),
            coco_instances=tuple(),
            seed=seed)

    n_inst = sample_n_instances(rng, n_min, n_max)

    coco_insts: list[CocoInstance] = []
    tlabels: list[str] = []

    for _ in range(n_inst):
        inst = sample_coco_instance(rng, coco_index)
        lab = sample_target_label(rng, targets)
        coco_insts.append(inst)
        tlabels.append(lab)

    return SampleDecision(
        split=split,
        city_id=city_id,
        is_synth=True,
        has_anomaly=True,
        n_instances=n_inst,
        target_labels=tuple(tlabels),
        coco_instances=tuple(coco_insts),
        seed=seed)

