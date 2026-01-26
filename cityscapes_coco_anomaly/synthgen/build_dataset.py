import time
import argparse
import numpy as np
from pathlib import Path

from cityscapes_coco_anomaly.synthgen.config import load_config, AppConfig
from cityscapes_coco_anomaly.synthgen.sampler import build_sample_decision, make_rng_for_sample
from cityscapes_coco_anomaly.synthgen.utils import name_to_trainid_map, decode_coco_segmentation_to_mask
from cityscapes_coco_anomaly.synthgen.cityscapes_index import build_cityscapes_index, iter_cityscapes_pairs
from cityscapes_coco_anomaly.synthgen.coco_index import build_coco_index, CocoIndex, load_coco_image_by_id

from cityscapes_coco_anomaly.synthgen.geometry import (parse_geometry_cfg,
                                                       compute_patch_geometry,
                                                       apply_geometry_to_patch)

from cityscapes_coco_anomaly.synthgen.blending import parse_blending_cfg, blend_patch_into_image
from cityscapes_coco_anomaly.synthgen.quality import parse_quality_cfg, sample_paste_location
from cityscapes_coco_anomaly.synthgen.targets import TargetsConfig, merge_alphas_to_gt_and_instances
from cityscapes_coco_anomaly.synthgen.export import export_sample, append_manifest_line

# ANSI colors for logs
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_CYAN = "\033[36m"


def extract_coco_patch_rgb_alpha(coco_index: CocoIndex, coco_images_dir: Path, inst) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      patch_rgb: uint8 h x w x 3
      alpha:     float32 h x w in [0,1]
    """

    # load full coco image
    coco_rgb = load_coco_image_by_id(coco_index, coco_images_dir, inst.image_id)

    # decode full-res mask (bool HxW)
    mask = decode_coco_segmentation_to_mask(inst.segmentation, inst.image_height, inst.image_width)

    x, y, w, h = inst.bbox_xywh
    x0 = int(max(0, np.floor(x)))
    y0 = int(max(0, np.floor(y)))
    x1 = int(min(inst.image_width, np.ceil(x + w)))
    y1 = int(min(inst.image_height, np.ceil(y + h)))

    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid bbox after rounding")

    patch_rgb = coco_rgb[y0:y1, x0:x1].copy()
    patch_mask = mask[y0:y1, x0:x1]

    alpha = patch_mask.astype(np.float32)
    return patch_rgb, alpha


def build_split(cfg: AppConfig, split: str, coco_index: CocoIndex):
    print(f"{C_BOLD}{C_CYAN}Building dataset for split `{split}`...{C_RESET}")

    # indexes
    cs_index = build_cityscapes_index(cfg.paths_cityscapes, split)

    syn = cfg.synthesis
    export_cfg = cfg.export

    # configs for modules
    geo_cfg = parse_geometry_cfg(syn)
    blend_cfg = parse_blending_cfg(syn)
    qual_cfg = parse_quality_cfg(syn)
    name2id = name_to_trainid_map()

    # targets config from YAML
    gt_cfg = export_cfg.get("gt_pixel", {})
    values = (gt_cfg.get("values", {}) if isinstance(gt_cfg, dict) else {})

    tcfg = TargetsConfig(
        normal_pixel_value=values.get("normal", 0),
        anomaly_pixel_value=values.get("anomaly", 1),
        ignore_pixel_value=values.get("ignore", 255))

    n_total = 0
    n_synth = 0
    n_anom = 0
    n_failed = 0

    for city_rgb, sem_trainids, city_id, left_path in iter_cityscapes_pairs(cs_index):
        n_total += 1

        # deterministic decision for this city_id
        decision = build_sample_decision(cfg, split, city_id, coco_index)

        # clean or synth-without-anomaly: just export empty targets
        if (not decision.is_synth) or (not decision.has_anomaly) or not decision.n_instances:
            H, W = city_rgb.shape[:2]
            gt_pixel, masks, labels = merge_alphas_to_gt_and_instances(H, W, [], tcfg)

            rel = export_sample(
                output_root=cfg.dataset.output_root,
                split=split,
                sample_id=city_id,
                rgb=city_rgb,
                gt_pixel=gt_pixel,
                masks=masks,
                labels=labels,
                export_cfg=export_cfg)

            record = {
                "split": split,
                "city_id": city_id,
                "is_synth": decision.is_synth,
                "has_anomaly": False,
                "n_instances": 0,
                "seed": decision.seed,
                "outputs": rel,
            }

            append_manifest_line(cfg.dataset.output_root, record, export_cfg=export_cfg)
            continue

        n_synth += 1

        # per-sample rng to sample locations consistently
        _, rng = make_rng_for_sample(city_id, cfg.dataset.seed)

        H, W = city_rgb.shape[:2]
        cur_img = city_rgb.copy()

        alpha_full_list: list[np.ndarray] = []
        placed = 0

        # for each instance
        for inst, tgt_label_name in zip(decision.coco_instances, decision.target_labels):

            # Extract patch from COCO
            patch_rgb, alpha = extract_coco_patch_rgb_alpha(coco_index, cfg.paths_coco.images_train, inst)

            # We use a y_center sampled uniformly; quality will resample x,y
            # This keeps geometry deterministic without needing a location first
            y_center = float(rng.integers(0, H))
            geom = compute_patch_geometry(
                rng=rng,
                city_H=H,
                city_W=W,
                patch_h=patch_rgb.shape[0],
                patch_w=patch_rgb.shape[1],
                target_y_center=y_center,
                geo_cfg=geo_cfg)

            # Apply resize/flip
            patch_rgb2, alpha2 = apply_geometry_to_patch(
                patch_rgb=patch_rgb,
                alpha=alpha,
                out_w=geom.out_w,
                out_h=geom.out_h,
                hflip=geom.hflip)

            # Sample a valid location given semantic constraints
            loc = sample_paste_location(
                rng=rng,
                sem_trainids=sem_trainids,
                target_label_name=tgt_label_name,
                patch_w=patch_rgb2.shape[1],
                patch_h=patch_rgb2.shape[0],
                name_to_trainid=name2id,
                cfg=qual_cfg)

            if loc is None:
                n_failed += 1
                continue

            x, y = loc

            # Blend into image
            cur_img, alpha_full = blend_patch_into_image(
                city_rgb=cur_img,
                patch_rgb=patch_rgb2,
                alpha=alpha2,
                x=x,
                y=y,
                cfg=blend_cfg)

            alpha_full_list.append(alpha_full)
            placed += 1

        if has_anom := placed > 0:
            n_anom += 1

        # Targets (union + per-instance)
        gt_pixel, masks, labels = merge_alphas_to_gt_and_instances(H, W, alpha_full_list, tcfg)

        rel = export_sample(
            output_root=cfg.dataset.output_root,
            split=split,
            sample_id=city_id,
            rgb=cur_img,
            gt_pixel=gt_pixel,
            masks=masks,
            labels=labels,
            export_cfg=export_cfg)

        record = {
            "split": split,
            "city_id": city_id,
            "is_synth": True,
            "has_anomaly": has_anom,
            "planned_instances": decision.n_instances,
            "placed_instances": placed,
            "seed": decision.seed,
            "target_labels": decision.target_labels,
            "coco_ann_ids": [ci.ann_id for ci in decision.coco_instances],
            "outputs": rel,
        }

        append_manifest_line(cfg.dataset.output_root, record, export_cfg=export_cfg)

        if n_total % 200 == 0:
            print(
                f"{C_DIM}{C_CYAN}[{split}]{C_RESET} {n_total:>5d} processed | "
                f"{C_YELLOW}synth{C_RESET} {n_synth:>5d} | "
                f"{C_RED}anomalous{C_RESET} {n_anom:>5d} | "
                f"failed placements {n_failed:>5d}")

    print(f"{C_BOLD}{C_GREEN}Done {split}:{C_RESET} total={n_total:>5d}, "
          f"synth={n_synth:>5d}, anomalous={n_anom:>5d}, failed placements={n_failed:>5d}")


def main():
    parser = argparse.ArgumentParser(description="Build Cityscapes + COCO synthetic anomaly dataset")

    parser.add_argument(
        "--config",
        type=str,
        default="../configs/synth_dataset_v1.yaml",
        help="Path to YAML config")

    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated list of splits to build")

    parser.add_argument(
        "--prepare_datasets",
        action="store_true",
        help="If set, download datasets according to YAML `datasets.*.download.enabled`")

    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download when --prepare_datasets is used")

    args = parser.parse_args()

    cfg = load_config(args.config, args.prepare_datasets, args.force_download)

    # Ensure output root exists
    cfg.dataset.output_root.mkdir(parents=True, exist_ok=True)

    # build coco pool
    coco_cfg = cfg.synthesis.get("coco", {})

    allowed_categories = coco_cfg.get("allowed_categories", [])
    if not isinstance(allowed_categories, list) or not len(allowed_categories):
        raise ValueError("synthesis.coco.allowed_categories must be a non-empty list")

    instance_filters = coco_cfg.get("instance_filters", {})
    if not isinstance(instance_filters, dict):
        instance_filters = {}

    t0 = time.perf_counter()
    coco_index = build_coco_index(
        coco_instances_json=cfg.paths_coco.instances_train_json,
        allowed_categories=[x for x in allowed_categories],
        instance_filters=instance_filters)

    if (dt := time.perf_counter() - t0) < 60:
        print(f"{C_BOLD}{C_BLUE}COCO pool size:{C_RESET} {len(coco_index.pool)} (built in {dt:.1f}s)")
    else:
        mins, secs = divmod(dt, 60.0)
        print(f"{C_BOLD}{C_BLUE}COCO pool size:{C_RESET} {len(coco_index.pool)} (built in {int(mins)}m {secs:.1f}s)")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        build_split(cfg, s, coco_index)


if __name__ == "__main__":
    main()
