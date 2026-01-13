import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
from typing import Optional, List

from .mask_classification_loss import MaskClassificationLoss
from .mask_classification_semantic import MaskClassificationSemantic


class MCS_Anomaly(MaskClassificationSemantic):

    def __init__(
            self,
            network: nn.Module,
            img_size: tuple[int, int],
            attn_mask_annealing_enabled: bool,
            attn_mask_annealing_start_steps: Optional[List[int]] = None,
            attn_mask_annealing_end_steps: Optional[List[int]] = None,
            num_points: int = 12544,
            num_classes: int = 19,
            ignore_idx: int = 255,
            lr: float = 1e-5,
            llrd: float = 0.8,
            llrd_l2_enabled: bool = True,
            lr_mult: float = 1.0,
            weight_decay: float = 0.05,
            oversample_ratio: float = 3.0,
            importance_sample_ratio: float = 0.75,
            poly_power: float = 0.9,
            warmup_steps: List[int] = [500, 1000],
            mask_coefficient: float = 2.0,
            dice_coefficient: float = 4.0,
            class_coefficient: float = 4.0,
            mask_thresh: float = 0.8,
            overlap_thresh: float = 0.8,
            ckpt_path: Optional[str] = None,
            delta_weights: bool = False,
            load_ckpt_class_head: bool = True,
            **kwargs
    ):

        no_object_coefficient = 1.0
        bg_coefficient = 1.0

        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            ignore_idx=ignore_idx,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            mask_thresh=mask_thresh,
            overlap_thresh=overlap_thresh,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
            no_object_coefficient=no_object_coefficient,
            **kwargs
        )

        self.criterion_anomalymask = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=2, # Normal (0), Anomaly (1) + NoObject (2)
            no_object_coefficient=no_object_coefficient,
            bg_coefficient=bg_coefficient,
        )

        num_layers = self.network.num_blocks + 1 if getattr(self.network, 'masked_attn_enabled', False) else 1
        self.init_metrics_semantic(self.ignore_idx, num_layers)

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for param in self.network.parameters():
            param.requires_grad = False

        # Unfreeze NORMALITY HEAD (trainable)
        if hasattr(self.network, 'normality_head'):
            print("Unfreezing normality_head params...")
            for param in self.network.normality_head.parameters():
                param.requires_grad = True

        # --- Balanced Strategy ---
        # Keep mask_head/upscale frozen to preserve semantic performance.
        # But unfreeze 'q' so queries can adapt to find normal regions.
        if hasattr(self.network, 'q'):
            print("Unfreezing query embeddings...")
            self.network.q.weight.requires_grad = True

    def _preprocess_images(self, imgs):
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0
        #imgs = (imgs - self.pixel_mean) / self.pixel_std
        return imgs

    def forward(self, x):
        x = self._preprocess_images(x)
        return self.network(x)

    def _unstack_targets(self, imgs, targets):
        if isinstance(targets, dict):
            batch_size = imgs.shape[0]
            return [{k: v[i] for k, v in targets.items()} for i in range(batch_size)]
        return targets

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        targets = self._unstack_targets(imgs, targets)

        # --- UPDATED PARADIGM ---
        # Class 0: Normal (Road, etc)
        # Class 1: Anomaly (Explicit supervision on anomalies)
        # Class 255: Void -> Map to No-Object (Implicitly ignored by not including in targets)
        targets_for_loss = []
        for t in targets:
            # Keep Normal(0) and Anomaly(1). Ignore Void(255).
            keep = (t["labels"] == 0) | (t["labels"] == 1)
            targets_for_loss.append({
                "masks": t["masks"][keep],
                "labels": t["labels"][keep] # Labels are already 0 and 1
            })

        mask_logits_per_block, class_logits_per_block, normality_logits_per_block = self(imgs)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits, normality_logits) in enumerate(
                list(zip(mask_logits_per_block, class_logits_per_block, normality_logits_per_block))
        ):
            losses = self.criterion_anomalymask(
                masks_queries_logits=mask_logits,
                class_queries_logits=normality_logits,
                targets=targets_for_loss,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        return self.criterion_anomalymask.loss_total(losses_all_blocks, self.log)

    def eval_step(self, batch, batch_idx=None, log_prefix=None):
        imgs, targets = batch
        targets = self._unstack_targets(imgs, targets)
        img_sizes = [img.shape[-2:] for img in imgs]

        crops, origins = self.window_imgs_semantic(imgs)

        mask_logits_per_layer, class_logits_per_layer, normality_logits_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits, normality_logits) in enumerate(
                list(zip(mask_logits_per_layer, class_logits_per_layer, normality_logits_per_layer))
        ):
            probs_normality = normality_logits.softmax(dim=-1)

            # --- NORMALITY HEAD DIRECT PREDICTION ---
            # normality_head output: [Normal, Anomaly, NoObject]
            # We want: Channel 0 = Normal, Channel 1 = Anomaly
            valid_probs = probs_normality[..., :2]

            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")

            # Combine Mask Probs (sigmoid) with Class Probs
            # [B, Q, H, W] x [B, Q, 2] -> [B, 2, H, W]
            crop_logits = torch.einsum(
                "bqhw, bqc -> bchw",
                mask_logits.sigmoid(),
                valid_probs
            )

            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 0:
                # Compute baseline MSP from frozen class_head for comparison
                baseline_logits = self._compute_baseline_msp(
                    mask_logits, class_logits, origins, img_sizes
                )
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], baseline_logits[0],
                    log_prefix, i, batch_idx
                )

    def _compute_baseline_msp(self, mask_logits, class_logits, origins, img_sizes):
        """
        Compute baseline anomaly detection using Maximum Softmax Probability (MSP)
        from frozen class_head predictions.

        MSP approach: Anomaly = 1 - max_prob(semantic_classes)
        Lower confidence on semantic classes → Higher anomaly score
        """
        # Get semantic probabilities from frozen class_head [B, Q, 20]
        semantic_probs = F.softmax(class_logits, dim=-1)

        # Max probability across 19 semantic classes (exclude void class at index 19)
        max_semantic_prob, _ = semantic_probs[..., :-1].max(dim=-1)  # [B, Q]

        # MSP anomaly score: Low semantic confidence = High anomaly
        # Create binary classification: [Normal, Anomaly]
        baseline_probs = torch.stack([
            max_semantic_prob,           # Normal (high semantic confidence)
            1.0 - max_semantic_prob      # Anomaly (low semantic confidence)
        ], dim=-1)  # [B, Q, 2]

        # Combine with masks same as normality head
        crop_logits = torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            baseline_probs
        )

        # Revert windowing
        logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)
        return logits

    def plot_semantic(self, img, target, logits, baseline_logits, prefix, layer_idx, batch_idx):
        import wandb

        # 1. Immagine Input (Raw) - Ensure [C, H, W] in range 0-1
        if img.dtype == torch.uint8:
            img_vis = img.float() / 255.0
        else:
            img_vis = img.clone()
            if img_vis.max() > 1.1:
                img_vis = img_vis / 255.0
        img_vis = torch.clamp(img_vis, 0, 1).cpu()

        # 2. Normality Head Prediction (Probabilità Anomalia) - logits shape: [2, H, W]
        # Channel 0 = Normal, Channel 1 = Anomaly
        if logits.dim() == 3 and logits.shape[0] == 2:
            prob_anomaly = torch.softmax(logits, dim=0)[1]  # Get anomaly channel
        else:
            prob_anomaly = torch.sigmoid(logits[0]) if logits.dim() == 3 else torch.sigmoid(logits)

        pred_vis = prob_anomaly.unsqueeze(0).repeat(3, 1, 1).cpu()
        pred_vis = torch.clamp(pred_vis, 0, 1)

        # 3. Baseline MSP Prediction
        if baseline_logits.dim() == 3 and baseline_logits.shape[0] == 2:
            baseline_prob_anomaly = torch.softmax(baseline_logits, dim=0)[1]
        else:
            baseline_prob_anomaly = torch.sigmoid(baseline_logits[0]) if baseline_logits.dim() == 3 else torch.sigmoid(baseline_logits)

        baseline_vis = baseline_prob_anomaly.unsqueeze(0).repeat(3, 1, 1).cpu()
        baseline_vis = torch.clamp(baseline_vis, 0, 1)

        # 4. Ground Truth - Anomaly=1 -> White, Background=0 -> Black, Void=255 -> Gray
        target_vis = target.clone().cpu().float()
        vis_map = torch.zeros_like(target_vis, dtype=torch.float32)

        vis_map[target_vis == 1] = 1.0       # Anomaly -> White
        vis_map[target_vis == self.ignore_idx] = 0.4  # Void -> Gray
        # Background (0) stays black (0.0)

        vis_t = vis_map.unsqueeze(0).repeat(3, 1, 1)

        # 5. Combina: [Input | Ground Truth | Normality Head | Baseline MSP]
        comparison = torch.cat([img_vis, vis_t, pred_vis, baseline_vis], dim=2)

        # 6. Log su WandB
        try:
            if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
                caption = f"{prefix}_L{layer_idx}_Input_GT_NormalityHead_BaselineMSP"
                self.logger.experiment.log({
                    f"val_images/{prefix}_layer_{layer_idx}": [
                        wandb.Image(comparison, caption=caption)
                    ]
                })
        except Exception as e:
            print(f"Warning: Could not log to wandb: {e}")

        # Debug locale (opzionale)
        if batch_idx == 0 and layer_idx == 0:
            save_image(comparison, f"debug_vis_{prefix}.png")
