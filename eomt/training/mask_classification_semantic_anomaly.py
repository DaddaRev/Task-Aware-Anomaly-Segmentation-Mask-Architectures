import io

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

        # Unfreeze ANOMALY HEAD (trainable)
        if hasattr(self.network, 'anomaly_head'):
            print("Unfreezing anomaly_head params...")
            for param in self.network.anomaly_head.parameters():
                param.requires_grad = True

        # --- Balanced Strategy ---
        # Keep mask_head/upscale frozen to preserve semantic performance.
        # But unfreeze 'q' so queries can adapt to find normal regions.
        # if hasattr(self.network, 'q'):
        #     print("Unfreezing query embeddings...")
        #     self.network.q.weight.requires_grad = True

    def _preprocess_images(self, imgs):
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0
        imgs = (imgs - self.pixel_mean) / self.pixel_std
        return imgs

    def forward(self, x):
        x = x / 255.0
        return self.network(x)

    def _unstack_targets(self, imgs, targets):
        if isinstance(targets, dict):
            batch_size = imgs.shape[0]
            return [{k: v[i] for k, v in targets.items()} for i in range(batch_size)]
        return targets

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        targets = self._unstack_targets(imgs, targets)

        # Prepare targets for anomaly mask loss
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

        # Forward Pass --> 20 class logits for each query (100 queries)
        mask_logits_per_block, class_logits_per_block, anomaly_logits_per_block = self(imgs)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits, anomaly_logits) in enumerate(
                list(zip(mask_logits_per_block, class_logits_per_block, anomaly_logits_per_block))
        ):
            losses = self.criterion_anomalymask(
                masks_queries_logits=mask_logits,
                class_queries_logits=anomaly_logits,
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

        # DEBUG --> Print input stats to check the preprocessing
        #print(f"Input Stats - Type: {imgs.dtype}, Min: {imgs.min()}, Max: {imgs.max()}, Mean: {imgs.float().mean():.2f}")

        crops, origins = self.window_imgs_semantic(imgs)

        mask_logits_per_layer, class_logits_per_layer, anomaly_logits_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits, anomaly_logits) in enumerate(
                list(zip(mask_logits_per_layer, class_logits_per_layer, anomaly_logits_per_layer))
        ):
            
            probs_anomaly = anomaly_logits.softmax(dim=-1)

            # --- ANOMALY HEAD DIRECT PREDICTION ---
            # anomaly_head output: [Normal, Anomaly, NoObject]
            # We want: Channel 0 = Normal, Channel 1 = Anomaly
            valid_probs = probs_anomaly[..., :2]

            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")

            # REMOVE THIS AFTER DEBUG:
            #crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)

            # Combine Mask Probs (sigmoid) with Class Probs
            # [B, Q, H, W] x [B, Q, 2] -> [B, 2, H, W]
            crop_logits = torch.einsum(
               "bqhw, bqc -> bchw",
               mask_logits.sigmoid(),
               valid_probs
            )

            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx in [0,8,16,64]:
                semantic_probs = class_logits.softmax(dim=-1)[..., :-1]

                crop_sem_logits = torch.einsum(
                    "bqhw, bqc -> bchw",
                    mask_logits.sigmoid(),
                    semantic_probs
                )

                sem_logits = self.revert_window_logits_semantic(crop_sem_logits, origins, img_sizes)
                self.plot_semantic_new(
                    imgs[0], targets[0], logits[0], sem_logits[0],
                    log_prefix, i, batch_idx
                )

            @torch.compiler.disable
            def plot_semantic_new(self, img, target, logits, fourth_panel_data, prefix, layer_idx, batch_idx):
                import wandb
                import matplotlib.pyplot as plt

                # 1. Immagine Input (Raw)
                if img.dtype == torch.uint8:
                    img_vis = img.float() / 255.0
                else:
                    img_vis = img.clone()
                    # Gestione range dinamico non standard
                    if img_vis.max() > 1.1:
                        img_vis = img_vis / 255.0
                img_vis = torch.clamp(img_vis, 0, 1).cpu()

                # 2. Logica di Visualizzazione Composita (RGB)
                # logits ha shape [2, H, W] -> [Normal, Anomaly]
                # Creiamo una mappa RGB di diagnostica
                if logits.dim() == 3 and logits.shape[0] == 2:
                    normal_map = torch.clamp(logits[0], 0, 1).cpu()
                    anomaly_map = torch.clamp(logits[1], 0, 1).cpu()

                    # R=Anomaly, G=Normal, B=0
                    # Risultato:
                    # - Verde brillante: Zona sicura e riconosciuta
                    # - Rosso brillante: Anomalia
                    # - Giallo: Incertezza (sovrapposizione)
                    # - Nero: "No Object" (il modello ignora questa zona)
                    pred_vis = torch.stack([anomaly_map, normal_map, torch.zeros_like(anomaly_map)], dim=0)
                else:
                    # Fallback per compatibilità vecchia
                    prob = torch.sigmoid(logits[0]) if logits.dim() == 3 else torch.sigmoid(logits)
                    pred_vis = prob.unsqueeze(0).repeat(3, 1, 1).cpu()

                pred_vis = torch.clamp(pred_vis, 0, 1)

                # 3. Ground Truth (Grigio=Void, Bianco=Anomaly, Nero=Normal)
                target_vis = target.clone().cpu().float()
                vis_map = torch.zeros_like(target_vis, dtype=torch.float32)
                vis_map[target_vis == 1] = 1.0  # Anomaly -> White
                vis_map[target_vis == self.ignore_idx] = 0.4  # Void -> Gray
                vis_t = vis_map.unsqueeze(0).repeat(3, 1, 1)

                # 4. Semantic Baseline (Quarto pannello opzionale)
                fourth_vis = None
                if fourth_panel_data is not None and fourth_panel_data.dim() == 3 and fourth_panel_data.shape[0] > 2:
                    sem_indices = torch.argmax(fourth_panel_data, dim=0).cpu()
                    # _apply_colormap deve essere disponibile o adattato
                    if hasattr(self, '_apply_colormap'):
                        fourth_vis = self._apply_colormap(sem_indices)
                    else:
                        # Fallback rapido semantica in scala di grigi se manca colormap
                        fourth_vis = (sem_indices.float() / fourth_panel_data.shape[0]).unsqueeze(0).repeat(3, 1, 1)

                # Composizione Plot
                num_plots = 4 if fourth_vis is not None else 3
                fig, axes = plt.subplots(1, num_plots, figsize=[5 * num_plots, 5], sharex=True, sharey=True)

                axes[0].imshow(img_vis.permute(1, 2, 0))
                axes[0].set_title("Input")

                axes[1].imshow(vis_t.permute(1, 2, 0))
                axes[1].set_title("GT (Wh=Anom, Gy=Void)")

                # Qui vedrai VERDE (Normal) vs ROSSO (Anomaly) vs NERO (Ignore)
                axes[2].imshow(pred_vis.permute(1, 2, 0))
                axes[2].set_title("Pred (G=Norm, R=Anom)")

                if fourth_vis is not None:
                    axes[3].imshow(fourth_vis.permute(1, 2, 0))
                    axes[3].set_title("Semantic Seg")

                for ax in axes: ax.axis("off")

                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, facecolor="black")
                plt.close(fig)
                buf.seek(0)

                block_postfix = self.block_postfix(layer_idx)
                name = f"{prefix}_pred_{batch_idx}{block_postfix}"

                # Log sicuro con controllo se logger esiste
                if hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'experiment'):
                    self.trainer.logger.experiment.log({name: [wandb.Image(Image.open(buf))]})

    '''
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

        eps = 1e-6
        if logits.dim() == 3 and logits.shape[0] == 2:
            # We use absolute Anomaly Mass (logits[1]) directly.
            # This naturally   "No Object" regions (where Anomaly Mass is low).
            prob_anomaly = logits[1]
        else:
            prob_anomaly = torch.sigmoid(logits[0]) if logits.dim() == 3 else torch.sigmoid(logits)

        pred_vis = prob_anomaly.unsqueeze(0).repeat(3, 1, 1).cpu()
        pred_vis = torch.clamp(pred_vis, 0, 1)

        # 3. Baseline MSP (FIXED)
        # Same logic for baseline: Use absolute Anomaly Mass
        if baseline_logits.dim() == 3 and baseline_logits.shape[0] == 2:
            baseline_prob_anomaly = baseline_logits[1]
        else:
            baseline_prob_anomaly = torch.sigmoid(baseline_logits[0]) if baseline_logits.dim() == 3 else torch.sigmoid(
                baseline_logits)

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
                from torchvision.transforms.functional import to_pil_image
                caption = f"{prefix}_L{layer_idx}_Input_GT_NormalityHead_BaselineMSP"
                # Convert to PIL Image to ensure robust serialization to WandB
                pil_comparison = to_pil_image(comparison.cpu())
                self.logger.experiment.log({
                    f"val_images/{prefix}_layer_{layer_idx}": [
                        wandb.Image(pil_comparison, caption=caption)
                    ]
                })
        except Exception as e:
            print(f"Warning: Could not log to wandb: {e}")

        # Debug locale (opzionale)
        if batch_idx == 0 and layer_idx == 0:
            save_image(comparison, f"debug_vis_{prefix}.png")
        '''

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

            # Combine with masks same as anomaly head
            crop_logits = torch.einsum(
                "bqhw, bqc -> bchw",
                mask_logits.sigmoid(),
                baseline_probs
            )

            # Revert windowing
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)
            return logits
