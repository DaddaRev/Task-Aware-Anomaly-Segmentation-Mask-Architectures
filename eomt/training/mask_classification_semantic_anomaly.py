import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from typing import Optional, List

from .mask_classification_loss import MaskClassificationLoss
from .mask_classification_semantic import MaskClassificationSemantic


class MCS_Anomaly(MaskClassificationSemantic):

    def __init__(
            self,
            network: nn.Module,
            img_size: tuple[int, int],
            num_classes: int,
            attn_mask_annealing_enabled: bool,
            attn_mask_annealing_start_steps: Optional[List[int]] = None,
            attn_mask_annealing_end_steps: Optional[List[int]] = None,
            **kwargs
    ):
        no_object_weight_val = kwargs.pop('no_object_weight', 0.1)

        kwargs.setdefault('mask_coefficient', 2.0)
        kwargs.setdefault('dice_coefficient', 2.0)
        kwargs.setdefault('class_coefficient', 2.0)
        kwargs['no_object_coefficient'] = no_object_weight_val
        self.num_classes = 2
        self.stuff_classes = range(2)

        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            **kwargs
        )

        num_layers = self.network.num_blocks + 1 if getattr(self.network, 'masked_attn_enabled', False) else 1
        self.init_metrics_semantic(self.ignore_idx, num_layers)

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess_images(self, imgs):
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0
        imgs = (imgs - self.pixel_mean) / self.pixel_std
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

        mask_logits_per_block, class_logits_per_block, anomaly_logits_per_block = self(imgs)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits, anomaly_logits) in enumerate(
                list(zip(mask_logits_per_block, class_logits_per_block, anomaly_logits_per_block))
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=anomaly_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        return self.criterion.loss_total(losses_all_blocks, self.log)

    def eval_step(self, batch, batch_idx=None, log_prefix=None):
        imgs, targets = batch
        targets = self._unstack_targets(imgs, targets)
        img_sizes = [img.shape[-2:] for img in imgs]

        crops, origins = self.window_imgs_semantic(imgs)

        mask_logits_per_layer, class_logits_per_layer, anomaly_logits_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits, anomaly_logits) in enumerate(
                list(zip(mask_logits_per_layer, class_logits_per_layer, anomaly_logits_per_layer))
        ):
            probs_anomaly = anomaly_logits.softmax(dim=-1)

            # --- FIX LOGICA CLASSI (Sfondo+Void=0, Anomalia=1) ---
            valid_probs = torch.stack([
                probs_anomaly[..., 0] + probs_anomaly[..., 2],  # Canale 0: Normal (Bg + Void)
                probs_anomaly[..., 1]  # Canale 1: Anomaly
            ], dim=-1)

            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")

            # Einsum corretto: [B, Q, H, W] * [B, Q, C] -> [B, C, H, W]
            crop_logits = torch.einsum(
                "bqhw, bqc -> bchw",
                mask_logits.sigmoid(),
                valid_probs
            )

            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

    def plot_semantic(self, img, target, logits, prefix, layer_idx, batch_idx):
        import wandb

        # 1. Denormalizza Immagine
        mean = self.pixel_mean.squeeze(0).cpu()
        std = self.pixel_std.squeeze(0).cpu()
        img_vis = img.clone().cpu() * std + mean
        img_vis = torch.clamp(img_vis, 0, 1)

        # 2. Predizione: Visualizza la PROBABILITÀ pura invece dell'argmax
        # logits[0] = Prob Sfondo, logits[1] = Prob Anomalia
        # Usiamo direttamente logits[1] che è un float tra 0 e 1
        anomaly_prob = logits[1].cpu()

        # Espandi a 3 canali per l'immagine RGB (Grayscale heat map)
        pred_vis = anomaly_prob.unsqueeze(0).repeat(3, 1, 1)

        # (Opzionale) Se vuoi enfatizzare i valori bassi, puoi usare:
        # pred_vis = torch.clamp(pred_vis * 2.0, 0, 1)

        # 3. Ground Truth (BG=0, Void=100/255, Anomaly=1.0)
        target_vis = target.clone().cpu()
        vis_t = torch.zeros_like(target_vis, dtype=torch.float32)
        vis_t[target_vis == 1] = 1.0  # Bianco (Anomalia)
        vis_t[target_vis == self.ignore_idx] = 100.0 / 255.0  # Grigio (Void)

        target_vis_rgb = vis_t.unsqueeze(0).repeat(3, 1, 1)

        # 4. Combina
        comparison = torch.cat([img_vis, target_vis_rgb, pred_vis], dim=2)

        # 5. Log su WandB direttamante
        if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
            caption = f"{prefix}_L{layer_idx}_img_gt_PROB"
            self.logger.experiment.log({
                f"val_images/{prefix}_layer_{layer_idx}": [
                    wandb.Image(comparison, caption=caption)
                ]
            })

        if batch_idx == 0:
            filename = f"vis_{prefix}_layer{layer_idx}.png"
            save_image(comparison, filename)