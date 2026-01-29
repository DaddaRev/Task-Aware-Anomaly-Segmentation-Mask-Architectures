import io

import torch
from torch import nn
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.utils import save_image
from typing import Optional, List
from .mask_classification_loss import MaskClassificationLoss
import torch.nn.functional as F

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
            weight_decay: float = 0.05,
            oversample_ratio: float = 3.0,
            importance_sample_ratio: float = 0.75,
            poly_power: float = 0.9,
            warmup_steps: Optional[List[int]] = None,
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

        if warmup_steps is None:
            warmup_steps = [500, 1000]

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
            **kwargs
        )

        self.criterion_anomalymask = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=2,  # Normal (0), Anomaly (1) + NoObject (2)
            no_object_coefficient=no_object_coefficient,
            bg_coefficient=bg_coefficient,
        )

        num_layers = self.network.num_blocks + 1 if getattr(self.network, 'masked_attn_enabled', False) else 1
        self.metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=2,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,
                )
                for _ in range(num_layers)
            ]
        )

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for param in self.network.parameters():
            param.requires_grad = False

        if hasattr(self.network, 'anomaly_head'):
            print("Unfreezing anomaly_head inside network...")
            for param in self.network.anomaly_head.parameters():
                param.requires_grad = True
        else:
            print("WARNING: 'anomaly_head' not found in network. Training might fail.")

        self.lr = 1e-4  
        self.weight_decay = 0.05

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

        # 1. Forward Pass of Backbone (frozen)
        with torch.no_grad():
            outputs = self(imgs)
            mask_logits = outputs[0][-1]
            class_logits = outputs[1][-1] 

        # 2. Compute Anomaly Score using the unified model method
        b, _, h, w = imgs.shape

        anomaly_logits = self.network.compute_anomaly_score(mask_logits, class_logits, (h, w))  # [B, H, W, 1]

        # 4. Prepare Targets (Binary Anomaly Mask)
        gt_anomaly_mask = self._prepare_dense_targets(targets, (h, w))

        # 5. Loss (BCE)
        loss = F.binary_cross_entropy_with_logits(
            anomaly_logits.squeeze(-1),
            gt_anomaly_mask.float(),
            pos_weight=torch.tensor([10.0]).to(imgs.device)  # Handle class imbalance
        )

        self.log("train_loss_anomaly", loss)

        if batch_idx % 100 == 0:
             self.plot_semantic(
                 imgs[0],
                 gt_anomaly_mask[0].cpu().numpy(),
                 anomaly_logits[0].sigmoid().detach().cpu().numpy(),
                 "train", # log_prefix
                 0,       # block_idx
                 batch_idx # batch_idx
             )

        return loss

    def _prepare_dense_targets(self, targets_list, shape):
        b, h, w = len(targets_list), shape[0], shape[1]
        gt_tensor = torch.zeros((b, h, w), device=self.device)

        for i, t in enumerate(targets_list):
            if "labels" in t and "masks" in t:
                # Ensure labels are on correct device
                labels = t["labels"].to(self.device)
                anomaly_indices = (labels == 1).nonzero(as_tuple=True)[0]

                if len(anomaly_indices) > 0:
                    anomaly_masks = t["masks"][anomaly_indices].to(self.device)  # [K, H, W]
                    if anomaly_masks.numel() > 0:
                        combined = anomaly_masks.sum(dim=0).bool()
                        gt_tensor[i] = combined.float()

        return gt_tensor

    def validation_step(self, batch, batch_idx=None, log_prefix=None):
        imgs, targets = batch
        targets = self._unstack_targets(imgs, targets)
        img_sizes = [img.shape[-2:] for img in imgs]

        crops, origins = self.window_imgs_semantic(imgs)


        outputs = self(crops)
        mask_logits_per_layer, class_logits_per_layer = outputs
        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)


        anomaly_logits_per_layer = []
        for masks, classes in zip(mask_logits_per_layer, class_logits_per_layer):
            h, w = crops.shape[-2:]
            anomaly_logits = self.network.compute_anomaly_score(masks, classes, (h, w))
            anomaly_logits_per_layer.append(anomaly_logits)

        for i, (mask_logits, class_logits, anomaly_logits) in enumerate(
                list(zip(mask_logits_per_layer, class_logits_per_layer, anomaly_logits_per_layer))
        ):
            layer_idx = i

            h, w = crops.shape[-2:]

            anom_logit = anomaly_logits.permute(0, 3, 1, 2)  
            zeros = torch.zeros_like(anom_logit)
            crop_logits = torch.cat([zeros, anom_logit], dim=1)  

            mask_logits_up = F.interpolate(mask_logits, size=(h, w), mode="bilinear", align_corners=False)

            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)
            self.update_metrics_semantic(logits, targets, i)

            if batch_idx in [0, 8, 16] and layer_idx == 0:
                self.plot_semantic_new(
                    imgs[0],
                    targets[0],
                    logits[0],
                    self._compute_baseline_msp(
                        mask_logits_up[0:1],  
                        class_logits[0:1],  
                        [origins[0]],  
                        [img_sizes[0]]  
                    )[0],
                    log_prefix,
                    i,
                    batch_idx
                )

    @torch.compiler.disable
    def plot_semantic_new(self, img, target, logits, fourth_panel_data, prefix, layer_idx, batch_idx):
        import wandb

        # Immagine Input (Raw) - Ensure [C, H, W] in range 0-1
        if img.dtype == torch.uint8:
            img_vis = img.float() / 255.0
        else:
            img_vis = img.clone()
            if img_vis.max() > 1.1:
                img_vis = img_vis / 255.0
        img_vis = torch.clamp(img_vis, 0, 1).cpu()

        eps = 1e-6
        if logits.dim() == 3 and logits.shape[0] == 2:
            score_anomaly = torch.clamp(logits[1], 0, 1).cpu()
            score_bg = torch.clamp(logits[0], 0, 1).cpu()
            pred_vis = torch.stack([score_anomaly, score_bg, torch.zeros_like(score_anomaly)], dim=0)
        else:
            prob = torch.sigmoid(logits[0]) if logits.dim() == 3 else torch.sigmoid(logits)
            pred_vis = prob.unsqueeze(0).repeat(3, 1, 1).cpu()

        pred_vis = torch.clamp(pred_vis, 0, 1)

        # Ground Truth - Anomaly=1 -> White, Background=0 -> Black, Void=255 -> Gray
        target_vis = target.clone().cpu().float()
        vis_map = torch.zeros_like(target_vis, dtype=torch.float32)
        vis_map[target_vis == 1] = 1.0 
        vis_map[target_vis == self.ignore_idx] = 0.4  
        vis_t = vis_map.unsqueeze(0).repeat(3, 1, 1)

        fourth_vis = None
        if fourth_panel_data.dim() == 3 and fourth_panel_data.shape[0] > 2:
            sem_indices = torch.argmax(fourth_panel_data, dim=0).cpu()  # [H, W]
            fourth_vis = self._apply_colormap(sem_indices)  # [3, H, W]

        elif fourth_panel_data.dim() == 3 and fourth_panel_data.shape[0] == 2:
            baseline_val = fourth_panel_data[1]
            fourth_vis = baseline_val.unsqueeze(0).repeat(3, 1, 1).cpu()

        else:
            fourth_vis = torch.sigmoid(fourth_panel_data).cpu().repeat(3, 1, 1)

        fourth_vis = torch.clamp(fourth_vis, 0, 1)

        # Combination: [Input | Ground Truth | Anomaly Head | Semantic Class]
        comparison = torch.cat([img_vis, vis_t, pred_vis, fourth_vis], dim=2)

        try:
            if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
                from torchvision.transforms.functional import to_pil_image
                caption = f"{prefix}_L{layer_idx}_Input_GT_AnomalyHead_SemanticClass"
                # Convert to PIL Image to ensure robust serialization to WandB
                pil_comparison = to_pil_image(comparison.cpu())
                self.logger.experiment.log({
                    f"val_images/{prefix}_layer_{layer_idx}": [
                        wandb.Image(pil_comparison, caption=caption)
                    ]
                })
        except Exception as e:
            print(f"Warning: Could not log to wandb: {e}")

        if batch_idx == 0 and layer_idx == 0:
            save_image(comparison, f"debug_vis_{prefix}.png")

    @torch.compiler.disable
    def _apply_colormap(self, tensor_indices):
        """
        Converts a 2D tensor of class indices [H, W] into a 3D RGB image tensor [3, H, W]
        using the 'tab20' colormap from Matplotlib.
        """
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap('tab20')
        num_colors = 20

        palette = torch.tensor(
            [cmap(i)[:3] for i in range(num_colors)], dtype=torch.float32
        ).to(tensor_indices.device)

        colored_img = palette[tensor_indices.long().clamp(0, num_colors - 1)]
        return colored_img.permute(2, 0, 1)

    def _compute_baseline_msp(self, mask_logits, class_logits, origins, img_sizes):
        """
            Compute baseline anomaly detection using Maximum Softmax Probability (MSP)
            from frozen class_head predictions.

            MSP approach: Anomaly = 1 - max_prob(semantic_classes)
            Lower confidence on semantic classes → Higher anomaly score
            """
        semantic_probs = F.softmax(class_logits, dim=-1)

        max_semantic_prob, _ = semantic_probs[..., :-1].max(dim=-1)  # [B, Q]
        baseline_probs = torch.stack([
            max_semantic_prob, 
            1.0 - max_semantic_prob 
        ], dim=-1)  

        # Combine with masks same as anomaly head
        crop_logits = torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            baseline_probs
        )

        logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)
        return logits
