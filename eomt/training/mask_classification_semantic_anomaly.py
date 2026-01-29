import io

import torch
from torch import nn
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.utils import save_image
from typing import Optional, List
from .mask_classification_loss import MaskClassificationLoss
# Add Anomaly Head definition here for clarity, or import it
import torch.nn.functional as F

# Fix unresolved reference: Import MaskClassificationSemantic
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
        print("learning rate:", lr)

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

        # self.criterion_anomalymask = MaskClassificationLoss(
        #     num_points=num_points,
        #     oversample_ratio=oversample_ratio,
        #     importance_sample_ratio=importance_sample_ratio,
        #     mask_coefficient=mask_coefficient,
        #     dice_coefficient=dice_coefficient,
        #     class_coefficient=class_coefficient,
        #     num_labels=2,  # Normal (0), Anomaly (1) + NoObject (2)
        #     no_object_coefficient=no_object_coefficient,
        #     bg_coefficient=bg_coefficient,
        # )

        num_layers = self.network.num_blocks + 1 if getattr(self.network, 'masked_attn_enabled', False) else 1

        # Override metrics initialization to use Anomaly-specific metrics (Binary Jaccard/IoU)
        # Standard MulticlassJaccardIndex initialized in super() uses num_classes=19
        # But here we evaluate Anomaly (Class 1) vs Normal (Class 0).
        # We need to re-initialize metrics for 2 classes (Binary segmentation essentially).
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

        # Freeze network, Unfreeze ONLY the internal anomaly_head
        for param in self.network.parameters():
            param.requires_grad = False

        # Assuming the unified model now has this attribute
        if hasattr(self.network, 'anomaly_head'):
            print("Unfreezing anomaly_head inside network...")
            for param in self.network.anomaly_head.parameters():
                param.requires_grad = True
        else:
            print("WARNING: 'anomaly_head' not found in network. Training might fail.")

        self.lr = 1e-4  # Maybe smaller for just the head?
        self.weight_decay = 0.05

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

        # 1. Forward Pass of Backbone (frozen)
        # We only need the valid output
        with torch.no_grad():
            outputs = self(imgs)
            # outputs is tuple: (mask_logits_list, class_logits_list, features_map)
            # We want the LAST layer output
            mask_logits = outputs[0][-1] # [B, Q, H/4, W/4] usually
            class_logits = outputs[1][-1] # [B, Q, C]
            # New feature output from EoMT_EXT
            vis_features = outputs[2] # [B, C_embed, H_grid, W_grid]

        # 2. Compute Anomaly Score using the unified model method
        # This handles Feature Extraction + MLP
        b, _, h, w = imgs.shape

        # Pass features to the anomaly head computation
        anomaly_logits = self.network.compute_anomaly_score(
            mask_logits,
            class_logits,
            (h, w),
            features=vis_features
        ) # [B, H, W, 1]

        # 4. Prepare Targets (Binary Anomaly Mask + Valid Mask)
        gt_anomaly_mask, valid_mask = self._prepare_dense_targets(targets, (h, w))

        # 5. Loss (BCE) handling ignored regions
        # Use reduction='none' to get pixel-wise loss, then mask it
        loss_pixel = F.binary_cross_entropy_with_logits(
            anomaly_logits.squeeze(-1),
            gt_anomaly_mask.float(),
            pos_weight=torch.tensor([10.0]).to(imgs.device),  # Handle class imbalance
            reduction='none'
        )

        # Apply valid mask (1.0 for valid, 0.0 for ignore)
        loss = (loss_pixel * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        self.log("train_loss_anomaly", loss)

        # Logging / Visualization (Every N steps)
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
        valid_mask = torch.ones((b, h, w), device=self.device)

        for i, t in enumerate(targets_list):
            # t["masks"] and t["labels"] could be on CPU or GPU
            if "labels" in t and "masks" in t:
                # Ensure labels are on correct device
                labels = t["labels"].to(self.device)
                masks = t["masks"].to(self.device)

                # 1. Handle Anomaly Class (Label 1)
                anomaly_indices = (labels == 1).nonzero(as_tuple=True)[0]
                if len(anomaly_indices) > 0:
                    anomaly_masks = masks[anomaly_indices]
                    if anomaly_masks.numel() > 0:
                        combined = anomaly_masks.sum(dim=0).bool()
                        gt_tensor[i] = combined.float()

                # 2. Handle Ignore/Void Class (Label 255/ignore_idx, or typically the void class)
                # We assume self.ignore_idx is what defines void.
                # In training_anomaly.py we see labels [0, 1, 255].
                # If specific label == self.ignore_idx, mask it out
                ignore_indices = (labels == self.ignore_idx).nonzero(as_tuple=True)[0]
                if len(ignore_indices) > 0:
                    ignore_masks = masks[ignore_indices]
                    if ignore_masks.numel() > 0:
                         combined_ignore = ignore_masks.sum(dim=0).bool()
                         # Set valid_mask to 0 where ignore class is present
                         valid_mask[i][combined_ignore] = 0.0

        return gt_tensor, valid_mask

    def validation_step(self, batch, batch_idx=None, log_prefix=None):
        imgs, targets = batch
        targets = self._unstack_targets(imgs, targets)
        img_sizes = [img.shape[-2:] for img in imgs]

        # DEBUG --> Print input stats to check the preprocessing
        #print(f"Input Stats - Type: {imgs.dtype}, Min: {imgs.min()}, Max: {imgs.max()}, Mean: {imgs.float().mean():.2f}")

        crops, origins = self.window_imgs_semantic(imgs)

        # Updated: Return 3 tuples, not tuple of lists
        # Wait, EoMT forward returns (mask_list, class_list) OR (mask_list, class_list, anomaly_list) ???
        # We need to update EoMT_EXT.forward signature.
        # But inside MCS training_step we used self.network(imgs) and picked outputs[0][-1].
        # Let's check EoMT_EXT.forward return value.
        outputs = self(crops)
        # Assuming EoMT_EXT returns (mask_list, class_list) as standard mask transformer
        # NEW: returns (mask_logits_per_layer, class_logits_per_layer, last_feat_map)
        mask_logits_per_layer, class_logits_per_layer, last_feat_map = outputs

        # We need anomaly scores per layer too if we want to visualize/eval per layer
        # But currently, we compute anomaly scores on the fly.

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        # Loop through layers
        # Construct anomaly_logits_per_layer manually
        anomaly_logits_per_layer = []
        for i, (masks, classes) in enumerate(zip(mask_logits_per_layer, class_logits_per_layer)):
             # Need image size of crops? crops is a tensor [B, C, H, W]
             h, w = crops.shape[-2:]

             # Only use features for the FINAL layer (where they match)
             # Intermediate layers don't have corresponding features returned by current forward()
             # So we pass None for intermediate layers
             feats_to_use = last_feat_map if i == len(mask_logits_per_layer) - 1 else None

             anomaly_logits = self.network.compute_anomaly_score(
                 masks,
                 classes,
                 (h, w),
                 features=feats_to_use
             )
             anomaly_logits_per_layer.append(anomaly_logits)

        for i, (mask_logits, class_logits, anomaly_logits) in enumerate(
                list(zip(mask_logits_per_layer, class_logits_per_layer, anomaly_logits_per_layer))
        ):
            # i is layer_idx
            layer_idx = i

            h, w = crops.shape[-2:]

            anom_logit = anomaly_logits.permute(0, 3, 1, 2)  # [B, 1, H, W]
            zeros = torch.zeros_like(anom_logit)
            crop_logits = torch.cat([zeros, anom_logit], dim=1)  # [B, 2, H, W]
            mask_logits_up = F.interpolate(mask_logits, size=(h, w), mode="bilinear", align_corners=False)

            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)
            self.update_metrics_semantic(logits, targets, i)

            # Use a slightly different batch index for visualization condition to avoid flooding logs
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
        vis_map[target_vis == 1] = 1.0  # Anomaly -> White
        vis_map[target_vis == self.ignore_idx] = 0.4  # Void -> Gray
        vis_t = vis_map.unsqueeze(0).repeat(3, 1, 1)

        # Semantic logits from baseline model (frozen class head)
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

        # Debug locale (opzionale)
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

        # Create binary classification: [Normal, Anomaly]
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
