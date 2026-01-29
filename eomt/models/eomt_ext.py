# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .scale_block import ScaleBlock

class PixelAnomalyHead(nn.Module):
    def __init__(self, input_dim=4, feature_dim=0, hidden_dim=64, dropout=0.1):
        super(PixelAnomalyHead, self).__init__()

        self.feature_dim = feature_dim

        if feature_dim > 0:
            self.feature_proj = nn.Sequential(
                nn.Conv2d(feature_dim, 32, kernel_size=1),
                nn.ReLU()
            )
            input_dim += 32 # 4 stats + 32 visual features

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, features=None):
        b, h, w, _ = x.shape

        if self.feature_dim > 0 and features is not None:
            vis_feat = self.feature_proj(features)

            if vis_feat.shape[-2:] != (h, w):
                vis_feat = F.interpolate(vis_feat, size=(h, w), mode="bilinear", align_corners=False)

            vis_feat = vis_feat.permute(0, 2, 3, 1)
            x = torch.cat([x, vis_feat], dim=-1)

        b, h, w, c = x.shape
        x_flat = x.reshape(-1, c)
        out = self.mlp(x_flat)   
        return out.reshape(b, h, w, 1)

class EoMT_EXT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q = nn.Embedding(num_q, self.encoder.backbone.embed_dim)
        print("EoMT_EXT: num_classes =", num_classes)

        self.class_head = nn.Linear(self.encoder.backbone.embed_dim, num_classes + 1)
        self.anomaly_head = PixelAnomalyHead(
            input_dim=4,
            feature_dim=self.encoder.backbone.embed_dim
        )

        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
        )

        patch_size = encoder.backbone.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.encoder.backbone.embed_dim) for _ in range(num_upscale)],
        )


    def _predict(self, x: torch.Tensor):
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)
        patch_tokens = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]

        # Reshape to [B, C, H, W]
        b, n, c = patch_tokens.shape
        h, w = self.encoder.backbone.patch_embed.grid_size
        feat_map = patch_tokens.transpose(1, 2).reshape(b, c, h, w)

        upscaled_feats = self.upscale(feat_map)

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), upscaled_feats
        )

        return mask_logits, class_logits, feat_map

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask, prob):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[
                :, : self.num_q, self.num_q + self.encoder.backbone.num_prefix_tokens :
            ][random_queries] = True

        return attn_mask

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rope: Optional[torch.Tensor],
    ):
        if rope is not None:
            if mask is not None:
                mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
            return module(x, mask, rope)[0]

        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))

        return x

    def _attn_mask(self, x: torch.Tensor, mask_logits: torch.Tensor, i: int):
        attn_mask = torch.ones(
            x.shape[0],
            x.shape[1],
            x.shape[1],
            dtype=torch.bool,
            device=x.device,
        )
        interpolated = F.interpolate(
            mask_logits,
            self.encoder.backbone.patch_embed.grid_size,
            mode="bilinear",
        )
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        attn_mask[
            :,
            : self.num_q,
            self.num_q + self.encoder.backbone.num_prefix_tokens :,
        ] = (
            interpolated > 0
        )
        attn_mask = self._disable_attn_mask(
            attn_mask,
            self.attn_mask_probs[
                i - len(self.encoder.backbone.blocks) + self.num_blocks
            ],
        )
        return attn_mask

    def forward(self, x: torch.Tensor):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        rope = None
        if hasattr(self.encoder.backbone, "rope_embeddings"):
            rope = self.encoder.backbone.rope_embeddings(x)

        x = self.encoder.backbone.patch_embed(x)

        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []
        # We need features from the last layer
        last_feat_map = None

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if (
                self.masked_attn_enabled
                and i >= len(self.encoder.backbone.blocks) - self.num_blocks
            ):
                mask_logits, class_logits, _ = self._predict(self.encoder.backbone.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                attn_mask = self._attn_mask(x, mask_logits, i)

            if hasattr(block, "attn"):
                attn = block.attn
            else:
                attn = block.attention
            attn_out = self._attn(attn, block.norm1(x), attn_mask, rope=rope)
            if hasattr(block, "ls1"):
                x = x + block.ls1(attn_out)
            elif hasattr(block, "layer_scale1"):
                x = x + block.layer_scale1(attn_out)

            mlp_out = block.mlp(block.norm2(x))
            if hasattr(block, "ls2"):
                x = x + block.ls2(mlp_out)
            elif hasattr(block, "layer_scale2"):
                x = x + block.layer_scale2(mlp_out)

        # Final prediction
        mask_logits, class_logits, last_feat_map = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        # Return (mask_list, class_list, last_feat_map)
        return (
            mask_logits_per_layer,
            class_logits_per_layer,
            last_feat_map
        )

    def compute_anomaly_score(self, mask_logits, class_logits, img_size, features=None):
        """
        Computes the pixel-wise anomaly score using the internal anomaly_head.
        Args:
            mask_logits: [B, Q, H_feat, W_feat]
            class_logits: [B, Q, C]
            img_size: tuple (H, W) target resolution
            features: [B, C_embed, H_grid, W_grid] Visual features from backbone
        """
        mask_logits_up = F.interpolate(mask_logits, size=img_size, mode="bilinear", align_corners=False)
        mask_probs = mask_logits_up.sigmoid() # [B, Q, H, W]

        class_probs = F.softmax(class_logits, dim=-1) # [B, Q, C]

        # Exclude "No Object" / "Void" class if it exists (usually last index)
        valid_class_probs = class_probs[..., :-1]
        semantic_map = torch.einsum("bqhw, bqc -> bchw", mask_probs, valid_class_probs)
        max_prob, _ = semantic_map.max(dim=1)

        semantic_map_clamped = torch.clamp(semantic_map, min=1e-7)
        entropy = -(semantic_map_clamped * torch.log(semantic_map_clamped)).sum(dim=1)

        energy_proxy = semantic_map.sum(dim=1)

        msp = 1.0 - max_prob

        # Stack Features: [B, H, W, 4]
        stat_features = torch.stack([max_prob, entropy, energy_proxy, msp], dim=-1)

        return self.anomaly_head(stat_features, features)
