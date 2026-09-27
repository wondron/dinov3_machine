# dino_finetune/model/dino_multitask.py
from __future__ import annotations

import logging
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino import DINOEncoderLoRA

logger = logging.getLogger(__name__)


class DINOEncoderLoRA_MultiTask(nn.Module):
    """
    共享 encoder 的多任务模型：
      - seg: 640 输入 -> DINOEncoderLoRA 输出 seg logits
      - cls: 224 输入 -> encoder.forward_features -> pooling -> cls_head
    """

    def __init__(
        self,
        seg_model: DINOEncoderLoRA,
        num_classes_cls: int,
        pool: Literal["cls_token", "mean"] = "cls_token",
        emb_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.seg_model = seg_model
        self.pool = str(pool)

        if self.pool not in {"cls_token", "mean"}:
            raise ValueError("pool 仅支持 cls_token / mean")

        if emb_dim is None:
            emb_dim = int(getattr(seg_model, "emb_dim", 0) or 0)
        if emb_dim <= 0:
            raise ValueError("emb_dim 无法确定，请显式传入 emb_dim")

        self.emb_dim = int(emb_dim)
        self.num_classes_cls = int(num_classes_cls)
        if self.num_classes_cls <= 0:
            raise ValueError("num_classes_cls 必须是大于 0 的整数")

        self.cls_head = nn.Linear(self.emb_dim, self.num_classes_cls)

        logger.info(
            "MultiTask 初始化完成：emb_dim=%d num_classes_cls=%d pool=%s",
            self.emb_dim,
            self.num_classes_cls,
            self.pool,
        )

    @property
    def encoder(self) -> nn.Module:
        """返回分割模型持有的共享 encoder，避免重复注册同一模块。"""
        return self.seg_model.encoder

    def forward_seg(self, x640: torch.Tensor) -> torch.Tensor:
        return self.seg_model(x640)

    def forward_cls(self, x224: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder.forward_features(x224)

        if self.pool == "cls_token":
            if "x_norm_clstoken" not in feats:
                raise KeyError("encoder.forward_features 缺少 x_norm_clstoken")
            emb = feats["x_norm_clstoken"]
        else:
            if "x_norm_patchtokens" not in feats:
                raise KeyError("encoder.forward_features 缺少 x_norm_patchtokens")
            emb = feats["x_norm_patchtokens"].mean(dim=1)

        emb = F.normalize(emb, p=2, dim=1)
        logits = self.cls_head(emb)
        return logits, emb

    def forward(self, x640: torch.Tensor, x224: torch.Tensor):
        seg_logits = self.forward_seg(x640)
        cls_logits, emb = self.forward_cls(x224)
        return seg_logits, cls_logits, emb

    def forward_patchtokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回 encoder.forward_features 的 x_norm_patchtokens，形状 (B, N, C)。
        这里只提供 token，不做 pooling，不做额外 normalize。
        """
        feats = self.encoder.forward_features(x)
        if "x_norm_patchtokens" not in feats:
            raise KeyError("encoder.forward_features 缺少 x_norm_patchtokens")
        return feats["x_norm_patchtokens"]

    def forward_token_grid(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 patch tokens 还原为网格特征，返回形状 (B, C, Ht, Wt)。
        Ht = Wt = int(sqrt(N))，并要求 Ht*Wt == N。
        """
        tok = self.forward_patchtokens(x)
        if tok.ndim != 3:
            raise ValueError(f"patch token 维度错误，期望 (B,N,C)，实际为 {tuple(tok.shape)}")

        bsz, num_tokens, channels = tok.shape
        ht = int(math.sqrt(num_tokens))
        wt = ht
        if ht * wt != num_tokens:
            raise ValueError(f"patch token 数量 N={num_tokens} 不能还原为平方网格")

        return tok.transpose(1, 2).contiguous().view(bsz, channels, ht, wt)
