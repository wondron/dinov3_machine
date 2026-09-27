import math
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRA

logger = logging.getLogger(__name__)


class DINOForClassification(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        emb_dim: int,
        num_classes: int,
        r: int = 3,
        use_lora: bool = False,
        pool: str = "cls_token",
    ) -> None:
        super().__init__()

        if emb_dim is None:
            raise ValueError("model_cls.emb_dim 不能为空")
        if num_classes is None:
            raise ValueError("num_classes 不能为空")

        self.emb_dim = int(emb_dim)
        self.num_classes = int(num_classes)
        self.use_lora = bool(use_lora)
        self.pool = str(pool)

        if self.pool not in {"cls_token", "mean"}:
            raise ValueError("model_cls.pool 仅支持：cls_token / mean")

        if self.use_lora:
            if r is None:
                raise ValueError("启用 LoRA 时必须指定 rank_r")
            if int(r) <= 0:
                raise ValueError("rank_r 必须是大于 0 的整数")

        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        if self.use_lora:
            self._inject_lora(int(r))

        self.head = nn.Linear(self.emb_dim, self.num_classes)

        enc_dim = getattr(self.encoder, "num_features", None)
        if enc_dim is not None and int(enc_dim) != self.emb_dim:
            logger.warning(
                "emb_dim(%d) 与 encoder.num_features(%d) 不一致，可能导致分类头维度错误",
                self.emb_dim,
                int(enc_dim),
            )

    @staticmethod
    def _create_lora_layer(dim: int, r: int) -> tuple[nn.Linear, nn.Linear]:
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def _inject_lora(self, r: int) -> None:
        if not hasattr(self.encoder, "blocks"):
            raise ValueError("当前 encoder 不支持 LoRA 注入（缺少 blocks 属性）")

        self.lora_layers = list(range(len(self.encoder.blocks)))
        self.w_a = []
        self.w_b = []

        for i, block in enumerate(self.encoder.blocks):
            if i not in self.lora_layers:
                continue
            w_qkv_linear = block.attn.qkv
            dim = w_qkv_linear.in_features

            w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
            w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

            self.w_a.extend([w_a_linear_q, w_a_linear_v])
            self.w_b.extend([w_b_linear_q, w_b_linear_v])

            block.attn.qkv = LoRA(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self._reset_lora_parameters()

    @staticmethod
    def _get_feature(features: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in features:
            raise KeyError(f"encoder.forward_features 输出中缺少 {key}")
        return features[key]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder.forward_features(x)

        if self.pool == "cls_token":
            embedding = self._get_feature(feats, "x_norm_clstoken")
        else:
            patch_tokens = self._get_feature(feats, "x_norm_patchtokens")
            embedding = patch_tokens.mean(dim=1)

        embedding = F.normalize(embedding, p=2, dim=1)
        cls_logits = self.head(embedding)
        return cls_logits, embedding

    @classmethod
    def from_config(
        cls,
        encoder: nn.Module,
        cfg: dict[str, Any],
        num_classes: int | None = None,
    ) -> "DINOForClassification":
        if "model_cls" not in cfg or cfg["model_cls"] is None:
            raise ValueError("config 中缺少 model_cls")

        model_cfg = cfg["model_cls"]
        if "emb_dim" not in model_cfg:
            raise ValueError("config.model_cls.emb_dim 为必填项")

        emb_dim = int(model_cfg["emb_dim"])
        pool = str(model_cfg.get("pool", "cls_token"))

        if "pool" not in model_cfg:
            logger.warning("未配置 model_cls.pool，默认使用 cls_token")

        if num_classes is None:
            if "num_classes" not in model_cfg or model_cfg["num_classes"] is None:
                raise ValueError("num_classes 未显式传入，且 config.model_cls.num_classes 缺失")
            num_classes = int(model_cfg["num_classes"])
            logger.info("num_classes 使用配置值：%d", num_classes)
        else:
            logger.info("num_classes 使用外部传入值：%d", int(num_classes))

        train_cfg = cfg.get("trainparams", {}) or {}
        use_lora = bool(train_cfg.get("use_lora", False))
        r = train_cfg.get("rank_r", 3)

        if use_lora and r is None:
            raise ValueError("启用 LoRA 时必须指定 trainparams.rank_r")

        return cls(
            encoder=encoder,
            emb_dim=emb_dim,
            num_classes=int(num_classes),
            r=int(r),
            use_lora=use_lora,
            pool=pool,
        )
