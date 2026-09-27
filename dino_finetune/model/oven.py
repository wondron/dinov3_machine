# dino_finetune/model/oven.py
"""一体机多任务视觉识别模型（设计文档第 2、4 节）。"""
from __future__ import annotations

import contextlib
import hashlib
import logging
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRA, inject_lora

logger = logging.getLogger(__name__)

POOLED_HEADS = ("is_oven", "food", "container", "accessory")
# 推理 / ONNX 导出统一的输出（顺序即 ONNX 输出顺序）
OUTPUT_KEYS = ("is_oven_prob", "food_prob", "container_prob", "accessory_prob", "rack_raw", "proj", "cls")


def inference_outputs(out: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    把 forward 输出整理成推理输出：四个分类头的 sigmoid 概率、未掩码的层位 logits
    （推理时型号由检索得到，层数掩码在后处理里做）、proj 与 cls 特征。
    """
    return {
        "is_oven_prob": torch.sigmoid(out["is_oven"].float()),
        "food_prob": torch.sigmoid(out["food"].float()),
        "container_prob": torch.sigmoid(out["container"].float()),
        "accessory_prob": torch.sigmoid(out["accessory"].float()),
        "rack_raw": out["rack_raw"].float(),
        "proj": out["proj"].float(),
        "cls": out["cls"].float(),
    }


class AttnPool(nn.Module):
    """一个可学习的 query 对 patch tokens 做 cross-attention，输出和 CLS 拼接：[B, 2D]。"""

    def __init__(self, dim: int = 1024, heads: int = 8) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:  # cls: [B, D]  patches: [B, N, D]
        out, _ = self.attn(self.q.expand(len(patches), -1, -1), patches, patches, need_weights=False)
        return torch.cat([cls, out.squeeze(1)], -1)


class PooledHead(nn.Module):
    """独立 attention pooling + MLP：Is-Oven / Food / Container / Accessory。"""

    def __init__(self, dim: int, attn_heads: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.pool = AttnPool(dim, attn_heads)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(cls, patches))


class RackHead(nn.Module):
    """层位头：输出 [B, max_rack + 1] 的未掩码 logits，0 = 底板层，k = 第 k 层导轨（基线不加型号条件）。"""

    def __init__(self, dim: int, attn_heads: int, hidden: int, num_pos: int, dropout: float) -> None:
        super().__init__()
        self.pool = AttnPool(dim, attn_heads)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.level = nn.Linear(hidden, num_pos)

    def forward(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        return self.level(self.mlp(self.pool(cls, patches)))


class ProjHead(nn.Module):
    """型号检索特征：MLP 1024 → 512 → 256，输出 L2 归一化。只用 CLS，不做 attention pooling。"""

    def __init__(self, dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(cls).float(), dim=-1)


def build_valid_mask(rack_count: torch.Tensor, floor_usable: torch.Tensor, num_pos: int) -> torch.Tensor:
    """rack_count: [B] long，floor_usable: [B] bool → [B, num_pos] bool。"""
    k = torch.arange(num_pos, device=rack_count.device)[None, :]
    valid = k <= rack_count[:, None]                     # 屏蔽超过该型号层数的导轨层
    valid[:, 0] &= floor_usable                          # 不能直接放底板的机型屏蔽第 0 类
    return valid                                         # 非一体机样本填 rack_count=max_rack、floor_usable=True


class OvenMultiTaskModel(nn.Module):
    """
    DINOv3 → CLS token + patch tokens
      ├─ Is-Oven / Food / Container / Accessory：各自独立的 attention pooling + MLP
      ├─ Rack：attention pooling + MLP，按真实（或检索到的）型号的 rack_count / floor_usable 掩码
      └─ Proj：CLS → MLP → L2 归一化特征，用于检索特征库
    骨干预训练权重冻结，用 LoRA（Q、V 低秩增量）微调；use_lora=False 时骨干完全冻结。骨干始终处于 eval 模式。
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        num_container: int,
        num_accessory: int,
        max_rack: int = 8,
        attn_heads: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        proj_hidden_dim: int = 512,
        proj_dim: int = 256,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_last_n_blocks: int = 0,
    ) -> None:
        super().__init__()
        dim = int(getattr(encoder, "num_features", 0) or 0)
        if dim <= 0:
            raise RuntimeError("无法从 encoder 获取 num_features，请检查 DINO 模型")
        self.encoder = encoder
        self.dim = dim
        self.max_rack = int(max_rack)
        self.num_pos = self.max_rack + 1

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.lora_blocks = inject_lora(self.encoder, lora_rank, lora_last_n_blocks) if use_lora else 0
        self.backbone_trainable = any(p.requires_grad for p in self.encoder.parameters())

        self.heads = nn.ModuleDict(
            {
                "is_oven": PooledHead(dim, attn_heads, hidden_dim, 1, dropout),
                "food": PooledHead(dim, attn_heads, hidden_dim, 1, dropout),
                "container": PooledHead(dim, attn_heads, hidden_dim, num_container, dropout),
                "accessory": PooledHead(dim, attn_heads, hidden_dim, num_accessory, dropout),
                "rack": RackHead(dim, attn_heads, hidden_dim, self.num_pos, dropout),
                "proj": ProjHead(dim, proj_hidden_dim, proj_dim),
            }
        )

    @classmethod
    def from_config(
        cls,
        encoder: nn.Module,
        cfg: Mapping[str, Any],
        *,
        num_container: int,
        num_accessory: int,
    ) -> "OvenMultiTaskModel":
        model_cfg = cfg["model"]
        patch = int(getattr(encoder, "patch_size", 16))
        img_h, img_w = cfg["input"]["img_dim"]
        if img_h % patch or img_w % patch:
            raise ValueError(f"input.img_dim={cfg['input']['img_dim']} 必须能被 patch_size={patch} 整除")
        return cls(
            encoder,
            num_container=num_container,
            num_accessory=num_accessory,
            max_rack=model_cfg["max_rack"],
            attn_heads=model_cfg["attn_pool_heads"],
            hidden_dim=model_cfg["head_hidden_dim"],
            dropout=model_cfg["head_dropout"],
            proj_hidden_dim=model_cfg["proj_hidden_dim"],
            proj_dim=model_cfg["proj_dim"],
            use_lora=model_cfg["use_lora"],
            lora_rank=model_cfg["lora_rank"],
            lora_last_n_blocks=model_cfg["lora_last_n_blocks"],
        )

    def train(self, mode: bool = True) -> "OvenMultiTaskModel":
        super().train(mode)
        self.encoder.eval()
        return self

    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backbone_trainable:
            feats = self.encoder.forward_features(images)
        else:
            with torch.no_grad():
                feats = self.encoder.forward_features(images)
        return feats["x_norm_clstoken"], feats["x_norm_patchtokens"]

    def forward(
        self,
        images: torch.Tensor,
        rack_count: torch.Tensor | None = None,
        floor_usable: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        返回各头 logits：is_oven / food [B]，container / accessory [B, C]，
        rack_raw [B, max_rack+1]（未掩码），rack（给了 rack_count 时按层数掩码，无效类为 -inf），
        proj [B, proj_dim]（L2 归一化），cls [B, D]（骨干 CLS，用于和 Proj 特征对比检索效果）。
        """
        cls, patches = self.extract(images)
        out = {name: self.heads[name](cls, patches) for name in POOLED_HEADS}
        out["is_oven"] = out["is_oven"].squeeze(-1)
        out["food"] = out["food"].squeeze(-1)
        out["rack_raw"] = self.heads["rack"](cls, patches)
        if rack_count is not None:
            if floor_usable is None:
                floor_usable = torch.ones_like(rack_count, dtype=torch.bool)
            valid = build_valid_mask(rack_count, floor_usable, self.num_pos)
            out["rack"] = out["rack_raw"].masked_fill(~valid, float("-inf"))
        out["proj"] = self.heads["proj"](cls)
        out["cls"] = cls
        return out

    # =========================
    # LoRA
    # =========================
    def _lora_modules(self) -> list[LoRA]:
        return [m for m in self.encoder.modules() if isinstance(m, LoRA)]

    @contextlib.contextmanager
    def lora_disabled(self) -> Iterator[None]:
        """临时关掉 LoRA，骨干等价于原始预训练的 DINOv3（用于和原始 DINOv3 特征对比）。"""
        modules = self._lora_modules()
        for m in modules:
            m.enabled = False
        try:
            yield
        finally:
            for m in modules:
                m.enabled = True

    def merge_lora(self) -> int:
        """把 LoRA 增量合并进 qkv 权重并去掉 LoRA 结构（导出用，输出不变），返回合并的 block 数。"""
        merged = 0
        for block in self.encoder.blocks:
            if isinstance(block.attn.qkv, LoRA):
                block.attn.qkv = block.attn.qkv.merged()
                merged += 1
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.backbone_trainable = False
        return merged

    # =========================
    # 参数分组与保存
    # =========================
    def head_parameters(self) -> dict[str, list[nn.Parameter]]:
        """按头分组的可训练参数（LoRA 单独一组），用于记录各头的梯度范数。"""
        groups = {name: [p for p in head.parameters() if p.requires_grad] for name, head in self.heads.items()}
        lora = [p for p in self.encoder.parameters() if p.requires_grad]
        if lora:
            groups["lora"] = lora
        return groups

    def _trainable_names(self) -> set[str]:
        return {name for name, p in self.named_parameters() if p.requires_grad}

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """只保存训练过的权重（各头 + LoRA），冻结的骨干由预训练权重重建。"""
        trainable = self._trainable_names()
        return {
            k: v.detach().cpu()
            for k, v in self.state_dict().items()
            if not k.startswith("encoder.") or k in trainable
        }

    def load_trainable_state_dict(
        self,
        state: Mapping[str, torch.Tensor],
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        """加载 trainable_state_dict；冻结骨干的权重本来就不在里面，不算缺失。"""
        missing, unexpected = self.load_state_dict(dict(state), strict=False)
        trainable = self._trainable_names()
        missing = [k for k in missing if not k.startswith("encoder.") or k in trainable]
        if strict and (missing or unexpected):
            raise RuntimeError(f"checkpoint 与模型结构不一致：缺失={missing[:10]} 多余={unexpected[:10]}")
        return missing, unexpected

    def gallery_fingerprint(self, weight_name: str) -> str:
        """特征库绑定的版本号：由预训练权重名、Proj Head 和 LoRA 的权重决定（需在 merge_lora 之前计算）。"""
        digest = hashlib.sha1(str(weight_name).encode("utf-8"))
        for key, tensor in sorted(self.trainable_state_dict().items()):
            if key.startswith(("heads.proj.", "encoder.")):
                digest.update(key.encode("utf-8"))
                digest.update(tensor.float().numpy().tobytes())
        return digest.hexdigest()[:16]
