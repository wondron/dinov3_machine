# dino_finetune/losses.py
"""各任务 loss（设计文档 4.3 / 4.5 / 第 5 节）。"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LOSS_TERMS


def masked_loss(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """只在有效样本上计算（按有效样本数平均）；batch 内没有有效样本时返回 0。"""
    if valid.any():
        return fn(pred[valid], target[valid])
    return torch.zeros((), device=pred.device)


def compute_pos_weight(targets: np.ndarray, known: np.ndarray, max_weight: float) -> torch.Tensor:
    """
    多标签 pos_weight = 负样本数 / 正样本数，限制在 [1/max_weight, max_weight]；
    训练集里全是正样本或没有正样本的类别无法平衡，取 1。
    """
    targets = targets[known]
    pos = targets.sum(axis=0)
    neg = len(targets) - pos
    balanced = (pos > 0) & (neg > 0)
    weight = np.where(balanced, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(np.clip(weight, 1.0 / max_weight, max_weight), dtype=torch.float32)


class MultiLabelLoss(nn.Module):
    """多标签 BCE / focal loss，可带 pos_weight（容器、附件头）。"""

    def __init__(self, pos_weight: torch.Tensor | None = None, kind: str = "bce", gamma: float = 2.0) -> None:
        super().__init__()
        self.kind = kind
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", pos_weight, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = logits.float(), targets.float()
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        if self.kind == "focal":
            p = torch.sigmoid(logits)
            p_t = p * targets + (1 - p) * (1 - targets)
            loss = loss * (1 - p_t).pow(self.gamma)
        return loss.mean()


class SupConLoss(nn.Module):
    """监督对比损失（Khosla et al., 2020）：同一 cavity_group 互为正样本；batch 内不足 2 个组时没有负样本，返回 0。"""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if feats.shape[0] < 2 or torch.unique(labels).numel() < 2:
            return torch.zeros((), device=feats.device)
        with torch.autocast(device_type=feats.device.type, enabled=False):
            feats = F.normalize(feats.float(), dim=-1)
            self_mask = torch.eye(len(feats), dtype=torch.bool, device=feats.device)
            logits = (feats @ feats.T / self.temperature).masked_fill(self_mask, float("-inf"))
            log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            pos = (labels[:, None] == labels[None, :]) & ~self_mask
            pos_count = pos.sum(1)
            anchors = pos_count > 0
            if not anchors.any():
                return torch.zeros((), device=feats.device)
            mean_log_prob_pos = log_prob.masked_fill(~pos, 0.0).sum(1)[anchors] / pos_count[anchors]
            return -mean_log_prob_pos.mean()


class ArcFaceLoss(nn.Module):
    """ArcFace（additive angular margin）：类别为 cavity_group，类中心是可训练参数。"""

    def __init__(self, dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.3) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=feats.device.type, enabled=False):
            cos = F.linear(F.normalize(feats.float(), dim=-1), F.normalize(self.weight.float(), dim=-1))
            cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
            sin = torch.sqrt(1.0 - cos * cos)
            phi = cos * self.cos_m - sin * self.sin_m
            phi = torch.where(cos > self.th, phi, cos - self.mm)
            one_hot = F.one_hot(labels, cos.shape[1]).bool()
            return F.cross_entropy(torch.where(one_hot, phi, cos) * self.scale, labels)


def rack_loss(logits: torch.Tensor, target: torch.Tensor, rack_count: torch.Tensor, smooth: float = 0.0) -> torch.Tensor:
    """
    logits: [B, NUM_POS]，已按层数掩码（无效类为 -inf）；target: [B]，取值 0..rack_count。
    smooth > 0 时做邻层软标签，只在导轨层之间平滑；第 0 类（底板层）不参与。
    """
    logits = logits.float()
    if smooth <= 0:
        return F.cross_entropy(logits, target)
    soft = F.one_hot(target, logits.size(1)).float() * (1 - smooth)
    for d in (-1, 1):
        nb = target + d
        ok = (target >= 1) & (nb >= 1) & (nb <= rack_count)   # 只在导轨层之间平滑
        idx = ok.nonzero(as_tuple=True)[0]
        soft[idx, nb[idx]] += smooth / 2
    soft = soft / soft.sum(-1, keepdim=True)
    logp = F.log_softmax(logits, -1).masked_fill(torch.isinf(logits), 0)
    return -(soft * logp).sum(-1).mean()


class MultiTaskLoss(nn.Module):
    """
    设计文档第 5 节的加权多任务 loss：
      Is-Oven BCE（全部）、Proj SupCon / ArcFace（is_oven=1）、Food BCE（有标注的样本）、
      Container / Accessory 多标签 BCE（有标注的样本）、层位 CE（is_oven=1 且层位不为 null，掩码用真实型号的层数）。
    返回 (加权总 loss, 各项未加权 loss)。
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        container_loss: MultiLabelLoss,
        accessory_loss: MultiLabelLoss,
        metric_loss: nn.Module,
        rack_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.weights = {term: float(weights[term]) for term in LOSS_TERMS}
        self.container_loss = container_loss
        self.accessory_loss = accessory_loss
        self.metric_loss = metric_loss
        self.rack_smoothing = float(rack_smoothing)

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        *,
        container_pos_weight: torch.Tensor | None,
        accessory_pos_weight: torch.Tensor | None,
        proj_dim: int,
        num_groups: int,
    ) -> "MultiTaskLoss":
        loss_cfg = cfg["loss"]
        if loss_cfg["metric"] == "arcface":
            metric: nn.Module = ArcFaceLoss(proj_dim, num_groups, loss_cfg["arcface_scale"], loss_cfg["arcface_margin"])
        else:
            metric = SupConLoss(loss_cfg["supcon_temperature"])
        return cls(
            loss_cfg["weights"],
            container_loss=MultiLabelLoss(container_pos_weight, loss_cfg["multilabel"], loss_cfg["focal_gamma"]),
            accessory_loss=MultiLabelLoss(accessory_pos_weight, loss_cfg["multilabel"], loss_cfg["focal_gamma"]),
            metric_loss=metric,
            rack_smoothing=loss_cfg["rack_smoothing"],
        )

    def _rack(self, out: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], valid: torch.Tensor) -> torch.Tensor:
        if not valid.any():
            return torch.zeros((), device=valid.device)
        return rack_loss(out["rack"][valid], batch["rack_level"][valid], batch["rack_count"][valid], self.rack_smoothing)

    def forward(
        self,
        out: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        is_oven = batch["is_oven"] > 0.5
        has_level = is_oven & batch["rack_known"]           # 一体机内部且非空腔
        bce = F.binary_cross_entropy_with_logits
        terms = {
            "is_oven": bce(out["is_oven"].float(), batch["is_oven"].float()),
            "proj": masked_loss(self.metric_loss, out["proj"], batch["group_id"], is_oven),
            "food": masked_loss(lambda p, t: bce(p.float(), t.float()), out["food"], batch["food"], batch["food_known"]),
            "container": masked_loss(self.container_loss, out["container"], batch["container"], batch["container_known"]),
            "accessory": masked_loss(self.accessory_loss, out["accessory"], batch["accessory"], batch["accessory_known"]),
            "rack": self._rack(out, batch, has_level),
        }
        total = sum(self.weights[term] * value for term, value in terms.items())
        return total, terms
