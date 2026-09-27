# dino_finetune/device.py
"""设备配置表（Device Profile）与检索式型号识别的特征库（设计文档 4.3 / 4.4）。"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

UNKNOWN_DEVICE = "未知型号"


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    rack_count: int
    floor_usable: bool
    cavity_group: str
    accessories: tuple[str, ...] | None  # None 表示尚未配置：推理时不做附件屏蔽

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["accessories"] = None if self.accessories is None else list(self.accessories)
        return data


def load_device_profile(
    path: str | Path,
    *,
    max_rack: int,
    accessory_classes: Sequence[str],
) -> dict[str, DeviceSpec]:
    """读取并校验 Device Profile；以 "_" 开头的键视为说明文字。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Device Profile 不存在：{path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Device Profile 格式错误：期望 dict，实际为 {type(raw).__name__}")

    known_accessories = set(accessory_classes)
    profile: dict[str, DeviceSpec] = {}
    for name, entry in raw.items():
        if str(name).startswith("_"):
            continue
        ctx = f"Device Profile[{name}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{ctx} 必须是 dict")

        rack_count = entry.get("rack_count")
        if isinstance(rack_count, bool) or not isinstance(rack_count, int) or not 1 <= rack_count <= max_rack:
            raise ValueError(
                f"{ctx}.rack_count 必须是 1～{max_rack} 的整数（上限为 model.max_rack），实际为 {rack_count!r}"
            )

        floor_usable = entry.get("floor_usable", True)
        if not isinstance(floor_usable, bool):
            raise ValueError(f"{ctx}.floor_usable 必须是 true / false，实际为 {floor_usable!r}")

        cavity_group = str(entry.get("cavity_group") or name).strip()

        accessories = entry.get("accessories")
        if accessories is not None:
            if not isinstance(accessories, list):
                raise ValueError(f"{ctx}.accessories 必须是列表或 null")
            unknown = [a for a in accessories if a not in known_accessories]
            if unknown:
                raise ValueError(f"{ctx}.accessories 中有不在附件类别表里的名称：{unknown}")
            accessories = tuple(accessories)

        profile[str(name)] = DeviceSpec(str(name), rack_count, floor_usable, cavity_group, accessories)

    if not profile:
        raise ValueError(f"Device Profile 为空：{path}")

    # 同一 cavity_group 内腔完全相同，层数和能否放底板必须一致
    first_of_group: dict[str, DeviceSpec] = {}
    for spec in profile.values():
        first = first_of_group.setdefault(spec.cavity_group, spec)
        if (first.rack_count, first.floor_usable) != (spec.rack_count, spec.floor_usable):
            raise ValueError(
                f"cavity_group={spec.cavity_group} 内的型号 {first.name} 与 {spec.name} "
                "的 rack_count / floor_usable 不一致"
            )
    return profile


@dataclass
class GalleryMatch:
    model: str             # 检索到的型号；top-1 相似度低于 tau 时为 UNKNOWN_DEVICE
    group: str | None      # cavity_group；未知型号为 None
    confidence: float      # 最佳 cavity_group 的票数占比
    top1_sim: float

    @property
    def known(self) -> bool:
        return self.group is not None


class DeviceGallery:
    """
    检索式型号识别的特征库：保留全部参考图特征，推理时 kNN 按相似度加权投票。
    - 内腔完全相同的型号属于同一个 cavity_group，投票按 cavity_group 汇总，结果以 cavity_group 为准；
    - 特征库与 Proj Head、骨干权重绑定版本，任何一方变动都要全部重建。
    """

    def __init__(self, group_of: Mapping[str, str] | None = None) -> None:
        self.group_of = dict(group_of or {})
        self.labels: list[str] = []
        self._chunks: list[torch.Tensor] = []
        self._bank: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def group(self, name: str) -> str:
        return self.group_of.get(name, name)

    @property
    def bank(self) -> torch.Tensor:
        if self._bank is None:
            if not self._chunks:
                raise RuntimeError("特征库为空")
            self._bank = torch.cat(self._chunks)
            self._chunks = [self._bank]
        return self._bank

    @torch.no_grad()
    def add(self, name: str, feats: torch.Tensor) -> None:  # feats: [N, D]
        feats = feats.reshape(-1, feats.shape[-1])
        self._chunks.append(F.normalize(feats.detach().float().cpu(), dim=-1))
        self.labels += [name] * len(feats)
        self._bank = None

    @torch.no_grad()
    def top1_similarity(self, queries: torch.Tensor) -> torch.Tensor:
        q = F.normalize(queries.detach().float().cpu().reshape(-1, queries.shape[-1]), dim=-1)
        return (q @ self.bank.T).max(dim=1).values

    @torch.no_grad()
    def query_batch(self, queries: torch.Tensor, k: int = 10, tau: float = 0.5) -> list[GalleryMatch]:
        q = F.normalize(queries.detach().float().cpu().reshape(-1, queries.shape[-1]), dim=-1)
        top_s, top_i = (q @ self.bank.T).topk(min(k, len(self)), dim=1)
        matches = []
        for scores, indices in zip(top_s.tolist(), top_i.tolist()):
            if scores[0] < tau:
                matches.append(GalleryMatch(UNKNOWN_DEVICE, None, 0.0, scores[0]))
                continue
            group_votes: dict[str, float] = {}
            model_votes: dict[str, float] = {}
            for s, i in zip(scores, indices):
                name = self.labels[i]
                weight = max(s, 0.0)
                group_votes[self.group(name)] = group_votes.get(self.group(name), 0.0) + weight
                model_votes[name] = model_votes.get(name, 0.0) + weight
            best_group = max(group_votes, key=group_votes.get)
            best_model = max((m for m in model_votes if self.group(m) == best_group), key=model_votes.get)
            total = sum(group_votes.values())
            confidence = group_votes[best_group] / total if total > 0 else 0.0
            matches.append(GalleryMatch(best_model, best_group, confidence, scores[0]))
        return matches

    def query(self, q: torch.Tensor, k: int = 10, tau: float = 0.5) -> GalleryMatch:  # q: [D]
        return self.query_batch(q.reshape(1, -1), k=k, tau=tau)[0]

    def without_group(self, group: str) -> "DeviceGallery":
        keep = [i for i, name in enumerate(self.labels) if self.group(name) != group]
        sub = DeviceGallery(self.group_of)
        if keep:
            sub._chunks = [self.bank[keep]]
            sub.labels = [self.labels[i] for i in keep]
        return sub

    def state_dict(self) -> dict[str, Any]:
        feats = self.bank.clone() if self.labels else torch.empty(0)
        return {"feats": feats, "labels": list(self.labels), "group_of": dict(self.group_of)}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "DeviceGallery":
        gallery = cls(state.get("group_of"))
        if state["labels"]:
            gallery._chunks = [state["feats"].float()]
            gallery.labels = list(state["labels"])
        return gallery


def calibrate_tau(
    gallery: DeviceGallery,
    feats: torch.Tensor,
    models: Sequence[str],
) -> dict[str, Any] | None:
    """
    快速标定未知型号阈值 tau（训练结束时的默认做法）：
      - 库内：验证集图片对完整特征库的 top-1 相似度，应当接受；
      - 库外：把图片所属 cavity_group 从特征库里临时拿掉后的 top-1 相似度，应当拒识。
    这是留一型号验证的近似（模型训练时见过该型号）；完整做法见 script/4-留一型号验证.py。
    特征库不足 2 个 cavity_group 时无法标定，返回 None。
    """
    groups = sorted({gallery.group(m) for m in gallery.labels})
    if len(groups) < 2:
        return None
    in_idx = [i for i, m in enumerate(models) if gallery.group(m) in groups]
    if not in_idx:
        return None

    s_in = gallery.top1_similarity(feats[in_idx])
    s_out = []
    for group in groups:
        idx = [i for i in in_idx if gallery.group(models[i]) == group]
        if idx:
            s_out.append(gallery.without_group(group).top1_similarity(feats[idx]))
    return choose_tau(s_in.numpy(), torch.cat(s_out).numpy())


def choose_tau(s_in: np.ndarray, s_out: np.ndarray) -> dict[str, Any]:
    """
    s_in：库内型号图片的 top-1 相似度（应当接受）；s_out：库外型号图片的 top-1 相似度（应当拒识为未知型号）。
    取两者平衡准确率最高的阈值，阈值落在相邻两个相似度的中点。
    """
    s_in, s_out = np.asarray(s_in, dtype=np.float64), np.asarray(s_out, dtype=np.float64)
    values = np.unique(np.concatenate([s_in, s_out]))
    candidates = np.concatenate([values[:1] - 1e-4, (values[1:] + values[:-1]) / 2, values[-1:] + 1e-4])
    accept_in = (s_in[None, :] >= candidates[:, None]).mean(1)
    reject_out = (s_out[None, :] < candidates[:, None]).mean(1)
    balanced = 0.5 * (accept_in + reject_out)
    best = int(np.argmax(balanced))
    return {
        "tau": float(candidates[best]),
        "balanced_acc": float(balanced[best]),
        "in_accept_rate": float(accept_in[best]),
        "out_reject_rate": float(reject_out[best]),
        "n_in": int(len(s_in)),
        "n_out": int(len(s_out)),
    }
