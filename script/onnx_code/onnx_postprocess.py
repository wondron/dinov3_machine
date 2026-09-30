"""独立部署后处理：只依赖 NumPy，不需要训练项目或 PyTorch。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

OUTPUT_KEYS = ("is_oven_prob", "food_prob", "container_prob", "accessory_prob", "rack_raw", "proj", "cls")
UNKNOWN_DEVICE = "未知型号"
NOT_OVEN_DEVICE = "无"


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    rack_count: int
    floor_usable: bool
    cavity_group: str
    accessories: tuple[str, ...] | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["accessories"] = None if self.accessories is None else list(self.accessories)
        return result


def load_device_profile(
    path: str | Path, *, max_rack: int, accessory_classes: Sequence[str]
) -> dict[str, DeviceSpec]:
    """读取设备约束；以 ``_`` 开头的顶层字段作为说明忽略。"""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Device Profile 必须是 JSON 对象")
    known_accessories = set(accessory_classes)
    profile: dict[str, DeviceSpec] = {}
    for name, entry in raw.items():
        if name.startswith("_"):
            continue
        ctx = f"Device Profile[{name}]"
        if not name.strip() or not isinstance(entry, dict):
            raise ValueError(f"{ctx} 必须具有非空型号名称和对象配置")
        rack_count = entry.get("rack_count")
        if isinstance(rack_count, bool) or not isinstance(rack_count, int) or not 1 <= rack_count <= max_rack:
            raise ValueError(f"{ctx}.rack_count 必须是 1～{max_rack} 的整数")
        floor_usable = entry.get("floor_usable", True)
        if not isinstance(floor_usable, bool):
            raise ValueError(f"{ctx}.floor_usable 必须是 true / false")
        cavity_group = str(entry.get("cavity_group") or name).strip()
        if not cavity_group:
            raise ValueError(f"{ctx}.cavity_group 不能为空")
        accessories = entry.get("accessories")
        if accessories is not None:
            if not isinstance(accessories, list) or any(not isinstance(a, str) for a in accessories):
                raise ValueError(f"{ctx}.accessories 必须是名称列表或 null")
            unknown = [a for a in accessories if a not in known_accessories]
            if unknown:
                raise ValueError(f"{ctx}.accessories 中存在未知附件类别：{unknown}")
            accessories = tuple(accessories)
        profile[name] = DeviceSpec(name, rack_count, floor_usable, cavity_group, accessories)
    if not profile:
        raise ValueError(f"Device Profile 为空：{path}")
    first_of_group: dict[str, DeviceSpec] = {}
    for spec in profile.values():
        first = first_of_group.setdefault(spec.cavity_group, spec)
        if (first.rack_count, first.floor_usable) != (spec.rack_count, spec.floor_usable):
            raise ValueError(
                f"cavity_group={spec.cavity_group} 内的型号 {first.name} 与 {spec.name} "
                "的 rack_count / floor_usable 不一致"
            )
    return profile


@dataclass(frozen=True)
class GalleryMatch:
    model: str
    group: str | None
    confidence: float
    top1_sim: float

    @property
    def known(self) -> bool:
        return self.group is not None


class DeviceGallery:
    """余弦 kNN：先按 cavity_group 投票，再选组内得票最高的型号。"""

    def __init__(self, group_of: Mapping[str, str] | None = None) -> None:
        self.group_of = dict(group_of or {})
        self.labels: list[str] = []
        self.bank = np.empty((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def group(self, name: str) -> str:
        return self.group_of.get(name, name)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "DeviceGallery":
        gallery = cls(state.get("group_of"))
        labels = state["labels"]
        if not isinstance(labels, list) or any(not isinstance(n, str) or not n.strip() for n in labels):
            raise ValueError("特征库 labels 必须是非空型号名称列表")
        bank = np.asarray(state["feats"], dtype=np.float32)
        if bank.ndim != 2 or bank.shape[0] != len(labels) or bank.shape[1] == 0:
            raise ValueError(f"特征库必须是 [样本数, 特征维度]，并与 labels 一一对应，实际为 {bank.shape}")
        if not np.isfinite(bank).all():
            raise ValueError("特征库包含 NaN 或无穷值")
        gallery.labels = list(labels)
        # 导出时已经归一化；重复归一化会改变阈值边界附近的结果。
        gallery.bank = np.ascontiguousarray(bank)
        return gallery

    def query_batch(self, queries: np.ndarray, k: int = 10, tau: float = 0.5) -> list[GalleryMatch]:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("knn_k 必须是正整数")
        if not len(self):
            raise ValueError("不能查询空特征库")
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self.bank.shape[1] or not np.isfinite(q).all():
            raise ValueError(f"检索特征必须是有限数值数组 [N, {self.bank.shape[1]}]，实际为 {q.shape}")
        # 与 torch.nn.functional.normalize 的 float32 / eps=1e-12 对齐。
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), np.float32(1e-12))
        similarities = q @ self.bank.T
        # 分数完全相同时按入库顺序处理；PyTorch topk 对并列项的顺序未作保证。
        top_indices = np.argsort(-similarities, axis=1, kind="stable")[:, : min(k, len(self))]
        matches = []
        for row, indices in zip(similarities, top_indices):
            scores = row[indices].tolist()
            if scores[0] < tau:
                matches.append(GalleryMatch(UNKNOWN_DEVICE, None, 0.0, scores[0]))
                continue
            group_votes: dict[str, float] = {}
            model_votes: dict[str, float] = {}
            for score, index in zip(scores, indices.tolist()):
                name = self.labels[index]
                group = self.group(name)
                weight = max(score, 0.0)
                group_votes[group] = group_votes.get(group, 0.0) + weight
                model_votes[name] = model_votes.get(name, 0.0) + weight
            best_group = max(group_votes, key=group_votes.get)
            best_model = max((n for n in model_votes if self.group(n) == best_group), key=model_votes.get)
            total = sum(group_votes.values())
            confidence = group_votes[best_group] / total if total > 0 else 0.0
            matches.append(GalleryMatch(best_model, best_group, confidence, scores[0]))
        return matches


def load_gallery_bundle(bundle_dir: str | Path) -> tuple[dict[str, DeviceGallery], dict[str, Any]]:
    """读取导出目录中的 gallery.json 和 .npy；禁止加载 pickle 数据。"""
    bundle = Path(bundle_dir).resolve()
    index = json.loads((bundle / "gallery.json").read_text(encoding="utf-8"))
    if not isinstance(index, dict) or not isinstance(index.get("features"), dict):
        raise ValueError("gallery.json 缺少 features 对象")
    group_of = index.get("group_of")
    meta = index.get("meta")
    if not isinstance(group_of, dict) or not isinstance(meta, dict):
        raise ValueError("gallery.json 缺少 group_of / meta 对象")
    if any(not isinstance(v, str) or not v.strip() for v in group_of.values()):
        raise ValueError("gallery.json 的 cavity_group 必须是非空名称")
    galleries = {}
    for name, entry in index["features"].items():
        if name not in ("proj", "cls"):
            raise ValueError(f"未知特征库名称：{name}")
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str) or "labels" not in entry:
            raise ValueError(f"gallery.json.features[{name}] 缺少 file / labels")
        path = (bundle / entry["file"]).resolve()
        if not path.is_relative_to(bundle):
            raise ValueError(f"特征库文件必须位于部署包目录内：{entry['file']}")
        galleries[name] = DeviceGallery.from_state_dict({
            "feats": np.load(path, allow_pickle=False), "labels": entry["labels"], "group_of": group_of,
        })
    return galleries, meta


def _round(value: float) -> float:
    return round(float(value), 4)


class OvenPostprocessor:
    """将模型的一个 batch 输出转换成业务结果，与原项目后处理字段保持一致。"""

    def __init__(
        self, *, calibration: Mapping[str, Any], profile: Mapping[str, DeviceSpec],
        container_classes: Sequence[str], accessory_classes: Sequence[str],
        galleries: Mapping[str, DeviceGallery],
    ) -> None:
        self.cal = calibration
        self.profile = profile
        self.container_classes = list(container_classes)
        self.accessory_classes = list(accessory_classes)
        self.feature = calibration["gallery_feature"]
        if self.feature not in ("proj", "cls"):
            raise ValueError(f"gallery_feature 必须是 proj 或 cls：{self.feature}")
        self.gallery = galleries.get(self.feature)
        self.container_thr = np.array([calibration["container_thresholds"][n] for n in self.container_classes])
        self.accessory_thr = np.array([calibration["accessory_thresholds"][n] for n in self.accessory_classes])
        for gallery in galleries.values():
            unknown = sorted(set(gallery.labels) - set(profile))
            if unknown:
                raise ValueError(f"特征库中的型号不在 Device Profile 中：{unknown}")
            for name in gallery.labels:
                if gallery.group(name) != profile[name].cavity_group:
                    raise ValueError(f"特征库与 Device Profile 的 cavity_group 不一致：{name}")

    def __call__(
        self, outputs: Mapping[str, np.ndarray], device_model: str | None = None, with_scores: bool = False
    ) -> list[dict[str, Any]]:
        if device_model is not None and device_model not in self.profile:
            raise ValueError(f"型号 {device_model} 不在 Device Profile 中")
        is_oven = outputs["is_oven_prob"] >= self.cal["is_oven_threshold"]
        matches: dict[int, GalleryMatch] = {}
        oven_rows = np.flatnonzero(is_oven)
        if device_model is None and len(oven_rows) and self.gallery is not None and len(self.gallery):
            found = self.gallery.query_batch(
                outputs[self.feature][oven_rows], k=self.cal["knn_k"], tau=self.cal["tau"][self.feature]
            )
            matches = dict(zip(oven_rows.tolist(), found))
        results = []
        for i in range(len(is_oven)):
            out = {key: outputs[key][i] for key in OUTPUT_KEYS}
            result = self._one(out, bool(is_oven[i]), matches.get(i), device_model)
            if with_scores:
                result["scores"] = self._scores(out)
            results.append(result)
        return results

    def _one(
        self, out: Mapping[str, np.ndarray], is_oven: bool, match: GalleryMatch | None, device_model: str | None
    ) -> dict[str, Any]:
        container = [n for n, p, t in zip(self.container_classes, out["container_prob"], self.container_thr) if p >= t]
        accessory_ok = out["accessory_prob"] >= self.accessory_thr
        result: dict[str, Any] = {
            "is_oven": is_oven, "device_model": None, "device_score": None,
            "food_exist": bool(out["food_prob"] >= self.cal["food_threshold"]),
            "container_type": container,
            "accessory_type": [n for n, ok in zip(self.accessory_classes, accessory_ok) if ok],
            "rack_level": None, "rack_level_score": None, "low_confidence": False,
        }
        if not is_oven:
            result["device_model"] = NOT_OVEN_DEVICE
            return result
        if device_model is not None:
            spec, score = self.profile[device_model], 1.0
        elif match is None or not match.known:
            result["device_model"] = UNKNOWN_DEVICE
            result["device_score"] = None if match is None else _round(match.top1_sim)
            result["pending_gallery"] = True
            return result
        else:
            spec, score = self.profile[match.model], match.confidence
        result["device_model"], result["device_score"] = spec.name, _round(score)
        if spec.accessories is not None:
            result["accessory_type"] = [
                n for n, ok in zip(self.accessory_classes, accessory_ok) if ok and n in spec.accessories
            ]
        if not (result["food_exist"] or result["container_type"] or result["accessory_type"]):
            return result
        valid = np.arange(len(out["rack_raw"])) <= spec.rack_count
        valid[0] &= spec.floor_usable
        logits = np.where(valid, out["rack_raw"], -np.inf)
        exp = np.exp(logits - logits.max())
        prob = exp / exp.sum()
        level = int(prob.argmax())
        result["rack_level"], result["rack_level_score"] = level, _round(prob[level])
        result["low_confidence"] = bool(
            prob[level] < self.cal["rack_conf_threshold"] or (level >= 1 and not result["accessory_type"])
        )
        return result

    def _scores(self, out: Mapping[str, np.ndarray]) -> dict[str, Any]:
        return {
            "is_oven": _round(out["is_oven_prob"]), "food": _round(out["food_prob"]),
            "container": {n: _round(p) for n, p in zip(self.container_classes, out["container_prob"])},
            "accessory": {n: _round(p) for n, p in zip(self.accessory_classes, out["accessory_prob"])},
        }
