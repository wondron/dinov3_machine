# dino_finetune/inference.py
"""
推理后处理（设计文档第 3 节），PT 推理、ONNX 推理和训练结束时的测试集预测共用：
  1. Is-Oven 判为"否"：device_model = "无"，层位 = null；食物、容器、附件照常输出，附件不做设备屏蔽；
  2. 检索型号：top-1 相似度 < tau 时输出"未知型号"，层位 = null，附件不屏蔽，并标记待补库；
  3. 型号已知：按 Device Profile 屏蔽该型号不支持的附件，再按各类阈值取正类；
  4. 层位：腔内没有任何物品时为 null；否则在该型号的有效层位里取 argmax，
     最大概率低于阈值、或"层位 ≥ 1 但附件为空"时标记低置信度；
  5. 食物、容器按各自的阈值输出。
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch

from .data import OvenTransforms, read_image_rgb
from .device import UNKNOWN_DEVICE, DeviceGallery, DeviceSpec, GalleryMatch
from .labels import IMG_EXTS
from .metrics import masked_softmax, rack_mask
from .model.oven import OUTPUT_KEYS, inference_outputs

NOT_OVEN_DEVICE = "无"


def _round(value: float) -> float:
    return round(float(value), 4)


class OvenPostprocessor:
    """输入为一个 batch 的推理输出（键见 model.oven.OUTPUT_KEYS，numpy 数组），输出设计文档格式的结果列表。"""

    def __init__(
        self,
        *,
        calibration: Mapping[str, Any],
        profile: Mapping[str, DeviceSpec],
        container_classes: Sequence[str],
        accessory_classes: Sequence[str],
        galleries: Mapping[str, DeviceGallery],
    ) -> None:
        self.cal = calibration
        self.profile = profile
        self.container_classes = list(container_classes)
        self.accessory_classes = list(accessory_classes)
        self.feature = calibration["gallery_feature"]
        self.gallery = galleries.get(self.feature)
        self.container_thr = np.array([calibration["container_thresholds"][n] for n in self.container_classes])
        self.accessory_thr = np.array([calibration["accessory_thresholds"][n] for n in self.accessory_classes])

    def __call__(
        self,
        outputs: Mapping[str, np.ndarray],
        device_model: str | None = None,
        with_scores: bool = False,
    ) -> list[dict[str, Any]]:
        """device_model：型号已知时（例如摄像头内置在一体机里）直接用该型号，跳过检索。"""
        if device_model is not None and device_model not in self.profile:
            raise ValueError(f"型号 {device_model} 不在 Device Profile 中")
        is_oven = outputs["is_oven_prob"] >= self.cal["is_oven_threshold"]

        matches: dict[int, GalleryMatch] = {}
        oven_rows = np.flatnonzero(is_oven)
        if device_model is None and len(oven_rows) and self.gallery is not None and len(self.gallery):
            feats = torch.from_numpy(np.ascontiguousarray(outputs[self.feature][oven_rows]))
            found = self.gallery.query_batch(feats, k=self.cal["knn_k"], tau=self.cal["tau"][self.feature])
            matches = dict(zip(oven_rows.tolist(), found))

        results = []
        for i in range(len(is_oven)):
            result = self._one({k: outputs[k][i] for k in OUTPUT_KEYS}, bool(is_oven[i]), matches.get(i), device_model)
            if with_scores:
                result["scores"] = self._scores({k: outputs[k][i] for k in OUTPUT_KEYS})
            results.append(result)
        return results

    def _one(
        self,
        out: Mapping[str, np.ndarray],
        is_oven: bool,
        match: GalleryMatch | None,
        device_model: str | None,
    ) -> dict[str, Any]:
        container = [n for n, p, t in zip(self.container_classes, out["container_prob"], self.container_thr) if p >= t]
        accessory_ok = out["accessory_prob"] >= self.accessory_thr
        result: dict[str, Any] = {
            "is_oven": is_oven,
            "device_model": None,
            "device_score": None,
            "food_exist": bool(out["food_prob"] >= self.cal["food_threshold"]),
            "container_type": container,
            "accessory_type": [n for n, ok in zip(self.accessory_classes, accessory_ok) if ok],
            "rack_level": None,
            "rack_level_score": None,
            "low_confidence": False,
        }
        if not is_oven:
            result["device_model"] = NOT_OVEN_DEVICE
            return result

        if device_model is not None:
            spec, score = self.profile[device_model], 1.0
        elif match is None or not match.known:
            result["device_model"] = UNKNOWN_DEVICE
            result["device_score"] = None if match is None else _round(match.top1_sim)
            result["pending_gallery"] = True  # 放入待补库池
            return result
        else:
            spec, score = self.profile[match.model], match.confidence
        result["device_model"], result["device_score"] = spec.name, _round(score)

        if spec.accessories is not None:  # 型号已知：屏蔽该型号不支持的附件
            result["accessory_type"] = [
                n for n, ok in zip(self.accessory_classes, accessory_ok) if ok and n in spec.accessories
            ]

        if not (result["food_exist"] or result["container_type"] or result["accessory_type"]):
            return result  # 空腔：层位 = null
        logits = np.where(rack_mask(spec, len(out["rack_raw"])), out["rack_raw"], -np.inf)
        prob = masked_softmax(logits[None])[0]
        level = int(prob.argmax())
        result["rack_level"], result["rack_level_score"] = level, _round(prob[level])
        result["low_confidence"] = bool(
            prob[level] < self.cal["rack_conf_threshold"] or (level >= 1 and not result["accessory_type"])
        )
        return result

    def _scores(self, out: Mapping[str, np.ndarray]) -> dict[str, Any]:
        """各头原始概率，便于核对阈值。"""
        return {
            "is_oven": _round(out["is_oven_prob"]),
            "food": _round(out["food_prob"]),
            "container": {n: _round(p) for n, p in zip(self.container_classes, out["container_prob"])},
            "accessory": {n: _round(p) for n, p in zip(self.accessory_classes, out["accessory_prob"])},
        }


class OvenPredictor:
    """PT 推理：预处理 → 模型 → 后处理。"""

    def __init__(self, run: Any, device: torch.device, use_amp: bool = True) -> None:  # run: engine.TrainedRun
        inp = run.cfg["input"]
        self.model = run.model.eval()
        self.device = device
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.transform = OvenTransforms(inp["img_dim"], inp["mean"], inp["std"], inp["img_interp"], is_train=False)
        self.post = OvenPostprocessor(
            calibration=run.calibration,
            profile=run.profile,
            container_classes=run.schema.container_classes,
            accessory_classes=run.schema.accessory_classes,
            galleries=run.galleries,
        )

    @torch.no_grad()
    def outputs(self, images: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
        x = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.amp.autocast(self.device.type, enabled=self.use_amp):
            out = inference_outputs(self.model(x))
        return {k: v.float().cpu().numpy() for k, v in out.items()}

    def predict(self, images: Sequence[np.ndarray], device_model: str | None = None, with_scores: bool = False) -> list[dict[str, Any]]:
        return self.post(self.outputs(images), device_model=device_model, with_scores=with_scores)


# =========================
# 部署包中的特征库（npy + json，部署侧不需要 torch）
# =========================
def save_gallery_bundle(galleries: Mapping[str, DeviceGallery], out_dir: Path, meta: Mapping[str, Any]) -> None:
    index: dict[str, Any] = {"meta": dict(meta), "group_of": {}, "features": {}}
    for name, gallery in galleries.items():
        if not len(gallery):
            continue
        np.save(out_dir / f"gallery_{name}.npy", gallery.bank.numpy().astype(np.float32))
        index["features"][name] = {"file": f"gallery_{name}.npy", "labels": gallery.labels}
        index["group_of"] = gallery.group_of
    (out_dir / "gallery.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def load_gallery_bundle(bundle_dir: Path) -> tuple[dict[str, DeviceGallery], dict[str, Any]]:
    index = json.loads((bundle_dir / "gallery.json").read_text(encoding="utf-8"))
    galleries = {}
    for name, entry in index["features"].items():
        feats = torch.from_numpy(np.load(bundle_dir / entry["file"]))
        galleries[name] = DeviceGallery.from_state_dict({"feats": feats, "labels": entry["labels"], "group_of": index["group_of"]})
    return galleries, index["meta"]


# =========================
# 批量推理的输入输出
# =========================
def list_images(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"输入不存在：{path}")
    images = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not images:
        raise RuntimeError(f"目录下没有图片：{path}")
    return images


def predict_files(
    paths: Sequence[Path],
    predict: Callable[[list[np.ndarray]], list[dict[str, Any]]],
    batch_size: int,
) -> Iterator[dict[str, Any]]:
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        for path, result in zip(chunk, predict([read_image_rgb(str(p)) for p in chunk])):
            yield {"image": str(path), **result}


def save_results(results: Sequence[Mapping[str, Any]], out_path: Path, pending_dir: Path | None = None) -> int:
    """保存结果 JSON；给了 pending_dir 时把未知型号的图片复制进去（待补库池），返回待补库图片数。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(list(results), ensure_ascii=False, indent=2), encoding="utf-8")
    pending = [r for r in results if r.get("pending_gallery")]
    if pending_dir is not None and pending:
        pending_dir.mkdir(parents=True, exist_ok=True)
        for r in pending:
            shutil.copy2(r["image"], pending_dir / Path(r["image"]).name)
    return len(pending)


def summarize_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "images": len(results),
        "device_model": dict(Counter(r["device_model"] for r in results)),
        "rack_level": dict(Counter(str(r["rack_level"]) for r in results)),
        "low_confidence": sum(bool(r["low_confidence"]) for r in results),
    }
