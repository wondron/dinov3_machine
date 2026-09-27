# dino_finetune/metrics.py
"""评估指标（设计文档第 9 节）与阶段 2 的阈值标定。输入都是整个评估集拼好的 numpy 数组。"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .device import DeviceGallery, DeviceSpec, GalleryMatch, calibrate_tau
from .labels import OvenLabel

FEATURE_KEYS = {"proj": "proj", "raw": "cls"}  # 检索特征名 → collect 出来的数组名


# -----------------------------
# 1) 基础指标
# -----------------------------
def _ratio(num: float, den: float) -> float | None:
    return float(num) / float(den) if den > 0 else None


def _distinct_desc(scores: np.ndarray, flags: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按分数降序排序，返回 (排序后分数, 排序后标记, 每个不同分数最后出现的位置)。"""
    order = np.argsort(-scores, kind="mergesort")
    scores, flags = scores[order], flags[order]
    last = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    return scores, flags, last


def average_precision(scores: np.ndarray, targets: np.ndarray) -> float | None:
    """AP，与 sklearn.average_precision_score 定义一致（按不同分数阈值累加）；没有正样本时返回 None。"""
    targets = np.asarray(targets).astype(bool)
    n_pos = int(targets.sum())
    if n_pos == 0:
        return None
    _, flags, last = _distinct_desc(np.asarray(scores, dtype=np.float64), targets)
    tp = np.cumsum(flags)[last]
    precision = tp / (last + 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def binary_stats(probs: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = np.asarray(probs) >= threshold
    t = np.asarray(targets).astype(bool)
    tp, fp = int((pred & t).sum()), int((pred & ~t).sum())
    fn, tn = int((~pred & t).sum()), int((~pred & ~t).sum())
    return {
        "n": int(len(t)),
        "positives": tp + fn,
        "acc": _ratio(tp + tn, len(t)),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "threshold": float(threshold),
    }


def multilabel_report(
    probs: np.ndarray,
    targets: np.ndarray,
    names: Sequence[str],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """
    每类 P / R / F1 / AP。评估集中某类没有负样本时 AP 恒为 1、没有区分意义，记为 None；
    mAP 只在同时有正、负样本的类别上取平均。
    """
    per_class: dict[str, Any] = {}
    aps = []
    for c, name in enumerate(names):
        stats = binary_stats(probs[:, c], targets[:, c], thresholds[c])
        ap = average_precision(probs[:, c], targets[:, c]) if 0 < stats["positives"] < stats["n"] else None
        if ap is not None:
            aps.append(ap)
        per_class[name] = {
            "ap": ap,
            "precision": stats["precision"],
            "recall": stats["recall"],
            "f1": stats["f1"],
            "support": stats["positives"],
            "negatives": stats["n"] - stats["positives"],
            "threshold": stats["threshold"],
        }
    return {"n": int(len(targets)), "map": float(np.mean(aps)) if aps else None, "per_class": per_class}


def masked_softmax(logits: np.ndarray) -> np.ndarray:
    """logits 中被掩码的类别为 -inf，对应概率为 0。"""
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def rack_mask(spec: DeviceSpec, num_pos: int) -> np.ndarray:
    valid = np.arange(num_pos) <= spec.rack_count
    valid[0] &= spec.floor_usable
    return valid


def rack_report(
    logits: np.ndarray,
    targets: np.ndarray,
    models: Sequence[str],
    profile: Mapping[str, DeviceSpec],
    conf_threshold: float,
) -> dict[str, Any]:
    """
    层位指标（logits 已按真实型号掩码）：精确层准确率、导轨层 ±1 准确率、底板层 vs 导轨层准确率、
    逐层准确率，以及按型号统计的 (N+1)×(N+1) 混淆矩阵（行 = 真实层位，列 = 预测层位）。
    """
    prob = masked_softmax(logits)
    pred, conf = prob.argmax(1), prob.max(1)
    correct = pred == targets
    rail = targets >= 1

    per_level = {
        str(level): {"n": int((targets == level).sum()), "acc": float(correct[targets == level].mean())}
        for level in sorted(set(targets.tolist()))
    }
    per_model = {}
    models = np.asarray(models, dtype=object)
    for model in sorted(set(models.tolist())):
        sel = models == model
        size = profile[model].rack_count + 1 if model in profile else logits.shape[1]
        confusion = np.zeros((size, size), dtype=np.int64)
        for t, p in zip(targets[sel], pred[sel]):
            if t < size and p < size:
                confusion[t, p] += 1
        per_model[model] = {"n": int(sel.sum()), "acc": float(correct[sel].mean()), "confusion": confusion.tolist()}

    return {
        "n": int(len(targets)),
        "acc": float(correct.mean()),
        "acc_pm1": float(((pred >= 1) & (np.abs(pred - targets) <= 1))[rail].mean()) if rail.any() else None,
        "floor_acc": float(((pred == 0) == (targets == 0)).mean()),
        "mean_conf": float(conf.mean()),
        "low_conf_rate": float((conf < conf_threshold).mean()),
        "per_level": per_level,
        "per_model": per_model,
    }


def retrieval_report(matches: Sequence[GalleryMatch], true_models: Sequence[str], gallery: DeviceGallery) -> dict[str, Any]:
    """型号检索：库内型号的 top-1 准确率（按 cavity_group 判定）、误拒率，库外型号的拒识率与混淆表。"""
    groups = {gallery.group(m) for m in gallery.labels}
    in_gallery = [gallery.group(m) in groups for m in true_models]
    correct = [m.known and m.group == gallery.group(t) for m, t in zip(matches, true_models)]
    rejected = [not m.known for m in matches]
    confusion: dict[str, Counter] = defaultdict(Counter)
    for match, true_model in zip(matches, true_models):
        confusion[true_model][match.model] += 1

    def mean_where(values: Sequence[bool], keep: Sequence[bool]) -> float | None:
        chosen = [v for v, k in zip(values, keep) if k]
        return float(np.mean(chosen)) if chosen else None

    unseen = [not x for x in in_gallery]
    return {
        "n": len(true_models),
        "top1": mean_where(correct, in_gallery),
        "unknown_rate": mean_where(rejected, in_gallery),
        "unseen_reject_rate": mean_where(rejected, unseen),
        "mean_top1_sim": float(np.mean([m.top1_sim for m in matches])) if matches else None,
        "confusion": {model: dict(counter) for model, counter in sorted(confusion.items())},
    }


def cascade_report(
    matches: Sequence[GalleryMatch],
    rows: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    models: Sequence[str],
    profile: Mapping[str, DeviceSpec],
    gallery: DeviceGallery,
) -> dict[str, Any]:
    """
    端到端级联误差：层位改用检索到的型号掩码（未知型号时层位输出 null，记为错误），
    统计型号识别错误时层位的出错比例。rows 为 matches 对应的数组行号。
    """
    num_pos = arrays["rack_raw"].shape[1]
    device_ok, rack_ok = [], []
    for match, row in zip(matches, rows):
        if not arrays["rack_known"][row]:
            continue
        device_ok.append(match.known and match.group == gallery.group(models[row]))
        if match.known and match.model in profile:
            logits = np.where(rack_mask(profile[match.model], num_pos), arrays["rack_raw"][row], -np.inf)
            rack_ok.append(int(logits.argmax()) == int(arrays["rack_level"][row]))
        else:
            rack_ok.append(False)
    if not device_ok:
        return {"n": 0, "rack_acc_e2e": None, "rack_err_given_device_err": None, "rack_err_given_device_ok": None}
    device_ok_arr, rack_ok_arr = np.array(device_ok), np.array(rack_ok)
    return {
        "n": len(device_ok),
        "device_acc": float(device_ok_arr.mean()),
        "rack_acc_e2e": float(rack_ok_arr.mean()),
        "rack_err_given_device_err": float((~rack_ok_arr[~device_ok_arr]).mean()) if (~device_ok_arr).any() else None,
        "rack_err_given_device_ok": float((~rack_ok_arr[device_ok_arr]).mean()) if device_ok_arr.any() else None,
    }


# -----------------------------
# 2) 整体评估
# -----------------------------
def default_calibration(
    eval_cfg: Mapping[str, Any],
    container_classes: Sequence[str],
    accessory_classes: Sequence[str],
) -> dict[str, Any]:
    """未标定时使用的阈值：多标签 / 二分类 / 层位置信度都用 default_threshold，tau 用配置值。"""
    default = float(eval_cfg["default_threshold"])
    return {
        "gallery_feature": eval_cfg["gallery_feature"],
        "knn_k": int(eval_cfg["knn_k"]),
        "tau": {name: float(eval_cfg["tau"]) for name in FEATURE_KEYS},
        "is_oven_threshold": default,
        "food_threshold": default,
        "container_thresholds": {name: default for name in container_classes},
        "accessory_thresholds": {name: default for name in accessory_classes},
        "rack_conf_threshold": default,
    }


def evaluate_outputs(
    arrays: Mapping[str, np.ndarray],
    labels: Sequence[OvenLabel],
    *,
    container_classes: Sequence[str],
    accessory_classes: Sequence[str],
    safety_classes: Sequence[str],
    profile: Mapping[str, DeviceSpec],
    galleries: Mapping[str, DeviceGallery],
    calibration: Mapping[str, Any],
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """返回 (日志 / 选模型用的扁平指标, 详细报告)。缺标注或无法计算的指标为 None。"""
    rows_all = np.arange(len(arrays["index"]))
    models = [labels[i].device_model for i in arrays["index"]]
    is_oven = arrays["is_oven"] > 0.5
    flat: dict[str, float | None] = {}
    report: dict[str, Any] = {"n": int(len(rows_all))}

    stats = binary_stats(arrays["is_oven_prob"], is_oven, calibration["is_oven_threshold"])
    report["is_oven"] = stats
    flat["is_oven_acc"], flat["is_oven_recall"] = stats["acc"], stats["recall"]

    known = arrays["food_known"]
    if known.any():
        stats = binary_stats(arrays["food_prob"][known], arrays["food"][known] > 0.5, calibration["food_threshold"])
        report["food"] = stats
        flat["food_acc"], flat["food_f1"] = stats["acc"], stats["f1"]

    known = arrays["container_known"]
    if known.any():
        thresholds = [calibration["container_thresholds"][n] for n in container_classes]
        result = multilabel_report(arrays["container_prob"][known], arrays["container"][known] > 0.5, container_classes, thresholds)
        result["safety_recall"] = {name: result["per_class"][name]["recall"] for name in safety_classes}
        report["container"] = result
        flat["container_map"] = result["map"]

    known = arrays["accessory_known"]
    thresholds = [calibration["accessory_thresholds"][n] for n in accessory_classes]
    report["accessory"] = {}
    for key, sel in (("all", known), ("in_oven", known & is_oven), ("out_oven", known & ~is_oven)):
        if sel.any():
            result = multilabel_report(arrays["accessory_prob"][sel], arrays["accessory"][sel] > 0.5, accessory_classes, thresholds)
            report["accessory"][key] = result
            flat["accessory_map" if key == "all" else f"accessory_map_{key}"] = result["map"]

    sel = is_oven & arrays["rack_known"]
    if sel.any():
        result = rack_report(
            arrays["rack_logits"][sel],
            arrays["rack_level"][sel],
            [models[i] for i in np.flatnonzero(sel)],
            profile,
            calibration["rack_conf_threshold"],
        )
        report["rack"] = result
        flat["rack_acc"], flat["rack_acc_pm1"], flat["rack_floor_acc"] = result["acc"], result["acc_pm1"], result["floor_acc"]

    oven_rows = np.flatnonzero(is_oven)
    for name, gallery in galleries.items():
        if len(gallery) == 0 or len(oven_rows) == 0:
            continue
        feats = torch.from_numpy(arrays[FEATURE_KEYS[name]][oven_rows])
        matches = gallery.query_batch(feats, k=calibration["knn_k"], tau=calibration["tau"][name])
        result = retrieval_report(matches, [models[i] for i in oven_rows], gallery)
        report[f"device_{name}"] = result
        flat[f"device_top1_{name}"] = result["top1"]
        if name == calibration["gallery_feature"]:
            flat["device_top1"] = result["top1"]
            cascade = cascade_report(matches, oven_rows, arrays, models, profile, gallery)
            report["cascade"] = cascade
            flat["rack_acc_e2e"] = cascade["rack_acc_e2e"]
            flat["rack_err_given_device_err"] = cascade["rack_err_given_device_err"]
    return flat, report


def weighted_score(metrics: Mapping[str, float | None], weights: Mapping[str, float]) -> float | None:
    """选 best checkpoint 的综合分：可用指标的加权平均，缺失的指标跳过。"""
    num = den = 0.0
    for key, weight in weights.items():
        value = metrics.get(key)
        if value is not None and weight > 0:
            num += weight * value
            den += weight
    return num / den if den > 0 else None


# -----------------------------
# 3) 阈值标定（阶段 2）
# -----------------------------
def best_f1_threshold(probs: np.ndarray, targets: np.ndarray, default: float) -> tuple[float, str]:
    """F1 最高的阈值（取与下一个更低分数的中点）；评估集中缺正样本或缺负样本时无法标定，返回默认值。"""
    t = np.asarray(targets).astype(bool)
    if t.all() or not t.any():
        return float(default), "default"
    scores, flags, last = _distinct_desc(np.asarray(probs, dtype=np.float64), t)
    tp = np.cumsum(flags)[last]
    fp = (last + 1) - tp
    f1 = 2 * tp / (2 * tp + fp + (t.sum() - tp))
    k = last[int(np.argmax(f1))]
    lower = scores[k + 1] if k + 1 < len(scores) else 0.0
    return float((scores[k] + lower) / 2), "max_f1"


def recall_target_threshold(probs: np.ndarray, targets: np.ndarray, target_recall: float, default: float) -> tuple[float, str]:
    """
    安全相关类别：满足召回率目标的最高阈值，且不高于默认阈值（宁可多报）。
    评估集中缺正样本或缺负样本时无法估计误报代价，返回默认值。
    """
    t = np.asarray(targets).astype(bool)
    if t.all() or not t.any():
        return float(default), "default"
    pos = np.sort(np.asarray(probs, dtype=np.float64)[t])[::-1]
    k = max(1, math.ceil(target_recall * len(pos) - 1e-9))
    return float(min(pos[k - 1], default)), "recall_target"


def rack_confidence_threshold(
    conf: np.ndarray,
    correct: np.ndarray,
    target_precision: float,
    default: float,
) -> tuple[float, str]:
    """层位置信度阈值：置信度不低于阈值的预测要达到目标准确率，取满足条件的最低阈值（覆盖最多样本）。"""
    if len(conf) == 0:
        return float(default), "default"
    scores, flags, last = _distinct_desc(np.asarray(conf, dtype=np.float64), np.asarray(correct).astype(bool))
    precision = np.cumsum(flags)[last] / (last + 1)
    meets = np.flatnonzero(precision >= target_precision)
    if len(meets) == 0:
        return float(default), "default(target_unreachable)"
    return float(scores[last[meets[-1]]]), "target_precision"


def calibrate_thresholds(
    arrays: Mapping[str, np.ndarray],
    labels: Sequence[OvenLabel],
    *,
    eval_cfg: Mapping[str, Any],
    container_classes: Sequence[str],
    accessory_classes: Sequence[str],
    safety_classes: Sequence[str],
    galleries: Mapping[str, DeviceGallery],
) -> dict[str, Any]:
    """
    阶段 2：在验证集上标定 is_oven / food 阈值、多标签逐类阈值（安全类别按召回率目标）、
    层位置信度阈值和未知型号阈值 tau。无法标定的项保留默认值，并在 sources 中注明。
    """
    default = float(eval_cfg["default_threshold"])
    cal = default_calibration(eval_cfg, container_classes, accessory_classes)
    sources: dict[str, str] = {}
    is_oven = arrays["is_oven"] > 0.5

    cal["is_oven_threshold"], sources["is_oven"] = best_f1_threshold(arrays["is_oven_prob"], is_oven, default)
    known = arrays["food_known"]
    cal["food_threshold"], sources["food"] = best_f1_threshold(arrays["food_prob"][known], arrays["food"][known] > 0.5, default)

    known = arrays["container_known"]
    for c, name in enumerate(container_classes):
        probs, targets = arrays["container_prob"][known, c], arrays["container"][known, c] > 0.5
        if name in safety_classes:
            thr, src = recall_target_threshold(probs, targets, eval_cfg["safety_recall_target"], default)
        else:
            thr, src = best_f1_threshold(probs, targets, default)
        cal["container_thresholds"][name], sources[f"container/{name}"] = thr, src

    known = arrays["accessory_known"]
    for c, name in enumerate(accessory_classes):
        thr, src = best_f1_threshold(arrays["accessory_prob"][known, c], arrays["accessory"][known, c] > 0.5, default)
        cal["accessory_thresholds"][name], sources[f"accessory/{name}"] = thr, src

    sel = is_oven & arrays["rack_known"]
    prob = masked_softmax(arrays["rack_logits"][sel]) if sel.any() else np.zeros((0, 1))
    cal["rack_conf_threshold"], sources["rack_conf"] = rack_confidence_threshold(
        prob.max(1) if len(prob) else np.zeros(0),
        prob.argmax(1) == arrays["rack_level"][sel] if len(prob) else np.zeros(0, dtype=bool),
        eval_cfg["rack_target_precision"],
        default,
    )

    oven_rows = np.flatnonzero(is_oven)
    models = [labels[arrays["index"][row]].device_model for row in oven_rows]
    cal["tau_calibration"] = {}
    for name, gallery in galleries.items():
        result = None
        if len(gallery) and len(oven_rows):
            result = calibrate_tau(gallery, torch.from_numpy(arrays[FEATURE_KEYS[name]][oven_rows]), models)
        if result is None:
            sources[f"tau/{name}"] = "default(需要 2 个以上 cavity_group)"
        else:
            cal["tau"][name] = result["tau"]
            sources[f"tau/{name}"] = "gallery_loo"
            cal["tau_calibration"][name] = result
    cal["sources"] = sources
    return cal
