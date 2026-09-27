# script/4-留一型号验证.py
"""
留一型号验证（设计文档第 9 节）：每个 cavity_group 轮流做一遍——
  1. 去掉该型号的全部图片，用与原训练相同的配置重新训练（LoRA + 各头 + Proj Head；可用 loo.epochs / --epochs 缩短）；
  2. 特征库只放其余型号的参考图，测该型号的图片能否被判为"未知型号"（拒识率）；
  3. 把该型号的参考图加入特征库（不重训），测它的识别率；
  4. 汇总所有折中"库内图片"与"库外图片"的 top-1 相似度标定 tau，并比较 proj / cls 两种检索特征
     （另附关掉 LoRA 的原始 DINOv3 CLS 作参照）。
查询图：被留出的型号优先用验证集图片，验证集里没有时从它的训练图里留出 loo.holdout_frac；库内型号只用验证集图片。
结果写入 <run>/loo_report.json，并把 tau 和推荐的检索特征写回 <run>/calibration.json（原文件备份为 calibration.before_loo.json）。

用法：
  python script/4-留一型号验证.py --run output/oven/<YYMMDD_HHMMSS>
  python script/4-留一型号验证.py --run output/oven/<run> --epochs 10 --groups G5A,G5B
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 把项目根目录加进去

import argparse
import copy
import json
import logging
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from dino_finetune.config import PROJECT_ROOT, resolve_path
from dino_finetune.data import build_eval_loader
from dino_finetune.device import DeviceGallery, DeviceSpec, choose_tau
from dino_finetune.engine import (
    FEATURES,
    build_model,
    collect_outputs,
    exclude_groups,
    load_checkpoint,
    load_profile_snapshot,
    load_run_config,
    make_dataset,
    select_gallery_indices,
)
from dino_finetune.labels import LabelSchema, OvenLabel, load_split
from dino_finetune.logging import setup_logging

logger = logging.getLogger("loo")

BASELINE = "dino"  # 关掉 LoRA 的原始 DINOv3 CLS，只作参照，不参与推荐


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="留一型号验证：标定 tau，评估新型号的拒识率与入库后识别率")
    parser.add_argument("--run", required=True, help="train.py 的输出目录（需要已完成阶段 2）")
    parser.add_argument("--epochs", type=int, default=None, help="每折训练轮数，默认取配置 loo.epochs（0 表示与原训练相同）")
    parser.add_argument("--groups", default=None, help="只做这些 cavity_group，逗号分隔；默认全部")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--retrain", action="store_true", help="忽略已完成的折，全部重新训练")
    parser.add_argument("--no_apply", action="store_true", help="只写 loo_report.json，不改 calibration.json")
    return parser.parse_args()


def safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def rate(values: Sequence[bool]) -> float | None:
    return float(np.mean(values)) if len(values) else None


# =========================
# 查询图与参考图
# =========================
def split_queries(
    train_labels: Sequence[OvenLabel],
    val_labels: Sequence[OvenLabel],
    profile: Mapping[str, DeviceSpec],
    holdout_frac: float,
    max_per_model: int,
    seed: int,
) -> tuple[dict[str, list[OvenLabel]], dict[str, list[OvenLabel]], dict[str, list[OvenLabel]]]:
    """返回 (参考图, 留出时的查询图, 库内时的查询图)，均按 cavity_group 分组。"""
    rng = np.random.default_rng(seed)
    train_by, val_by = defaultdict(list), defaultdict(list)
    for labels, by_group in ((train_labels, train_by), (val_labels, val_by)):
        for lb in labels:
            if lb.is_oven:
                by_group[profile[lb.device_model].cavity_group].append(lb)

    references, out_queries, in_queries = {}, {}, {}
    for g, items in sorted(train_by.items()):
        if val_by.get(g):
            refs, out_queries[g] = items, val_by[g]
        else:
            order = rng.permutation(len(items))
            n_hold = min(max(1, int(round(len(items) * holdout_frac))), len(items) - 1)
            out_queries[g] = [items[i] for i in order[:n_hold]]
            refs = [items[i] for i in order[n_hold:]]
        references[g] = [refs[i] for i in select_gallery_indices(refs, max_per_model, seed)]
        in_queries[g] = val_by.get(g, [])  # 库内型号的查询图必须是训练没见过的验证集图片
    return references, out_queries, in_queries


# =========================
# 每一折
# =========================
def train_fold(cfg: Mapping[str, Any], run_dir: Path, fold_dir: Path, group: str, epochs: int, device: str) -> None:
    fold_cfg = copy.deepcopy(dict(cfg))
    fold_cfg["data"]["exclude_groups"] = sorted(set(cfg["data"]["exclude_groups"]) | {group})
    fold_cfg["eval"]["run_stage2"] = False
    fold_cfg["eval"]["run_test"] = False
    fold_cfg["device_profile"] = str(run_dir / "device_profile.json")
    if epochs > 0:
        fold_cfg["trainparams"]["epochs"] = epochs
    fold_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = fold_dir / "fold_config.yaml"
    cfg_path.write_text(yaml.safe_dump(fold_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    cmd = [sys.executable, str(PROJECT_ROOT / "train.py"), "--config", str(cfg_path), "--output_dir", str(fold_dir), "--device", device]
    logger.info("训练第 %s 折（留出 %s）：%s", fold_dir.name, group, " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    (fold_dir / "train_done.json").write_text(json.dumps({"group": group}), encoding="utf-8")


def evaluate_fold(
    held: str,
    feats: Mapping[str, np.ndarray],
    row_of: Mapping[str, int],
    references: Mapping[str, list[OvenLabel]],
    out_queries: Mapping[str, list[OvenLabel]],
    in_queries: Mapping[str, list[OvenLabel]],
    profile: Mapping[str, DeviceSpec],
    knn_k: int,
) -> dict[str, Any]:
    """
    gallery_a：除被留出型号外的参考图；gallery_b：再加上被留出型号的参考图。
    记录不加 tau 时的 top-1 相似度与投票结果，tau 汇总所有折后再定。
    """
    group_of = {name: spec.cavity_group for name, spec in profile.items()}
    in_items = [lb for g, items in sorted(in_queries.items()) if g != held for lb in items]
    out_rows = [row_of[lb.image_path] for lb in out_queries[held]]
    in_rows = [row_of[lb.image_path] for lb in in_items]
    result = {}
    for name, x in feats.items():
        gallery_a, gallery_b = DeviceGallery(group_of), DeviceGallery(group_of)
        for g, refs in sorted(references.items()):
            by_model = defaultdict(list)
            for lb in refs:
                by_model[lb.device_model].append(row_of[lb.image_path])
            for model_name, model_rows in sorted(by_model.items()):
                if g != held:
                    gallery_a.add(model_name, torch.from_numpy(x[model_rows]))
                gallery_b.add(model_name, torch.from_numpy(x[model_rows]))

        q_out = torch.from_numpy(x[out_rows])
        before = gallery_a.query_batch(q_out, k=knn_k, tau=-2.0)  # tau=-2：不拒识，只取投票结果
        after = gallery_b.query_batch(q_out, k=knn_k, tau=-2.0)
        in_matches = gallery_a.query_batch(torch.from_numpy(x[in_rows]), k=knn_k, tau=-2.0) if in_rows else []
        result[name] = {
            "out_top1": [m.top1_sim for m in before],
            "out_pred": [m.group for m in before],
            "add_top1": [m.top1_sim for m in after],
            "add_pred": [m.group for m in after],
            "in_top1": [m.top1_sim for m in in_matches],
            "in_correct": [m.group == group_of[lb.device_model] for m, lb in zip(in_matches, in_items)],
        }
    return result


# =========================
# 汇总
# =========================
def summarize(folds: Mapping[str, Mapping[str, Any]], profile: Mapping[str, DeviceSpec], feature: str) -> dict[str, Any]:
    s_in = np.array([s for fold in folds.values() for s in fold[feature]["in_top1"]])
    s_out = np.array([s for fold in folds.values() for s in fold[feature]["out_top1"]])
    if len(s_in) == 0 or len(s_out) == 0:
        return {"tau": None, "note": "缺少库内或库外查询图，无法标定"}
    calib = choose_tau(s_in, s_out)
    tau = calib["tau"]

    per_group, in_hits = {}, []
    for g, fold in folds.items():
        d = fold[feature]
        out_top1, add_top1 = np.array(d["out_top1"]), np.array(d["add_top1"])
        accepted = out_top1 >= tau
        in_hits += [ok and s >= tau for s, ok in zip(d["in_top1"], d["in_correct"])]
        per_group[g] = {
            "models": sorted(name for name, spec in profile.items() if spec.cavity_group == g),
            "queries": int(len(out_top1)),
            "reject_rate": float((~accepted).mean()),
            "confused_with": dict(Counter(p for p, a in zip(d["out_pred"], accepted) if a)),
            "recognition_after_add": float(((add_top1 >= tau) & (np.array(d["add_pred"]) == g)).mean()),
            "mean_top1_before_add": float(out_top1.mean()),
            "mean_top1_after_add": float(add_top1.mean()),
        }
    summary = {
        "tau": tau,
        "tau_calibration": calib,
        "reject_rate": rate([v["reject_rate"] for v in per_group.values()]),
        "recognition_after_add": rate([v["recognition_after_add"] for v in per_group.values()]),
        "in_gallery_top1": rate(in_hits),
        "per_group": per_group,
    }
    parts = [summary["reject_rate"], summary["recognition_after_add"], summary["in_gallery_top1"]]
    summary["score"] = rate([p for p in parts if p is not None])
    return summary


def apply_to_calibration(run_dir: Path, report: Mapping[str, Any]) -> None:
    cal_path = run_dir / "calibration.json"
    calibration = json.loads(cal_path.read_text(encoding="utf-8"))
    backup = run_dir / "calibration.before_loo.json"
    if not backup.exists():
        backup.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in FEATURES:
        summary = report["features"].get(name, {})
        if summary.get("tau") is not None:
            calibration["tau"][name] = summary["tau"]
            calibration["tau_calibration"][name] = {**summary["tau_calibration"], "method": "loo"}
            calibration["sources"][f"tau/{name}"] = "loo"
    if report["recommended_feature"]:
        calibration["gallery_feature"] = report["recommended_feature"]
    calibration["loo"] = {
        "report": "loo_report.json",
        "groups": sorted(report["folds"]),
        "recommended_feature": report["recommended_feature"],
    }
    cal_path.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已更新 %s（原文件备份在 %s）：tau=%s gallery_feature=%s", cal_path, backup.name, calibration["tau"], calibration["gallery_feature"])


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run).expanduser().resolve()
    loo_dir = run_dir / "loo"
    loo_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(name="loo", log_file=str(loo_dir / "loo.log"), use_shanghai_time=True)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    cfg = load_run_config(run_dir)
    profile = load_profile_snapshot(run_dir, cfg)
    schema = LabelSchema.from_config(cfg)
    loo_cfg, tp, ev, data_cfg = cfg["loo"], cfg["trainparams"], cfg["eval"], cfg["data"]
    epochs = args.epochs if args.epochs is not None else loo_cfg["epochs"]

    # =========================
    # 1) 数据与分组
    # =========================
    roots = [resolve_path(root) for root in data_cfg["root"]]
    split_labels = {}
    for key in ("train_split", "val_split"):
        labels, _ = load_split(roots, data_cfg[key], schema, profile, on_error=data_cfg["on_error"], strict=data_cfg["strict"])
        split_labels[key] = exclude_groups(labels, profile, data_cfg["exclude_groups"])
    train_labels, val_labels = split_labels["train_split"], split_labels["val_split"]

    counts = Counter(profile[lb.device_model].cavity_group for lb in train_labels if lb.is_oven)
    if len(counts) < 2:
        raise SystemExit(f"训练集只有 {len(counts)} 个 cavity_group（{dict(counts)}），留一型号验证至少需要 2 个")
    eligible = sorted(g for g, c in counts.items() if c >= loo_cfg["min_group_images"])
    groups = [g.strip() for g in args.groups.split(",")] if args.groups else eligible
    unknown = [g for g in groups if g not in eligible]
    if unknown:
        raise SystemExit(f"这些 cavity_group 不能做留一：{unknown}（可选：{eligible}，训练图数：{dict(counts)}）")

    references, out_queries, in_queries = split_queries(
        train_labels, val_labels, profile, loo_cfg["holdout_frac"], ev["gallery_max_per_model"], tp["seed"]
    )
    eval_labels = list({lb.image_path: lb for items in (references, out_queries, in_queries) for v in items.values() for lb in v}.values())
    row_of = {lb.image_path: i for i, lb in enumerate(eval_labels)}
    loader = build_eval_loader(
        make_dataset(eval_labels, cfg, profile, is_train=False),
        batch_size=tp["batch_size"],
        num_workers=tp["num_workers_eval"],
        pin_memory=device.type == "cuda",
    )
    logger.info(
        "留一型号验证：%d 折 %s，每折 %s 轮；参考图 %s；留出查询图 %s；库内查询图 %s",
        len(groups), groups, epochs or tp["epochs"],
        {g: len(v) for g, v in references.items()}, {g: len(v) for g, v in out_queries.items()},
        {g: len(v) for g, v in in_queries.items()},
    )

    # =========================
    # 2) 逐折训练与评估
    # =========================
    use_amp = bool(tp["use_amp"] and device.type == "cuda")
    baseline_feats: np.ndarray | None = None
    folds: dict[str, Any] = {}
    for group in groups:
        fold_dir = loo_dir / safe_name(group)
        result_path = fold_dir / "loo_fold.json"
        done = (fold_dir / "train_done.json").is_file()
        if args.retrain or not done:
            train_fold(cfg, run_dir, fold_dir, group, epochs, args.device)
        elif result_path.is_file():
            folds[group] = json.loads(result_path.read_text(encoding="utf-8"))
            logger.info("第 %s 折已完成，复用结果（--retrain 可重做）", group)
            continue

        fold_cfg = load_run_config(fold_dir)
        model = build_model(fold_cfg, device)
        _, ckpt = load_checkpoint(fold_dir / "ckpt_best.pt")
        model.load_trainable_state_dict(ckpt["model"])
        arrays, _ = collect_outputs(model, loader, device, use_amp)
        if baseline_feats is None:
            with model.lora_disabled():
                baseline_feats = collect_outputs(model, loader, device, use_amp)[0]["cls"]
        del model, ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()

        feats = {name: arrays[name] for name in FEATURES}
        feats[BASELINE] = baseline_feats
        folds[group] = evaluate_fold(group, feats, row_of, references, out_queries, in_queries, profile, ev["knn_k"])
        result_path.write_text(json.dumps(folds[group], ensure_ascii=False), encoding="utf-8")
        before = folds[group][ev["gallery_feature"]]
        logger.info(
            "第 %s 折完成：留出型号 top-1 相似度（入库前）均值=%.4f，入库后=%.4f",
            group, float(np.mean(before["out_top1"])), float(np.mean(before["add_top1"])),
        )

    # =========================
    # 3) 汇总、推荐特征、写回标定
    # =========================
    summaries = {name: summarize(folds, profile, name) for name in (*FEATURES, BASELINE)}
    candidates = {name: summaries[name]["score"] for name in FEATURES if summaries[name].get("score") is not None}
    recommended = max(candidates, key=candidates.get) if candidates else None
    report = {
        "groups": groups,
        "epochs": epochs or tp["epochs"],
        "recommended_feature": recommended,
        "features": summaries,
        "folds": {g: {"out_queries": len(out_queries[g]), "references": len(references[g])} for g in folds},
        "note": f"{BASELINE} 为关掉 LoRA 的原始 DINOv3 CLS，仅作参照；score 为拒识率、入库后识别率、库内 top-1 的平均",
    }
    (run_dir / "loo_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, s in summaries.items():
        if s.get("tau") is None:
            logger.info("%-4s：%s", name, s["note"])
            continue
        logger.info(
            "%-4s：tau=%.4f 拒识率=%.3f 入库后识别率=%.3f 库内top1=%s score=%.3f",
            name, s["tau"], s["reject_rate"], s["recognition_after_add"],
            "-" if s["in_gallery_top1"] is None else f"{s['in_gallery_top1']:.3f}", s["score"],
        )
    logger.info("推荐检索特征：%s；报告：%s", recommended, run_dir / "loo_report.json")
    if not args.no_apply:
        apply_to_calibration(run_dir, report)


if __name__ == "__main__":
    main()
