# train.py
"""
一体机多任务视觉识别模型 · 训练入口（设计文档：docs/oven-multitask-design.md）

阶段 1：DINOv3 预训练权重冻结、LoRA 微调，Is-Oven / Food / Container / Accessory / Rack 各头与 Proj Head 联合训练；
        Rack Head 不加型号条件，只用真实型号的层数掩码。
阶段 2：训练结束后用 ckpt_best.pt 建特征库（gallery.pt），在验证集上标定 tau、多标签逐类阈值、
        层位置信度阈值（calibration.json），再用这些阈值评估测试集（test_report.json / test_predictions.json）。
        tau 更准确的标定见 script/4-留一型号验证.py（每个型号轮流留出、重训后标定）。

用法：
  python train.py --config configs/default_oven.yaml --device auto
  python train.py --resume output/oven/<run>/ckpt_last.pt
  python train.py --config <配置> --init output/oven/<run>/ckpt_best.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from dino_finetune.config import get_dino_paths, load_config, resolve_path, validate_config
from dino_finetune.data import build_eval_loader, build_train_loader
from dino_finetune.device import DeviceGallery, DeviceSpec, load_device_profile
from dino_finetune.engine import (
    build_galleries,
    build_model,
    collect_outputs,
    exclude_groups,
    group_names_of,
    load_checkpoint,
    make_dataset,
    select_gallery_indices,
    to_device,
)
from dino_finetune.inference import OvenPostprocessor
from dino_finetune.labels import LabelSchema, OvenLabel, load_split, summarize_labels
from dino_finetune.logging import setup_logging
from dino_finetune.losses import MultiTaskLoss, compute_pos_weight
from dino_finetune.metrics import calibrate_thresholds, default_calibration, evaluate_outputs, weighted_score
from dino_finetune.model.oven import OvenMultiTaskModel
from dino_finetune.utils.training_monitor import TrainingMonitor

logger = logging.getLogger("train")

EPOCH_LOG_KEYS = (
    "is_oven_acc",
    "food_acc",
    "container_map",
    "accessory_map",
    "rack_acc",
    "rack_acc_pm1",
    "rack_floor_acc",
    "device_top1_proj",
    "device_top1_cls",
    "rack_acc_e2e",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一体机多任务视觉识别模型训练")
    parser.add_argument("--config", default=None, help="配置文件；默认 configs/default_oven.yaml，续训时默认用 checkpoint 里保存的配置")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--resume", default=None, help="续训 checkpoint（ckpt_last.pt），输出沿用其所在目录")
    parser.add_argument("--init", default=None, help="只加载模型权重（不含优化器状态），从已有 checkpoint 开始新一轮训练")
    parser.add_argument("--output_dir", default=None, help="输出目录，默认 <output.root>/<YYMMDD_HHMMSS>")
    args = parser.parse_args()
    if args.resume and args.init:
        parser.error("--resume 与 --init 不能同时使用")
    return args


def pick_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 --device cuda，但当前环境没有可用的 CUDA")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _next_or_restart(data_iter, data_loader: DataLoader):
    """读取下一个 batch；迭代结束时重建 iterator，不缓存历史 batch。"""
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(data_loader)
        try:
            return next(data_iter), data_iter
        except StopIteration as exc:
            raise RuntimeError("训练 DataLoader 为空，无法开始训练") from exc


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def format_metrics(metrics: Mapping[str, float | None], keys: Sequence[str]) -> str:
    return " ".join(f"{key}={_fmt(metrics.get(key))}" for key in keys)


def check_compatible(ckpt: Mapping[str, Any], cfg: Mapping[str, Any], group_names: Sequence[str], *, check_groups: bool) -> None:
    for key in ("container", "accessory"):
        if list(ckpt.get("labels", {}).get(key, [])) != list(cfg["labels"][key]):
            raise ValueError(f"checkpoint 的 labels.{key} 类别表与当前配置不一致，不能加载")
    if int(ckpt.get("max_rack", -1)) != int(cfg["model"]["max_rack"]):
        raise ValueError(f"checkpoint 的 max_rack={ckpt.get('max_rack')} 与当前配置 {cfg['model']['max_rack']} 不一致")
    if check_groups and list(ckpt.get("group_names", [])) != list(group_names):
        raise ValueError("checkpoint 的 cavity_group 列表与当前 Device Profile 不一致，ArcFace 类中心无法续训")


# =========================
# 数据相关
# =========================
def log_label_summary(split: str, labels: Sequence[OvenLabel]) -> None:
    s = summarize_labels(labels)
    logger.info(
        "%s：样本=%d 一体机=%d 非一体机=%d 型号=%s",
        split, s["num_samples"], s["is_oven"], s["non_oven"], s["device_model"],
    )
    logger.info("%s：食物=%s 层位=%s", split, s["food_exist"], s["rack_level"])
    logger.info("%s：容器=%s", split, s["container"])
    logger.info("%s：附件=%s", split, s["accessory"])


def warn_data_coverage(
    train_labels: Sequence[OvenLabel],
    val_labels: Sequence[OvenLabel],
    profile: Mapping[str, DeviceSpec],
) -> None:
    """提示当前数据覆盖不到的任务，便于理解为什么某些 loss / 指标没有意义。"""
    groups = {profile[lb.device_model].cavity_group for lb in train_labels if lb.is_oven}
    if len(groups) < 2:
        logger.warning(
            "训练集只有 %d 个 cavity_group：Proj Head 的对比学习没有负样本（proj loss 恒为 0），"
            "tau 也无法标定，将使用 eval.tau 默认值",
            len(groups),
        )
    if all(lb.is_oven for lb in train_labels):
        logger.warning("训练集没有非一体机图片：Is-Oven 头只见过正样本")
    food = [lb.food_exist for lb in train_labels if lb.food_exist is not None]
    if food and len(set(food)) == 1:
        logger.warning("训练集食物标签只有一种取值（%s）：Food 头学不到区分能力", "全部有食物" if food[0] else "全部无食物")
    for key, title in (("container", "容器"), ("accessory", "附件")):
        seen = {n for lb in train_labels for n in (getattr(lb, key) or [])}
        unseen = sorted({n for lb in val_labels for n in (getattr(lb, key) or [])} - seen)
        if unseen:
            logger.warning("验证集出现训练集没有的%s类别：%s", title, unseen)


# =========================
# 训练
# =========================
def build_optimizer(model: OvenMultiTaskModel, criterion: nn.Module, tp: Mapping[str, Any]) -> torch.optim.Optimizer:
    decay, no_decay, lora = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            lora.append(param)
        elif param.ndim <= 1:
            no_decay.append(param)
        else:
            decay.append(param)
    decay += [p for p in criterion.parameters() if p.requires_grad]  # ArcFace 类中心

    groups = [
        {"name": "heads", "params": decay, "lr": tp["lr"], "weight_decay": tp["weight_decay"]},
        {"name": "heads_no_decay", "params": no_decay, "lr": tp["lr"], "weight_decay": 0.0},
        {"name": "lora", "params": lora, "lr": tp["lr_lora"], "weight_decay": tp["weight_decay"]},
    ]
    groups = [group for group in groups if group["params"]]
    for group in groups:
        logger.info(
            "参数组 %s：%d 个张量，%.2fM 参数，lr=%.3g weight_decay=%.3g",
            group["name"], len(group["params"]), sum(p.numel() for p in group["params"]) / 1e6,
            group["lr"], group["weight_decay"],
        )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, lr: float, min_lr: float):
    """按 step 的 warmup + cosine；各参数组按同一比例衰减到 min_lr / lr。"""
    warmup_steps = min(warmup_steps, total_steps)
    min_lr_ratio = float(min_lr) / float(lr)

    def lr_lambda(step: int) -> float:
        step = min(step, total_steps - 1)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def clip_head_grads(groups: Mapping[str, Sequence[nn.Parameter]], max_norm: float) -> dict[str, float]:
    """
    逐组裁剪梯度并返回裁剪前各组的梯度 L2 范数（需在 scaler.unscale_ 之后调用）。
    各头参数互不共享，逐头裁剪可以避免某个头的大梯度（如 ArcFace）把其他头的更新一起压小；max_norm <= 0 时只统计不裁剪。
    """
    norms = {}
    for name, params in groups.items():
        params = [p for p in params if p.grad is not None]
        if not params:
            norms[name] = 0.0
        elif max_norm > 0:
            norms[name] = float(nn.utils.clip_grad_norm_(params, max_norm=max_norm))
        else:
            norms[name] = float(torch.stack([p.grad.detach().float().norm() for p in params]).norm())
    return norms


def main() -> None:
    args = parse_args()
    setup_logging(name="train", use_shanghai_time=True)

    # =========================
    # 1) 配置、输出目录与设备
    # =========================
    resume_path, resume_ckpt = None, None
    if args.resume:
        resume_path, resume_ckpt = load_checkpoint(args.resume)
        if args.config:
            cfg = load_config(args.config)
        else:
            cfg = resume_ckpt["cfg"]
            validate_config(cfg)
        run_dir = resume_path.parent
    else:
        cfg = load_config(args.config)
        if args.output_dir:
            run_dir = Path(args.output_dir).expanduser().resolve()
        else:
            run_dir = resolve_path(cfg["output"]["root"]) / datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(name="train", log_file=str(run_dir / "train.log"), use_shanghai_time=True)

    device = pick_device(args.device)
    tp, ev = cfg["trainparams"], cfg["eval"]
    use_amp = bool(tp["use_amp"] and device.type == "cuda")
    set_seed(tp["seed"])
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    logger.info("输出目录：%s", run_dir)
    if resume_path is not None:
        logger.info("续训 checkpoint：%s", resume_path)

    # =========================
    # 2) 类别表、Device Profile 与标注
    # =========================
    schema = LabelSchema.from_config(cfg)
    max_rack = cfg["model"]["max_rack"]
    profile = load_device_profile(
        resolve_path(cfg["device_profile"]),
        max_rack=max_rack,
        accessory_classes=schema.accessory_classes,
    )
    group_names = group_names_of(profile)
    logger.info("Device Profile：%d 个型号，%d 个 cavity_group", len(profile), len(group_names))
    write_json(run_dir / "device_profile.json", {name: spec.to_dict() for name, spec in profile.items()})

    data_cfg = cfg["data"]
    roots = [resolve_path(root) for root in data_cfg["root"]]
    splits = {"train": data_cfg["train_split"], "val": data_cfg["val_split"]}
    if data_cfg["test_split"] and ev["run_test"]:
        splits["test"] = data_cfg["test_split"]
    split_labels: dict[str, list[OvenLabel]] = {}
    label_issues: dict[str, list[dict[str, str]]] = {}
    for key, split in splits.items():
        labels, issues = load_split(roots, split, schema, profile, on_error=data_cfg["on_error"], strict=data_cfg["strict"])
        label_issues[split] = [asdict(issue) for issue in issues]
        if data_cfg["exclude_groups"]:
            kept = exclude_groups(labels, profile, data_cfg["exclude_groups"])
            logger.info("%s：排除 cavity_group=%s 的 %d 张图", split, data_cfg["exclude_groups"], len(labels) - len(kept))
            labels = kept
        if not labels:
            raise RuntimeError(f"{split} 划分没有可用样本")
        split_labels[key] = labels
        log_label_summary(split, labels)
    write_json(run_dir / "label_issues.json", label_issues)
    warn_data_coverage(split_labels["train"], split_labels["val"], profile)

    # =========================
    # 3) 数据集与 DataLoader
    # =========================
    train_ds = make_dataset(split_labels["train"], cfg, profile, is_train=True)
    train_eval_ds = make_dataset(split_labels["train"], cfg, profile, is_train=False)  # 建特征库用，不做增强
    val_ds = make_dataset(split_labels["val"], cfg, profile, is_train=False)
    test_ds = make_dataset(split_labels["test"], cfg, profile, is_train=False) if "test" in split_labels else None
    logger.info("训练预处理：%s", train_ds.transform.describe())

    batch_size = tp["batch_size"]
    steps_per_epoch = tp["steps_per_epoch"] or math.ceil(len(train_ds) / batch_size)
    pin_memory = device.type == "cuda"
    eval_kwargs = dict(batch_size=batch_size, num_workers=tp["num_workers_eval"], pin_memory=pin_memory)
    train_loader = build_train_loader(
        train_ds,
        batch_size=batch_size,
        sampler_cfg=cfg["sampler"],
        steps_per_epoch=steps_per_epoch,
        num_workers=tp["num_workers"],
        seed=tp["seed"],
        pin_memory=pin_memory,
    )
    val_loader = build_eval_loader(val_ds, persistent=True, **eval_kwargs)
    gallery_indices = select_gallery_indices(train_eval_ds.labels, ev["gallery_max_per_model"], tp["seed"])
    gallery_loader = (
        build_eval_loader(Subset(train_eval_ds, gallery_indices), persistent=True, **eval_kwargs)
        if gallery_indices
        else None
    )
    test_loader = build_eval_loader(test_ds, **eval_kwargs) if test_ds is not None else None
    logger.info(
        "dataloader：train=%d 张（每轮 %d 步）val=%d 张 特征库参考图=%d 张 test=%s 张",
        len(train_ds), steps_per_epoch, len(val_ds), len(gallery_indices), len(test_ds) if test_ds else "-",
    )

    # =========================
    # 4) 模型与 loss
    # =========================
    _, weight_path = get_dino_paths(cfg)
    model = build_model(cfg, device)
    model_cfg = cfg["model"]
    logger.info(
        "模型：%s，可训练参数 %.2fM",
        f"LoRA 微调（{model.lora_blocks} 个 block，rank={model_cfg['lora_rank']}）" if model_cfg["use_lora"] else "骨干完全冻结",
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
    )

    loss_cfg = cfg["loss"]
    pos_weights: dict[str, torch.Tensor | None] = {"container": None, "accessory": None}
    if loss_cfg["pos_weight"] == "auto":
        for key, classes in (("container", schema.container_classes), ("accessory", schema.accessory_classes)):
            pos_weights[key] = compute_pos_weight(
                getattr(train_ds, key), getattr(train_ds, f"{key}_known"), loss_cfg["pos_weight_max"]
            )
            logger.info("%s pos_weight：%s", key, {n: round(float(w), 3) for n, w in zip(classes, pos_weights[key])})
    criterion = MultiTaskLoss.from_config(
        cfg,
        container_pos_weight=pos_weights["container"],
        accessory_pos_weight=pos_weights["accessory"],
        proj_dim=model_cfg["proj_dim"],
        num_groups=len(group_names),
    ).to(device)

    # =========================
    # 5) 优化器、调度器
    # =========================
    optimizer = build_optimizer(model, criterion, tp)
    total_steps = tp["epochs"] * steps_per_epoch
    scheduler = build_scheduler(optimizer, total_steps, tp["warmup_steps"], tp["lr"], tp["min_lr"])
    scaler = torch.amp.GradScaler(enabled=use_amp)
    grad_groups = model.head_parameters()
    grad_groups["proj"] = grad_groups["proj"] + [p for p in criterion.parameters() if p.requires_grad]  # ArcFace 类中心
    monitor = TrainingMonitor(run_dir)

    # =========================
    # 6) 续训 / 初始化
    # =========================
    start_epoch, global_step = 0, 0
    best_score, best_val_loss = float("-inf"), float("inf")
    best_path, last_path = run_dir / "ckpt_best.pt", run_dir / "ckpt_last.pt"
    if resume_ckpt is not None:
        check_compatible(resume_ckpt, cfg, group_names, check_groups=loss_cfg["metric"] == "arcface")
        model.load_trainable_state_dict(resume_ckpt["model"])
        criterion.load_state_dict(resume_ckpt["criterion"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        if use_amp and resume_ckpt.get("scaler"):
            scaler.load_state_dict(resume_ckpt["scaler"])
        monitor.load_state_dict(resume_ckpt.get("monitor"))
        start_epoch = int(resume_ckpt["epoch"]) + 1
        global_step = int(resume_ckpt["global_step"])
        best_score = float(resume_ckpt["best_score"])
        best_val_loss = float(resume_ckpt.get("best_val_loss", float("inf")))
        logger.info("从第 %d 轮继续：global_step=%d best_score=%.4f", start_epoch + 1, global_step, best_score)
        del resume_ckpt
    elif args.init:
        init_path, init_ckpt = load_checkpoint(args.init)
        check_compatible(init_ckpt, cfg, group_names, check_groups=False)
        missing, unexpected = model.load_trainable_state_dict(init_ckpt["model"], strict=False)
        logger.info(
            "从 %s 初始化模型权重：缺失=%d（新加入的 LoRA 等参数，保持初始值）多余=%d",
            init_path, len(missing), len(unexpected),
        )
        del init_ckpt

    logger.info("========== 训练参数配置 ==========")
    logger.info("epochs=%d batch_size=%d steps_per_epoch=%d total_steps=%d", tp["epochs"], batch_size, steps_per_epoch, total_steps)
    logger.info("lr=%.3g lr_lora=%.3g weight_decay=%.3g min_lr=%.3g warmup_steps=%d", tp["lr"], tp["lr_lora"], tp["weight_decay"], tp["min_lr"], tp["warmup_steps"])
    logger.info("use_amp=%s grad_clip=%.3f 采样=%s", use_amp, tp["grad_clip"], cfg["sampler"]["type"])
    logger.info("loss 权重=%s metric=%s multilabel=%s rack_smoothing=%.3f", loss_cfg["weights"], loss_cfg["metric"], loss_cfg["multilabel"], loss_cfg["rack_smoothing"])
    logger.info("best 综合分权重=%s", ev["score_weights"])
    logger.info("=================================")

    # =========================
    # 7) 训练循环
    # =========================
    container_classes, accessory_classes = schema.container_classes, schema.accessory_classes
    safety_classes = cfg["labels"]["safety_container"]
    eval_kwargs_metrics = dict(
        container_classes=container_classes,
        accessory_classes=accessory_classes,
        safety_classes=safety_classes,
        profile=profile,
    )
    default_cal = default_calibration(ev, container_classes, accessory_classes)

    if start_epoch >= tp["epochs"]:
        logger.info("checkpoint 已完成配置中的全部 %d 轮训练，直接进入阶段 2", tp["epochs"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, tp["epochs"]):
        model.train()
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        count, bad_steps = 0, 0
        train_iter = iter(train_loader)

        for step in range(steps_per_epoch):
            batch, train_iter = _next_or_restart(train_iter, train_loader)
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                out = model(batch["image"], batch["rack_count"], batch["floor_usable"])
                loss, terms = criterion(out, batch)

            if not torch.isfinite(loss):
                bad_steps += 1
                names = [Path(train_ds.labels[i].image_path).name for i in batch["index"].tolist()[:8]]
                logger.warning(
                    "轮次=%d 步=%d loss 非有限值，跳过该步：%s；样本：%s",
                    epoch + 1, step, {k: float(v) for k, v in terms.items()}, names,
                )
                if bad_steps >= 10:
                    raise RuntimeError("同一轮内已有 10 步 loss 非有限值，请检查标注与学习率")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norms = clip_head_grads(grad_groups, tp["grad_clip"])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if not use_amp or scaler.get_scale() >= scale_before:  # GradScaler 因梯度溢出跳过本步时，学习率调度也不前进
                scheduler.step()

            global_step += 1
            count += 1
            step_values = {"loss": float(loss), **{f"loss_{t}": float(v) for t, v in terms.items()}}
            if all(math.isfinite(v) for v in grad_norms.values()):  # AMP 溢出的步会被 GradScaler 跳过，不计入梯度范数
                step_values.update({f"gn_{name}": value for name, value in grad_norms.items()})
            for key, value in step_values.items():
                sums[key] += value
                counts[key] += 1

            if step % tp["log_every"] == 0 or step == steps_per_epoch - 1:
                avg = {key: sums[key] / counts[key] for key in sums}
                logger.info(
                    "轮次=%d 步=%d/%d 总损失=%.4f | %s | 梯度范数 %s | lrs=[%s]",
                    epoch + 1, step + 1, steps_per_epoch, avg["loss"],
                    " ".join(f"{t}={avg[f'loss_{t}']:.4f}" for t in terms),
                    " ".join(f"{n}={avg[f'gn_{n}']:.3g}" if f"gn_{n}" in avg else f"{n}=-" for n in grad_norms),
                    ", ".join(f"{group['lr']:.3g}" for group in optimizer.param_groups),
                )

        if count == 0:
            raise RuntimeError(f"第 {epoch + 1} 轮没有有效的训练步")
        train_metrics = {f"train_{key}": sums[key] / counts[key] for key in sums}

        # ===== val =====
        galleries: dict[str, DeviceGallery] = {}
        if gallery_loader is not None:
            gallery_arrays, _ = collect_outputs(model, gallery_loader, device, use_amp)
            galleries = build_galleries(gallery_arrays, train_eval_ds.labels, profile)
        val_arrays, val_losses = collect_outputs(model, val_loader, device, use_amp, criterion)
        val_flat, _ = evaluate_outputs(
            val_arrays, val_ds.labels, galleries=galleries, calibration=default_cal, **eval_kwargs_metrics
        )
        score = weighted_score(val_flat, ev["score_weights"])
        if score is None:
            score = -val_losses["loss"]

        epoch_metrics: dict[str, float | None] = {
            **train_metrics,
            **{f"val_{key}": value for key, value in val_losses.items()},
            **{f"val_{key}": value for key, value in val_flat.items()},
            "score": score,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        monitor.update(epoch, epoch_metrics)
        memory = f" 显存峰值={torch.cuda.max_memory_allocated(device) / 2**30:.1f}G" if device.type == "cuda" else ""
        logger.info(
            "轮次=%d 完成：train_loss=%.4f val_loss=%.4f | %s | score=%.4f%s",
            epoch + 1, train_metrics["train_loss"], val_losses["loss"], format_metrics(val_flat, EPOCH_LOG_KEYS), score, memory,
        )

        # 综合分更高即为最佳；验证集小、指标饱和时综合分常常持平，此时取验证 loss 更低的
        val_loss = val_losses["loss"]
        is_best = score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9 and val_loss < best_val_loss)
        if is_best:
            best_score, best_val_loss = score, val_loss
        checkpoint = {
            "model": model.trainable_state_dict(),
            "criterion": criterion.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "cfg": cfg,
            "labels": {"container": container_classes, "accessory": accessory_classes},
            "group_names": group_names,
            "device_profile": {name: spec.to_dict() for name, spec in profile.items()},
            "max_rack": max_rack,
            "metrics": {key: value for key, value in epoch_metrics.items() if value is not None},
            "score": score,
            "best_score": best_score,
            "best_val_loss": best_val_loss,
            "monitor": monitor.state_dict(),
        }
        if is_best:
            torch.save({**checkpoint, "type": "best"}, best_path)
            logger.info("已更新最佳模型：score=%.4f val_loss=%.4f -> %s", best_score, best_val_loss, best_path)
        torch.save(
            {
                **checkpoint,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "type": "last",
            },
            last_path,
        )
        logger.info("已覆盖保存最新 checkpoint：%s", last_path)

    # =========================
    # 8) 阶段 2：建特征库、标定阈值、评估测试集
    # =========================
    if not ev["run_stage2"]:
        logger.info("eval.run_stage2=false，跳过阶段 2。输出目录：%s", run_dir)
        return
    if not best_path.is_file():
        logger.warning("没有 ckpt_best.pt，跳过阶段 2")
        return
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_trainable_state_dict(best_ckpt["model"])
    best_epoch = int(best_ckpt["epoch"])
    logger.info("阶段 2：加载 %s（第 %d 轮，score=%.4f）", best_path.name, best_epoch + 1, float(best_ckpt["score"]))
    del best_ckpt

    galleries = {}
    if gallery_loader is not None:
        gallery_arrays, _ = collect_outputs(model, gallery_loader, device, use_amp)
        galleries = build_galleries(gallery_arrays, train_eval_ds.labels, profile)
    fingerprint = model.gallery_fingerprint(Path(weight_path).name)
    version = {"fingerprint": fingerprint, "checkpoint": best_path.name, "epoch": best_epoch}
    torch.save(
        {
            "meta": {
                **version,
                "weight": Path(weight_path).name,
                "img_dim": cfg["input"]["img_dim"],
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "num_refs": dict(Counter(galleries["proj"].labels)) if galleries else {},
            },
            **{name: gallery.state_dict() for name, gallery in galleries.items()},
        },
        run_dir / "gallery.pt",
    )
    logger.info("特征库已保存：%s（版本 %s）", run_dir / "gallery.pt", fingerprint)

    val_arrays, _ = collect_outputs(model, val_loader, device, use_amp)
    calibration = calibrate_thresholds(
        val_arrays,
        val_ds.labels,
        eval_cfg=ev,
        container_classes=container_classes,
        accessory_classes=accessory_classes,
        safety_classes=safety_classes,
        galleries=galleries,
    )
    calibration.update(version)
    write_json(run_dir / "calibration.json", calibration)
    calibrated = sorted(key for key, src in calibration["sources"].items() if not src.startswith("default"))
    logger.info(
        "阈值标定完成：tau=%s rack_conf=%.4f，已标定 %d 项，其余用默认值（详见 calibration.json）：%s",
        calibration["tau"], calibration["rack_conf_threshold"], len(calibrated), calibrated,
    )

    val_flat, val_report = evaluate_outputs(
        val_arrays, val_ds.labels, galleries=galleries, calibration=calibration, **eval_kwargs_metrics
    )
    write_json(run_dir / "val_report.json", {**version, **val_report})
    logger.info("验证集（标定后阈值）：%s", format_metrics(val_flat, EPOCH_LOG_KEYS))

    if test_loader is not None:
        test_arrays, test_losses = collect_outputs(model, test_loader, device, use_amp, criterion)
        test_flat, test_report = evaluate_outputs(
            test_arrays, test_ds.labels, galleries=galleries, calibration=calibration, **eval_kwargs_metrics
        )
        write_json(run_dir / "test_report.json", {**version, "loss": test_losses, **test_report})
        logger.info("测试集：loss=%.4f %s", test_losses["loss"], format_metrics(test_flat, EPOCH_LOG_KEYS))

        # 按设计文档第 3 节的完整推理流程给出测试集逐张预测，附上标注便于对照
        post = OvenPostprocessor(
            calibration=calibration,
            profile=profile,
            container_classes=container_classes,
            accessory_classes=accessory_classes,
            galleries=galleries,
        )
        predictions = []
        for index, result in zip(test_arrays["index"], post(test_arrays)):
            label = test_ds.labels[int(index)]
            predictions.append({"image": label.image_path, **result, "label": asdict(label)})
        write_json(run_dir / "test_predictions.json", predictions)
    logger.info("全部完成，输出目录：%s", run_dir)


if __name__ == "__main__":
    main()
