# train_mul.py
from __future__ import annotations

import math, logging, argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from dino_finetune.logging import setup_logging
from dino_finetune.config import load_config, default_config_path, resolve_interp, get_dino_paths
from dino_finetune.metrics import (
    SegmentationMetricAccumulator,
    soft_dice_loss,
    binary_soft_dice_loss_fg,
)
from dino_finetune.utils.ckpt_cls import build_encoder
from dino_finetune.utils.training_monitor import TrainingMonitor
from dino_finetune import (
    DINOEncoderLoRA,
    get_dataloader,                  # seg dataloader（来自 data.py）
)

from dino_finetune.data_cls import get_cls_dataloader  # 你现成分类 dl
from dino_finetune.model.dino_multitask import DINOEncoderLoRA_MultiTask

logger = logging.getLogger("train_mul")


def _next_or_restart(data_iter, data_loader, task_name: str):
    """读取下一个 batch；迭代结束时重建 iterator，不缓存历史 batch。"""
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(data_loader)
        try:
            return next(data_iter), data_iter
        except StopIteration as exc:
            raise RuntimeError(f"{task_name} DataLoader 为空，无法开始训练") from exc


def _normalize_int_mapping(mapping: Any) -> dict[int, int]:
    if not isinstance(mapping, dict):
        return {}
    try:
        return {int(key): int(value) for key, value in mapping.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint 中的整数映射格式错误") from exc


def topk_acc(logits: torch.Tensor, labels: torch.Tensor, ks=(1, 5)) -> dict[str, float]:
    with torch.no_grad():
        maxk = min(max(ks), int(logits.shape[1]))
        pred = logits.topk(maxk, dim=1).indices  # (B, maxk)
        correct = pred.eq(labels.view(-1, 1))
        out = {}
        for k in ks:
            k_eff = min(int(k), maxk)
            out[f"top{k}"] = float(correct[:, :k_eff].any(dim=1).float().mean().item())
        return out


def main():
    setup_logging(name="train_mul", use_shanghai_time=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config_path("mul"))
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--resume", default=None, help="续训 checkpoint 路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("使用设备：%s", device)

    # =========================
    # 1) seg 配置（640）
    # =========================
    input_cfg = cfg["input"]
    seg_img_dim = tuple(input_cfg["img_dim"])  # (H,W)
    mean = input_cfg["mean"]
    std = input_cfg["std"]
    img_interp = resolve_interp(str(input_cfg.get("img_interp", "linear")))
    msk_interp = resolve_interp(str(input_cfg.get("mask_interp", "nearest")))

    dataset_cfg = cfg["dataset"]
    seg_root = dataset_cfg["root"]
    seg_type = str(dataset_cfg.get("type", "binary")).strip().lower()

    model_cfg = cfg["model"]
    n_classes_seg = int(model_cfg["n_classes"])
    dataset_classes = {"voc": 21, "ade20k": 150, "binary": 2}
    if seg_type in dataset_classes and dataset_classes[seg_type] != n_classes_seg:
        raise ValueError(
            f"n_classes 与数据集类型不一致：dataset.type={seg_type} 期望 {dataset_classes[seg_type]}，"
            f"实际为 {n_classes_seg}"
        )

    post_cfg = cfg.get("postprocess", {}) or {}
    ignore_index = int(post_cfg.get("ignore_index", 255))
    iou_mode = str(post_cfg.get("iou_mode", "argmax")).strip().lower()
    seg_threshold = float(post_cfg.get("thr", 0.5))
    if iou_mode == "prob_threshold" and n_classes_seg != 2:
        raise ValueError("prob_threshold 验证模式仅支持二分类分割")

    # =========================
    # 2) cls 配置（224）
    # =========================
    input_cls_cfg = cfg["input_cls"]
    ds_cls_cfg = cfg["dataset_cls"]
    cls_root = ds_cls_cfg["root"]
    cls_type = str(ds_cls_cfg["type"]).strip().lower()
    logger.info("分类数据源：type=%s root=%s", cls_type, cls_root)

    model_cls_cfg = cfg["model_cls"]
    emb_dim_cfg = int(model_cls_cfg["emb_dim"])
    pool = str(model_cls_cfg.get("pool", "cls_token"))

    # =========================
    # 3) train params
    # =========================
    tp = cfg["trainparams"]

    def read_trainparam(key: str, cast):
        """读取必填的 config.trainparams 参数并转换类型。"""
        if key not in tp or tp[key] is None:
            raise ValueError(f"config.trainparams.{key} 为必填项")
        raw_value = tp[key]
        try:
            return cast(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"config.trainparams.{key} 类型错误：{raw_value}") from exc

    epochs = read_trainparam("epochs", int)
    batch_seg = read_trainparam("batch_size_seg", int)
    batch_cls = read_trainparam("batch_size_cls", int)
    configured_steps_per_epoch = read_trainparam("steps_per_epoch", int)
    num_workers_cls = read_trainparam("num_workers_cls", int)
    lr = read_trainparam("lr", float)
    weight_decay = read_trainparam("weight_decay", float)
    min_lr = read_trainparam("min_lr", float)
    warmup_steps = read_trainparam("warmup_steps", int)
    use_amp = bool(read_trainparam("use_amp", int))
    grad_clip = read_trainparam("grad_clip", float)

    dice_w = read_trainparam("dice_weight", float)
    cls_w = read_trainparam("loss_cls_weight", float)
    label_smoothing = read_trainparam("label_smoothing", float)
    rank_r = read_trainparam("rank_r", int)
    use_lora = read_trainparam("use_lora", bool)
    use_fpn = read_trainparam("use_fpn", bool)
    lr_decoder = read_trainparam("lr_decoder", float)
    lr_cls = read_trainparam("lr_cls", float)
    lr_lora = read_trainparam("lr_lora", float)
    wd_decoder = read_trainparam("wd_decoder", float)
    wd_cls = read_trainparam("wd_cls", float)
    wd_lora = read_trainparam("wd_lora", float)
    use_amp = bool(use_amp and device.type == "cuda")

    best_score_seg_weight = read_trainparam("best_score_seg_weight", float)
    if not 0.0 <= best_score_seg_weight <= 1.0:
        raise ValueError("best_score_seg_weight 必须在 0 到 1 之间")
    if not use_amp:
        logger.info("AMP 已关闭（仅在 CUDA 上启用）")

    logger.info("========== 训练参数配置 ==========")
    logger.info("epochs=%d", epochs)
    logger.info("batch_size(seg)=%d batch_size(cls)=%d", batch_seg, batch_cls)

    logger.info("lr=%.6g weight_decay=%.6g min_lr=%.6g", lr, weight_decay, min_lr)
    logger.info("warmup_steps=%d use_amp=%s grad_clip=%.3f", warmup_steps, use_amp, grad_clip)

    logger.info("dice_weight=%.3f cls_weight=%.3f label_smoothing=%.3f",
                dice_w, cls_w, label_smoothing)

    logger.info("验证范围=完整验证集 best_score分割权重=%.3f", best_score_seg_weight)
    logger.info("分割验证模式=%s 阈值=%.3f", iou_mode, seg_threshold)

    logger.info("device=%s", device)
    logger.info("=================================")

    # =========================
    # 4) run dir
    # =========================
    resume_ckpt = None
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"续训 checkpoint 不存在：{resume_path}")
        run_dir = resume_path.parent
        logger.info("续训 checkpoint：%s", resume_path)
    else:
        run_dir = Path("output") / "03-multi" / datetime.now().strftime("%y%m%d")
    run_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 5) dataloaders
    # =========================
    seg_train_loader, seg_val_loader = get_dataloader(
        seg_type,
        img_dim=seg_img_dim,
        batch_size=batch_seg,
        mean=mean,
        std=std,
        img_interp=img_interp,
        msk_interp=msk_interp,
        root_path=seg_root,
        n_classes=n_classes_seg,
        ignore_index=ignore_index,
    )

    train_split = str(ds_cls_cfg.get("train_split", "train"))
    valid_split = str(ds_cls_cfg.get("valid_split", "valid"))

    logger.info("分类数据集：train_split=%s valid_split=%s", train_split, valid_split)
    cls_train_loader = get_cls_dataloader(
        dataroot=cls_root,
        split=train_split,
        input_cfg=input_cls_cfg,
        batch_size=batch_cls,
        shuffle=True,
        num_workers=num_workers_cls,
        ds_cfg=ds_cls_cfg,
        pin_memory=True,
    )
    cls_val_loader = get_cls_dataloader(
        dataroot=cls_root,
        split=valid_split,
        input_cfg=input_cls_cfg,
        batch_size=batch_cls,
        shuffle=False,
        num_workers=num_workers_cls,
        ds_cfg=ds_cls_cfg,
        pin_memory=True,
    )

    num_classes_cls = int(cls_train_loader.dataset.num_classes)
    leaf_id_to_idx = {int(i): int(i) for i in range(num_classes_cls)}
    class_names = list(cls_train_loader.dataset.class_names)
    val_class_names = list(cls_val_loader.dataset.class_names)
    if class_names != val_class_names:
        raise ValueError("分类训练集与验证集的类别名称或顺序不一致")
    class_to_idx = {
        str(name): int(index)
        for name, index in cls_train_loader.dataset.class_to_leaf_id.items()
    }
    for ds in (cls_train_loader.dataset, cls_val_loader.dataset):
        setattr(ds, "leaf_id_to_idx", leaf_id_to_idx)
        setattr(ds, "strict_label_map", True)
    logger.info("local_folder 模式：跳过 preflight，直接使用目录类别映射")


    logger.info(
        "dataloader：seg_train=%d seg_val=%d cls_train=%d cls_val=%d num_classes_cls=%d",
        len(seg_train_loader), len(seg_val_loader), len(cls_train_loader), len(cls_val_loader), num_classes_cls
    )
    if len(seg_train_loader) <= 0 or len(cls_train_loader) <= 0:
        raise RuntimeError("训练 DataLoader 为空，请检查数据量与 batch size")
    if len(seg_val_loader) <= 0 or len(cls_val_loader) <= 0:
        raise RuntimeError("验证 DataLoader 为空，请检查验证集数据")
    steps_per_epoch = (
        configured_steps_per_epoch
        if configured_steps_per_epoch > 0
        else min(len(seg_train_loader), len(cls_train_loader))
    )
    logger.info(
        "每轮训练步数=%d，约等于 seg=%.2f 轮，cls=%.2f 轮",
        steps_per_epoch,
        steps_per_epoch / len(seg_train_loader),
        steps_per_epoch / len(cls_train_loader),
    )
    
    # =========================
    # 4) build model
    # =========================
    dino_local_repo, weight_path = get_dino_paths(cfg)
    encoder = build_encoder(cfg, device, dino_local_repo=dino_local_repo, weight_path=weight_path)
    enc_dim = int(getattr(encoder, "num_features", 0) or 0)
    if enc_dim <= 0:
        raise RuntimeError("无法从 encoder 获取 num_features，请检查 DINO 模型")
    if emb_dim_cfg != enc_dim:
        logger.warning("model_cls.emb_dim=%d 与 encoder.num_features=%d 不一致，已使用 encoder 维度", emb_dim_cfg, enc_dim)
    emb_dim = enc_dim

    seg_model = DINOEncoderLoRA(
        encoder=encoder,
        r=rank_r,
        emb_dim=emb_dim,
        img_dim=seg_img_dim,
        n_classes=n_classes_seg,
        use_lora=use_lora,
        use_fpn=use_fpn,
    ).to(device)

    model = DINOEncoderLoRA_MultiTask(
        seg_model=seg_model,
        num_classes_cls=num_classes_cls,
        pool=pool,
        emb_dim=emb_dim,
    ).to(device)

    if resume_path is not None:
        resume_ckpt = torch.load(resume_path, map_location="cpu")
        if not isinstance(resume_ckpt, dict):
            raise ValueError("续训 checkpoint 格式错误：期望字典")

        saved_num_classes = int(resume_ckpt.get("num_classes_cls", num_classes_cls))
        if saved_num_classes != num_classes_cls:
            raise ValueError(
                f"续训类别数不一致：checkpoint={saved_num_classes} 当前数据集={num_classes_cls}"
            )
        saved_leaf_id_to_idx = _normalize_int_mapping(resume_ckpt.get("leaf_id_to_idx"))
        if saved_leaf_id_to_idx and saved_leaf_id_to_idx != leaf_id_to_idx:
            raise ValueError("续训 checkpoint 的 leaf_id_to_idx 与当前数据集不一致")
        saved_class_to_idx = resume_ckpt.get("class_to_idx")
        if isinstance(saved_class_to_idx, dict):
            normalized_class_to_idx = {
                str(name): int(index) for name, index in saved_class_to_idx.items()
            }
            if normalized_class_to_idx != class_to_idx:
                raise ValueError("续训 checkpoint 的 class_to_idx 与当前数据集不一致")
        else:
            logger.warning("续训 checkpoint 未保存类别名称映射，请确认数据集类别顺序未变化")
    

    # =========================
    # 6) losses
    # =========================
    ce_ignore = ignore_index if ignore_index is not None else -100
    seg_ce = nn.CrossEntropyLoss(ignore_index=ce_ignore).to(device)
    cls_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    # =========================
    # 7) optimizer groups（沿用 train_seg.py 的规则 + cls_head）
    # =========================
    lora_params, decoder_params, cls_params, other_params = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = name.lower()
        if "cls_head" in n:
            cls_params.append(p)
        elif ("lora" in n) or ("linear_a" in n) or ("linear_b" in n):
            lora_params.append(p)
        elif ("fpn" in n) or ("decoder" in n) or ("head" in n):
            decoder_params.append(p)
        else:
            other_params.append(p)

    logger.info("参数分组：decoder=%d lora=%d cls=%d other=%d",
                len(decoder_params), len(lora_params), len(cls_params), len(other_params))

    param_groups = []
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": lr_decoder, "weight_decay": wd_decoder})
    if cls_params:
        param_groups.append({"params": cls_params, "lr": lr_cls, "weight_decay": wd_cls})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr_lora, "weight_decay": wd_lora})
    if other_params:
        param_groups.append({"params": other_params, "lr": lr, "weight_decay": weight_decay})

    optimizer = optim.AdamW(param_groups)

    # scheduler（step-based warmup+cosine，按 train_seg.py 的逻辑）
    total_steps = epochs * steps_per_epoch
    warmup_steps = min(warmup_steps, total_steps)

    def lr_lambda(step: int) -> float:
        step = min(step, total_steps - 1)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_lr_ratio = float(min_lr) / float(lr)
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = torch.amp.GradScaler(enabled=use_amp)
    non_blocking = device.type == "cuda"
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    monitor = TrainingMonitor(run_dir)

    # =========================
    # 8) train loop
    # =========================
    global_step = 0
    start_epoch = 0

    best_top1 = float("-inf")
    best_iou = float("-inf")
    best_score = float("-inf")

    best_checkpoint_path = run_dir / "ckpt_best.pt"
    last_checkpoint_path = run_dir / "ckpt_last.pt"

    if resume_ckpt is not None:
        model_state = resume_ckpt.get("model")
        if not isinstance(model_state, dict):
            raise ValueError("续训 checkpoint 缺少 model 权重")
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        logger.info("续训模型加载完成：缺失=%d 多余=%d", len(missing), len(unexpected))

        if isinstance(resume_ckpt.get("optimizer"), dict):
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        else:
            logger.warning("续训 checkpoint 不含 optimizer，将使用新优化器状态")
        if isinstance(resume_ckpt.get("scheduler"), dict):
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        else:
            logger.warning("续训 checkpoint 不含 scheduler，将使用新调度器状态")
        if use_amp and isinstance(resume_ckpt.get("scaler"), dict):
            scaler.load_state_dict(resume_ckpt["scaler"])

        monitor.load_state_dict(resume_ckpt.get("monitor"))
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        best_top1 = float(resume_ckpt.get("best_top1", resume_ckpt.get("val_top1", best_top1)))
        best_iou = float(
            resume_ckpt.get(
                "best_iou",
                resume_ckpt.get("val_miou", resume_ckpt.get("val_iou", best_iou)),
            )
        )
        best_score = float(resume_ckpt.get("best_score", resume_ckpt.get("score", best_score)))
        logger.info(
            "从第 %d 轮继续：global_step=%d best_top1=%.4f best_miou=%.4f best_score=%.4f",
            start_epoch + 1,
            global_step,
            best_top1,
            best_iou,
            best_score,
        )
        del resume_ckpt

    if start_epoch >= epochs:
        logger.info("checkpoint 已完成配置中的全部 %d 轮训练，无需继续", epochs)
        return

    for epoch in range(start_epoch, epochs):
        model.train()
        ep_loss, ep_seg, ep_cls = 0.0, 0.0, 0.0
        ep_top1, ep_top5 = 0.0, 0.0
        cnt = 0
        seg_train_iter = iter(seg_train_loader)
        cls_train_iter = iter(cls_train_loader)

        for step in range(steps_per_epoch):
            (seg_imgs_cpu, seg_masks_cpu), seg_train_iter = _next_or_restart(
                seg_train_iter,
                seg_train_loader,
                "分割训练",
            )
            (cls_imgs_cpu, cls_labels_cpu, _metas), cls_train_iter = _next_or_restart(
                cls_train_iter,
                cls_train_loader,
                "分类训练",
            )

            optimizer.zero_grad(set_to_none=True)

            seg_imgs = seg_imgs_cpu.to(
                device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )
            seg_masks = seg_masks_cpu.to(
                device,
                dtype=torch.long,
                non_blocking=non_blocking,
            )
            with torch.amp.autocast(amp_device, enabled=use_amp):
                seg_logits = model.forward_seg(seg_imgs)
                ce = seg_ce(seg_logits, seg_masks)
                if n_classes_seg == 2:
                    dice = binary_soft_dice_loss_fg(
                        seg_logits,
                        seg_masks,
                        ignore_index=ignore_index,
                    )
                else:
                    dice = soft_dice_loss(
                        seg_logits,
                        seg_masks,
                        ignore_index=ignore_index,
                    )
                loss_seg = ce + dice_w * dice

            scaler.scale(loss_seg).backward()
            loss_seg_detached = loss_seg.detach()
            seg_loss_value = float(loss_seg_detached.item())
            del seg_logits, ce, dice, loss_seg, seg_imgs, seg_masks

            cls_imgs = cls_imgs_cpu.to(
                device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )
            cls_labels = cls_labels_cpu.to(
                device,
                dtype=torch.long,
                non_blocking=non_blocking,
            )
            with torch.amp.autocast(amp_device, enabled=use_amp):
                cls_logits, embedding = model.forward_cls(cls_imgs)
                loss_cls = cls_ce(cls_logits, cls_labels)
                weighted_cls_loss = cls_w * loss_cls

            scaler.scale(weighted_cls_loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            acc = topk_acc(cls_logits, cls_labels, ks=(1, 5))
            loss = loss_seg_detached + weighted_cls_loss.detach()

            ep_loss += float(loss.item())
            ep_seg += seg_loss_value
            ep_cls += float(loss_cls.item())
            ep_top1 += acc["top1"]
            ep_top5 += acc["top5"]
            cnt += 1
            global_step += 1
            del cls_logits, embedding, loss_cls, weighted_cls_loss, loss, cls_imgs, cls_labels

            if step % 50 == 0:
                lrs = [pg.get("lr", 0.0) for pg in optimizer.param_groups]
                lr_str = "[" + ", ".join([f"{x:.6g}" for x in lrs]) + "]"
                logger.info(
                    "轮次=%d 步=%d 总损失=%.4f 分割=%.4f 分类=%.4f top1=%.3f top5=%.3f lrs=%s",
                    epoch + 1, step,
                    ep_loss / max(cnt, 1),
                    ep_seg / max(cnt, 1),
                    ep_cls / max(cnt, 1),
                    ep_top1 / max(cnt, 1),
                    ep_top5 / max(cnt, 1),
                    lr_str,
                )

        # ===== val =====
        model.eval()
        seg_metric = SegmentationMetricAccumulator(
            num_classes=n_classes_seg,
            ignore_index=ignore_index,
            mode=iou_mode,
            threshold=seg_threshold,
        )
        with torch.no_grad():
            # seg val
            val_seg_loss_sum = 0.0
            val_ce_sum = 0.0
            val_dice_sum = 0.0
            val_seg_samples = 0
            for vimgs, vmasks in seg_val_loader:
                vimgs = vimgs.to(device, dtype=torch.float32, non_blocking=non_blocking)
                vmasks = vmasks.to(device, dtype=torch.long, non_blocking=non_blocking)

                with torch.amp.autocast(amp_device, enabled=use_amp):
                    vlogits = model.forward_seg(vimgs)
                    val_ce_batch = seg_ce(vlogits, vmasks)
                    if n_classes_seg == 2:
                        val_dice_batch = binary_soft_dice_loss_fg(
                            vlogits,
                            vmasks,
                            ignore_index=ignore_index,
                        )
                    else:
                        val_dice_batch = soft_dice_loss(
                            vlogits,
                            vmasks,
                            ignore_index=ignore_index,
                        )
                    val_seg_loss_batch = val_ce_batch + dice_w * val_dice_batch

                seg_metric.update(vlogits, vmasks)
                batch_samples = int(vimgs.shape[0])
                val_seg_loss_sum += float(val_seg_loss_batch.item()) * batch_samples
                val_ce_sum += float(val_ce_batch.item()) * batch_samples
                val_dice_sum += float(val_dice_batch.item()) * batch_samples
                val_seg_samples += batch_samples

            val_seg_denom = max(val_seg_samples, 1)
            val_seg_loss = val_seg_loss_sum / val_seg_denom
            val_ce = val_ce_sum / val_seg_denom
            val_dice = val_dice_sum / val_seg_denom
            seg_metrics = seg_metric.compute()
            val_miou = seg_metrics["miou"]
            val_fg_iou = seg_metrics["fg_iou"]
            val_pixel_acc = seg_metrics["pixel_acc"]

            # cls val
            val_cls_loss_sum = 0.0
            val_top1_sum = 0.0
            val_top5_sum = 0.0
            val_cls_samples = 0
            for vimgs, vlabels, _m in cls_val_loader:
                vimgs = vimgs.to(device, dtype=torch.float32, non_blocking=non_blocking)
                vlabels = vlabels.to(device, dtype=torch.long, non_blocking=non_blocking)

                with torch.amp.autocast(amp_device, enabled=use_amp):
                    vlogits, _emb = model.forward_cls(vimgs)
                    val_cls_loss_batch = cls_ce(vlogits, vlabels)
                acc = topk_acc(vlogits, vlabels, ks=(1, 5))
                batch_samples = int(vlabels.shape[0])
                val_cls_loss_sum += float(val_cls_loss_batch.item()) * batch_samples
                val_top1_sum += acc["top1"] * batch_samples
                val_top5_sum += acc["top5"] * batch_samples
                val_cls_samples += batch_samples

            val_cls_denom = max(val_cls_samples, 1)
            val_cls_loss = val_cls_loss_sum / val_cls_denom
            vtop1 = val_top1_sum / val_cls_denom
            vtop5 = val_top5_sum / val_cls_denom

        val_loss = val_seg_loss + cls_w * val_cls_loss
        score = best_score_seg_weight * val_miou + (1.0 - best_score_seg_weight) * vtop1
        train_loss = ep_loss / max(cnt, 1)
        train_seg_loss = ep_seg / max(cnt, 1)
        train_cls_loss = ep_cls / max(cnt, 1)
        train_top1 = ep_top1 / max(cnt, 1)
        train_top5 = ep_top5 / max(cnt, 1)

        logger.info(
            "轮次=%d 完成：train_loss=%.4f val_loss=%.4f val_seg_loss=%.4f "
            "val_cls_loss=%.4f val_ce=%.4f val_dice=%.4f val_miou=%.4f "
            "val_fg_iou=%.4f val_pixel_acc=%.4f val_top1=%.4f val_top5=%.4f "
            "score=%.4f score分割权重=%.2f",
            epoch + 1,
            train_loss,
            val_loss,
            val_seg_loss,
            val_cls_loss,
            val_ce,
            val_dice,
            val_miou,
            val_fg_iou,
            val_pixel_acc,
            vtop1,
            vtop5,
            score,
            best_score_seg_weight,
        )

        is_best_cls = vtop1 > best_top1
        is_best_seg = val_miou > best_iou
        is_best_score = score > best_score
        if is_best_cls:
            best_top1 = vtop1
        if is_best_seg:
            best_iou = val_miou
        if is_best_score:
            best_score = score

        epoch_metrics = {
            "train_loss": train_loss,
            "train_seg_loss": train_seg_loss,
            "train_cls_loss": train_cls_loss,
            "train_top1": train_top1,
            "train_top5": train_top5,
            "val_loss": val_loss,
            "val_seg_loss": val_seg_loss,
            "val_cls_loss": val_cls_loss,
            "val_ce": val_ce,
            "val_dice": val_dice,
            "val_miou": val_miou,
            "val_fg_iou": val_fg_iou,
            "val_pixel_acc": val_pixel_acc,
            "val_top1": vtop1,
            "val_top5": vtop5,
            "score": score,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        monitor.update(epoch, epoch_metrics)

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "cfg": cfg,
            **epoch_metrics,
            "val_iou": val_miou,
            "best_score_seg_weight": best_score_seg_weight,
            "best_top1": best_top1,
            "best_iou": best_iou,
            "best_score": best_score,
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "leaf_id_to_idx": leaf_id_to_idx,
            "num_classes_cls": num_classes_cls,
            "monitor": monitor.state_dict(),
        }

        if is_best_score:
            torch.save(
                {**checkpoint, "metric": best_score, "type": "best_score"},
                best_checkpoint_path,
            )
            logger.info("已更新最佳综合模型：score=%.4f -> %s", best_score, best_checkpoint_path)

        training_checkpoint = {
            **checkpoint,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "type": "last",
        }
        torch.save(training_checkpoint, last_checkpoint_path)
        logger.info("已覆盖保存最新 checkpoint：%s", last_checkpoint_path)


if __name__ == "__main__":
    main()
