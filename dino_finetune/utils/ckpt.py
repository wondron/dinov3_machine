from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def pick_state_dict(
    ckpt: dict[str, Any],
    *,
    extra_keys: Iterable[str] | None = None,
    min_tensors: int = 50,
) -> dict[str, Any]:
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint 类型错误，期望 dict，实际为 {type(ckpt).__name__}")

    tensor_like = sum(1 for v in ckpt.values() if hasattr(v, "shape"))
    if tensor_like >= int(min_tensors):
        return ckpt

    keys = [
        "model",
        "state_dict",
        "teacher",
        "student",
        "backbone",
        "net",
        "module",
    ]
    if extra_keys:
        keys.extend(list(extra_keys))

    for k in keys:
        v = ckpt.get(k, None)
        if isinstance(v, dict) and len(v) > 0:
            return pick_state_dict(v, extra_keys=extra_keys, min_tensors=min_tensors)
    return ckpt


def strip_prefix(sd: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def auto_align_and_load(model: nn.Module, ckpt_sd: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Load a complete backbone, validating all state before copying any tensors."""
    model_sd = model.state_dict()
    model_keys = set(model_sd)
    candidates = [
        "",
        "module.",
        "model.",
        "backbone.",
        "teacher.",
        "student.",
        "teacher.backbone.",
        "student.backbone.",
        "teacher.model.",
        "student.model.",
        "teacher.module.",
        "student.module.",
        "teacher.backbone.module.",
        "student.backbone.module.",
    ]

    best_sd = None
    best_prefix = ""
    best_match = -1

    for p in candidates:
        sd_try = ckpt_sd if p == "" else strip_prefix(ckpt_sd, p)
        if not sd_try:
            continue
        match = sum((k in model_keys) for k in sd_try.keys())
        if match > best_match:
            best_match = match
            best_prefix = p
            best_sd = sd_try

    if best_sd is None or best_match <= 0:
        raise RuntimeError("拒绝加载骨干 checkpoint：无法匹配到任何模型参数，模型未修改")

    # load_state_dict can copy valid tensors before reporting other errors.
    # Validate parameters AND persistent buffers first, so an invalid checkpoint
    # never leaves a partially loaded (and subsequently frozen) backbone.
    missing = sorted(model_keys - set(best_sd))
    unexpected = sorted(set(best_sd) - model_keys)
    invalid = []
    for key, expected in model_sd.items():
        if key not in best_sd:
            continue
        value = best_sd[key]
        if not isinstance(value, torch.Tensor):
            invalid.append(f"{key}: 期望 Tensor，实际为 {type(value).__name__}")
        elif value.shape != expected.shape:
            invalid.append(f"{key}: 形状不匹配，期望 {tuple(expected.shape)}，实际为 {tuple(value.shape)}")
        elif value.layout != expected.layout:
            invalid.append(f"{key}: 布局不匹配，期望 {expected.layout}，实际为 {value.layout}")
        elif value.is_meta:
            invalid.append(f"{key}: meta Tensor 没有可加载的权重数据")

    if missing or invalid:
        details = []
        if missing:
            details.append(f"缺失参数或持久缓冲区 {len(missing)} 项（前 20 项）：{missing[:20]}")
        if invalid:
            details.append(f"无效权重 {len(invalid)} 项（前 20 项）：{invalid[:20]}")
        raise RuntimeError(
            f"拒绝加载骨干 checkpoint（前缀={best_prefix!r}，匹配={best_match}/{len(model_keys)}）："
            + "；".join(details)
            + "。骨干权重必须完整且兼容，模型未修改"
        )

    # Full training checkpoints may also contain task heads. Only backbone keys
    # are loaded, but their coverage is strict. Floating-point dtype conversion
    # remains supported, e.g. BF16 pretrained weights into an FP32 encoder.
    state_to_load = {key: best_sd[key] for key in model_sd}
    model.load_state_dict(state_to_load, strict=True)
    logger.info(
        "加载 ckpt 完成：前缀=%s, 匹配=%d, 缺失=%d, 多余=%d",
        best_prefix, best_match, len(missing), len(unexpected)
    )
    if len(unexpected) > 0:
        logger.warning("忽略非骨干参数（前 20 个）：%s", unexpected[:20])
    return missing, unexpected


def build_encoder(
    cfg: dict[str, Any],
    device: torch.device,
    *,
    dino_local_repo: str,
    weight_path: str | Path,
) -> nn.Module:
    model_cfg = cfg.get("model", {}) or {}
    dino_type = str(model_cfg.get("dino_type", "")).strip()
    size = str(model_cfg.get("size", "")).strip()
    if not dino_type or not size:
        raise ValueError("config.model.dino_type 与 config.model.size 为必填项")

    patch_size = 16 if dino_type == "dinov3" else 14
    backbones = {
        "small": f"{dino_type}_vits{patch_size}{'_reg' if dino_type == 'dinov2' else ''}",
        "base": f"{dino_type}_vitb{patch_size}{'_reg' if dino_type == 'dinov2' else ''}",
        "large": f"{dino_type}_vitl{patch_size}{'_reg' if dino_type == 'dinov2' else ''}",
        "giant": f"{dino_type}_vitg{patch_size}{'_reg' if dino_type == 'dinov2' else ''}",
        "huge": f"{dino_type}_vith{patch_size}{'plus' if dino_type == 'dinov3' else ''}{'_reg' if dino_type == 'dinov2' else ''}",
    }
    if size not in backbones:
        raise ValueError(f"未知的 model.size={size}，可选：{sorted(backbones)}")

    backbone_name = backbones[size]
    logger.info("加载 DINO encoder：%s", backbone_name)

    hub_backbones = Path(dino_local_repo) / dino_type / "hub" / "backbones.py"
    if hub_backbones.is_file():
        # 直接从 <repo>/<dino_type>/hub/backbones.py 构建骨干：hubconf.py 会顺带导入分割、检测等模块，
        # 需要 torchmetrics 等只做训练骨干时用不到的依赖
        if str(dino_local_repo) not in sys.path:
            sys.path.insert(0, str(dino_local_repo))
        module = importlib.import_module(f"{dino_type}.hub.backbones")
        encoder = getattr(module, backbone_name)(pretrained=False).to(device)
    else:
        try:
            encoder = torch.hub.load(
                repo_or_dir=dino_local_repo,
                model=backbone_name,
                source="local",
                pretrained=False,
            ).to(device)
        except TypeError:
            encoder = torch.hub.load(
                repo_or_dir=dino_local_repo,
                model=backbone_name,
                source="local",
            ).to(device)

    weight_path = Path(weight_path)
    if not weight_path.is_file():
        raise FileNotFoundError(f"权重文件不存在：{weight_path}")

    ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=True)
    ckpt_sd = pick_state_dict(ckpt)
    auto_align_and_load(encoder, ckpt_sd)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder
