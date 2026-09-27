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
    model_keys = set(model.state_dict().keys())
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
        msd = model.state_dict()
        filtered = {
            k: v for k, v in ckpt_sd.items()
            if k in msd and hasattr(v, "shape") and v.shape == msd[k].shape
        }
        if not filtered:
            raise RuntimeError("无法从 checkpoint 中匹配到任何模型参数")
        best_sd = filtered
        best_prefix = "(shape-filter)"
        best_match = len(filtered)

    missing, unexpected = model.load_state_dict(best_sd, strict=False)
    logger.info(
        "加载 ckpt 完成：前缀=%s, 匹配=%d, 缺失=%d, 多余=%d",
        best_prefix, best_match, len(missing), len(unexpected)
    )
    if len(missing) > 0:
        logger.warning("缺失参数（前 20 个）：%s", missing[:20])
    if len(unexpected) > 0:
        logger.warning("多余参数（前 20 个）：%s", unexpected[:20])
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
