# dino_finetune/engine.py
"""训练、留一型号验证、推理和导出共用的流程：构建模型与数据集、在整个数据集上收集输出、建特征库、加载训练产物。"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .config import get_dino_paths, validate_config
from .data import TARGET_KEYS, OvenDataset, OvenTransforms
from .device import DeviceGallery, DeviceSpec, load_device_profile
from .labels import LabelSchema, OvenLabel
from .losses import MultiTaskLoss
from .model.oven import OvenMultiTaskModel, inference_outputs
from .utils.ckpt import build_encoder

logger = logging.getLogger(__name__)

FEATURES = ("proj", "cls")  # 型号检索可用的特征：Proj Head 输出 / 骨干 CLS


def to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    non_blocking = device.type == "cuda"
    return {k: v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_checkpoint(path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"checkpoint 格式错误：{path}")
    return path, ckpt


def build_model(cfg: Mapping[str, Any], device: torch.device) -> OvenMultiTaskModel:
    """加载预训练 DINOv3 并按配置搭好各任务头与 LoRA（未加载训练权重）。"""
    dino_local_repo, weight_path = get_dino_paths(cfg)
    encoder = build_encoder(cfg, device, dino_local_repo=dino_local_repo, weight_path=weight_path)
    return OvenMultiTaskModel.from_config(
        encoder,
        cfg,
        num_container=len(cfg["labels"]["container"]),
        num_accessory=len(cfg["labels"]["accessory"]),
    ).to(device)


def group_names_of(profile: Mapping[str, DeviceSpec]) -> list[str]:
    return sorted({spec.cavity_group for spec in profile.values()})


def make_dataset(
    labels: Sequence[OvenLabel],
    cfg: Mapping[str, Any],
    profile: Mapping[str, DeviceSpec],
    *,
    is_train: bool,
) -> OvenDataset:
    inp = cfg["input"]
    transform = OvenTransforms(
        inp["img_dim"], inp["mean"], inp["std"], inp["img_interp"], is_train=is_train, aug_cfg=inp["train_aug"]
    )
    return OvenDataset(
        labels,
        transform,
        container_classes=cfg["labels"]["container"],
        accessory_classes=cfg["labels"]["accessory"],
        profile=profile,
        group_names=group_names_of(profile),
        max_rack=cfg["model"]["max_rack"],
    )


def exclude_groups(
    labels: Sequence[OvenLabel],
    profile: Mapping[str, DeviceSpec],
    groups: Sequence[str],
) -> list[OvenLabel]:
    """去掉属于指定 cavity_group 的一体机样本（留一型号验证时，被留出的型号不参与训练和选模型）。"""
    groups = set(groups)
    return [lb for lb in labels if not (lb.is_oven and profile[lb.device_model].cavity_group in groups)]


def select_gallery_indices(labels: Sequence[OvenLabel], max_per_model: int, seed: int) -> list[int]:
    """每个型号最多取 max_per_model 张图作为特征库参考图。"""
    rng = np.random.default_rng(seed)
    by_model: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        if label.is_oven:
            by_model[label.device_model].append(i)
    selected: list[int] = []
    for model_name in sorted(by_model):
        idx = by_model[model_name]
        if len(idx) > max_per_model:
            idx = sorted(rng.choice(idx, size=max_per_model, replace=False).tolist())
        selected.extend(idx)
    return selected


@torch.no_grad()
def collect_outputs(
    model: OvenMultiTaskModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    criterion: MultiTaskLoss | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """
    在整个数据集上前向，收集推理输出（各头概率、未掩码层位 logits、proj / cls 特征）、
    按真实型号掩码的层位 logits（rack_logits）和标签；给了 criterion 时按各任务的
    有效样本数汇总平均 loss，再按任务权重计算总 loss。SupCon 按有效 anchor 汇总，
    其正负样本仍来自各自的 batch，因此该项本身仍受 batch 组成影响。
    """
    model.eval()
    chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    loss_sums: dict[str, float] = defaultdict(float)
    loss_counts: dict[str, int] = defaultdict(int)
    for batch in loader:
        batch = to_device(batch, device)
        with torch.amp.autocast(device.type, enabled=use_amp):
            out = model(batch["image"], batch["rack_count"], batch["floor_usable"])
            if criterion is not None:
                _, terms = criterion(out, batch)
        if criterion is not None:
            effective_counts = criterion.effective_counts(batch)
            for term, value in terms.items():
                count = effective_counts[term]
                loss_counts[term] += count
                if count:
                    loss_sums[term] += float(value) * count
        for key, value in inference_outputs(out).items():
            chunks[key].append(value.cpu())
        chunks["rack_logits"].append(out["rack"].float().cpu())
        for key in TARGET_KEYS:
            chunks[key].append(batch[key].cpu())
    arrays = {key: torch.cat(values).numpy() for key, values in chunks.items()}
    losses = {f"loss_{term}": loss_sums[term] / max(count, 1) for term, count in loss_counts.items()}
    if criterion is not None and loss_counts:
        losses["loss"] = sum(criterion.weights[term] * losses[f"loss_{term}"] for term in loss_counts)
    return arrays, losses


def build_galleries(
    arrays: Mapping[str, np.ndarray],
    labels: Sequence[OvenLabel],
    profile: Mapping[str, DeviceSpec],
) -> dict[str, DeviceGallery]:
    """用参考图的 proj 特征和骨干 CLS 特征各建一个特征库，两者的检索效果都会评估。"""
    group_of = {name: spec.cavity_group for name, spec in profile.items()}
    galleries = {name: DeviceGallery(group_of) for name in FEATURES}
    rows_by_model: dict[str, list[int]] = defaultdict(list)
    for row, index in enumerate(arrays["index"]):
        label = labels[int(index)]
        if label.is_oven:
            rows_by_model[label.device_model].append(row)
    for model_name, rows in sorted(rows_by_model.items()):
        for name in FEATURES:
            galleries[name].add(model_name, torch.from_numpy(arrays[name][rows]))
    return galleries


# =========================
# 训练产物
# =========================
@dataclass
class TrainedRun:
    run_dir: Path
    cfg: dict[str, Any]
    schema: LabelSchema
    profile: dict[str, DeviceSpec]
    model: OvenMultiTaskModel
    calibration: dict[str, Any]
    galleries: dict[str, DeviceGallery]
    gallery_meta: dict[str, Any]


def load_profile_snapshot(run_dir: Path, cfg: Mapping[str, Any]) -> dict[str, DeviceSpec]:
    return load_device_profile(
        run_dir / "device_profile.json",
        max_rack=cfg["model"]["max_rack"],
        accessory_classes=cfg["labels"]["accessory"],
    )


def load_run_config(run_dir: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def load_run(run_dir: str | Path, device: torch.device, *, ckpt_name: str = "ckpt_best.pt") -> TrainedRun:
    """加载 train.py 的输出目录：配置、Device Profile 快照、权重、特征库和阈值，并核对特征库版本。"""
    run_dir = Path(run_dir).expanduser().resolve()
    for name in ("config.yaml", "device_profile.json", ckpt_name, "gallery.pt", "calibration.json"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"训练输出不完整，缺少 {run_dir / name}（需要跑完 train.py 的阶段 2）")

    cfg = load_run_config(run_dir)
    profile = load_profile_snapshot(run_dir, cfg)
    model = build_model(cfg, device)
    _, ckpt = load_checkpoint(run_dir / ckpt_name)
    model.load_trainable_state_dict(ckpt["model"])
    model.eval()

    gallery_state = torch.load(run_dir / "gallery.pt", map_location="cpu", weights_only=True)
    meta = dict(gallery_state["meta"])
    fingerprint = model.gallery_fingerprint(Path(get_dino_paths(cfg)[1]).name)
    if meta.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"特征库版本 {meta.get('fingerprint')} 与 {ckpt_name} 的权重版本 {fingerprint} 不一致，"
            "骨干或 Proj Head 变动后必须重建特征库"
        )
    galleries = {name: DeviceGallery.from_state_dict(gallery_state[name]) for name in FEATURES if name in gallery_state}
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    if calibration.get("fingerprint") != fingerprint:
        raise RuntimeError("calibration.json 与当前权重版本不一致，请重新标定")
    return TrainedRun(run_dir, cfg, LabelSchema.from_config(cfg), profile, model, calibration, galleries, meta)
