from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_IMPORTS = {
    "LoRA": ("dino_finetune.model.lora", "LoRA"),
    "DINOEncoderLoRA": ("dino_finetune.model.dino", "DINOEncoderLoRA"),
    "LinearClassifier": ("dino_finetune.model.linear_decoder", "LinearClassifier"),
    "FPNDecoder": ("dino_finetune.model.fpn_decoder", "FPNDecoder"),
    "get_dataloader": ("dino_finetune.data", "get_dataloader"),
    "visualize_overlay": ("dino_finetune.visualization", "visualize_overlay"),
    "compute_iou_metric": ("dino_finetune.metrics", "compute_iou_metric"),
    "get_corruption_transforms": ("dino_finetune.corruption", "get_corruption_transforms"),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"模块 {__name__!r} 没有属性 {name!r}")

    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
