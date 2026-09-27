from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_IMPORTS = {
    "LoRA": ("dino_finetune.model.lora", "LoRA"),
    "OvenMultiTaskModel": ("dino_finetune.model.oven", "OvenMultiTaskModel"),
    "OvenDataset": ("dino_finetune.data", "OvenDataset"),
    "OvenTransforms": ("dino_finetune.data", "OvenTransforms"),
    "OvenLabel": ("dino_finetune.labels", "OvenLabel"),
    "LabelSchema": ("dino_finetune.labels", "LabelSchema"),
    "load_split": ("dino_finetune.labels", "load_split"),
    "DeviceSpec": ("dino_finetune.device", "DeviceSpec"),
    "DeviceGallery": ("dino_finetune.device", "DeviceGallery"),
    "load_device_profile": ("dino_finetune.device", "load_device_profile"),
    "MultiTaskLoss": ("dino_finetune.losses", "MultiTaskLoss"),
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
