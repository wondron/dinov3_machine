# dino_finetune/data.py
"""一体机多任务数据：图像预处理、Dataset、PK 采样与 DataLoader 工厂。"""
from __future__ import annotations

import logging
from typing import Any, Iterator, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import resolve_interp
from .device import DeviceSpec
from .labels import OvenLabel

logger = logging.getLogger(__name__)


def read_image_rgb(path: str) -> np.ndarray:
    """用 imdecode 读图（兼容 Windows 中文路径），返回 RGB uint8 (H, W, 3)。"""
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"图片解码失败：{path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# -----------------------------
# 1) transforms
# -----------------------------
class OvenTransforms:
    """
    train（一体机场景的增强约束，设计文档 4.5）：
      - 四条边各自随机裁掉 0～max_crop_frac：只有轻微缩放和平移，不会切掉底板和导轨；
      - 左右翻转；不做上下翻转，也不做旋转；
      - 亮度 / 对比度 / 饱和度 / 色调扰动，覆盖灯亮、灯暗。
    eval：固定 resize + normalize。
    输入 RGB uint8 (H, W, 3)，输出 torch.float32 (3, H, W)。
    """

    def __init__(
        self,
        resize_hw: Sequence[int],
        mean: Sequence[float],
        std: Sequence[float],
        img_interp: str = "linear",
        is_train: bool = False,
        aug_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.interp = resolve_interp(img_interp)

        aug_cfg = dict(aug_cfg or {})
        self.augment = bool(is_train) and bool(aug_cfg.get("enabled", True))
        self.max_crop_frac = float(aug_cfg.get("max_crop_frac", 0.05))
        self.hflip_p = float(aug_cfg.get("hflip_p", 0.5))
        self.color_jitter_p = float(aug_cfg.get("color_jitter_p", 0.8))
        self.brightness = float(aug_cfg.get("brightness", 0.3))
        self.contrast = float(aug_cfg.get("contrast", 0.2))
        self.saturation = float(aug_cfg.get("saturation", 0.15))
        self.hue = float(aug_cfg.get("hue", 0.02))

    def describe(self) -> str:
        if not self.augment:
            return f"mode=eval, resize_hw={self.resize_hw}"
        return (
            f"mode=train, resize_hw={self.resize_hw}, max_crop_frac={self.max_crop_frac}, "
            f"hflip_p={self.hflip_p}, color_jitter_p={self.color_jitter_p}, brightness={self.brightness}, "
            f"contrast={self.contrast}, saturation={self.saturation}, hue={self.hue}"
        )

    def _edge_crop(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        top, bottom = (int(np.random.uniform(0, self.max_crop_frac) * h) for _ in range(2))
        left, right = (int(np.random.uniform(0, self.max_crop_frac) * w) for _ in range(2))
        return image[top : h - bottom, left : w - right]

    @staticmethod
    def _factor(strength: float) -> float:
        return float(np.random.uniform(1.0 - strength, 1.0 + strength))

    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        return np.clip(image.astype(np.float32) * self._factor(self.brightness), 0, 255).astype(np.uint8)

    def _adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        mean = float(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean())
        out = (image.astype(np.float32) - mean) * self._factor(self.contrast) + mean
        return np.clip(out, 0, 255).astype(np.uint8)

    def _adjust_saturation(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)[:, :, None]
        out = gray + (image.astype(np.float32) - gray) * self._factor(self.saturation)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _adjust_hue(self, image: np.ndarray) -> np.ndarray:
        shift = int(round(float(np.random.uniform(-self.hue, self.hue)) * 180.0))  # OpenCV uint8 的 H 取值 0..179
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + shift) % 180).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        ops = []
        if self.brightness > 0:
            ops.append(self._adjust_brightness)
        if self.contrast > 0:
            ops.append(self._adjust_contrast)
        if self.saturation > 0:
            ops.append(self._adjust_saturation)
        if self.hue > 0:
            ops.append(self._adjust_hue)
        for i in np.random.permutation(len(ops)):
            image = ops[i](image)
        return image

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if self.augment and self.max_crop_frac > 0:
            image = self._edge_crop(image)

        h, w = self.resize_hw
        img = cv2.resize(image, (w, h), interpolation=self.interp)
        if self.augment:
            if np.random.random() < self.hflip_p:
                img = np.ascontiguousarray(img[:, ::-1])
            if np.random.random() < self.color_jitter_p:
                img = self._color_jitter(img)

        img = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


# -----------------------------
# 2) Dataset
# -----------------------------
TARGET_KEYS = (
    "index",
    "is_oven",
    "food",
    "food_known",
    "container",
    "container_known",
    "accessory",
    "accessory_known",
    "rack_level",
    "rack_known",
)


class OvenDataset(Dataset):
    """
    把 OvenLabel 编码成各任务的训练目标。null 标签填占位值，并用 *_known 掩码标出，最终由 loss 掩码排除。
    非一体机样本：rack_count = max_rack、floor_usable = True、group_id = -1（不参与层位和型号 loss）。
    """

    def __init__(
        self,
        labels: Sequence[OvenLabel],
        transform: OvenTransforms,
        *,
        container_classes: Sequence[str],
        accessory_classes: Sequence[str],
        profile: Mapping[str, DeviceSpec],
        group_names: Sequence[str],
        max_rack: int,
    ) -> None:
        self.labels = list(labels)
        self.transform = transform
        container_index = {name: i for i, name in enumerate(container_classes)}
        accessory_index = {name: i for i, name in enumerate(accessory_classes)}
        group_index = {name: i for i, name in enumerate(group_names)}

        n = len(self.labels)
        self.is_oven = np.zeros(n, dtype=np.float32)
        self.food = np.zeros(n, dtype=np.float32)
        self.food_known = np.zeros(n, dtype=bool)
        self.container = np.zeros((n, len(container_classes)), dtype=np.float32)
        self.container_known = np.zeros(n, dtype=bool)
        self.accessory = np.zeros((n, len(accessory_classes)), dtype=np.float32)
        self.accessory_known = np.zeros(n, dtype=bool)
        self.rack_level = np.zeros(n, dtype=np.int64)
        self.rack_known = np.zeros(n, dtype=bool)
        self.rack_count = np.full(n, int(max_rack), dtype=np.int64)
        self.floor_usable = np.ones(n, dtype=bool)
        self.group_id = np.full(n, -1, dtype=np.int64)

        for i, label in enumerate(self.labels):
            if label.is_oven:
                spec = profile[label.device_model]
                self.is_oven[i] = 1.0
                self.rack_count[i] = spec.rack_count
                self.floor_usable[i] = spec.floor_usable
                self.group_id[i] = group_index[spec.cavity_group]
                if label.rack_level is not None:
                    self.rack_level[i] = label.rack_level
                    self.rack_known[i] = True
            if label.food_exist is not None:
                self.food[i] = float(label.food_exist)
                self.food_known[i] = True
            if label.container is not None:
                self.container_known[i] = True
                for name in label.container:
                    self.container[i, container_index[name]] = 1.0
            if label.accessory is not None:
                self.accessory_known[i] = True
                for name in label.accessory:
                    self.accessory[i, accessory_index[name]] = 1.0

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = self.transform(read_image_rgb(self.labels[index].image_path))
        return {
            "image": image,
            "index": index,
            "is_oven": torch.tensor(self.is_oven[index]),
            "food": torch.tensor(self.food[index]),
            "food_known": torch.tensor(self.food_known[index]),
            "container": torch.from_numpy(self.container[index]),
            "container_known": torch.tensor(self.container_known[index]),
            "accessory": torch.from_numpy(self.accessory[index]),
            "accessory_known": torch.tensor(self.accessory_known[index]),
            "rack_level": torch.tensor(self.rack_level[index]),
            "rack_known": torch.tensor(self.rack_known[index]),
            "rack_count": torch.tensor(self.rack_count[index]),
            "floor_usable": torch.tensor(self.floor_usable[index]),
            "group_id": torch.tensor(self.group_id[index]),
        }


# -----------------------------
# 3) PK 采样
# -----------------------------
class _Cycle:
    """按打乱后的顺序循环取样，一轮取完再重新打乱。"""

    def __init__(self, indices: np.ndarray, rng: np.random.Generator) -> None:
        self.indices = indices
        self.rng = rng
        self.order: np.ndarray = indices[:0]
        self.pos = 0

    def take(self, n: int) -> list[int]:
        out: list[int] = []
        while len(out) < n:
            if self.pos >= len(self.order):
                self.order = self.rng.permutation(self.indices)
                self.pos = 0
            out.append(int(self.order[self.pos]))
            self.pos += 1
        return out


class PKBatchSampler(Sampler[list[int]]):
    """
    PK 采样（设计文档 4.3）：每个 batch 随机取 P 个 cavity_group，每组取 K 张一体机图片，再补一定比例的非一体机图片。
    各组被选中的概率相同，数据少的机型不会被数据多的机型淹没（设计文档 4.5 的按型号均衡采样）。
    组内样本不足 K 张时会在同一 batch 内重复出现（增强不同）。
    """

    def __init__(
        self,
        group_ids: Sequence[int],
        batch_size: int,
        p_groups: int,
        non_oven_ratio: float,
        num_batches: int,
        seed: int = 0,
        require_supcon: bool = False,
    ) -> None:
        group_ids = np.asarray(group_ids)
        self.pools = {int(g): np.flatnonzero(group_ids == g) for g in np.unique(group_ids[group_ids >= 0])}
        self.non_oven = np.flatnonzero(group_ids < 0)
        if not self.pools and len(self.non_oven) == 0:
            raise ValueError("训练集为空，无法采样")

        self.require_supcon = bool(require_supcon)
        if self.require_supcon:
            if batch_size < 4:
                raise ValueError(
                    "SupCon + PK 采样需要 trainparams.batch_size >= 4，"
                    "以保证至少 2 个 cavity_group、每组至少 2 张一体机图片；"
                    "请增大训练 batch_size，或将 loss.weights.proj 设为 0。"
                )
            if p_groups < 2:
                raise ValueError("SupCon + PK 采样需要 sampler.p_groups >= 2，请增大 p_groups。")
            if len(self.pools) < 2:
                raise ValueError(
                    f"SupCon + PK 采样需要训练集中至少 2 个 cavity_group，实际为 {len(self.pools)}；"
                    "请补充不同组的一体机样本，或将 loss.weights.proj 设为 0。"
                )

        if not self.pools:
            n_non_oven = batch_size
        elif len(self.non_oven) == 0:
            n_non_oven = 0
        else:
            requested_non_oven = int(round(batch_size * non_oven_ratio))
            min_oven = 4 if self.require_supcon else 2
            n_non_oven = min(requested_non_oven, max(batch_size - min_oven, 0))
            if self.require_supcon and n_non_oven != requested_non_oven:
                logger.warning(
                    "SupCon + PK：非一体机配额从 %d 调整为 %d（batch_size=%d），"
                    "为至少 2 个组各保留 2 张一体机图片。",
                    requested_non_oven, n_non_oven, batch_size,
                )
        self.n_non_oven = n_non_oven
        self.n_oven = batch_size - n_non_oven
        self.p = min(int(p_groups), len(self.pools))
        if self.require_supcon:
            self.p = min(self.p, self.n_oven // 2)
            if self.p != int(p_groups):
                logger.warning(
                    "SupCon + PK：P 从 %d 调整为 %d（可用组数=%d，一体机配额=%d），"
                    "保证每组至少 2 张一体机图片。",
                    p_groups, self.p, len(self.pools), self.n_oven,
                )
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.epoch = 0

    def describe(self) -> str:
        return (
            f"PK 采样：groups={len(self.pools)} P={self.p} 一体机/batch={self.n_oven} "
            f"非一体机/batch={self.n_non_oven} batches={self.num_batches}"
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        cycles = {g: _Cycle(idx, rng) for g, idx in self.pools.items()}
        non_oven = _Cycle(self.non_oven, rng) if len(self.non_oven) else None
        groups = np.array(sorted(self.pools))
        for _ in range(self.num_batches):
            batch: list[int] = []
            if self.p > 0:
                base, extra = divmod(self.n_oven, self.p)
                for j, g in enumerate(rng.choice(groups, size=self.p, replace=False)):
                    batch += cycles[int(g)].take(base + (1 if j < extra else 0))
            if non_oven is not None and self.n_non_oven > 0:
                batch += non_oven.take(self.n_non_oven)
            yield batch


# -----------------------------
# 4) DataLoader 工厂
# -----------------------------
def build_train_loader(
    dataset: OvenDataset,
    *,
    batch_size: int,
    sampler_cfg: Mapping[str, Any],
    steps_per_epoch: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
    require_supcon: bool = False,
) -> DataLoader:
    common = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    if sampler_cfg["type"] == "pk":
        batch_sampler = PKBatchSampler(
            dataset.group_id,
            batch_size=batch_size,
            p_groups=sampler_cfg["p_groups"],
            non_oven_ratio=sampler_cfg["non_oven_ratio"],
            num_batches=steps_per_epoch,
            seed=seed,
            require_supcon=require_supcon,
        )
        logger.info(batch_sampler.describe())
        return DataLoader(dataset, batch_sampler=batch_sampler, **common)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) >= batch_size,
        **common,
    )


def build_eval_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent and num_workers > 0,
    )
