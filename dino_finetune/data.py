# data.py
# 完整版：OpenCV 读图 + torchvision v2 统一 Resize/Normalize（image bilinear+antialias，mask nearest）
# 可选：val 使用 Albumentations 的 corruption（ImageNet-C 风格）后再走同一套 v2 Normalize，保证分布对齐

import os
import zipfile
import logging
import urllib.request
from typing import Optional, Tuple, Callable

import cv2
import numpy as np
import torch

import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation

from .corruption import get_corruption_transforms


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


class SegTransforms:
    def __init__(
        self,
        resize_hw: Tuple[int, int],
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        img_interp: int = cv2.INTER_LINEAR,     # 与 ONNX 一致：bilinear
        msk_interp: int = cv2.INTER_NEAREST,    # mask 最近邻
    ):
        self.resize_hw = resize_hw  # (H,W)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.img_interp = img_interp
        self.msk_interp = msk_interp

    def __call__(self, image_np: np.ndarray, mask_np: Optional[np.ndarray] = None):
        """
        image_np: RGB uint8 HWC
        mask_np : (H,W) int / (H,W,C) 可选
        return:
          img: torch.float32 (3,H,W) 经过 /255 + imagenet normalize
          msk: torch.long (H,W)（若提供）
        """
        h, w = self.resize_hw

        # ---- image ----
        image_np = np.ascontiguousarray(image_np)
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        img = cv2.resize(image_np, (w, h), interpolation=self.img_interp).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).contiguous().float()

        if mask_np is None:
            return img_t

        # ---- mask ----
        mask_np = np.ascontiguousarray(mask_np)

        # (H,W,C) -> (H,W) index
        if mask_np.ndim == 3:
            mask_np = np.argmax(mask_np, axis=-1).astype(np.int64)

        # ✅ 关键：不要转 uint8，避免潜在截断；用 int32 resize 再转 int64
        m_rs = cv2.resize(mask_np.astype(np.int32), (w, h), interpolation=self.msk_interp)
        m_t = torch.from_numpy(m_rs.astype(np.int64)).contiguous().long()

        return img_t, m_t



class CorruptThenSegTfm:
    def __init__(self, img_dim: Tuple[int, int], severity: int, v2_tfm: SegTransforms):
        self.corr: A.Compose = get_corruption_transforms(img_dim, severity)  # OneOf + Resize
        self.v2_tfm = v2_tfm

    def __call__(self, image_np: np.ndarray, mask_np: np.ndarray):
        out = self.corr(image=image_np, mask=mask_np)
        return self.v2_tfm(out["image"], out["mask"])


# -----------------------------
# 4) ADE20K Dataset
# -----------------------------
class ADE20kDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "training",  # "training" or "validation"
        transform: Optional[Callable] = None,
    ):
        self.root = root
        self.split = split
        self.n_classes = 150
        self.transform = transform

        ade_root = os.path.join(root, "ADEChallengeData2016")
        self.images_dir = os.path.join(ade_root, "images", split)
        self.masks_dir  = os.path.join(ade_root, "annotations", split)

        if not os.path.exists(self.images_dir) or not os.path.exists(self.masks_dir):
            self.download_and_extract_dataset()

        self.image_files = sorted(os.listdir(self.images_dir))

    def download_and_extract_dataset(self) -> None:
        dataset_url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
        zip_path = os.path.join(self.root, "ADEChallengeData2016.zip")
        os.makedirs(self.root, exist_ok=True)

        logging.info("Downloading ADE20K...")
        urllib.request.urlretrieve(dataset_url, zip_path)

        logging.info("Extracting ADE20K...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.root)

        logging.info("ADE20K extracted!")
        os.remove(zip_path)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int):
        img_name = self.image_files[index]
        img_path  = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace(".jpg", ".png"))

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"ADE image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"ADE mask not found: {mask_path}")

        # ADE 标注一般是 1..150，转成 0..149
        mask = mask.astype(np.int64) - 1

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            # 兜底：不建议走这里
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask


# -----------------------------
# 5) Binary Seg Dataset（自定义二分类）
# -----------------------------
class BinarySegDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "training",  # "training" or "validation"
        transform: Optional[Callable] = None,
        n_classes: int | None = 2,
        ignore_index: int | None = 255,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.n_classes = n_classes
        self.ignore_index = ignore_index

        self.images_dir = os.path.join(root, "images", split)
        self.masks_dir  = os.path.join(root, "annotations", split)
        self.image_files = sorted(os.listdir(self.images_dir))

    def __len__(self):
        return len(self.image_files)

    def _validate_mask(self, mask: np.ndarray, mask_path: str) -> None:
        if self.n_classes is None:
            return

        valid = mask if self.ignore_index is None else mask[mask != self.ignore_index]
        if valid.size == 0:
            raise ValueError(f"标注没有有效像素：{mask_path}")

        min_label = int(valid.min())
        max_label = int(valid.max())
        if min_label < 0 or max_label >= int(self.n_classes):
            unique = np.unique(mask).tolist()
            raise ValueError(
                f"二分类标注值越界：{mask_path}，期望有效标签范围为 0..{int(self.n_classes) - 1}，"
                f"ignore_index={self.ignore_index}，实际 unique={unique}。"
                "请先将前景标注转换为类别 id=1，或确认 config.model.n_classes 与标注类别一致。"
            )

    def __getitem__(self, index: int):
        img_name = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_name)

        base = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.masks_dir, base + ".png")

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"mask not found: {mask_path}")

        # 期望：0/1（可选 255 ignore）
        mask = mask.astype(np.int64)

        self._validate_mask(mask, mask_path)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            # 兜底：不建议走这里
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask


class MulticlassSegDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "training",  # "training" or "validation"
        transform: Optional[Callable] = None,
        n_classes: int | None = None,
        ignore_index: int | None = 255,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.n_classes = n_classes
        self.ignore_index = ignore_index

        self.images_dir = os.path.join(root, "images", split)
        self.masks_dir = os.path.join(root, "annotations", split)
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"图像目录不存在：{self.images_dir}")
        if not os.path.isdir(self.masks_dir):
            raise FileNotFoundError(f"标注目录不存在：{self.masks_dir}")

        self.image_files = sorted(os.listdir(self.images_dir))
        if len(self.image_files) == 0:
            raise RuntimeError(f"图像目录为空：{self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    def _validate_mask(self, mask: np.ndarray, mask_path: str) -> None:
        if self.n_classes is None:
            return

        valid = mask if self.ignore_index is None else mask[mask != self.ignore_index]
        if valid.size == 0:
            raise ValueError(f"标注没有有效像素：{mask_path}")

        min_label = int(valid.min())
        max_label = int(valid.max())
        if min_label < 0 or max_label >= int(self.n_classes):
            unique = np.unique(mask).tolist()
            raise ValueError(
                f"标注值越界：{mask_path}，期望有效标签范围为 0..{int(self.n_classes) - 1}，实际 unique={unique}"
            )

    def __getitem__(self, index: int):
        img_name = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_name)

        base = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.masks_dir, base + ".png")

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"图像不存在或无法读取：{img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"标注不存在或无法读取：{mask_path}")
        if mask.ndim == 3:
            if not np.all(mask == mask[:, :, :1]):
                raise ValueError(f"多类别标注必须是单通道类别 id 图：{mask_path}")
            mask = mask[:, :, 0]

        mask = mask.astype(np.int64)
        self._validate_mask(mask, mask_path)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask


# -----------------------------
# 6) DataLoader 工厂
# -----------------------------
def get_dataloader(
    dataset_name: str,
    img_dim: Tuple[int, int] = (490, 490),  # (H,W)
    batch_size: int = 6,
    corruption_severity: Optional[int] = None,  # 1..5，仅建议 val 用
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    img_interp: int = cv2.INTER_LINEAR,
    msk_interp: int = cv2.INTER_NEAREST,
    num_workers_train: int = 32,
    num_workers_val: int = 8,
    pin_memory: bool = True,
    root_path: str = "./data",
    n_classes: int | None = None,
    ignore_index: int | None = 255,
) -> Tuple[DataLoader, DataLoader]:
    """
    返回 (train_loader, val_loader)

    预处理策略：
      - train: v2 Resize+Normalize（分割版：mask nearest）
      - val  : 默认同 train
      - val+corruption: 先 Albumentations corruption（OneOf + Resize）再 v2 Normalize（保证分布一致）
    """
    assert dataset_name in ["ade20k", "binary", "multiclass"], "数据集名称不在[ade20k, binary, multiclass]中"

    tfm_clean = SegTransforms(
        resize_hw=(img_dim[0], img_dim[1]),
        mean=mean,
        std=std,
        img_interp=img_interp,
        msk_interp=msk_interp,
    )
    tfm_val = tfm_clean
    if corruption_severity is not None:
        tfm_val = CorruptThenSegTfm(img_dim=img_dim, severity=corruption_severity, v2_tfm=tfm_clean)

    if dataset_name == "ade20k":
        train_dataset = ADE20kDataset(
            root=root_path,
            split="training",
            transform=tfm_clean,
        )
        val_dataset = ADE20kDataset(
            root=root_path,
            split="validation",
            transform=tfm_val,
        )

    elif dataset_name == "binary":
        train_dataset = BinarySegDataset(
            root=root_path,
            split="training",
            transform=tfm_clean,
            n_classes=n_classes,
            ignore_index=ignore_index,
        )
        val_dataset = BinarySegDataset(
            root=root_path,
            split="validation",
            transform=tfm_val,
            n_classes=n_classes,
            ignore_index=ignore_index,
        )
    else:  # multiclass
        if n_classes is None or int(n_classes) <= 1:
            raise ValueError("multiclass 数据集必须传入 model.n_classes，且需要包含背景 0，取值应大于 1")
        train_dataset = MulticlassSegDataset(
            root=root_path,
            split="training",
            transform=tfm_clean,
            n_classes=int(n_classes),
            ignore_index=ignore_index,
        )
        val_dataset = MulticlassSegDataset(
            root=root_path,
            split="validation",
            transform=tfm_val,
            n_classes=int(n_classes),
            ignore_index=ignore_index,
        )
    # ✅ 建议：先别用 32 workers，太容易 CPU 内存/碎片化爆炸
    nw_tr = min(num_workers_train, 8)
    nw_va = min(num_workers_val, 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=nw_tr,
        persistent_workers=(nw_tr > 0),
        prefetch_factor=2 if nw_tr > 0 else None,   # ✅ 控制预取，降CPU内存峰值
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,                              # ✅ 可选：让 batch 恒定，训练更稳
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=nw_va,
        persistent_workers=(nw_va > 0),
        prefetch_factor=2 if nw_va > 0 else None,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader
