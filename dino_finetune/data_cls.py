from __future__ import annotations

import logging, math, cv2, torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from torch.utils.data import Dataset, DataLoader, get_worker_info
from .config import resolve_interp

logger = logging.getLogger(__name__)

# -----------------------------
# 0) 分类 transforms（不依赖 SegTransforms）
# -----------------------------
class ClsTransforms:
    """
    分类专用 transforms。

    train:
      - RandomResizedCrop
      - RandomHorizontalFlip
      - 轻量 ColorJitter
      - normalize
      - HWC -> CHW

    valid/test:
      - 固定 resize
      - normalize
      - HWC -> CHW

    说明：
      - 输入、输出都沿用当前项目的 RGB uint8 -> torch.float32 逻辑。
      - 食物类别通常与颜色有关，因此默认 hue 和 saturation 扰动较轻。
    """

    def __init__(
        self,
        resize_hw: Tuple[int, int],
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        img_interp: int = cv2.INTER_LINEAR,
        is_train: bool = False,
        aug_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))  # (H, W)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.img_interp = int(img_interp)
        self.is_train = bool(is_train)

        aug_cfg = aug_cfg or {}
        self.aug_enabled = bool(aug_cfg.get("enabled", True))

        self.random_resized_crop = bool( aug_cfg.get("random_resized_crop", True) )
        self.crop_scale = self._parse_range( aug_cfg.get("crop_scale", (0.80, 1.00)), name="crop_scale", lower_bound=0.0,)
        self.crop_ratio = self._parse_range( aug_cfg.get("crop_ratio", (0.90, 1.10)), name="crop_ratio", lower_bound=0.0,)
        
        self.hflip_p = self._parse_probability( aug_cfg.get("hflip_p", 0.5), name="hflip_p", )
        self.color_jitter_p = self._parse_probability( aug_cfg.get("color_jitter_p", 0.8), name="color_jitter_p",)
        self.brightness = self._parse_nonnegative(aug_cfg.get("brightness", 0.15), name="brightness",)
        self.contrast = self._parse_nonnegative(aug_cfg.get("contrast", 0.15), name="contrast",)
        self.saturation = self._parse_nonnegative( aug_cfg.get("saturation", 0.10), name="saturation", )
        self.hue = self._parse_nonnegative( aug_cfg.get("hue", 0.02), name="hue", )
        
        if self.hue > 0.5:
            raise ValueError("hue 必须位于 [0, 0.5]")

    @staticmethod
    def _parse_nonnegative(value: Any, name: str) -> float:
        value = float(value)
        if value < 0:
            raise ValueError(f"{name} 必须 >= 0，当前值={value}")
        return value

    @staticmethod
    def _parse_probability(value: Any, name: str) -> float:
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} 必须位于 [0, 1]，当前值={value}")
        return value

    @staticmethod
    def _parse_range( value: Any, name: str, lower_bound: float,) -> Tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} 必须是长度为 2 的 list/tuple")

        low, high = float(value[0]), float(value[1])
        if low <= lower_bound or high <= lower_bound or low > high:
            raise ValueError(f"{name} 非法：要求 {lower_bound} < low <= high，当前值=({low}, {high})")
        return low, high

    def describe(self) -> str:
        if not self.is_train or not self.aug_enabled:
            return ( f"mode=eval, resize_hw={self.resize_hw}, augmentation=disabled" )

        return (
            f"mode=train, resize_hw={self.resize_hw}, "
            f"random_resized_crop={self.random_resized_crop}, "
            f"crop_scale={self.crop_scale}, crop_ratio={self.crop_ratio}, "
            f"hflip_p={self.hflip_p}, color_jitter_p={self.color_jitter_p}, "
            f"brightness={self.brightness}, contrast={self.contrast}, "
            f"saturation={self.saturation}, hue={self.hue}"
        )

    def _fixed_resize(self, image: np.ndarray) -> np.ndarray:
        out_h, out_w = self.resize_hw
        return cv2.resize(image, (out_w, out_h), interpolation=self.img_interp,)

    def _random_resized_crop(self, image: np.ndarray) -> np.ndarray:
        """
        参考 torchvision RandomResizedCrop 的采样方式实现，
        避免额外引入 torchvision/PIL 预处理链。
        """
        src_h, src_w = image.shape[:2]
        area = float(src_h * src_w)
        log_ratio = (math.log(self.crop_ratio[0]), math.log(self.crop_ratio[1]), )

        # 尝试随机采样合法 crop
        for _ in range(10):
            target_area = area * float( np.random.uniform(self.crop_scale[0], self.crop_scale[1]) )
            aspect_ratio = math.exp( float(np.random.uniform(log_ratio[0], log_ratio[1])) )
            crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < crop_w <= src_w and 0 < crop_h <= src_h:
                top = int(np.random.randint(0, src_h - crop_h + 1))
                left = int(np.random.randint(0, src_w - crop_w + 1))
                crop = image[top : top + crop_h, left : left + crop_w]
                return self._fixed_resize(crop)

        # 随机采样失败时使用中心裁剪兜底
        in_ratio = src_w / max(src_h, 1)
        min_ratio, max_ratio = self.crop_ratio

        if in_ratio < min_ratio:
            crop_w = src_w
            crop_h = int(round(crop_w / min_ratio))
        elif in_ratio > max_ratio:
            crop_h = src_h
            crop_w = int(round(crop_h * max_ratio))
        else:
            crop_w = src_w
            crop_h = src_h

        crop_h = min(max(crop_h, 1), src_h)
        crop_w = min(max(crop_w, 1), src_w)
        top = max((src_h - crop_h) // 2, 0)
        left = max((src_w - crop_w) // 2, 0)

        crop = image[top : top + crop_h, left : left + crop_w]
        return self._fixed_resize(crop)

    @staticmethod
    def _adjust_brightness( image: np.ndarray, strength: float,) -> np.ndarray:
        if strength <= 0:
            return image
        factor = float(np.random.uniform(1.0 - strength, 1.0 + strength))
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype( np.uint8 )

    @staticmethod
    def _adjust_contrast( image: np.ndarray, strength: float, ) -> np.ndarray:
        if strength <= 0:
            return image
        factor = float(np.random.uniform(1.0 - strength, 1.0 + strength))
        gray_mean = float( cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean())
        out = ( image.astype(np.float32) - gray_mean ) * factor + gray_mean
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _adjust_saturation( image: np.ndarray, strength: float, ) -> np.ndarray:
        if strength <= 0:
            return image
        factor = float(np.random.uniform(1.0 - strength, 1.0 + strength))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
        out = gray_rgb + ( image.astype(np.float32) - gray_rgb ) * factor
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _adjust_hue(
        image: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        if strength <= 0:
            return image

        # OpenCV uint8 HSV 的 H 范围为 [0, 179]
        hue_shift = int(round(float(np.random.uniform(-strength, strength)) * 180.0))
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(np.int16)
        hsv[:, :, 0] = ((h + hue_shift) % 180).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        operations = [
            lambda x: self._adjust_brightness(x, self.brightness),
            lambda x: self._adjust_contrast(x, self.contrast),
            lambda x: self._adjust_saturation(x, self.saturation),
            lambda x: self._adjust_hue(x, self.hue),
        ]
        np.random.shuffle(operations)

        out = image
        for op in operations:
            out = op(out)
        return out

    def __call__(self, image_np: np.ndarray) -> torch.Tensor:
        image_np = np.ascontiguousarray(image_np)
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        if self.is_train and self.aug_enabled:
            if self.random_resized_crop:
                img = self._random_resized_crop(image_np)
            else:
                img = self._fixed_resize(image_np)

            if self.hflip_p > 0 and np.random.random() < self.hflip_p:
                img = np.ascontiguousarray(img[:, ::-1])

            if ( self.color_jitter_p > 0 and np.random.random() < self.color_jitter_p ):
                img = self._color_jitter(img)
        else:
            # valid/test：始终固定 resize，不使用任何随机增强
            img = self._fixed_resize(image_np)

        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img_t = torch.from_numpy( np.transpose(img, (2, 0, 1)) ).contiguous().float()
        return img_t

# -----------------------------
# 1) meta
# -----------------------------
@dataclass
class ClsSampleMeta:
    image_id: int
    leaf_id: int
    class_name: str
    image_path: str
    path_hierarchy: Optional[List[str]] = None


# -----------------------------
# 2) Dataset
# -----------------------------
class FoodClsDatasetLocalFolder(Dataset):
    """
    本地文件夹版分类数据集

    目录结构建议：
      root/train/<class_name>/*.jpg
      root/valid/<class_name>/*.jpg
      root/test/<class_name>/*.jpg

    传入本类的 root 应该是某个 split 的目录，例如：
      root = /path/to/class_folder_split/train
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: str,
        resize_hw: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        img_interp: str = "linear",
        strict: bool = False,
        log_every_n_warnings: int = 50,
        class_sorted: bool = True,
        recursive: bool = True,
        is_train: bool = False,
        aug_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.root = Path(root).resolve()
        self.strict = bool(strict)
        self._warn_decode = 0
        self._log_every_n_warnings = max(int(log_every_n_warnings), 1)

        if not self.root.exists():
            raise FileNotFoundError(f"分类根目录不存在：{self.root}")
        if not self.root.is_dir():
            raise FileNotFoundError(f"分类根路径不是目录：{self.root}")

        logger.info(
            "初始化 FoodClsDatasetLocalFolder："
            f"root={self.root}, resize_hw={resize_hw}, img_interp={img_interp}, "
            f"class_sorted={class_sorted}, recursive={recursive}, "
            f"is_train={is_train}"
        )

        interp_cv2 = resolve_interp(img_interp)
        self.tfm = ClsTransforms(
            resize_hw=resize_hw,
            mean=mean,
            std=std,
            img_interp=interp_cv2,
            is_train=is_train,
            aug_cfg=aug_cfg,
        )
        logger.info("分类预处理配置：%s", self.tfm.describe())

        class_dirs = [p for p in self.root.iterdir() if p.is_dir()]
        if class_sorted:
            class_dirs = sorted(class_dirs, key=lambda x: x.name)

        if not class_dirs:
            raise ValueError(f"分类根目录下未找到任何类别子目录：{self.root}")

        self.class_names: List[str] = [p.name for p in class_dirs]
        self.class_to_leaf_id: Dict[str, int] = { class_name: idx for idx, class_name in enumerate(self.class_names) }
        self.leaf_id_to_class: Dict[int, str] = { idx: class_name for idx, class_name in enumerate(self.class_names) }
        self.num_classes = len(self.class_names)

        # 兼容训练入口中的类别预检逻辑：
        # 直接从元数据获取类别，避免训练前调用 __getitem__
        # 遍历并解码十万级图片。
        self.leaf_ids = set(self.leaf_id_to_class.keys())
        self.leaf_to_name = dict(self.leaf_id_to_class)

        self.samples: List[Dict[str, Any]] = []
        image_id = 0

        for class_dir in class_dirs:
            class_name = class_dir.name
            leaf_id = self.class_to_leaf_id[class_name]

            file_iter = class_dir.rglob("*") if recursive else class_dir.glob("*")
            img_files = [ p for p in file_iter if p.is_file() and p.suffix.lower() in self.IMG_EXTS]
            img_files = sorted(img_files)

            if not img_files:
                logger.warning(f"类别目录下未找到图片，跳过该类：class='{class_name}', dir={class_dir}")
                continue

            for img_path in img_files:
                rel_parts = list(img_path.relative_to(self.root).parts[:-1])
                self.samples.append({
                    "image_id": image_id,
                    "leaf_id": int(leaf_id),
                    "class_name": class_name,
                    "image_path": str(img_path),
                    "path_hierarchy": rel_parts if rel_parts else [class_name],
                })
                image_id += 1

        if not self.samples:
            raise ValueError(f"未找到任何可用图片样本：{self.root}")

        logger.info( f"本地分类数据集扫描完成：类别数={self.num_classes}, 样本数={len(self.samples)}")

        self.leaf_id_to_idx = None
        self.strict_label_map = False

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_image_by_cv2(image_path: str) -> np.ndarray:
        """
        使用 OpenCV 读取图片，并转换成 RGB uint8(HWC)
        """
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cv2.imread 读取失败：{image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def _map_leaf_to_idx(self, leaf_id: int) -> int:
        mp = getattr(self, "leaf_id_to_idx", None)
        if mp is None:
            return int(leaf_id)
        if leaf_id not in mp:
            if bool(getattr(self, "strict_label_map", False)):
                raise ValueError(f"验证/训练集出现未在映射中的 leaf_id：{leaf_id}，请检查 train/valid 映射一致性")
            return -1
        return int(mp[leaf_id])

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        image_id = int(sample["image_id"])
        leaf_id = int(sample["leaf_id"])
        class_name = str(sample["class_name"])
        image_path = str(sample["image_path"])
        path_hierarchy = sample.get("path_hierarchy", None)

        try:
            rgb = self._read_image_by_cv2(image_path)
        except Exception as e:
            if self.strict:
                raise ValueError(f"读取图片失败：image_path={image_path}，错误：{e}")

            self._warn_decode += 1
            if self._warn_decode <= 3 or (self._warn_decode % self._log_every_n_warnings == 0):
                logger.warning(
                    "读取图片失败，使用全零图兜底："
                    f"image_path={image_path}, 错误={type(e).__name__}: {e}, "
                    f"累计失败次数={self._warn_decode}"
                )
            rgb = np.zeros((self.tfm.resize_hw[0], self.tfm.resize_hw[1], 3), dtype=np.uint8)

        img_t = self.tfm(rgb)

        class_idx = self._map_leaf_to_idx(leaf_id)
        meta = ClsSampleMeta( 
            image_id=image_id,
            leaf_id=leaf_id,
            class_name=class_name,
            image_path=image_path,
            path_hierarchy=path_hierarchy,
        )
        md = meta.__dict__
        md["class_idx"] = int(class_idx)

        return img_t, int(class_idx), md


def get_cls_dataloader(
    dataroot: str,
    split: str,
    input_cfg: Dict[str, Any],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    ds_cfg: Optional[Dict[str, Any]] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    本地目录分类 DataLoader 工厂。

    参数说明：
      - dataroot: 分类根目录，内部按 split 拼接
      - split: 数据集划分名称，如 train / valid / test
    """
    ds_cfg = ds_cfg or {}

    img_dim = input_cfg["img_dim"]
    mean = input_cfg["mean"]
    std = input_cfg["std"]
    img_interp = input_cfg.get("img_interp", "linear")

    strict = bool(ds_cfg.get("strict", False))
    log_every_n_warnings = int(ds_cfg.get("log_every_n_warnings", 50))

    class_sorted = bool(ds_cfg.get("class_sorted", True))
    recursive = bool(ds_cfg.get("recursive", True))

    train_split = str(ds_cfg.get("train_split", "train"))
    is_train = str(split) == train_split
    train_aug_cfg = input_cfg.get("train_aug", {}) or {}

    split_root = (Path(dataroot) / split).resolve()
    ds = FoodClsDatasetLocalFolder(
        root=str(split_root),
        resize_hw=(int(img_dim[0]), int(img_dim[1])),
        mean=tuple(mean),
        std=tuple(std),
        img_interp=str(img_interp),
        strict=strict,
        log_every_n_warnings=log_every_n_warnings,
        class_sorted=class_sorted,
        recursive=recursive,
        is_train=is_train,
        aug_cfg=train_aug_cfg,
    )

    log_dataset_desc = (
        "构建 DataLoader："
        f"split={split}, split_root={split_root}, batch_size={batch_size}, "
        f"shuffle={shuffle}, num_workers={num_workers}, "
        f"samples={len(ds)}, num_classes={ds.num_classes}, "
        f"class_sorted={class_sorted}, recursive={recursive}, is_train={is_train}"
    )

    def collate_fn(batch):
        imgs, labels, metas = zip(*batch)
        imgs_t = torch.stack(imgs, dim=0)
        labels_t = torch.tensor(labels, dtype=torch.long)
        return imgs_t, labels_t, list(metas)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
        drop_last=False,
    )

    logger.info(log_dataset_desc)
    return dl
