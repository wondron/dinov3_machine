"""
6-inferOnnx_iou_analysis.py

ONNXRuntime 二分类分割推理脚本，支持：
1. --input 直接传入单张图像路径；
2. --input 传入图像目录，自动读取常见图像格式；
3. 自动或显式指定标注目录，计算逐图 IoU 指标；
4. iou.csv 按 mIoU 从小到大保存；
5. 输出 IoU 汇总、区间分布、阈值通过率及最低/最高样本分析。

默认输出：
  test_images/2_onnx_out/YYMMDD/
    masks/
    overlays/
    probs/                         # 使用 --save_prob_npy 时生成
    iou.csv                        # 按 mIoU 从小到大
    iou_summary.csv
    iou_distribution.csv
    iou_threshold_pass_rate.csv
    iou_lowest.csv
    iou_highest.csv
    iou_analysis.txt
    area_bucket_metrics.csv          # 按 GT 食物面积比例分桶的聚合指标
    area_bucket_analysis.txt         # 分桶指标文本报告
    missing_labels.csv
"""

import argparse
import csv
import glob
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dino_finetune.config import default_config_path, load_config, resolve_interp


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 按 GT 前景面积占有效像素的比例分桶。区间定义严格对应：
# 0、(0, 0.01]、(0.01, 0.1]、(0.1, 0.3]、(0.3, 0.7]、
# (0.7, 0.9]、(0.9, 0.95]、(0.95, 1.0]。
AREA_BUCKETS = [
    ("0", 0.0, 0.0),
    ("(0, 0.01]", 0.0, 0.01),
    ("(0.01, 0.1]", 0.01, 0.1),
    ("(0.1, 0.3]", 0.1, 0.3),
    ("(0.3, 0.7]", 0.3, 0.7),
    ("(0.7, 0.9]", 0.7, 0.9),
    ("(0.9, 0.95]", 0.9, 0.95),
    ("(0.95, 1.0]", 0.95, 1.0),
]
AREA_BUCKET_NAMES = [item[0] for item in AREA_BUCKETS]


# ============================================================
# 基础工具
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_image_path(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def list_images(
    in_dir: str,
    pattern: Optional[str] = None,
    recursive: bool = False,
) -> List[str]:
    """读取目录下的图像。

    - pattern 非空时，按 glob pattern 读取，例如 ``*.jpg``；
    - pattern 为空时，自动读取所有常见图像格式；
    - recursive=True 时递归读取。
    """
    root = Path(in_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"输入目录不存在：{in_dir}")

    paths: List[Path] = []
    if pattern:
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        paths = [p for p in iterator if p.is_file() and is_image_path(str(p))]
    else:
        iterator: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
        paths = [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]

    return sorted(str(p.resolve()) for p in paths)


def resolve_input_paths(
    input_path: str,
    pattern: Optional[str],
    recursive: bool,
) -> Tuple[List[str], str, bool]:
    """解析单图或目录输入。

    Returns:
        paths: 待推理图像路径列表
        input_dir: 图像所属目录
        single_image: 是否为单张图像模式
    """
    path = Path(input_path).expanduser()

    if path.is_file():
        if not is_image_path(str(path)):
            raise ValueError(
                f"--input 指向文件，但不是支持的图像格式：{path}\n"
                f"支持格式：{sorted(IMAGE_SUFFIXES)}"
            )
        return [str(path.resolve())], str(path.resolve().parent), True

    if path.is_dir():
        paths = list_images(str(path), pattern=pattern, recursive=recursive)
        if not paths:
            pattern_msg = pattern if pattern else "全部常见图像格式"
            raise FileNotFoundError(
                f"输入目录未找到图像：{path.resolve()}，pattern={pattern_msg}，recursive={recursive}"
            )
        return paths, str(path.resolve()), False

    raise FileNotFoundError(f"--input 路径不存在：{input_path}")


def get_providers(prefer_trt: bool = False) -> List[str]:
    avail = ort.get_available_providers()
    providers: List[str] = []

    if prefer_trt and "TensorrtExecutionProvider" in avail:
        providers.append("TensorrtExecutionProvider")
    if "CUDAExecutionProvider" in avail:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in avail:
        providers.append("CPUExecutionProvider")

    if not providers:
        raise RuntimeError(f"没有可用的 ONNXRuntime provider：{avail}")
    return providers


# ============================================================
# 预处理和后处理
# ============================================================
def letterbox_resize(
    img: np.ndarray,
    new_hw: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
    interp: int = cv2.INTER_LINEAR,
):
    target_h, target_w = new_hw
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    out = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )
    meta = (scale, pad_top, pad_left, pad_bottom, pad_right)
    return out, meta, (h, w)


def unletterbox_mask(
    mask_hw: np.ndarray,
    meta: Tuple[float, int, int, int, int],
    orig_hw: Tuple[int, int],
    interp: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    _, pad_top, pad_left, pad_bottom, pad_right = meta
    h, w = mask_hw.shape[:2]
    orig_h, orig_w = orig_hw

    y1, y2 = pad_top, h - pad_bottom
    x1, x2 = pad_left, w - pad_right
    cropped = mask_hw[y1:y2, x1:x2]

    if cropped.size == 0:
        raise ValueError(
            f"unletterbox 后裁剪为空：mask_shape={mask_hw.shape}, meta={meta}"
        )

    return cv2.resize(cropped, (orig_w, orig_h), interpolation=interp)


def preprocess(
    bgr: np.ndarray,
    img_dim: Tuple[int, int],
    letterbox: bool = False,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
    img_interp: int = cv2.INTER_LINEAR,
):
    if bgr is None or bgr.ndim != 3:
        raise ValueError("输入图像为空或维度不正确")

    orig_h, orig_w = bgr.shape[:2]

    if letterbox:
        resized, meta, _ = letterbox_resize(bgr, img_dim, interp=img_interp)
        info = {
            "orig_hw": (orig_h, orig_w),
            "letterbox": True,
            "meta": meta,
            "img_dim": img_dim,
        }
    else:
        resized = cv2.resize(
            bgr,
            (img_dim[1], img_dim[0]),
            interpolation=img_interp,
        )
        info = {
            "orig_hw": (orig_h, orig_w),
            "letterbox": False,
            "meta": None,
            "img_dim": img_dim,
        }

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0

    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    x = (x - mean_arr) / std_arr

    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0).astype(np.float32)
    return x, info


def postprocess(
    logits: np.ndarray,
    info: dict,
    threshold: float = 0.5,
    img_interp: int = cv2.INTER_LINEAR,
    msk_interp: int = cv2.INTER_NEAREST,
):
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 2:
        raise ValueError(f"期望 logits shape=(1,2,H,W)，实际为 {logits.shape}")

    x0 = logits[0, 0].astype(np.float32)
    x1 = logits[0, 1].astype(np.float32)

    # 数值稳定的二分类 softmax 前景概率。
    max_logits = np.maximum(x0, x1)
    exp0 = np.exp(x0 - max_logits)
    exp1 = np.exp(x1 - max_logits)
    prob_model = exp1 / (exp0 + exp1 + 1e-12)

    mask_model = (prob_model >= threshold).astype(np.uint8) * 255
    orig_h, orig_w = info["orig_hw"]

    if info.get("letterbox", False):
        meta = info["meta"]
        prob_up = unletterbox_mask(
            prob_model,
            meta,
            (orig_h, orig_w),
            interp=img_interp,
        )
        mask_u8 = unletterbox_mask(
            mask_model,
            meta,
            (orig_h, orig_w),
            interp=msk_interp,
        )
    else:
        prob_up = cv2.resize(
            prob_model,
            (orig_w, orig_h),
            interpolation=img_interp,
        )
        mask_u8 = cv2.resize(
            mask_model,
            (orig_w, orig_h),
            interpolation=msk_interp,
        )

    mask_u8 = np.where(mask_u8 > 0, 255, 0).astype(np.uint8)
    return mask_u8, prob_up.astype(np.float32), prob_model.astype(np.float32)


def overlay_mask(
    bgr: np.ndarray,
    mask_u8: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = bgr.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)
    mask_bool = mask_u8 > 0
    overlay[mask_bool] = (
        overlay[mask_bool].astype(np.float32) * (1.0 - alpha)
        + color.astype(np.float32) * alpha
    ).astype(np.uint8)
    return overlay


# ============================================================
# 标注读取和 IoU 指标
# ============================================================
def build_label_map(label_dir: str, recursive: bool = False) -> Dict[str, str]:
    """按文件 stem 建立标注索引。"""
    label_paths = list_images(label_dir, pattern=None, recursive=recursive)
    label_map: Dict[str, str] = {}

    for path in label_paths:
        stem = Path(path).stem
        if stem in label_map:
            tqdm.write(
                f"[警告] 标注文件 stem 重复，将使用后出现的文件："
                f"{label_map[stem]} -> {path}"
            )
        label_map[stem] = path

    return label_map


def read_label_as_index(
    label_path: str,
    ignore_index: Optional[int] = 255,
) -> np.ndarray:
    """读取二分类标注，返回 int64(H,W)。

    支持：
    - 0/1，且可包含 ignore_index=255；
    - 当 ignore_index=None 时，0/255 会自动转换为 0/1。
    """
    gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"读取标注失败：{label_path}")

    gt = gt.astype(np.int64)
    unique_values = set(np.unique(gt).tolist())
    allowed = {0, 1, 255}

    if not unique_values.issubset(allowed):
        raise ValueError(
            f"标注值必须是 {sorted(allowed)} 的子集，"
            f"实际为 {sorted(unique_values)}：{label_path}"
        )

    if ignore_index == 255:
        return gt

    if 255 in unique_values:
        if unique_values.issubset({0, 255}):
            return (gt > 0).astype(np.int64)
        raise ValueError(
            f"标注同时包含 0/1/255，但 ignore_index={ignore_index}，"
            f"无法确定 255 是前景还是忽略值：{label_path}"
        )

    return gt


def safe_div(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def get_gt_area_bucket(gt_fg_ratio: float) -> str:
    """按照 GT 前景面积比例返回固定分桶名称。"""
    if not math.isfinite(gt_fg_ratio):
        raise ValueError(f"GT 前景面积比例不是有效数值：{gt_fg_ratio}")

    # 浮点计算可能出现极小误差，先限制到 [0, 1]。
    ratio = min(max(float(gt_fg_ratio), 0.0), 1.0)
    if ratio <= 1e-12:
        return "0"
    if ratio <= 0.01:
        return "(0, 0.01]"
    if ratio <= 0.1:
        return "(0.01, 0.1]"
    if ratio <= 0.3:
        return "(0.1, 0.3]"
    if ratio <= 0.7:
        return "(0.3, 0.7]"
    if ratio <= 0.9:
        return "(0.7, 0.9]"
    if ratio <= 0.95:
        return "(0.9, 0.95]"
    return "(0.95, 1.0]"


def compute_binary_segmentation_metrics(
    pred_mask_u8: np.ndarray,
    gt: np.ndarray,
    ignore_index: Optional[int] = 255,
) -> Dict[str, object]:
    """计算二分类分割指标和面积分桶字段。

    说明：
    - mIoU 与原脚本逻辑一致，只对 union>0 的类别求平均；
    - GT/预测面积比例使用有效像素作为分母；
    - 空图的 false_positive_area_ratio = FP / 整图像素数；
    - GT 面积比例位于 (0.95, 1.0] 时，
      false_negative_area_ratio = FN / 整图像素数；
    - 非适用样本的额外比例返回 NaN，写入 CSV 时显示为空白。
    """
    if pred_mask_u8.shape != gt.shape:
        raise ValueError(
            f"预测和标注尺寸不一致：pred={pred_mask_u8.shape}, gt={gt.shape}"
        )

    pred_fg_raw = pred_mask_u8 > 0
    gt_fg_raw = gt == 1

    if ignore_index is None:
        valid = np.ones(gt.shape, dtype=bool)
    else:
        valid = gt != ignore_index

    pred_fg = pred_fg_raw & valid
    gt_fg = gt_fg_raw & valid
    pred_bg = (~pred_fg_raw) & valid
    gt_bg = (~gt_fg_raw) & valid

    tp = int(np.logical_and(pred_fg, gt_fg).sum())
    fp = int(np.logical_and(pred_fg, gt_bg).sum())
    fn = int(np.logical_and(pred_bg, gt_fg).sum())
    tn = int(np.logical_and(pred_bg, gt_bg).sum())

    fg_union = tp + fp + fn
    bg_union = tn + fp + fn

    fg_iou = safe_div(tp, fg_union)
    bg_iou = safe_div(tn, bg_union)

    valid_ious = [value for value in (bg_iou, fg_iou) if math.isfinite(value)]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)

    image_pixels = int(gt.size)
    valid_pixels = int(valid.sum())
    pred_fg_pixels = int(pred_fg.sum())
    gt_fg_pixels = int(gt_fg.sum())
    pred_fg_ratio = safe_div(pred_fg_pixels, valid_pixels)
    gt_fg_ratio = safe_div(gt_fg_pixels, valid_pixels)

    if not math.isfinite(gt_fg_ratio):
        raise ValueError("标注中没有有效像素，无法计算面积比例")

    gt_area_bucket = get_gt_area_bucket(gt_fg_ratio)

    false_positive_area_ratio = (
        safe_div(fp, image_pixels)
        if gt_area_bucket == "0"
        else float("nan")
    )
    false_negative_area_ratio = (
        safe_div(fn, image_pixels)
        if gt_area_bucket == "(0.95, 1.0]"
        else float("nan")
    )

    return {
        "miou": miou,
        "fg_iou": fg_iou,
        "bg_iou": bg_iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "image_pixels": image_pixels,
        "valid_pixels": valid_pixels,
        "pred_fg_pixels": pred_fg_pixels,
        "gt_fg_pixels": gt_fg_pixels,
        "pred_fg_ratio": pred_fg_ratio,
        "gt_fg_ratio": gt_fg_ratio,
        "gt_area_bucket": gt_area_bucket,
        "false_positive_area_ratio": false_positive_area_ratio,
        "false_negative_area_ratio": false_negative_area_ratio,
    }


# ============================================================
# 单图推理
# ============================================================
def infer_and_save_one(
    sess: ort.InferenceSession,
    input_path: str,
    mask_dir: str,
    ov_dir: Optional[str],
    prob_dir: Optional[str],
    img_dim: Tuple[int, int],
    threshold: float,
    letterbox: bool,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    img_interp: int,
    msk_interp: int,
    save_prob_npy: bool,
) -> Dict[str, object]:
    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"读取图像失败：{input_path}")

    x, info = preprocess(
        bgr,
        img_dim=img_dim,
        letterbox=letterbox,
        mean=mean,
        std=std,
        img_interp=img_interp,
    )

    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: x})
    if not outputs:
        raise RuntimeError("ONNX 模型没有返回任何输出")

    logits = outputs[0]
    mask_u8, prob_up, prob_model = postprocess(
        logits,
        info,
        threshold=threshold,
        img_interp=img_interp,
        msk_interp=msk_interp,
    )

    base = Path(input_path).stem
    mask_path = os.path.join(mask_dir, f"{base}.png")
    if not cv2.imwrite(mask_path, mask_u8):
        raise IOError(f"保存 mask 失败：{mask_path}")

    overlay_path: Optional[str] = None
    if ov_dir is not None:
        overlay = overlay_mask(bgr, mask_u8, alpha=0.45)
        overlay_path = os.path.join(ov_dir, f"{base}.png")
        if not cv2.imwrite(overlay_path, overlay):
            raise IOError(f"保存 overlay 失败：{overlay_path}")

    prob_path: Optional[str] = None
    if save_prob_npy and prob_dir is not None:
        prob_path = os.path.join(prob_dir, f"{base}.npy")
        np.save(prob_path, prob_up.astype(np.float32))

    return {
        "base": base,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "prob_path": prob_path,
        "mask_u8": mask_u8,
        "prob_model": prob_model,
        "logits": logits.astype(np.float32),
        "prob_up": prob_up,
    }


# ============================================================
# IoU 结果保存和分析
# ============================================================
IOU_COLUMNS = [
    "rank",
    "image",
    "image_path",
    "label",
    "label_path",
    "gt_area_bucket",
    "miou",
    "fg_iou",
    "bg_iou",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "tn",
    "image_pixels",
    "valid_pixels",
    "pred_fg_pixels",
    "gt_fg_pixels",
    "pred_fg_ratio",
    "gt_fg_ratio",
    "false_positive_area_ratio",
    "false_negative_area_ratio",
    "mask_path",
    "overlay_path",
]



def csv_value(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.6f}"
    if value is None:
        return ""
    return value


def write_records_csv(
    path: str,
    records: List[Dict[str, object]],
    columns: List[str],
) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: csv_value(record.get(key)) for key in columns})


def finite_array(records: List[Dict[str, object]], key: str) -> np.ndarray:
    values: List[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def metric_summary_row(
    metric: str,
    values: np.ndarray,
) -> Dict[str, object]:
    if values.size == 0:
        return {
            "metric": metric,
            "count": 0,
            "min": "",
            "max": "",
            "mean": "",
            "median": "",
            "std": "",
            "p05": "",
            "p10": "",
            "p25": "",
            "p75": "",
            "p90": "",
            "p95": "",
        }

    return {
        "metric": metric,
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }



AREA_BUCKET_COLUMNS = [
    "bucket_order",
    "gt_area_bucket",
    "sample_count",
    "fg_iou",
    "bg_iou",
    "miou",
    "precision",
    "recall",
    "pred_fg_ratio",
    "gt_fg_ratio",
    "false_positive_area_ratio",
    "false_negative_area_ratio",
    "tp",
    "fp",
    "fn",
    "tn",
    "image_pixels",
    "valid_pixels",
    "pred_fg_pixels",
    "gt_fg_pixels",
]


def create_area_bucket_accumulators() -> Dict[str, Dict[str, int]]:
    """创建固定顺序的面积分桶累计器。"""
    return {
        name: {
            "sample_count": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "image_pixels": 0,
            "valid_pixels": 0,
            "pred_fg_pixels": 0,
            "gt_fg_pixels": 0,
        }
        for name in AREA_BUCKET_NAMES
    }


def update_area_bucket_accumulators(
    accumulators: Dict[str, Dict[str, int]],
    record: Dict[str, object],
) -> None:
    bucket = str(record["gt_area_bucket"])
    if bucket not in accumulators:
        raise KeyError(f"未知面积分桶：{bucket}")

    target = accumulators[bucket]
    target["sample_count"] += 1
    for key in (
        "tp",
        "fp",
        "fn",
        "tn",
        "image_pixels",
        "valid_pixels",
        "pred_fg_pixels",
        "gt_fg_pixels",
    ):
        target[key] += int(record[key])


def build_area_bucket_rows(
    accumulators: Dict[str, Dict[str, int]],
) -> List[Dict[str, object]]:
    """从累计混淆矩阵生成每个面积区间的像素级聚合指标。"""
    rows: List[Dict[str, object]] = []

    for order, bucket_name in enumerate(AREA_BUCKET_NAMES):
        item = accumulators[bucket_name]
        tp = int(item["tp"])
        fp = int(item["fp"])
        fn = int(item["fn"])
        tn = int(item["tn"])
        image_pixels = int(item["image_pixels"])
        valid_pixels = int(item["valid_pixels"])
        pred_fg_pixels = int(item["pred_fg_pixels"])
        gt_fg_pixels = int(item["gt_fg_pixels"])

        fg_iou = safe_div(tp, tp + fp + fn)
        bg_iou = safe_div(tn, tn + fp + fn)
        valid_ious = [value for value in (bg_iou, fg_iou) if math.isfinite(value)]
        miou = float(np.mean(valid_ious)) if valid_ious else float("nan")

        rows.append(
            {
                "bucket_order": order,
                "gt_area_bucket": bucket_name,
                "sample_count": int(item["sample_count"]),
                "fg_iou": fg_iou,
                "bg_iou": bg_iou,
                "miou": miou,
                "precision": safe_div(tp, tp + fp),
                "recall": safe_div(tp, tp + fn),
                "pred_fg_ratio": safe_div(pred_fg_pixels, valid_pixels),
                "gt_fg_ratio": safe_div(gt_fg_pixels, valid_pixels),
                "false_positive_area_ratio": (
                    safe_div(fp, image_pixels)
                    if bucket_name == "0"
                    else float("nan")
                ),
                "false_negative_area_ratio": (
                    safe_div(fn, image_pixels)
                    if bucket_name == "(0.95, 1.0]"
                    else float("nan")
                ),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "image_pixels": image_pixels,
                "valid_pixels": valid_pixels,
                "pred_fg_pixels": pred_fg_pixels,
                "gt_fg_pixels": gt_fg_pixels,
            }
        )

    return rows


def metric_text(value: object) -> str:
    if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
        return f"{float(value):.6f}"
    return "N/A"


def print_area_bucket_summary(
    rows: List[Dict[str, object]],
    title: str,
) -> None:
    """通过 tqdm.write 输出分桶统计，不破坏进度条。"""
    tqdm.write("")
    tqdm.write(title)
    tqdm.write("-" * 154)
    tqdm.write(
        f"{'GT面积区间':<16} {'数量':>7} {'fgIoU':>10} {'bgIoU':>10} "
        f"{'mIoU':>10} {'Precision':>10} {'Recall':>10} "
        f"{'PredRatio':>11} {'GTRatio':>11} {'FP-area':>10} {'FN-area':>10}"
    )
    tqdm.write("-" * 154)
    for row in rows:
        tqdm.write(
            f"{str(row['gt_area_bucket']):<16} "
            f"{int(row['sample_count']):>7d} "
            f"{metric_text(row['fg_iou']):>10} "
            f"{metric_text(row['bg_iou']):>10} "
            f"{metric_text(row['miou']):>10} "
            f"{metric_text(row['precision']):>10} "
            f"{metric_text(row['recall']):>10} "
            f"{metric_text(row['pred_fg_ratio']):>11} "
            f"{metric_text(row['gt_fg_ratio']):>11} "
            f"{metric_text(row['false_positive_area_ratio']):>10} "
            f"{metric_text(row['false_negative_area_ratio']):>10}"
        )
    tqdm.write("-" * 154)


def save_area_bucket_analysis(
    out_dir: str,
    accumulators: Dict[str, Dict[str, int]],
) -> Dict[str, str]:
    rows = build_area_bucket_rows(accumulators)

    bucket_csv = os.path.join(out_dir, "area_bucket_metrics.csv")
    write_records_csv(bucket_csv, rows, AREA_BUCKET_COLUMNS)

    bucket_txt = os.path.join(out_dir, "area_bucket_analysis.txt")
    with open(bucket_txt, "w", encoding="utf-8") as file:
        file.write("按 GT 食物面积比例分桶的二分类分割指标\n")
        file.write("=" * 154 + "\n")
        file.write(
            "统计方式：每个桶内先累计 TP/FP/FN/TN，再计算像素级聚合指标；"
            "面积比例以有效像素为分母。\n"
        )
        file.write(
            "空图 false_positive_area_ratio = FP / 整图像素数；"
            "(0.95, 1.0] 桶 false_negative_area_ratio = FN / 整图像素数。\n\n"
        )
        file.write(
            f"{'GT面积区间':<16} {'数量':>7} {'fgIoU':>10} {'bgIoU':>10} "
            f"{'mIoU':>10} {'Precision':>10} {'Recall':>10} "
            f"{'PredRatio':>11} {'GTRatio':>11} {'FP-area':>10} {'FN-area':>10}\n"
        )
        file.write("-" * 154 + "\n")
        for row in rows:
            file.write(
                f"{str(row['gt_area_bucket']):<16} "
                f"{int(row['sample_count']):>7d} "
                f"{metric_text(row['fg_iou']):>10} "
                f"{metric_text(row['bg_iou']):>10} "
                f"{metric_text(row['miou']):>10} "
                f"{metric_text(row['precision']):>10} "
                f"{metric_text(row['recall']):>10} "
                f"{metric_text(row['pred_fg_ratio']):>11} "
                f"{metric_text(row['gt_fg_ratio']):>11} "
                f"{metric_text(row['false_positive_area_ratio']):>10} "
                f"{metric_text(row['false_negative_area_ratio']):>10}\n"
            )

    return {
        "area_bucket_csv": bucket_csv,
        "area_bucket_txt": bucket_txt,
    }

def save_iou_analysis(
    out_dir: str,
    records: List[Dict[str, object]],
    top_k: int,
    area_bucket_accumulators: Dict[str, Dict[str, int]],
) -> Dict[str, str]:
    """按 mIoU 从小到大保存，并生成统计分析文件。"""
    sorted_records = sorted(records, key=lambda item: float(item["miou"]))
    for rank, record in enumerate(sorted_records, start=1):
        record["rank"] = rank

    iou_csv = os.path.join(out_dir, "iou.csv")
    write_records_csv(iou_csv, sorted_records, IOU_COLUMNS)

    metric_names = ["miou", "fg_iou", "bg_iou", "precision", "recall", "f1"]
    summary_rows = [
        metric_summary_row(metric, finite_array(sorted_records, metric))
        for metric in metric_names
    ]
    summary_columns = [
        "metric",
        "count",
        "min",
        "max",
        "mean",
        "median",
        "std",
        "p05",
        "p10",
        "p25",
        "p75",
        "p90",
        "p95",
    ]
    summary_csv = os.path.join(out_dir, "iou_summary.csv")
    write_records_csv(summary_csv, summary_rows, summary_columns)

    miou_values = finite_array(sorted_records, "miou")

    distribution_rows: List[Dict[str, object]] = []
    bin_edges = np.arange(0.0, 1.000001, 0.1)
    for index in range(len(bin_edges) - 1):
        lower = float(bin_edges[index])
        upper = float(bin_edges[index + 1])
        if index == len(bin_edges) - 2:
            count = int(np.logical_and(miou_values >= lower, miou_values <= upper).sum())
            interval = f"[{lower:.1f}, {upper:.1f}]"
        else:
            count = int(np.logical_and(miou_values >= lower, miou_values < upper).sum())
            interval = f"[{lower:.1f}, {upper:.1f})"

        distribution_rows.append(
            {
                "interval": interval,
                "min_inclusive": lower,
                "max": upper,
                "count": count,
                "ratio": safe_div(count, int(miou_values.size)),
            }
        )

    distribution_csv = os.path.join(out_dir, "iou_distribution.csv")
    write_records_csv(
        distribution_csv,
        distribution_rows,
        ["interval", "min_inclusive", "max", "count", "ratio"],
    )

    pass_rows: List[Dict[str, object]] = []
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        pass_count = int((miou_values >= threshold).sum())
        pass_rows.append(
            {
                "threshold": threshold,
                "pass_count": pass_count,
                "total": int(miou_values.size),
                "pass_rate": safe_div(pass_count, int(miou_values.size)),
            }
        )

    pass_rate_csv = os.path.join(out_dir, "iou_threshold_pass_rate.csv")
    write_records_csv(
        pass_rate_csv,
        pass_rows,
        ["threshold", "pass_count", "total", "pass_rate"],
    )

    actual_top_k = min(max(top_k, 1), len(sorted_records))
    lowest_records = sorted_records[:actual_top_k]
    highest_records = list(reversed(sorted_records[-actual_top_k:]))

    lowest_csv = os.path.join(out_dir, "iou_lowest.csv")
    highest_csv = os.path.join(out_dir, "iou_highest.csv")
    write_records_csv(lowest_csv, lowest_records, IOU_COLUMNS)
    write_records_csv(highest_csv, highest_records, IOU_COLUMNS)

    analysis_txt = os.path.join(out_dir, "iou_analysis.txt")
    with open(analysis_txt, "w", encoding="utf-8") as file:
        file.write("IoU 分析报告\n")
        file.write("=" * 80 + "\n")
        file.write(f"有效标注样本数：{len(sorted_records)}\n")
        file.write("iou.csv 排序方式：按 mIoU 从小到大\n\n")

        for row in summary_rows:
            if int(row["count"]) == 0:
                continue
            file.write(
                f"{row['metric']}: "
                f"min={float(row['min']):.6f}, "
                f"max={float(row['max']):.6f}, "
                f"mean={float(row['mean']):.6f}, "
                f"median={float(row['median']):.6f}, "
                f"std={float(row['std']):.6f}, "
                f"p10={float(row['p10']):.6f}, "
                f"p90={float(row['p90']):.6f}\n"
            )

        file.write("\n不同 mIoU 阈值通过率\n")
        file.write("-" * 80 + "\n")
        for row in pass_rows:
            pass_rate = row["pass_rate"]
            rate_text = "N/A" if not math.isfinite(float(pass_rate)) else f"{float(pass_rate):.2%}"
            file.write(
                f"mIoU >= {float(row['threshold']):.2f}: "
                f"{row['pass_count']}/{row['total']} ({rate_text})\n"
            )

        file.write(f"\n最低 {actual_top_k} 个样本\n")
        file.write("-" * 80 + "\n")
        for record in lowest_records:
            file.write(
                f"rank={record['rank']:>5}  "
                f"mIoU={float(record['miou']):.6f}  "
                f"fgIoU={csv_value(record['fg_iou'])}  "
                f"F1={csv_value(record['f1'])}  "
                f"image={record['image']}\n"
            )

        file.write(f"\n最高 {actual_top_k} 个样本\n")
        file.write("-" * 80 + "\n")
        for record in highest_records:
            file.write(
                f"rank={record['rank']:>5}  "
                f"mIoU={float(record['miou']):.6f}  "
                f"fgIoU={csv_value(record['fg_iou'])}  "
                f"F1={csv_value(record['f1'])}  "
                f"image={record['image']}\n"
            )

    area_bucket_paths = save_area_bucket_analysis(
        out_dir=out_dir,
        accumulators=area_bucket_accumulators,
    )

    return {
        "iou_csv": iou_csv,
        "summary_csv": summary_csv,
        "distribution_csv": distribution_csv,
        "pass_rate_csv": pass_rate_csv,
        "lowest_csv": lowest_csv,
        "highest_csv": highest_csv,
        "analysis_txt": analysis_txt,
        **area_bucket_paths,
    }


# ============================================================
# 主函数
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ONNX 二分类分割推理：支持单图/目录输入和 IoU 分析"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path("seg"),
        help="YAML 配置路径",
    )
    parser.add_argument("--onnx", required=True, help="ONNX 模型路径")
    parser.add_argument(
        "--input",
        required=True,
        help="单张图像路径或图像目录",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="目录输入时的 glob pattern；默认读取全部常见图像格式",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="输入为目录时递归搜索图像",
    )
    parser.add_argument(
        "--label_dir",
        default=None,
        help="标注目录；未指定时按 images/xxx -> annotations/xxx 自动推导",
    )
    parser.add_argument(
        "--label_recursive",
        action="store_true",
        help="递归读取标注目录",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="输出目录；默认 test_images/2_onnx_out/YYMMDD",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="兼容旧参数；实际阈值仍以 config 中 postprocess.thr 为准",
    )
    parser.add_argument(
        "--letterbox",
        action="store_true",
        help="兼容旧参数；实际设置仍以 config 为准",
    )
    parser.add_argument(
        "--save_prob_npy",
        action="store_true",
        help="保存原图尺寸的前景概率图 .npy",
    )
    parser.add_argument(
        "--no_overlay",
        action="store_true",
        help="不保存预测叠加图",
    )
    parser.add_argument(
        "--prefer_trt",
        action="store_true",
        help="优先使用 TensorRTExecutionProvider",
    )
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=255,
        help="兼容旧参数；实际设置仍以 config 为准",
    )
    parser.add_argument(
        "--analysis_topk",
        type=int,
        default=20,
        help="最低/最高 IoU 样本各保存多少条，默认 20",
    )
    parser.add_argument(
        "--area_bucket_log_interval",
        type=int,
        default=1000,
        help=(
            "推理期间每处理多少个有标注样本输出一次面积分桶累计指标；"
            "默认 1000，设置为 0 表示仅在结束时输出"
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_input = cfg["input"]
    cfg_model = cfg["model"]
    cfg_post = cfg["postprocess"]

    if (
        cfg_post.get("mask_mode") != "prob_threshold"
        or cfg_post.get("iou_mode") != "prob_threshold"
    ):
        raise ValueError(
            "config 中 mask_mode 和 iou_mode 必须均为 'prob_threshold'"
        )

    img_dim = tuple(cfg_input["img_dim"])
    mean = tuple(cfg_input["mean"])
    std = tuple(cfg_input["std"])
    img_interp = resolve_interp(cfg_input["img_interp"])
    msk_interp = resolve_interp(cfg_input["mask_interp"])
    threshold = float(cfg_post["thr"])
    cfg_ignore_index = int(cfg_post["ignore_index"])
    ignore_index = None if cfg_ignore_index < 0 else cfg_ignore_index
    n_classes = int(cfg_model["n_classes"])
    letterbox = bool(cfg_input.get("letterbox", False))

    if n_classes != 2:
        raise ValueError(f"当前脚本仅支持二分类，config n_classes={n_classes}")

    if abs(args.threshold - threshold) > 1e-9:
        tqdm.write(
            f"[config] 忽略 CLI threshold={args.threshold}，使用 config thr={threshold}"
        )
    if args.letterbox != letterbox and args.letterbox:
        tqdm.write(
            f"[config] 忽略 CLI letterbox=True，使用 config letterbox={letterbox}"
        )
    if args.ignore_index >= 0 and args.ignore_index != cfg_ignore_index:
        tqdm.write(
            f"[config] 忽略 CLI ignore_index={args.ignore_index}，"
            f"使用 config ignore_index={cfg_ignore_index}"
        )

    paths, input_dir, single_image = resolve_input_paths(
        input_path=args.input,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    date_tag = datetime.now().strftime("%y%m%d")
    out_dir = args.out_dir or os.path.join("test_images", "2_onnx_out", date_tag)
    out_dir = str(Path(out_dir).resolve())
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    
    
    mask_dir = os.path.join(out_dir, "masks")
    overlay_dir = os.path.join(out_dir, "overlays")
    prob_dir = os.path.join(out_dir, "probs")

    ensure_dir(out_dir)
    ensure_dir(mask_dir)
    if not args.no_overlay:
        ensure_dir(overlay_dir)
    if args.save_prob_npy:
        ensure_dir(prob_dir)

    # 单图和目录模式都允许手工指定 label_dir。
    if args.label_dir:
        label_dir = str(Path(args.label_dir).expanduser().resolve())
    else:
        input_dir_path = Path(input_dir).resolve()
        label_dir = str(
            input_dir_path.parent.parent / "annotations" / input_dir_path.name
        )

    has_label = os.path.isdir(label_dir)
    label_map = (
        build_label_map(label_dir, recursive=args.label_recursive)
        if has_label
        else {}
    )

    tqdm.write("-" * 80)
    tqdm.write(f"输入模式：{'单张图像' if single_image else '图像目录'}")
    tqdm.write(f"输入路径：{Path(args.input).expanduser().resolve()}")
    tqdm.write(f"待推理图像数：{len(paths)}")
    tqdm.write(f"输出目录：{out_dir}")

    if has_label:
        tqdm.write(f"标注目录：{label_dir}")
        tqdm.write(f"标注数量：{len(label_map)}")
    else:
        tqdm.write(f"未检测到标注目录：{label_dir}，将只推理、不计算 IoU")

    providers = get_providers(prefer_trt=args.prefer_trt)
    sess = ort.InferenceSession(args.onnx, providers=providers)

    model_input = sess.get_inputs()[0]
    tqdm.write(f"ONNXRuntime providers：{sess.get_providers()}")
    tqdm.write(
        f"模型输入：name={model_input.name}, shape={model_input.shape}, type={model_input.type}"
    )
    tqdm.write(
        f"配置：img_dim={img_dim}, threshold={threshold}, letterbox={letterbox}, "
        f"ignore_index={ignore_index}"
    )
    tqdm.write("-" * 80)

    iou_records: List[Dict[str, object]] = []
    missing_label_records: List[Dict[str, object]] = []
    area_bucket_accumulators = create_area_bucket_accumulators()

    success_count = 0
    failed_count = 0
    miou_running_sum = 0.0
    current_bucket: Optional[str] = None
    current_gt_ratio = float("nan")
    current_pred_ratio = float("nan")

    progress = tqdm(
        paths,
        total=len(paths),
        desc="ONNX 推理",
        unit="img",
        dynamic_ncols=True,
    )

    for image_path in progress:
        filename = os.path.basename(image_path)
        progress.set_description(f"ONNX 推理 | {filename[:36]}")

        try:
            result = infer_and_save_one(
                sess=sess,
                input_path=image_path,
                mask_dir=mask_dir,
                ov_dir=None if args.no_overlay else overlay_dir,
                prob_dir=prob_dir if args.save_prob_npy else None,
                img_dim=img_dim,
                threshold=threshold,
                letterbox=letterbox,
                mean=mean,
                std=std,
                img_interp=img_interp,
                msk_interp=msk_interp,
                save_prob_npy=args.save_prob_npy,
            )

            base = str(result["base"])
            if has_label and base in label_map:
                label_path = label_map[base]
                gt = read_label_as_index(
                    label_path,
                    ignore_index=ignore_index,
                )
                metrics = compute_binary_segmentation_metrics(
                    pred_mask_u8=result["mask_u8"],
                    gt=gt,
                    ignore_index=ignore_index,
                )

                record: Dict[str, object] = {
                    "image": filename,
                    "image_path": image_path,
                    "label": os.path.basename(label_path),
                    "label_path": label_path,
                    "mask_path": result["mask_path"],
                    "overlay_path": result["overlay_path"],
                }
                record.update(metrics)
                iou_records.append(record)
                update_area_bucket_accumulators(area_bucket_accumulators, record)

                miou_running_sum += float(record["miou"])
                current_bucket = str(record["gt_area_bucket"])
                current_gt_ratio = float(record["gt_fg_ratio"])
                current_pred_ratio = float(record["pred_fg_ratio"])

                if (
                    args.area_bucket_log_interval > 0
                    and len(iou_records) % args.area_bucket_log_interval == 0
                ):
                    print_area_bucket_summary(
                        build_area_bucket_rows(area_bucket_accumulators),
                        title=(
                            f"面积分桶累计统计：已处理 {len(iou_records)} 个有标注样本"
                        ),
                    )

            elif has_label:
                missing_label_records.append(
                    {
                        "image": filename,
                        "image_path": image_path,
                        "expected_stem": base,
                    }
                )

            success_count += 1

        except Exception as exc:
            failed_count += 1
            tqdm.write(
                f"[失败] {image_path} | {type(exc).__name__}: {exc}"
            )

        postfix = {
            "success": success_count,
            "failed": failed_count,
            "labeled": len(iou_records),
        }
        if iou_records:
            postfix["mIoU"] = f"{miou_running_sum / len(iou_records):.4f}"
        if current_bucket is not None:
            postfix["bucket"] = current_bucket
            postfix["GT_area"] = f"{current_gt_ratio:.4f}"
            postfix["Pred_area"] = f"{current_pred_ratio:.4f}"
        if has_label:
            postfix["no_label"] = len(missing_label_records)
        progress.set_postfix(postfix)

    progress.close()

    missing_label_csv = os.path.join(out_dir, "missing_labels.csv")
    write_records_csv(
        missing_label_csv,
        missing_label_records,
        ["image", "image_path", "expected_stem"],
    )

    tqdm.write("-" * 80)
    tqdm.write(
        f"推理统计：总数={len(paths)}，成功={success_count}，失败={failed_count}，"
        f"有标注={len(iou_records)}，缺少标注={len(missing_label_records)}"
    )

    if iou_records:
        analysis_paths = save_iou_analysis(
            out_dir=out_dir,
            records=iou_records,
            top_k=args.analysis_topk,
            area_bucket_accumulators=area_bucket_accumulators,
        )

        miou_values = finite_array(iou_records, "miou")
        fg_iou_values = finite_array(iou_records, "fg_iou")

        tqdm.write(
            f"mIoU：min={np.min(miou_values):.6f}，"
            f"max={np.max(miou_values):.6f}，"
            f"mean={np.mean(miou_values):.6f}，"
            f"median={np.median(miou_values):.6f}"
        )
        if fg_iou_values.size > 0:
            tqdm.write(
                f"前景 IoU：min={np.min(fg_iou_values):.6f}，"
                f"max={np.max(fg_iou_values):.6f}，"
                f"mean={np.mean(fg_iou_values):.6f}"
            )

        print_area_bucket_summary(
            build_area_bucket_rows(area_bucket_accumulators),
            title="面积分桶最终统计",
        )

        tqdm.write(f"IoU 明细（min -> max）：{analysis_paths['iou_csv']}")
        tqdm.write(f"IoU 汇总统计：{analysis_paths['summary_csv']}")
        tqdm.write(f"IoU 区间分布：{analysis_paths['distribution_csv']}")
        tqdm.write(f"IoU 阈值通过率：{analysis_paths['pass_rate_csv']}")
        tqdm.write(f"IoU 文本分析：{analysis_paths['analysis_txt']}")
        tqdm.write(f"面积分桶 CSV：{analysis_paths['area_bucket_csv']}")
        tqdm.write(f"面积分桶文本报告：{analysis_paths['area_bucket_txt']}")
    else:
        tqdm.write("未匹配到有效标注，跳过 IoU 分析")

    tqdm.write(f"推理完成：{out_dir}")


if __name__ == "__main__":
    main()
