from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dino_finetune.config import default_config_path, load_config, resolve_interp


LOGGER = logging.getLogger("多任务ONNX分割推理")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def recreate_output_dir(out_dir: Path) -> None:
    protected_dirs = {ROOT.resolve(), Path(out_dir.anchor).resolve()}
    if out_dir in protected_dirs:
        raise ValueError(f"拒绝删除危险的输出目录：{out_dir}")
    if out_dir.exists():
        if not out_dir.is_dir():
            raise NotADirectoryError(f"输出路径不是目录：{out_dir}")
        shutil.rmtree(out_dir)
        LOGGER.info("已删除原输出目录：%s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)


def resolve_images(input_path: Path, recursive: bool) -> tuple[list[Path], Path]:
    path = input_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"输入文件不是支持的图片格式：{path}")
        return [path], path.parent
    if not path.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{path}")
    iterator = path.rglob("*") if recursive else path.iterdir()
    images = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"输入目录中未找到图片：{path}")
    return images, path


def preprocess(image: np.ndarray, input_cfg: dict) -> np.ndarray:
    height, width = (int(v) for v in input_cfg["img_dim"])
    interpolation = resolve_interp(str(input_cfg["img_interp"]))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=interpolation).astype(np.float32) / 255.0
    mean = np.asarray(input_cfg["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(input_cfg["std"], dtype=np.float32).reshape(1, 1, 3)
    normalized = (rgb - mean) / std
    return np.transpose(normalized, (2, 0, 1))[None].astype(np.float32)


def softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / (np.sum(exp, axis=axis, keepdims=True) + 1e-12)


def resize_probabilities(probs: np.ndarray, size_wh: tuple[int, int], interpolation: int) -> np.ndarray:
    return np.stack(
        [cv2.resize(channel, size_wh, interpolation=interpolation) for channel in probs],
        axis=0,
    )


def postprocess(
    logits: np.ndarray,
    original_hw: tuple[int, int],
    n_classes: int,
    mask_mode: str,
    threshold: float,
    img_interp: int,
    mask_interp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != n_classes:
        raise ValueError(f"分割输出形状错误：期望 (1,{n_classes},H,W)，实际 {logits.shape}")
    probs = softmax(logits.astype(np.float32), axis=1)[0]
    original_h, original_w = original_hw

    if mask_mode == "prob_threshold":
        if n_classes != 2:
            raise ValueError("prob_threshold 仅支持二分类分割，多分类请配置 mask_mode=argmax")
        prob_up = cv2.resize(probs[1], (original_w, original_h), interpolation=img_interp)
        mask_index = (prob_up >= threshold).astype(np.uint8)
        return mask_index * 255, mask_index, prob_up.astype(np.float32)

    if mask_mode == "argmax":
        mask_model = np.argmax(probs, axis=0).astype(np.uint16)
        mask_index = cv2.resize(mask_model, (original_w, original_h), interpolation=mask_interp)
        mask_index = mask_index.astype(np.uint8 if n_classes <= 256 else np.uint16)
        probs_up = resize_probabilities(probs, (original_w, original_h), img_interp)
        return mask_index, mask_index, probs_up.astype(np.float32)

    raise ValueError(f"不支持的分割后处理模式：{mask_mode}")


def build_palette(n_classes: int) -> np.ndarray:
    palette = np.zeros((max(n_classes, 1), 3), dtype=np.uint8)
    if n_classes > 1:
        palette[1] = (0, 0, 255)
    for class_id in range(2, n_classes):
        hsv = np.uint8([[[((class_id - 1) * 37) % 180, 210, 255]]])
        palette[class_id] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return palette


def overlay_mask(image: np.ndarray, mask_index: np.ndarray, n_classes: int, alpha: float = 0.45) -> np.ndarray:
    result = image.copy()
    palette = build_palette(n_classes)
    for class_id in range(1, n_classes):
        selected = mask_index == class_id
        if np.any(selected):
            result[selected] = (
                result[selected].astype(np.float32) * (1.0 - alpha)
                + palette[class_id].astype(np.float32) * alpha
            ).astype(np.uint8)
    return result


def select_providers(ort: object, provider: str) -> list[str]:
    available = ort.get_available_providers()
    if provider == "auto":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
    mapping = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }
    selected = mapping[provider]
    if selected not in available:
        raise RuntimeError(f"指定的 ONNXRuntime provider 不可用：{selected}；当前可用={available}")
    return [selected, "CPUExecutionProvider"] if selected != "CPUExecutionProvider" else [selected]


def main() -> None:
    date_tag = datetime.now().strftime("%y%m%d")
    parser = argparse.ArgumentParser(description="多任务 ONNX 模型分割推理（支持单文件和文件夹）")
    parser.add_argument("--config", default=default_config_path("mul"), help="多任务 YAML 配置文件")
    parser.add_argument("--onnx", required=True, help="multitask_seg.onnx 路径")
    parser.add_argument("--input", required=True, help="单张图片或图片文件夹")
    parser.add_argument(
        "--out_dir",
        default=f"z_infer_res/2_onnx_out/3_multi/1_segment/{date_tag}",
        help="输出目录",
    )
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda", "tensorrt"), default="auto")
    parser.add_argument("--recursive", action="store_true", help="递归读取输入文件夹")
    parser.add_argument("--save_prob_npy", action="store_true", help="保存分割概率 npy")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("未安装 onnxruntime，请先安装 onnxruntime 或 onnxruntime-gpu") from exc

    cfg = load_config(args.config)
    onnx_path = Path(args.onnx).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX 模型不存在：{onnx_path}")
    images, input_root = resolve_images(Path(args.input), args.recursive)
    out_dir = Path(args.out_dir).expanduser().resolve()
    recreate_output_dir(out_dir)

    input_cfg = cfg["input"]
    post_cfg = cfg["postprocess"]
    n_classes = int(cfg["model"]["n_classes"])
    mask_mode = str(post_cfg["mask_mode"]).strip().lower()
    threshold = float(post_cfg["thr"])
    img_interp = resolve_interp(str(input_cfg["img_interp"]))
    mask_interp = resolve_interp(str(input_cfg["mask_interp"]))

    session = ort.InferenceSession(str(onnx_path), providers=select_providers(ort, args.provider))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    LOGGER.info("ONNX providers：%s；输入=%s；输出=%s", session.get_providers(), input_name, output_name)

    success = 0
    for image_path in tqdm(images, desc="分割推理", unit="张"):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            tqdm.write(f"错误：图片读取失败，已跳过：{image_path}")
            continue
        logits = session.run([output_name], {input_name: preprocess(image, input_cfg)})[0]
        mask_save, mask_index, probabilities = postprocess(
            logits,
            image.shape[:2],
            n_classes,
            mask_mode,
            threshold,
            img_interp,
            mask_interp,
        )
        relative = image_path.relative_to(input_root).with_suffix("")
        mask_path = out_dir / "masks" / relative.with_suffix(".png")
        overlay_path = out_dir / "overlays" / relative.with_suffix(".jpg")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_path), mask_save):
            raise OSError(f"分割 mask 保存失败：{mask_path}")
        if not cv2.imwrite(str(overlay_path), overlay_mask(image, mask_index, n_classes)):
            raise OSError(f"分割叠加图保存失败：{overlay_path}")
        if args.save_prob_npy:
            prob_path = out_dir / "probs" / relative.with_suffix(".npy")
            prob_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(prob_path), probabilities)
        success += 1

    if success == 0:
        raise RuntimeError("没有成功完成任何图片的分割推理")
    LOGGER.info("分割推理完成：成功=%d，总数=%d，输出=%s", success, len(images), out_dir)


if __name__ == "__main__":
    main()
