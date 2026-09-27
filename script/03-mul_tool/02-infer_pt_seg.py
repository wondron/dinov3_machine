from __future__ import annotations

import argparse
import logging
import shutil, math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dino_finetune import DINOEncoderLoRA
from dino_finetune.config import default_config_path, get_dino_paths, load_config, resolve_interp
from dino_finetune.data import SegTransforms
from dino_finetune.model.dino_multitask import DINOEncoderLoRA_MultiTask
from dino_finetune.utils.ckpt_cls import build_encoder


LOGGER = logging.getLogger("多任务PT分割推理")
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


def normalize_optional_string_list(
    value: Any,
) -> list[str] | None:
    """
    转换为 list[str] | None。

    示例：
        None        -> None

        [10]        -> ["10"]
        [10, 20]    -> ["10", "20"]
        ["10"]      -> ["10"]
        ["大"]      -> ["大"]

        10          -> ["10"]
        "10"        -> ["10"]

        []          -> []
    """

    # null 是合法值
    if value is None:
        return None

    # ========================================================
    # list
    # ========================================================

    if isinstance(value, list):
        result: list[str] = []

        for item in value:

            if item is None:
                continue

            if isinstance(item, bool):
                continue

            # 字符串
            if isinstance(item, str):
                text = item.strip()

                if text:
                    result.append(text)

                continue

            # int
            if isinstance(item, int):
                result.append(str(item))
                continue

            # float
            if isinstance(item, float):

                if not math.isfinite(item):
                    continue

                # 10.0 -> "10"
                if item.is_integer():
                    result.append(str(int(item)))
                else:
                    result.append(str(item))

        return result

    # ========================================================
    # 单个字符串
    # ========================================================

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return None

        return [text]

    # ========================================================
    # 单个 int
    # ========================================================

    if isinstance(value, int) and not isinstance(value, bool):
        return [str(value)]

    # ========================================================
    # 单个 float
    # ========================================================

    if isinstance(value, float):

        if not math.isfinite(value):
            return None

        if value.is_integer():
            return [str(int(value))]

        return [str(value)]

    return None


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


def pick_model_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(k, str) for k in checkpoint):
            return checkpoint
    raise TypeError("checkpoint 格式错误：未找到模型权重字典")


def infer_num_classes_cls(cfg: dict, checkpoint: Any, state_dict: dict[str, torch.Tensor]) -> int:
    if isinstance(checkpoint, dict) and int(checkpoint.get("num_classes_cls", 0) or 0) > 0:
        return int(checkpoint["num_classes_cls"])
    weight = state_dict.get("cls_head.weight")
    if hasattr(weight, "shape") and len(weight.shape) == 2:
        return int(weight.shape[0])
    value = int((cfg.get("model_cls", {}) or {}).get("num_classes", 0) or 0)
    if value <= 0:
        raise ValueError("无法确定分类类别数：checkpoint 缺少 cls_head.weight 和 num_classes_cls")
    return value


def build_multitask_model(
    cfg: dict,
    checkpoint_path: Path,
    device: torch.device,
) -> DINOEncoderLoRA_MultiTask:
    dino_local_repo, weight_path = get_dino_paths(cfg)
    encoder = build_encoder(cfg, device, dino_local_repo=dino_local_repo, weight_path=weight_path)
    emb_dim = int(getattr(encoder, "num_features", 0) or 0)
    if emb_dim <= 0:
        raise RuntimeError("无法从 encoder 获取 num_features")

    train_cfg = cfg["trainparams"]
    model_cfg = cfg["model"]
    seg_model = DINOEncoderLoRA(
        encoder=encoder,
        r=int(train_cfg["rank_r"]),
        emb_dim=emb_dim,
        img_dim=tuple(cfg["input"]["img_dim"]),
        n_classes=int(model_cfg["n_classes"]),
        use_lora=bool(train_cfg["use_lora"]),
        use_fpn=bool(train_cfg["use_fpn"]),
    ).to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = pick_model_state_dict(checkpoint)
    model = DINOEncoderLoRA_MultiTask(
        seg_model=seg_model,
        num_classes_cls=infer_num_classes_cls(cfg, checkpoint, state_dict),
        pool=str(cfg["model_cls"]["pool"]),
        emb_dim=emb_dim,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    LOGGER.info("多任务权重加载完成：缺失参数=%d，多余参数=%d", len(missing), len(unexpected))
    model.eval()
    return model


def resize_probabilities(probs: np.ndarray, size_wh: tuple[int, int], interpolation: int) -> np.ndarray:
    return np.stack(
        [cv2.resize(channel, size_wh, interpolation=interpolation) for channel in probs],
        axis=0,
    )


def postprocess(
    logits: torch.Tensor,
    original_hw: tuple[int, int],
    n_classes: int,
    mask_mode: str,
    threshold: float,
    img_interp: int,
    mask_interp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != n_classes:
        raise ValueError(f"分割输出形状错误：期望 (1,{n_classes},H,W)，实际 {tuple(logits.shape)}")
    probs = torch.softmax(logits, dim=1)[0].float().cpu().numpy()
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


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境不可用")
    return torch.device(name)


def main() -> None:
    date_tag = datetime.now().strftime("%y%m%d")
    parser = argparse.ArgumentParser(description="多任务 PT 模型分割推理（支持单文件和文件夹）")
    parser.add_argument("--config", default=default_config_path("mul"), help="多任务 YAML 配置文件")
    parser.add_argument("--ckpt", required=True, help="多任务 PT checkpoint")
    parser.add_argument("--input", required=True, help="单张图片或图片文件夹")
    parser.add_argument(
        "--out_dir",
        default=f"z_infer_res/1_pt_out/3_multi/1_segment/{date_tag}",
        help="输出目录",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="推理设备")
    parser.add_argument("--recursive", action="store_true", help="递归读取输入文件夹")
    parser.add_argument("--save_prob_npy", action="store_true", help="保存分割概率 npy")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config(args.config)
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"多任务 checkpoint 不存在：{checkpoint_path}")

    images, input_root = resolve_images(Path(args.input), args.recursive)
    out_dir = Path(args.out_dir).expanduser().resolve()
    recreate_output_dir(out_dir)
    device = resolve_device(args.device)
    LOGGER.info("使用设备：%s；待推理图片：%d 张", device, len(images))

    input_cfg = cfg["input"]
    post_cfg = cfg["postprocess"]
    n_classes = int(cfg["model"]["n_classes"])
    mask_mode = str(post_cfg["mask_mode"]).strip().lower()
    threshold = float(post_cfg["thr"])
    img_interp = resolve_interp(str(input_cfg["img_interp"]))
    mask_interp = resolve_interp(str(input_cfg["mask_interp"]))
    transform = SegTransforms(
        resize_hw=tuple(input_cfg["img_dim"]),
        mean=tuple(input_cfg["mean"]),
        std=tuple(input_cfg["std"]),
        img_interp=img_interp,
        msk_interp=mask_interp,
    )
    model = build_multitask_model(cfg, checkpoint_path, device)

    success = 0
    for image_path in tqdm(images, desc="分割推理", unit="张"):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            tqdm.write(f"错误：图片读取失败，已跳过：{image_path}")
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model.forward_seg(tensor)
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
