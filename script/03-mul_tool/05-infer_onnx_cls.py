from __future__ import annotations

import argparse
import json
import logging
import shutil
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

from dino_finetune.config import default_config_path, load_config, resolve_interp
from dino_finetune.data_cls import ClsTransforms


LOGGER = logging.getLogger("多任务ONNX分类推理")
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


def resolve_images(input_path: Path, recursive: bool) -> list[Path]:
    path = input_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"输入文件不是支持的图片格式：{path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{path}")
    iterator = path.rglob("*") if recursive else path.iterdir()
    images = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"输入目录中未找到图片：{path}")
    return images


def labels_from_data(data: Any) -> tuple[dict[int, str], dict[int, int]]:
    idx_to_name: dict[int, str] = {}
    idx_to_leaf_id: dict[int, int] = {}
    if not isinstance(data, dict):
        return idx_to_name, idx_to_leaf_id

    if isinstance(data.get("idx_to_class_name"), dict):
        idx_to_name.update({int(k): str(v) for k, v in data["idx_to_class_name"].items()})
    if isinstance(data.get("idx_to_leaf_id"), dict):
        idx_to_leaf_id.update({int(k): int(v) for k, v in data["idx_to_leaf_id"].items()})
    class_names = data.get("class_names")
    if isinstance(class_names, (list, tuple)):
        idx_to_name.update({index: str(name) for index, name in enumerate(class_names)})
    class_to_idx = data.get("class_to_idx")
    if isinstance(class_to_idx, dict):
        idx_to_name.update({int(index): str(name) for name, index in class_to_idx.items()})
    leaf_id_to_idx = data.get("leaf_id_to_idx")
    if isinstance(leaf_id_to_idx, dict):
        idx_to_leaf_id.update({int(index): int(leaf_id) for leaf_id, index in leaf_id_to_idx.items()})
    return idx_to_name, idx_to_leaf_id


def load_labels(mapping_path: Path | None) -> tuple[dict[int, str], dict[int, int]]:
    if mapping_path is None:
        return {}, {}
    if not mapping_path.is_file():
        raise FileNotFoundError(f"类别映射文件不存在：{mapping_path}")
    if mapping_path.suffix.lower() == ".json":
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
    else:
        data = torch.load(str(mapping_path), map_location="cpu")
    return labels_from_data(data)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-12)


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
    parser = argparse.ArgumentParser(description="多任务 ONNX 模型分类 Top-K 推理（支持单文件和文件夹）")
    parser.add_argument("--config", default=default_config_path("mul"), help="多任务 YAML 配置文件")
    parser.add_argument("--onnx", required=True, help="multitask_cls.onnx 路径")
    parser.add_argument("--input", required=True, help="单张图片或图片文件夹")
    parser.add_argument(
        "--out_dir",
        default=f"z_infer_res/2_onnx_out/3_multi/2_classify/{date_tag}",
        help="输出目录",
    )
    parser.add_argument(
        "--mapping",
        default="",
        help="可选 leaf_id_map.json 或多任务 PT checkpoint，默认读取 ONNX 同目录 leaf_id_map.json",
    )
    parser.add_argument("--topk", type=int, default=3, help="输出概率最高的 K 个类别（默认：3）")
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda", "tensorrt"), default="auto")
    parser.add_argument("--recursive", action="store_true", help="递归读取输入文件夹")
    parser.add_argument("--save_embedding", action="store_true", help="在 JSONL 中保存分类 embedding")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.topk <= 0:
        raise ValueError("topk 必须大于 0")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("未安装 onnxruntime，请先安装 onnxruntime 或 onnxruntime-gpu") from exc

    cfg = load_config(args.config)
    onnx_path = Path(args.onnx).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX 模型不存在：{onnx_path}")
    images = resolve_images(Path(args.input), args.recursive)
    out_dir = Path(args.out_dir).expanduser().resolve()
    recreate_output_dir(out_dir)
    output_path = out_dir / "preds_topk.jsonl"

    if args.mapping:
        mapping_path = Path(args.mapping).expanduser().resolve()
    else:
        mapping_path = onnx_path.parent / "leaf_id_map.json"
        LOGGER.info("自动使用类别映射：%s", mapping_path)
    idx_to_name, idx_to_leaf_id = load_labels(mapping_path)

    input_cfg = cfg["input_cls"]
    transform = ClsTransforms(
        resize_hw=tuple(input_cfg["img_dim"]),
        mean=tuple(input_cfg["mean"]),
        std=tuple(input_cfg["std"]),
        img_interp=resolve_interp(str(input_cfg["img_interp"])),
        is_train=False,
    )
    session = ort.InferenceSession(str(onnx_path), providers=select_providers(ort, args.provider))
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    logits_name = "cls_logits" if "cls_logits" in output_names else output_names[0]
    embedding_name = "cls_emb" if "cls_emb" in output_names else (output_names[1] if len(output_names) > 1 else None)
    requested_outputs = [logits_name]
    if args.save_embedding and embedding_name is not None:
        requested_outputs.append(embedding_name)
    LOGGER.info("ONNX providers：%s；输入=%s；输出=%s", session.get_providers(), input_name, requested_outputs)

    success = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for image_path in tqdm(images, desc="分类推理", unit="张"):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                tqdm.write(f"错误：图片读取失败，已跳过：{image_path}")
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb).unsqueeze(0).numpy().astype(np.float32)
            outputs = session.run(requested_outputs, {input_name: tensor})
            probabilities = softmax(outputs[0].astype(np.float32))[0]
            k = min(args.topk, int(probabilities.shape[0]))
            indices = np.argsort(-probabilities)[:k]
            top_items = []
            for rank, index in enumerate(indices, start=1):
                class_index = int(index)
                top_items.append(
                    {
                        "rank": rank,
                        "class_idx": class_index,
                        "leaf_id": int(idx_to_leaf_id.get(class_index, class_index)),
                        "class_name": idx_to_name.get(class_index, str(class_index)),
                        "prob": float(probabilities[class_index]),
                    }
                )
            record: dict[str, Any] = {"image_path": str(image_path), "topk": top_items}
            if args.save_embedding:
                if embedding_name is None:
                    tqdm.write(f"警告：ONNX 模型没有 embedding 输出，已跳过保存：{image_path.name}")
                else:
                    record["embedding"] = outputs[1][0].astype(np.float32).tolist()
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary = " | ".join(
                f"#{item['rank']} {item['class_name']}({item['leaf_id']}) p={item['prob']:.4f}"
                for item in top_items
            )
            tqdm.write(f"分类完成：{image_path.name} -> {summary}")
            success += 1

    if success == 0:
        raise RuntimeError("没有成功完成任何图片的分类推理")
    LOGGER.info("分类推理完成：成功=%d，总数=%d，结果=%s", success, len(images), output_path)


if __name__ == "__main__":
    main()
