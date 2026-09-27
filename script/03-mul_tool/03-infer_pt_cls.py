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
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dino_finetune import DINOEncoderLoRA
from dino_finetune.config import default_config_path, get_dino_paths, load_config, resolve_interp
from dino_finetune.data_cls import ClsTransforms
from dino_finetune.model.dino_multitask import DINOEncoderLoRA_MultiTask
from dino_finetune.utils.ckpt_cls import build_encoder


LOGGER = logging.getLogger("多任务PT分类推理")
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


def pick_model_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(k, str) for k in checkpoint):
            return checkpoint
    raise TypeError("checkpoint 格式错误：未找到模型权重字典")


def infer_num_classes(cfg: dict, checkpoint: Any, state_dict: dict[str, torch.Tensor]) -> int:
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
    checkpoint: Any,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> DINOEncoderLoRA_MultiTask:
    dino_local_repo, weight_path = get_dino_paths(cfg)
    encoder = build_encoder(cfg, device, dino_local_repo=dino_local_repo, weight_path=weight_path)
    emb_dim = int(getattr(encoder, "num_features", 0) or 0)
    if emb_dim <= 0:
        raise RuntimeError("无法从 encoder 获取 num_features")

    train_cfg = cfg["trainparams"]
    seg_model = DINOEncoderLoRA(
        encoder=encoder,
        r=int(train_cfg["rank_r"]),
        emb_dim=emb_dim,
        img_dim=tuple(cfg["input"]["img_dim"]),
        n_classes=int(cfg["model"]["n_classes"]),
        use_lora=bool(train_cfg["use_lora"]),
        use_fpn=bool(train_cfg["use_fpn"]),
    ).to(device)
    model = DINOEncoderLoRA_MultiTask(
        seg_model=seg_model,
        num_classes_cls=infer_num_classes(cfg, checkpoint, state_dict),
        pool=str(cfg["model_cls"]["pool"]),
        emb_dim=emb_dim,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    LOGGER.info("多任务权重加载完成：缺失参数=%d，多余参数=%d", len(missing), len(unexpected))
    model.eval()
    return model


def labels_from_checkpoint(checkpoint: Any) -> tuple[dict[int, str], dict[int, int]]:
    idx_to_name: dict[int, str] = {}
    idx_to_leaf_id: dict[int, int] = {}
    if not isinstance(checkpoint, dict):
        return idx_to_name, idx_to_leaf_id

    class_names = checkpoint.get("class_names")
    if isinstance(class_names, (list, tuple)):
        idx_to_name.update({index: str(name) for index, name in enumerate(class_names)})
    class_to_idx = checkpoint.get("class_to_idx")
    if isinstance(class_to_idx, dict):
        idx_to_name.update({int(index): str(name) for name, index in class_to_idx.items()})
    leaf_id_to_idx = checkpoint.get("leaf_id_to_idx")
    if isinstance(leaf_id_to_idx, dict):
        idx_to_leaf_id.update({int(index): int(leaf_id) for leaf_id, index in leaf_id_to_idx.items()})
    return idx_to_name, idx_to_leaf_id


def update_labels_from_json(
    mapping_path: Path,
    idx_to_name: dict[int, str],
    idx_to_leaf_id: dict[int, int],
) -> None:
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    if isinstance(data.get("idx_to_class_name"), dict):
        idx_to_name.update({int(k): str(v) for k, v in data["idx_to_class_name"].items()})
    if isinstance(data.get("idx_to_leaf_id"), dict):
        idx_to_leaf_id.update({int(k): int(v) for k, v in data["idx_to_leaf_id"].items()})


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境不可用")
    return torch.device(name)


def main() -> None:
    date_tag = datetime.now().strftime("%y%m%d")
    parser = argparse.ArgumentParser(description="多任务 PT 模型分类 Top-K 推理（支持单文件和文件夹）")
    parser.add_argument("--config", default=default_config_path("mul"), help="多任务 YAML 配置文件")
    parser.add_argument("--ckpt", required=True, help="多任务 PT checkpoint")
    parser.add_argument("--input", required=True, help="单张图片或图片文件夹")
    parser.add_argument(
        "--out_dir",
        default=f"z_infer_res/1_pt_out/3_multi/2_classify/{date_tag}",
        help="输出目录",
    )
    parser.add_argument("--mapping", default="", help="可选 leaf_id_map.json，用于覆盖类别映射")
    parser.add_argument("--topk", type=int, default=3, help="输出概率最高的 K 个类别（默认：3）")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="推理设备")
    parser.add_argument("--recursive", action="store_true", help="递归读取输入文件夹")
    parser.add_argument("--save_embedding", action="store_true", help="在 JSONL 中保存分类 embedding")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.topk <= 0:
        raise ValueError("topk 必须大于 0")
    cfg = load_config(args.config)
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"多任务 checkpoint 不存在：{checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = pick_model_state_dict(checkpoint)
    idx_to_name, idx_to_leaf_id = labels_from_checkpoint(checkpoint)
    if args.mapping:
        mapping_path = Path(args.mapping).expanduser().resolve()
        if not mapping_path.is_file():
            raise FileNotFoundError(f"类别映射文件不存在：{mapping_path}")
        update_labels_from_json(mapping_path, idx_to_name, idx_to_leaf_id)

    images = resolve_images(Path(args.input), args.recursive)
    out_dir = Path(args.out_dir).expanduser().resolve()
    recreate_output_dir(out_dir)
    output_path = out_dir / "preds_topk.jsonl"
    device = resolve_device(args.device)
    LOGGER.info("使用设备：%s；待推理图片：%d 张", device, len(images))

    model = build_multitask_model(cfg, checkpoint, state_dict, device)
    input_cfg = cfg["input_cls"]
    transform = ClsTransforms(
        resize_hw=tuple(input_cfg["img_dim"]),
        mean=tuple(input_cfg["mean"]),
        std=tuple(input_cfg["std"]),
        img_interp=resolve_interp(str(input_cfg["img_interp"])),
        is_train=False,
    )

    success = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for image_path in tqdm(images, desc="分类推理", unit="张"):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                tqdm.write(f"错误：图片读取失败，已跳过：{image_path}")
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits, embedding = model.forward_cls(tensor)
                probabilities = torch.softmax(logits, dim=1)
                k = min(args.topk, int(probabilities.shape[1]))
                values, indices = torch.topk(probabilities, k=k, dim=1)

            top_items = []
            for rank, (index, probability) in enumerate(
                zip(indices[0].cpu().tolist(), values[0].cpu().tolist()), start=1
            ):
                class_index = int(index)
                top_items.append(
                    {
                        "rank": rank,
                        "class_idx": class_index,
                        "leaf_id": int(idx_to_leaf_id.get(class_index, class_index)),
                        "class_name": idx_to_name.get(class_index, str(class_index)),
                        "prob": float(probability),
                    }
                )
            record: dict[str, Any] = {"image_path": str(image_path), "topk": top_items}
            if args.save_embedding:
                record["embedding"] = embedding[0].float().cpu().tolist()
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
