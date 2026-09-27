# tools/infer_cls_top5.py
from __future__ import annotations
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 把项目根目录加进去

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from dino_finetune.config import load_config, default_config_path, resolve_interp, get_dino_paths
from dino_finetune.logging import setup_logging
from dino_finetune.model.dino_cls import DINOForClassification
from dino_finetune.utils.ckpt_cls import build_encoder

logger = logging.getLogger(__name__)



# -----------------------------
# 1) encoder 加载（与你训练一致）
# -----------------------------
def _preprocess_one(img_path: Path, input_cfg: Dict[str, Any]) -> torch.Tensor:
    if not img_path.is_file():
        raise FileNotFoundError(f"找不到图片：{img_path}")

    img_dim = input_cfg["img_dim"]  # [H, W]
    mean = np.array(input_cfg["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(input_cfg["std"], dtype=np.float32).reshape(1, 1, 3)
    interp = resolve_interp(str(input_cfg.get("img_interp", "linear")))

    # OpenCV 读 BGR -> RGB
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"图片读取失败：{img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w = int(img_dim[0]), int(img_dim[1])
    rgb = cv2.resize(rgb, (w, h), interpolation=interp).astype(np.float32) / 255.0
    rgb = (rgb - mean) / std
    chw = np.transpose(rgb, (2, 0, 1)).copy()
    t = torch.from_numpy(chw).float().unsqueeze(0)  # (1,3,H,W)
    return t


# -----------------------------
# 3) mapping 读取
# -----------------------------
def _load_mapping(mapping_path: Path) -> Dict[str, Any]:
    if not mapping_path.is_file():
        raise FileNotFoundError(f"找不到 leaf_id_map.json：{mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as f:
        mp = json.load(f)

    # 统一成 int key 的 dict 方便用
    idx_to_leaf_id = {int(k): int(v) for k, v in mp["idx_to_leaf_id"].items()}
    idx_to_class_name = {int(k): str(v) for k, v in mp["idx_to_class_name"].items()}
    num_classes = int(mp["num_classes"])
    return {
        "num_classes": num_classes,
        "idx_to_leaf_id": idx_to_leaf_id,
        "idx_to_class_name": idx_to_class_name,
        "raw": mp,
    }


def _infer_mapping_path_from_ckpt(ckpt_path: Path, ckpt: Dict[str, Any]) -> Path:
    # 优先使用 ckpt 内记录
    mp = ckpt.get("mapping_path", None)
    if isinstance(mp, str) and mp:
        cand = Path(mp)
        if cand.is_file():
            return cand

    # fallback：同目录 leaf_id_map.json
    cand2 = ckpt_path.parent / "leaf_id_map.json"
    if cand2.is_file():
        return cand2

    raise FileNotFoundError("无法找到 mapping：ckpt 内无 mapping_path，且同目录不存在 leaf_id_map.json")


# -----------------------------
# 4) 推理与 Top-K
# -----------------------------
@torch.no_grad()
def _predict_topk(
    model: torch.nn.Module,
    x: torch.Tensor,
    k: int,
) -> Tuple[List[int], List[float]]:
    logits, _ = model(x)
    probs = torch.softmax(logits, dim=1)  # (1,C)
    k = int(min(k, probs.shape[1]))
    topv, topi = torch.topk(probs, k=k, dim=1)
    idxs = topi[0].detach().cpu().tolist()
    vals = topv[0].detach().cpu().tolist()
    return idxs, vals


def _iter_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [p for p in path.rglob("*") if p.suffix.lower() in exts]
        files.sort()
        return files
    raise FileNotFoundError(f"输入路径不存在：{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="分类推理 Top-5（输出 leaf_id 与类名）")
    parser.add_argument("--config", type=str, default=default_config_path("cls"), help="配置文件路径")
    parser.add_argument("--ckpt", type=str, required=True, help="ckpt_best.pt 或 ckpt_last.pt 路径")
    parser.add_argument("--input", type=str, required=True, help="单张图片路径 或 图片目录")
    parser.add_argument("--topk", type=int, default=5, help="top-k，默认 5")
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda")
    parser.add_argument("--out", type=str, default="", help="输出目录（默认 ckpt 同目录）")
    args = parser.parse_args()

    setup_logging(name=__name__, level=logging.INFO)
    logging.getLogger(__name__).propagate = False

    cfg = load_config(args.config)
    dino_local_repo, weight_path = get_dino_paths(cfg)
    if "input_cls" not in cfg or cfg["input_cls"] is None:
        raise RuntimeError("config 中未配置 input_cls")
    input_cfg = cfg["input_cls"]

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("使用设备：%s", device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到 ckpt：{ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("ckpt 格式不符合预期：需要包含 model_state_dict")

    mapping_path = _infer_mapping_path_from_ckpt(ckpt_path, ckpt)
    mapping = _load_mapping(mapping_path)
    num_classes = int(mapping["num_classes"])
    logger.info("加载 mapping：%s | num_classes=%d", str(mapping_path), num_classes)

    # build model
    encoder = build_encoder(cfg, device, dino_local_repo=dino_local_repo, weight_path=weight_path)
    model = DINOForClassification.from_config(
        encoder=encoder,
        cfg=cfg,
        num_classes=num_classes,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    logger.info("已加载分类 ckpt：%s", str(ckpt_path))

    # output dir
    out_dir = Path(args.out) if args.out else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "preds_top5.jsonl"
    logger.info("输出文件：%s", str(out_jsonl))

    # infer
    input_path = Path(args.input)
    img_files = _iter_images(input_path)
    if not img_files:
        raise ValueError("未找到任何图片")

    idx_to_leaf_id = mapping["idx_to_leaf_id"]
    idx_to_class_name = mapping["idx_to_class_name"]
    topk = int(args.topk)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for p in img_files:
            x = _preprocess_one(p, input_cfg).to(device, non_blocking=True)
            idxs, probs = _predict_topk(model, x, k=topk)

            top_items = []
            for rank, (idx, prob) in enumerate(zip(idxs, probs), 1):
                leaf_id = int(idx_to_leaf_id.get(int(idx), -1))
                name = str(idx_to_class_name.get(int(idx), str(leaf_id)))
                top_items.append({
                    "rank": rank,
                    "class_idx": int(idx),
                    "leaf_id": leaf_id,
                    "class_name": name,
                    "prob": float(prob),
                })

            record = {
                "image_path": str(p),
                "topk": top_items,
            }

            # 控制台打印（中文）
            pretty = " | ".join(
                [f"#{it['rank']} {it['class_name']}({it['leaf_id']}) p={it['prob']:.4f}" for it in top_items]
            )
            logger.info("Top-%d: %s -> %s", topk, p.name, pretty)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("推理完成：共 %d 张图片", len(img_files))


if __name__ == "__main__":
    main()
