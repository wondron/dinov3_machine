# script/13-infer_onnx_cls_top5.py
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from dino_finetune.config import load_config, default_config_path
from dino_finetune.data_cls import get_cls_dataloader
from dino_finetune.logging import setup_logging
from dino_finetune.utils.label_mapping import normalize_label_mapping

logger = logging.getLogger(__name__)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def _load_mapping(mapping_path: Path) -> Dict[str, Any]:
    if not mapping_path.is_file():
        raise FileNotFoundError(f"类别映射文件不存在：{mapping_path}")
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"类别映射读取失败：{mapping_path}，原因：{exc}") from exc
    return normalize_label_mapping(mapping)


def _resolve_mapping_path(onnx_path: Path, mapping_arg: str) -> Path:
    if mapping_arg:
        mapping_path = Path(mapping_arg).expanduser().resolve()
    else:
        mapping_path = onnx_path.parent / "leaf_id_map.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"类别映射文件不存在：{mapping_path}；请将 leaf_id_map.json 放在 ONNX 同目录，"
            "或使用 --mapping 指定"
        )
    return mapping_path


def main() -> None:
    parser = argparse.ArgumentParser("onnx inference top5 (classification)")
    parser.add_argument("--config", type=str, default=default_config_path("cls"), help="配置文件")
    parser.add_argument("--onnx", type=str, required=True, help="onnx 路径（logits+embedding 输出）")
    parser.add_argument("--mapping", type=str, default="", help="可选类别映射，默认读取 ONNX 同目录 leaf_id_map.json")
    parser.add_argument("--split", type=str, default="valid", help="train/valid/test（取决于你的数据）")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_batches", type=int, default=5, help="最多跑多少个 batch，0=全量")
    parser.add_argument("--provider", type=str, default="auto", help="auto/cpu/cuda")
    args = parser.parse_args()

    setup_logging(name=__name__, level=logging.INFO)
    cfg = load_config(args.config)
    onnx_path = Path(args.onnx).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX 模型不存在：{onnx_path}")
    mapping_path = _resolve_mapping_path(onnx_path, args.mapping)
    logger.info("使用类别映射：%s", mapping_path)

    # dataloader
    ds_cfg = cfg["dataset_cls"]
    input_cfg = cfg["input_cls"]
    dl = get_cls_dataloader(
        dataroot=str(ds_cfg["root"]),
        split=str(args.split),
        input_cfg=input_cfg,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        ds_cfg=ds_cfg,
        pin_memory=True,
    )

    mp = _load_mapping(mapping_path)
    idx_to_name = {int(k): str(v) for k, v in (mp.get("idx_to_class_name") or {}).items()}

    # onnxruntime
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError("未安装 onnxruntime，请先 pip install onnxruntime 或 onnxruntime-gpu") from e

    providers = ort.get_available_providers()
    logger.info("onnxruntime 可用 providers：%s", providers)

    if args.provider == "cpu":
        use_providers = ["CPUExecutionProvider"]
    elif args.provider == "cuda":
        use_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        # auto
        use_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(onnx_path), providers=use_providers)
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    logger.info("ONNX 输入=%s 输出=%s providers=%s", in_name, out_names, use_providers)

    topk = int(args.topk)
    max_batches = int(args.max_batches)

    for bi, (images, labels, metas) in enumerate(dl):
        if max_batches > 0 and bi >= max_batches:
            break

        x = images.numpy().astype(np.float32)  # (B,3,H,W)
        outs = sess.run(None, {in_name: x})
        # 约定：输出顺序 logits, embedding
        logits = outs[0]
        probs = _softmax(logits, axis=1)  # (B,C)

        for i in range(min(len(metas), 5)):  # 每个 batch 只展示前 5 个样本，避免刷屏
            p = probs[i]
            idxs = np.argsort(-p)[:topk]
            meta = metas[i]
            gt = int(labels[i].item())

            msg = [f"样本[{bi}:{i}] gt_idx={gt} gt_name={idx_to_name.get(gt, str(gt))}"]
            for r, k in enumerate(idxs, 1):
                msg.append(f"  top{r}: idx={int(k)} name={idx_to_name.get(int(k), str(int(k)))} prob={float(p[k]):.4f}")
            logger.info("\n".join(msg))

    logger.info("✅ 推理结束")


if __name__ == "__main__":
    main()
