# script/14-compare_pt_onnx_cls.py
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from dino_finetune.config import load_config, default_config_path, get_dino_paths
from dino_finetune.data_cls import get_cls_dataloader
from dino_finetune.logging import setup_logging
from dino_finetune.model.dino_cls import DINOForClassification
from dino_finetune.utils.ckpt_cls import build_encoder, pick_state_dict, auto_align_and_load

logger = logging.getLogger(__name__)



def _cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # a,b: (B,D)
    an = np.linalg.norm(a, axis=1) + eps
    bn = np.linalg.norm(b, axis=1) + eps
    return (a * b).sum(axis=1) / (an * bn)


def main() -> None:
    parser = argparse.ArgumentParser("compare pytorch vs onnx (classification)")
    parser.add_argument("--config", type=str, default=default_config_path("cls"))
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--mapping", type=str, default="", help="leaf_id_map.json（可选，用于覆盖 num_classes）")
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--provider", type=str, default="auto", help="auto/cpu/cuda")
    args = parser.parse_args()

    setup_logging(name=__name__, level=logging.INFO)
    cfg = load_config(args.config)
    dino_local_repo, weight_path = get_dino_paths(cfg)

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

    # onnxruntime
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError("未安装 onnxruntime，请先 pip install onnxruntime 或 onnxruntime-gpu") from e

    providers = ort.get_available_providers()
    if args.provider == "cpu":
        use_providers = ["CPUExecutionProvider"]
    elif args.provider == "cuda":
        use_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        use_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(args.onnx), providers=use_providers)
    in_name = sess.get_inputs()[0].name

    # build PT model
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    logger.info("PT 使用设备：%s", device)

    num_classes = None
    if args.mapping:
        mp = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
        num_classes = int(mp["num_classes"])
        logger.info("num_classes 使用 mapping：%d", num_classes)

    encoder = build_encoder(cfg, device, dino_local_repo=dino_local_repo, weight_path=weight_path)
    model = DINOForClassification.from_config(encoder=encoder, cfg=cfg, num_classes=num_classes).to(device).eval()

    ckpt = torch.load(str(args.ckpt), map_location="cpu")
    ckpt_sd = pick_state_dict(ckpt, extra_keys=["model_state_dict"])
    auto_align_and_load(model, ckpt_sd)

    worst = {"logits_max": -1.0, "emb_max": -1.0, "cos_min": 2.0, "batch": -1}

    max_batches = int(args.max_batches)
    for bi, (images, _, _) in enumerate(dl):
        if max_batches > 0 and bi >= max_batches:
            break

        x_pt = images.to(device, non_blocking=True).float()
        x_np = images.numpy().astype(np.float32)

        with torch.no_grad():
            pt_logits, pt_emb = model(x_pt)
        pt_logits_np = pt_logits.detach().cpu().numpy()
        pt_emb_np = pt_emb.detach().cpu().numpy()

        onnx_logits_np, onnx_emb_np = sess.run(None, {in_name: x_np})

        logits_diff = np.abs(pt_logits_np - onnx_logits_np)
        emb_diff = np.abs(pt_emb_np - onnx_emb_np)

        logits_max = float(logits_diff.max())
        logits_mean = float(logits_diff.mean())
        emb_max = float(emb_diff.max())
        emb_mean = float(emb_diff.mean())

        cos = _cosine_sim(pt_emb_np, onnx_emb_np)
        cos_min = float(cos.min())
        cos_mean = float(cos.mean())

        logger.info(
            "batch=%d | logits: max=%.6g mean=%.6g | emb: max=%.6g mean=%.6g | cos(min/mean)=%.6f/%.6f",
            bi, logits_max, logits_mean, emb_max, emb_mean, cos_min, cos_mean
        )

        if logits_max > worst["logits_max"] or emb_max > worst["emb_max"] or cos_min < worst["cos_min"]:
            worst = {"logits_max": logits_max, "emb_max": emb_max, "cos_min": cos_min, "batch": bi}

    logger.info("✅ 一致性验证完成：worst_batch=%d logits_max=%.6g emb_max=%.6g cos_min=%.6f",
                worst["batch"], worst["logits_max"], worst["emb_max"], worst["cos_min"])
    logger.info("若差异偏大：优先检查 opset、是否有 fp16、是否有不同的归一化/预处理、onnxruntime provider。")


if __name__ == "__main__":
    main()
