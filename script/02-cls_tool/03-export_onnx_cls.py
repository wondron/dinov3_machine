# script/12-export_onnx_cls.py
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn

# 让 script 直接运行时也能 import 项目包
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from dino_finetune.config import load_config, default_config_path, get_dino_paths
from dino_finetune.logging import setup_logging
from dino_finetune.model.dino_cls import DINOForClassification
from dino_finetune.utils.ckpt_cls import build_encoder, pick_state_dict, auto_align_and_load
from dino_finetune.utils.label_mapping import normalize_label_mapping, save_label_mapping

logger = logging.getLogger(__name__)



class OnnxExportWrapper(nn.Module):
    """确保 forward 输出是 (logits, embedding)，并且都是 tensor。"""
    def __init__(self, model: DINOForClassification):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, emb = self.model(x)
        return logits, emb


def resolve_mapping_path(mapping_arg: str, ckpt_path: Path, ckpt: dict[str, Any]) -> Path:
    if mapping_arg:
        mapping_path = Path(mapping_arg).expanduser().resolve()
        if not mapping_path.is_file():
            raise FileNotFoundError(f"类别映射文件不存在：{mapping_path}")
        return mapping_path

    candidates: list[Path] = []
    saved_mapping_path = ckpt.get("mapping_path")
    if isinstance(saved_mapping_path, str) and saved_mapping_path.strip():
        saved_path = Path(saved_mapping_path).expanduser()
        candidates.append(saved_path)
        if not saved_path.is_absolute():
            candidates.append(ckpt_path.parent / saved_path)
        candidates.append(ckpt_path.parent / saved_path.name)
    candidates.append(ckpt_path.parent / "leaf_id_map.json")

    checked: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.add(candidate)
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"无法自动找到类别映射：请将 leaf_id_map.json 放在 checkpoint 同目录，或使用 --mapping 指定；"
        f"checkpoint={ckpt_path}"
    )


def load_mapping(mapping_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"类别映射读取失败：{mapping_path}，原因：{exc}") from exc
    return normalize_label_mapping(data)


def main() -> None:
    parser = argparse.ArgumentParser("export onnx for classification (logits+embedding)")
    parser.add_argument("--config", type=str, default=default_config_path("cls"), help="配置文件路径")
    parser.add_argument("--ckpt", type=str, required=True, help="ckpt_best.pt 或 ckpt_last.pt")
    parser.add_argument(
        "--mapping",
        type=str,
        default="",
        help="可选类别映射，默认从 checkpoint 记录或同目录 leaf_id_map.json 读取",
    )
    parser.add_argument("--out", type=str, default="output_cls.onnx", help="输出 onnx 路径")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--batch", type=int, default=1, help="导出 dummy batch")
    args = parser.parse_args()

    setup_logging(name=__name__, level=logging.INFO)
    cfg = load_config(args.config)
    dino_local_repo, weight_path = get_dino_paths(cfg)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    logger.info("使用设备：%s", device)

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"ckpt 不存在：{ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint 类型错误，期望 dict，实际为 {type(ckpt).__name__}")

    mapping_source_path = resolve_mapping_path(args.mapping, ckpt_path, ckpt)
    mapping = load_mapping(mapping_source_path)
    num_classes = int(mapping["num_classes"])
    logger.info("类别映射读取完成：%s；类别数=%d", mapping_source_path, num_classes)

    encoder = build_encoder(cfg, device, dino_local_repo=dino_local_repo, weight_path=weight_path)
    model = DINOForClassification.from_config(encoder=encoder, cfg=cfg, num_classes=num_classes).to(device)

    ckpt_sd = pick_state_dict(ckpt, extra_keys=["model_state_dict"])
    auto_align_and_load(model, ckpt_sd)

    model.eval()
    wrapper = OnnxExportWrapper(model).to(device).eval()

    # dummy input
    input_cfg = cfg["input_cls"]
    h, w = int(input_cfg["img_dim"][0]), int(input_cfg["img_dim"][1])
    dummy = torch.randn(int(args.batch), 3, h, w, device=device, dtype=torch.float32)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["images"]
    output_names = ["logits", "embedding"]

    dynamic_axes = {
        "images": {0: "batch"},
        "logits": {0: "batch"},
        "embedding": {0: "batch"},
    }

    logger.info("开始导出 ONNX：%s", str(out_path))
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=int(args.opset),
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    mapping_output_path = out_path.parent / "leaf_id_map.json"
    save_label_mapping(mapping, mapping_output_path)
    logger.info("✅ 导出完成：%s", str(out_path))
    logger.info("类别映射已保存：%s", mapping_output_path)
    logger.info("输出：logits=[B,C], embedding=[B,emb_dim]")


if __name__ == "__main__":
    main()
