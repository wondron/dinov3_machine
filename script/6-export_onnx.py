# script/6-export_onnx.py
"""
导出 ONNX 部署包到 <run>/onnx/：
  - 先把 LoRA 增量合并进 qkv 权重（网络结构与原始 ViT 相同，推理不增加计算），再导出；
  - 用真实图片逐项对比 PT 与 ONNX 的输出（含 batch>1），并比较完整后处理后的结果是否一致，写入 export_check.json；
  - 特征库、阈值、Device Profile、预处理参数一起放进部署包，版本号与权重绑定。

部署包内容：
  oven.onnx          输入 image [B,3,H,W] float32（RGB，/255 后按 mean/std 归一化；H、W 固定，batch 动态）
                     输出 is_oven_prob / food_prob [B]，container_prob [B,C]，accessory_prob [B,A]，
                     rack_raw [B,max_rack+1]（未掩码层位 logits，按检索到的型号掩码在后处理里做），proj [B,256]，cls [B,1024]
  gallery_*.npy + gallery.json    特征库（L2 归一化特征、对应型号、cavity_group）
  calibration.json / device_profile.json / meta.json / export_check.json

用法：
  python script/6-export_onnx.py --run output/oven/<run>
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 把项目根目录加进去

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dino_finetune.config import resolve_path
from dino_finetune.data import OvenTransforms, read_image_rgb
from dino_finetune.engine import load_run
from dino_finetune.inference import OvenPostprocessor, list_images, save_gallery_bundle
from dino_finetune.labels import load_split
from dino_finetune.logging import setup_logging
from dino_finetune.model.oven import OUTPUT_KEYS, inference_outputs

logger = logging.getLogger("export_onnx")

# PT 与 ONNX 输出的允许误差：概率 / logits 看最大绝对误差，特征看最小余弦相似度
TOLERANCE = {"prob": 1e-3, "logits": 1e-2, "cosine": 0.9999}


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = inference_outputs(self.model(image))
        return tuple(out[key] for key in OUTPUT_KEYS)


def sample_images(run, check_input: str | None, count: int) -> list[Path]:
    if check_input:
        return list_images(check_input)[:count]
    data_cfg = run.cfg["data"]
    roots = [resolve_path(root) for root in data_cfg["root"]]
    paths: list[Path] = []
    for split in (data_cfg["val_split"], data_cfg["test_split"], data_cfg["train_split"]):
        if split and len(paths) < count:
            labels, _ = load_split(roots, split, run.schema, run.profile, on_error="skip")
            paths += [Path(lb.image_path) for lb in labels[: count - len(paths)]]
    return paths


def compare(ref: dict[str, np.ndarray], got: dict[str, np.ndarray]) -> dict[str, dict[str, float | bool]]:
    result = {}
    for key in OUTPUT_KEYS:
        if key in ("proj", "cls"):
            a, b = ref[key], got[key]
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
            result[key] = {"min_cosine": float(cos.min()), "ok": bool(cos.min() >= TOLERANCE["cosine"])}
        else:
            diff = float(np.abs(ref[key] - got[key]).max())
            limit = TOLERANCE["logits" if key == "rack_raw" else "prob"]
            result[key] = {"max_abs_diff": diff, "ok": diff <= limit}
    return result


def export_onnx(model: nn.Module, dummy: torch.Tensor, out_path: Path, opset: int) -> None:
    """先导出到临时文件、检查通过后再替换，避免留下不完整的模型。"""
    with tempfile.NamedTemporaryFile(prefix=f".{out_path.stem}.", suffix=".tmp.onnx", dir=out_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.onnx.export(
            ExportWrapper(model).eval(),
            (dummy,),
            str(tmp_path),
            opset_version=opset,
            input_names=["image"],
            output_names=list(OUTPUT_KEYS),
            dynamic_axes={"image": {0: "B"}, **{key: {0: "B"} for key in OUTPUT_KEYS}},
            do_constant_folding=True,
        )
        import onnx

        onnx.checker.check_model(str(tmp_path))
        tmp_path.replace(out_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ONNX 导出或完整性检查失败：{out_path}，原因：{exc}") from exc
    logger.info("导出完成：%s（%.1f MB）", out_path, out_path.stat().st_size / 2**20)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 ONNX 部署包")
    parser.add_argument("--run", required=True, help="train.py 的输出目录")
    parser.add_argument("--ckpt", default="ckpt_best.pt")
    parser.add_argument("--out", default=None, help="部署包目录，默认 <run>/onnx")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--check_input", default=None, help="对齐检查用的图片或目录，默认取验证集 / 测试集图片")
    parser.add_argument("--check_images", type=int, default=8, help="对齐检查的图片数（一次按 batch 推理，同时检查动态 batch）")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    setup_logging(name="export_onnx", use_shanghai_time=True)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    run = load_run(args.run, device, ckpt_name=args.ckpt)
    out_dir = Path(args.out) if args.out else run.run_dir / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    inp = run.cfg["input"]
    model = run.model.float().eval()

    # =========================
    # 1) 合并 LoRA，确认输出不变
    # =========================
    transform = OvenTransforms(inp["img_dim"], inp["mean"], inp["std"], inp["img_interp"], is_train=False)
    paths = sample_images(run, args.check_input, args.check_images)
    x = torch.stack([transform(read_image_rgb(str(p))) for p in paths]).to(device)
    with torch.no_grad():
        ref = {k: v.cpu().numpy() for k, v in inference_outputs(model(x)).items()}
        merged_blocks = model.merge_lora()
        merged = {k: v.cpu().numpy() for k, v in inference_outputs(model(x)).items()}
    merge_check = compare(ref, merged)
    logger.info("LoRA 已合并进 %d 个 block 的 qkv，合并前后输出对比：%s", merged_blocks, merge_check)

    # =========================
    # 2) 导出并用 onnxruntime 对齐
    # =========================
    onnx_path = out_dir / "oven.onnx"
    export_onnx(model, x[:1], onnx_path, args.opset)

    import onnxruntime as ort

    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    got = dict(zip(OUTPUT_KEYS, session.run(None, {"image": x.cpu().numpy()})))
    onnx_check = compare(ref, got)

    post = OvenPostprocessor(
        calibration=run.calibration,
        profile=run.profile,
        container_classes=run.schema.container_classes,
        accessory_classes=run.schema.accessory_classes,
        galleries=run.galleries,
    )
    decisions = lambda r: {k: v for k, v in r.items() if not k.endswith("_score")}  # noqa: E731  只比较判定结果
    mismatched = [str(p) for p, a, b in zip(paths, post(ref), post(got)) if decisions(a) != decisions(b)]
    passed = all(v["ok"] for v in onnx_check.values()) and not mismatched
    check = {
        "passed": passed,
        "images": [str(p) for p in paths],
        "batch": len(paths),
        "providers": session.get_providers(),
        "tolerance": TOLERANCE,
        "lora_merge": merge_check,
        "onnx_vs_pt": onnx_check,
        "postprocess_mismatch": mismatched,
    }
    (out_dir / "export_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    log = logger.info if passed else logger.warning
    log("PT 与 ONNX 对齐%s：%s；后处理结果不一致 %d/%d 张", "通过" if passed else "未通过", onnx_check, len(mismatched), len(paths))

    # =========================
    # 3) 部署包其余文件
    # =========================
    save_gallery_bundle(run.galleries, out_dir, run.gallery_meta)
    for name in ("calibration.json", "device_profile.json"):
        shutil.copy2(run.run_dir / name, out_dir / name)
    meta = {
        "fingerprint": run.calibration["fingerprint"],
        "checkpoint": args.ckpt,
        "input": {
            "name": "image",
            "img_dim": inp["img_dim"],
            "mean": inp["mean"],
            "std": inp["std"],
            "img_interp": inp["img_interp"],
            "color": "RGB",
            "layout": "NCHW",
        },
        "outputs": list(OUTPUT_KEYS),
        "container_classes": run.schema.container_classes,
        "accessory_classes": run.schema.accessory_classes,
        "max_rack": run.cfg["model"]["max_rack"],
        "opset": args.opset,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("部署包：%s（%s）", out_dir, ", ".join(sorted(p.name for p in out_dir.iterdir())))


if __name__ == "__main__":
    main()
