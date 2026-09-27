# script/7-infer_onnx.py
"""
ONNX 推理：只依赖 6-export_onnx.py 生成的部署包，预处理和后处理（设计文档第 3 节）与 PT 推理完全一致。

用法：
  python script/7-infer_onnx.py --onnx_dir output/oven/<run>/onnx --input data/model/0904/test
  python script/7-infer_onnx.py --onnx_dir <部署包> --input a.jpg --device_model C87-i7Pro
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 把项目根目录加进去

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from dino_finetune.data import OvenTransforms
from dino_finetune.device import load_device_profile
from dino_finetune.inference import (
    OvenPostprocessor,
    list_images,
    load_gallery_bundle,
    predict_files,
    save_results,
    summarize_results,
)
from dino_finetune.logging import setup_logging

logger = logging.getLogger("infer_onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description="一体机多任务模型 ONNX 推理")
    parser.add_argument("--onnx_dir", required=True, help="6-export_onnx.py 生成的部署包目录")
    parser.add_argument("--input", required=True, help="单张图片或图片目录（递归查找）")
    parser.add_argument("--out", default=None, help="结果 JSON，默认 <onnx_dir>/predictions/<输入名>.json")
    parser.add_argument("--device_model", default=None, help="已知设备型号时直接使用，跳过特征库检索")
    parser.add_argument("--pending_dir", default=None, help="把判为未知型号的图片复制到这个目录（待补库池）")
    parser.add_argument("--scores", action="store_true", help="结果里附上各头原始概率")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--provider", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    setup_logging(name="infer_onnx", use_shanghai_time=True)

    import onnxruntime as ort

    bundle = Path(args.onnx_dir)
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    calibration = json.loads((bundle / "calibration.json").read_text(encoding="utf-8"))
    galleries, gallery_meta = load_gallery_bundle(bundle)
    versions = {meta["fingerprint"], calibration["fingerprint"], gallery_meta["fingerprint"]}
    if len(versions) != 1:
        raise RuntimeError(f"部署包内的模型 / 阈值 / 特征库版本不一致：{versions}")
    profile = load_device_profile(
        bundle / "device_profile.json", max_rack=meta["max_rack"], accessory_classes=meta["accessory_classes"]
    )
    post = OvenPostprocessor(
        calibration=calibration,
        profile=profile,
        container_classes=meta["container_classes"],
        accessory_classes=meta["accessory_classes"],
        galleries=galleries,
    )

    inp = meta["input"]
    transform = OvenTransforms(inp["img_dim"], inp["mean"], inp["std"], inp["img_interp"], is_train=False)
    wanted = {"auto": ("CUDAExecutionProvider", "CPUExecutionProvider"), "cuda": ("CUDAExecutionProvider",), "cpu": ("CPUExecutionProvider",)}[args.provider]
    providers = [p for p in wanted if p in ort.get_available_providers()]
    if not providers:
        raise RuntimeError(f"onnxruntime 没有可用的 {wanted}，当前可用：{ort.get_available_providers()}")
    session = ort.InferenceSession(str(bundle / "oven.onnx"), providers=providers)
    logger.info("已加载部署包 %s（版本 %s，%s）", bundle, meta["fingerprint"], session.get_providers()[0])

    def predict(images: list[np.ndarray]) -> list[dict]:
        x = np.stack([transform(img).numpy() for img in images])
        outputs = dict(zip(meta["outputs"], session.run(None, {inp["name"]: x})))
        return post(outputs, device_model=args.device_model, with_scores=args.scores)

    results = list(predict_files(list_images(args.input), predict, args.batch_size))
    out = Path(args.out) if args.out else bundle / "predictions" / f"{Path(args.input).stem}.json"
    pending = save_results(results, out, Path(args.pending_dir) if args.pending_dir else None)
    logger.info("完成 %d 张：%s", len(results), summarize_results(results))
    if pending:
        logger.info("%d 张判为未知型号%s", pending, f"，已复制到 {args.pending_dir}" if args.pending_dir else "（可用 --pending_dir 放入待补库池）")
    logger.info("结果：%s", out)


if __name__ == "__main__":
    main()
