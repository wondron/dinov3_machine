# script/5-infer.py
"""
PT 推理（设计文档第 3 节的完整流程）：加载 train.py 的输出目录（权重、特征库、阈值、Device Profile 快照，并核对特征库版本），
对单张图片或整个目录推理，结果按设计文档的输出格式保存为 JSON。

用法：
  python script/5-infer.py --run output/oven/<run> --input data/model/0904/test
  python script/5-infer.py --run output/oven/<run> --input a.jpg --device_model C87-i7Pro   # 型号已知时跳过检索
  python script/5-infer.py --run output/oven/<run> --input <目录> --pending_dir <待补库池目录> --scores
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 把项目根目录加进去

import argparse
import logging
from pathlib import Path

import torch

from dino_finetune.engine import load_run
from dino_finetune.inference import OvenPredictor, list_images, predict_files, save_results, summarize_results
from dino_finetune.logging import setup_logging

logger = logging.getLogger("infer")


def main() -> None:
    parser = argparse.ArgumentParser(description="一体机多任务模型 PT 推理")
    parser.add_argument("--run", required=True, help="train.py 的输出目录")
    parser.add_argument("--input", required=True, help="单张图片或图片目录（递归查找）")
    parser.add_argument("--out", default=None, help="结果 JSON，默认 <run>/predictions/<输入名>.json")
    parser.add_argument("--ckpt", default="ckpt_best.pt", help="输出目录中的权重文件名")
    parser.add_argument("--device_model", default=None, help="已知设备型号时直接使用，跳过特征库检索")
    parser.add_argument("--pending_dir", default=None, help="把判为未知型号的图片复制到这个目录（待补库池）")
    parser.add_argument("--scores", action="store_true", help="结果里附上各头原始概率")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    setup_logging(name="infer", use_shanghai_time=True)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    run = load_run(args.run, device, ckpt_name=args.ckpt)
    predictor = OvenPredictor(run, device)
    logger.info(
        "已加载 %s：检索特征=%s tau=%.4f 特征库=%d 张参考图",
        run.run_dir, run.calibration["gallery_feature"], run.calibration["tau"][run.calibration["gallery_feature"]],
        len(run.galleries.get(run.calibration["gallery_feature"], [])),
    )

    paths = list_images(args.input)
    results = list(
        predict_files(
            paths,
            lambda images: predictor.predict(images, device_model=args.device_model, with_scores=args.scores),
            args.batch_size,
        )
    )
    out = Path(args.out) if args.out else run.run_dir / "predictions" / f"{Path(args.input).stem}.json"
    pending = save_results(results, out, Path(args.pending_dir) if args.pending_dir else None)
    logger.info("完成 %d 张：%s", len(results), summarize_results(results))
    if pending:
        logger.info("%d 张判为未知型号%s", pending, f"，已复制到 {args.pending_dir}" if args.pending_dir else "（可用 --pending_dir 放入待补库池）")
    logger.info("结果：%s", out)


if __name__ == "__main__":
    main()
