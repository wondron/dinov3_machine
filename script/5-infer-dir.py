# script/5-infer-dir.py
"""
PT 目录推理：修改下方配置后直接运行，不使用命令行参数。

用法：
  python script/5-infer-dir.py

递归查找 INPUT_DIR 下的图片，固定 batch_size=1，输出格式与 5-infer.py 一致。
相对路径均以项目根目录为基准；图像预处理使用训练输出目录中的配置。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from dino_finetune.engine import load_run
from dino_finetune.inference import OvenPredictor, list_images, predict_files, save_results, summarize_results
from dino_finetune.logging import setup_logging


# =========================
# 配置：运行前按实际路径修改
# =========================
RUN_DIR = "output/oven/260929"       # train.py 的完整输出目录，需要完成阶段 2
INPUT_DIR = "/data/wangzhuo/66-newdata/00-dataset/03-多属性/02-traindata/01-dinov3/20260929/test/3afcac815e2fc464fe1640f20211d91f.jpg"         # 图片目录，递归包含子目录
OUTPUT_JSON = None                  # None：保存到 <RUN_DIR>/predictions/<输入目录名>.json
CKPT_NAME = "ckpt_best.pt"           # 权重必须与特征库、阈值的版本一致
DEVICE = "cuda"                     # auto / cpu / cuda
DEVICE_MODEL = None                 # None：自动检索；已知型号可填 "C87-i7Pro"
WITH_SCORES = False                 # 是否附上各头原始概率
PENDING_DIR = None                  # 可选：把判为未知型号的图片复制到该目录

logger = logging.getLogger("infer_dir")


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def main() -> None:
    setup_logging(name="infer_dir", use_shanghai_time=True)

    input_path = project_path(INPUT_DIR)

    if input_path.is_file():
        paths = [input_path]
        input_dir = input_path.parent
    elif input_path.is_dir():
        paths = list_images(input_path)
    else:
        raise FileNotFoundError(f"路径不存在：{input_path}")

    if DEVICE not in ("auto", "cpu", "cuda"):
        raise ValueError(f"DEVICE 必须是 auto / cpu / cuda，实际为 {DEVICE!r}")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else DEVICE
    device = torch.device(device_name)
    run = load_run(project_path(RUN_DIR), device, ckpt_name=CKPT_NAME)
    predictor = OvenPredictor(run, device)
    logger.info(
        "已加载 %s：检索特征=%s tau=%.4f 特征库=%d 张参考图",
        run.run_dir, run.calibration["gallery_feature"], run.calibration["tau"][run.calibration["gallery_feature"]],
        len(run.galleries.get(run.calibration["gallery_feature"], [])),
    )
    logger.info("输入目录：%s，共 %d 张图片，推理设备：%s", input_dir, len(paths), device)

    results = list(
        predict_files(
            paths,
            lambda images: predictor.predict(images, device_model=DEVICE_MODEL, with_scores=WITH_SCORES),
            batch_size=1,
        )
    )
    out = project_path(OUTPUT_JSON) if OUTPUT_JSON else run.run_dir / "predictions" / f"{input_dir.name}.json"
    pending_dir = project_path(PENDING_DIR) if PENDING_DIR else None
    pending = save_results(results, out, pending_dir)
    logger.info("完成 %d 张：%s", len(results), summarize_results(results))
    if pending:
        logger.info(
            "%d 张判为未知型号%s",
            pending,
            f"，已复制到 {pending_dir}" if pending_dir else "（可设置 PENDING_DIR 放入待补库池）",
        )
    logger.info("结果：%s", out)


if __name__ == "__main__":
    main()
