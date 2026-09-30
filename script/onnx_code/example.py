"""单图 / 目录推理示例；复制整个 onnx_code 目录后也可直接执行。"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

if __package__:
    from .c_onnx_classify import OnnxClassifier
else:
    from c_onnx_classify import OnnxClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="独立 ONNX 一体机推理")
    parser.add_argument("--onnx_dir", required=True, help="6-export_onnx.py 导出的完整目录")
    parser.add_argument("--input", required=True, help="图片路径或图片目录")
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--device_model", default=None, help="已知型号，必须存在于 device_profile.json")
    parser.add_argument("--scores", action="store_true", help="附上各任务原始概率")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", default=None, help="可选：结果 JSON 保存路径")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    classifier = OnnxClassifier({"CLASS_CONFIG": {
        "onnx_dir": args.onnx_dir, "provider": args.provider, "device_id": args.device_id,
        "device_model": args.device_model, "with_scores": args.scores,
    }})
    classifier.warmup(args.warmup)
    source = Path(args.input)
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
        if not paths:
            raise ValueError(f"目录下没有图片：{source}")
    else:
        raise FileNotFoundError(f"输入不存在：{source}")
    results = [{"image": str(path), **classifier.detect(path)} for path in paths]
    content = json.dumps(results, ensure_ascii=False, indent=2)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        logging.info("完成 %d 张，结果已保存：%s", len(results), target)
    else:
        print(content)


if __name__ == "__main__":
    main()
