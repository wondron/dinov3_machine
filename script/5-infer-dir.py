# script/5-infer-dir.py

from __future__ import annotations

import sys
import json, shutil
from pathlib import Path
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dino_finetune.engine import load_run
from dino_finetune.inference import (
    OvenPredictor,
    list_images,
    predict_files,
)


# =========================
# 配置
# =========================
RUN_DIR = "output/oven/260929"
INPUT_PATH = "/data/wangzhuo/01-code/02-project/dinov3_machine/data/0904/test/"

# 可视化保存目录
VIS_DIR = None
CKPT_NAME = "ckpt_best.pt"
DEVICE = "cuda"
DEVICE_MODEL = None
WITH_SCORES = False

DISPLAY_NAME = {
    "is_oven": "存在食材",
    "device_model": "设备型号",
    "container_type": "容器",
    "accessory_type": "附件",
    "rack_level": "层位",
}

ignore_keys = ["rack_level_score", "device_score", "food_exist", "low_confidence"]

def format_key(k):
    return DISPLAY_NAME.get(k, k)

# =========================
# path
# =========================
def project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path ).resolve()

def get_input_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path], input_path.parent

    if input_path.is_dir():
        return (list_images(input_path), input_path)

    raise FileNotFoundError(input_path)


def format_value(value,) -> str:
    if value is None:
        return "无"

    if isinstance(value, list):
        if len(value) == 0:
            return "无"
        return ",".join(str(x) for x in value)
    return str(value)


# =========================
# json标注
# =========================
def load_annotation(img_path):
    """
    根据图片找同名json
    xxx.jpg
    xxx.json
    """
    json_path = (img_path.with_suffix(".json"))
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8" ) as f:
            data = json.load(f)
        return data.get("annotations", {})
    except Exception:
        return None

# =========================
# 可视化
# =========================
def get_font(size=32):
    fonts = ["/data/wangzhuo/01-code/02-project/dinov3_machine/configs/fonts/SourceHanSansCN-VF.ttf"]
    for f in fonts:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def draw_result(
    image_path: Path,
    result: dict,
    save_path: Path,
    error_keys: set[str] | None = None,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = get_font(28)

    if error_keys is None:
        error_keys = set()

    lines = []
    colors = []

    for key, value in result.items():
        if key in ("image", "image_path") or key in ignore_keys:
            continue

        name = DISPLAY_NAME.get(key, key)
        lines.append(f"{name}: {format_value(value)}")

        if key in error_keys:
            colors.append((255, 80, 80))
        else:
            colors.append((255, 255, 255))

    text = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=8,)
    padding = 15
    box = (0, 0, bbox[2] + padding * 2, bbox[3] + padding * 2,)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0),)
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(box, fill=(0, 0, 0, 160),)
    img = Image.alpha_composite(img.convert("RGBA"), overlay,)
    draw = ImageDraw.Draw(img)
    y = padding
    for line, color in zip(lines, colors):
        draw.text((padding, y), line, font=font, fill=color,)
        y += font.size + 8

    save_path.parent.mkdir(parents=True, exist_ok=True,)
    img.convert("RGB").save(save_path, quality=95,)


# =========================
# 评估
# =========================
def normalize_value(v):
    if v is None:
        return None
    if isinstance(v,list):
        return sorted([str(x) for x in v])
    return str(v)

def evaluate_results(run, results):
    metrics = {
        "device_model": [0, 0],
        "container_type": [0, 0],
        "accessory_type": [0, 0],
        "rack_level": [0, 0],
    }

    if VIS_DIR:
        vis_dir = project_path(VIS_DIR)
    else:
        vis_dir = run.run_dir / "vis" / run.input_dir_name

    shutil.rmtree(vis_dir, ignore_errors=True)
    true_dir = vis_dir / "true"
    true_dir.mkdir(parents=True, exist_ok=True)
    false_dir = vis_dir / "false"
    false_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(f"开始评估，共 {len(results)} 张图片")
    for item in tqdm(results, desc="Evaluating", ncols=100):
        img = Path(item["image"])
        gt = load_annotation(img)
        if gt is None:
            tqdm.write(f"跳过无标注文件: {img}")
            continue

        err_num = 0
        error_keys = set()
        error_logs = []
        for key in metrics:
            pred = normalize_value(item.get(key))
            label = normalize_value(gt.get(key))
            if key == "rack_level" and isinstance(label, list) and len(label) == 1:
                label = label[0]
            if key == "container_type" and label == ["无"]:
                label = []
                
            if key == "accessory_type" and label == ["无"]:
                label = []
            
            if key == 'container_type' and label == ["纸质容器"]:
                label = ["油纸"]
            
            if key == 'container_type' and pred == ["纸质容器"]:
                pred = ["油纸"]
            
            metrics[key][1] += 1
            if pred == label:
                metrics[key][0] += 1
            else:
                err_num += 1
                error_keys.add(key)
                error_logs.append(f"{key}: pred={pred}, gt={label}")

        if err_num > 0:
            draw_result(img, item, false_dir / img.name, error_keys)
            tqdm.write(f"错误: {img.name} | {'; '.join(error_logs)}")
        else:
            draw_result(img, item, true_dir / img.name)

    result_metrics = {k: (v[0], v[1], round(v[0] / v[1], 4) if v[1] else 0) for k, v in metrics.items()}
    tqdm.write(f"评估完成: {result_metrics}")
    return result_metrics

# =========================
# main
# =========================
def main():
    input_path = project_path(INPUT_PATH)
    paths, input_dir = get_input_images(input_path)
    tqdm.write(f"输入: {input_path}")
    tqdm.write(f"图片数量: {len(paths)}")

    device = torch.device(DEVICE)
    run = load_run(project_path(RUN_DIR), device, ckpt_name=CKPT_NAME)
    run.input_dir_name = input_dir.name

    predictor = OvenPredictor(run, device)
    tqdm.write("模型加载完成")
    results=[]
    pred_iter=predict_files(paths, lambda images:predictor.predict(images, device_model=DEVICE_MODEL, with_scores=WITH_SCORES), batch_size=1)

    for item in tqdm(pred_iter, total=len(paths), desc="推理"):
        results.append(item)
        
    # =====================
    # 评估 + 可视化
    # =====================
    score=evaluate_results(run, results)
    tqdm.write("========== Accuracy ==========")

    for k,v in score.items():
        tqdm.write(f"{k}: {v[0]}/{v[1]} acc={v[2]}")

    tqdm.write("==============================")

if __name__=="__main__":
    main()