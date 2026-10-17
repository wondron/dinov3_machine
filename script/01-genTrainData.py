"""整理多任务训练数据，按设备分层划分；外部“未知”设备仍属于内腔图片。

源标注不修改。输出仍兼容 annotations 1.x 格式，并补齐明确的训练标签：
未知设备保留食物/容器/附件，is_oven=true，rack_level=null；
已出现字段的 null 表示确认没有，缺失字段和“遮挡”保留为未标注。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


# ==================================================
# 配置
# ==================================================
# 原始数据路径
SRC_PATH = r"/data/wangzhuo/66-newdata/00-dataset/03-多属性/01-oridata"
# 输出路径
DST_PATH = r"/data/wangzhuo/66-newdata/00-dataset/03-多属性/02-traindata/01-dinov3"
# 图片格式
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
UNKNOWN_DEVICE_TOKENS = {"未知", "未知型号"}
NONE_TOKENS = {"", "无"}

# 数据比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 随机种子
SEED = 42

# 日期目录
DATE_STR = datetime.now().strftime("%Y%m%d")

# ==================================================
# 根据json寻找图片
# ==================================================
def find_image(json_path, raw=None):
    """优先使用标注中的文件名，再找同名图片，兼容大写扩展名。"""
    json_path = Path(json_path)
    raw = raw or {}
    ann = raw.get("annotations", raw)
    for fields in (raw, ann):
        for key in ("image", "image_name"):
            value = fields.get(key)
            if isinstance(value, str) and value.strip():
                path = Path(value.strip().replace("\\", "/"))
                # 先尊重有效的相对目录，再兼容另一台机器留下的绝对路径。
                for candidate in (json_path.parent / path, json_path.parent / path.name):
                    if candidate.is_file():
                        return candidate
    for ext in IMAGE_EXTS + [ext.upper() for ext in IMAGE_EXTS]:
        candidate = json_path.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


def _is_none(value):
    return value is None or (isinstance(value, str) and value.strip() in NONE_TOKENS)


def _normalize_names(value, field):
    if _is_none(value):
        return []
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list):
        raise ValueError(f"{field} 必须是字符串、列表或 null")
    names = []
    for name in values:
        if _is_none(name):
            continue
        if not isinstance(name, str):
            raise ValueError(f"{field} 中存在非字符串类别: {name!r}")
        name = name.strip()
        if name not in names:
            names.append(name)
    return names


def _normalize_rack(value):
    values = value if isinstance(value, list) else [value]
    values = [v for v in values if not _is_none(v)]
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"rack_level 只能有一个层位: {value!r}")
    level = values[0]
    if isinstance(level, bool):
        raise ValueError(f"rack_level 不能是布尔值: {value!r}")
    if isinstance(level, str) and level.strip().isdigit():
        level = int(level.strip())
    elif isinstance(level, float) and level.is_integer():
        level = int(level)
    if not isinstance(level, int) or level < 0:
        raise ValueError(f"rack_level 必须是非负整数或无: {value!r}")
    return level


def normalize_annotation(raw):
    """保留业务字段，只规范化训练所需标签；不修改传入对象。"""
    if not isinstance(raw, dict):
        raise ValueError("JSON 顶层必须是对象")
    result = copy.deepcopy(raw)
    if "annotations" in result and not isinstance(result["annotations"], dict):
        raise ValueError("annotations 必须是对象")
    ann = result.get("annotations", result)

    device = ann.get("device_model")
    if _is_none(device):
        device = None
    elif not isinstance(device, str):
        raise ValueError("device_model 必须是字符串或 null")
    else:
        device = device.strip()
    external = device in UNKNOWN_DEVICE_TOKENS
    if external:
        device = "未知"
    ann["device_model"] = device

    is_oven = ann.get("is_oven")
    if is_oven is None:
        is_oven = device is not None
    if not isinstance(is_oven, bool):
        raise ValueError("is_oven 必须是 true / false")
    if external and not is_oven:
        raise ValueError("“未知”已约定为外部设备内腔，不能标为 is_oven=false")
    if not is_oven and device is not None:
        raise ValueError("is_oven=false 时不能有设备型号")
    if is_oven and device is None:
        raise ValueError("is_oven=true 时需要明确设备型号或“未知”")
    ann["is_oven"] = is_oven

    if "food_exist" in ann:
        if ann["food_exist"] is None:
            ann["food_exist"] = False
        elif not isinstance(ann["food_exist"], bool):
            raise ValueError("food_exist 必须是 true / false / null")
    elif "food_name" in ann:
        food_names = _normalize_names(ann["food_name"], "food_name")
        if "遮挡" not in food_names:
            ann["food_exist"] = bool(food_names)

    for field in ("container_type", "accessory_type"):
        if field in ann:
            ann[field] = _normalize_names(ann[field], field)

    # 无层位不等于底板层 0；外部设备不套用公司层位定义。
    if external or not is_oven:
        ann["rack_level"] = None
    elif "rack_level" in ann:
        ann["rack_level"] = _normalize_rack(ann["rack_level"])
    return result

# ==================================================
# 递归扫描数据
# ==================================================

def collect_pairs(src_path):
    src_path = Path(src_path)
    json_files = sorted(src_path.rglob("*.json"))
    tqdm.write(f"发现标注文件: {len(json_files)}")
    pairs = []
    for json_file in tqdm(json_files, desc="扫描json", unit="file"):
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8-sig"))
            normalized = normalize_annotation(raw)
            image_file = find_image(json_file, raw)
            if image_file is None:
                raise ValueError("找不到图片")
        except (OSError, UnicodeError, ValueError) as exc:
            tqdm.write(f"[跳过] {json_file}: {exc}")
            continue
        pairs.append({"json": json_file, "image": image_file, "raw": normalized})

    # 扁平化后同名文件不能互相覆盖，跨 split 也使用统一名称。
    stems = Counter(item["json"].stem.casefold() for item in pairs)
    used = set()
    for item in pairs:
        stem = item["json"].stem
        if stems[stem.casefold()] > 1:
            relative = item["json"].relative_to(src_path).as_posix()
            stem += "_" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
        base, index = stem, 1
        while stem.casefold() in used:
            stem = f"{base}_{index}"
            index += 1
        used.add(stem.casefold())
        item["output_stem"] = stem

    tqdm.write(f"有效样本数量: {len(pairs)}")
    return pairs


# ==================================================
# 划分 train val test
# ==================================================
def _split_counts(total):
    ratios = (TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    if any(not math.isfinite(r) or r < 0 for r in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("train / val / test 比例必须非负且总和为 1")
    counts = [int(total * ratio) for ratio in ratios]
    for index in sorted(range(3), key=lambda i: (total * ratios[i] - counts[i], -i), reverse=True)[:total - sum(counts)]:
        counts[index] += 1
    active = [i for i, ratio in enumerate(ratios) if ratio > 0]
    if total >= len(active):
        for index in active:
            if counts[index] == 0:
                donor = max(active, key=lambda i: counts[i])
                counts[donor] -= 1
                counts[index] += 1
    return counts


def split_dataset(data, seed=SEED):
    """按型号分层，同一图片的多份标注绑定划分；不打乱调用方列表。"""
    by_image = defaultdict(list)
    for item in data:
        by_image[Path(item["image"]).resolve()].append(item)
    groups = defaultdict(list)
    for items in by_image.values():
        # 即使同一图片被标成不同型号，也不能让它跨 train / val / test。
        models = set()
        for item in items:
            ann = item["raw"].get("annotations", item["raw"])
            models.add(ann["device_model"] or "无")
        groups[tuple(sorted(models))].append(items)
    rng = random.Random(seed)
    splits = ([], [], [])
    for name in sorted(groups):
        items = groups[name].copy()
        rng.shuffle(items)
        counts = _split_counts(len(items))
        start = 0
        for target, count in zip(splits, counts):
            for image_items in items[start:start + count]:
                target.extend(image_items)
            start += count
    for items in splits:
        rng.shuffle(items)
    return splits

# ==================================================
# 拷贝数据
# ==================================================
def copy_dataset(items, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in tqdm(items, desc=f"复制 {output_dir.name}", unit="sample"):
        image_path = item["image"]
        json_path = item["json"]
        stem = item["output_stem"]
        image_name = stem + image_path.suffix.lower()
        image_target = output_dir / image_name
        json_target = output_dir / (stem + ".json")
        if image_target.exists() or json_target.exists():
            raise FileExistsError(f"拒绝覆盖已有样本: {json_target}")
        raw = copy.deepcopy(item["raw"])
        raw["source_json"] = str(json_path.resolve())
        raw["source_image"] = str(image_path.resolve())
        raw["image"] = raw["image_name"] = image_name
        raw["image_root"] = str(output_dir.resolve())
        shutil.copy2(image_path, image_target)
        json_target.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def choose_output_root(dst_path, date_str):
    """同日重跑创建新目录，不删除已有训练集。"""
    datetime.strptime(date_str, "%Y%m%d")
    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError("日期必须为 YYYYMMDD")
    output = Path(dst_path) / date_str
    if output.exists():
        base = f"{date_str}_{datetime.now():%H%M%S}"
        output = Path(dst_path) / base
        index = 1
        while output.exists():
            output = Path(dst_path) / f"{base}_{index}"
            index += 1
    return output


def summarize(items):
    models = Counter(item["raw"].get("annotations", item["raw"])["device_model"] or "无" for item in items)
    return {"total": len(items), "device_model": dict(sorted(models.items())), "external_unknown": models["未知"]}

# ==================================================
# main
# ==================================================
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path(SRC_PATH), help="原始标注和图片目录")
    parser.add_argument("--dst", type=Path, default=Path(DST_PATH), help="输出根目录，下面创建日期目录")
    parser.add_argument("--date", default=DATE_STR, help="输出日期目录，YYYYMMDD")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    src_path = args.src.resolve()
    if not src_path.is_dir():
        raise FileNotFoundError(f"原始数据目录不存在: {src_path}")
    # 输出不能位于输入中，防止再次扫描时混入上次的生成结果。
    dst_path = args.dst.resolve()
    if dst_path == src_path or src_path in dst_path.parents:
        raise ValueError("输出根目录不能位于原始数据目录内")
    output_root = choose_output_root(dst_path, args.date)
    tqdm.write("开始扫描数据...")
    data = collect_pairs(src_path)
    if not data:
        raise RuntimeError("没有可用样本，未生成训练集")
    train_data, val_data, test_data = split_dataset(data, seed=args.seed)
    tqdm.write("\n数据划分:")
    tqdm.write(f"train : {len(train_data)}")
    tqdm.write(f"val   : {len(val_data)}")
    tqdm.write(f"test  : {len(test_data)}")

    tqdm.write(f"\n输出目录:")
    tqdm.write(str(output_root))

    copy_dataset(train_data, output_root / "train")
    copy_dataset(val_data, output_root / "val")
    copy_dataset(test_data, output_root / "test")
    summary = {name: summarize(items) for name, items in (("train", train_data), ("val", val_data), ("test", test_data))}
    (output_root / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, counts in summary.items():
        tqdm.write(f"{name}: 型号分布={counts['device_model']}，外部未知={counts['external_unknown']}")
    tqdm.write("\n全部完成!")


if __name__ == "__main__":
    main()
