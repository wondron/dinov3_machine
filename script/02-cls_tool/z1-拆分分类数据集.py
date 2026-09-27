# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from pypinyin import lazy_pinyin
from tqdm import tqdm


# ============================================================
# 参数区
# ============================================================

# 原始数据目录：
# ROOT/
# ├── 类别1/
# │   ├── 001.jpg
# │   └── 002.jpg
# └── 类别2/
#     ├── 001.jpg
#     └── 002.jpg
ROOT = Path(r"F:\10-single_zfood_data\260725\original")
# 输出根目录
OUT_BASE_ROOT = Path(r"E:\01-code\dinov3_finetune\data/1-classify")

# 每天输出到一个日期目录，例如：260727
DATE_TAG = datetime.now().strftime("%y%m%d")
OUT_ROOT = OUT_BASE_ROOT / DATE_TAG

# 数据集切分比例
TRAIN_RATIO = 0.90
VALID_RATIO = 0.05
TEST_RATIO = 0.05

# 随机种子
SEED = 42

# False：复制文件
# True：移动文件
MOVE_FILES = False

# 移动文件会修改原始数据。
# 只有同时设置 MOVE_FILES=True 和 ALLOW_MOVE_FILES=True 才允许移动。
ALLOW_MOVE_FILES = False

# 是否递归读取类别目录中的图片
#
# False：
# 仅扫描 ROOT/类别名/*.jpg
#
# True：
# 递归扫描 ROOT/类别名/**/*.jpg
RECURSIVE = False

# 是否清空当天的输出目录
CLEAR_EXISTING = True


# 复制文件时是否保留原文件的修改时间等元数据
#
# False 通常更快，适合训练数据复制。
# True 使用 shutil.copy2。
PRESERVE_METADATA = False


# ============================================================
# 性能参数
# ============================================================

# 并行复制线程数
COPY_WORKERS = 16

# 每次最多向线程池提交多少个任务
# 防止一次性提交十几万或几十万个 Future 占用过多内存
COPY_BATCH_SIZE = 5000

# 最多在终端打印多少条复制失败信息
MAX_FAILURE_LOGS = 20


# ============================================================
# 数据检查参数
# ============================================================

# 图片数少于该值时输出警告
MIN_CLASS_IMAGES_WARNING = 20


# 支持的图片后缀
IMG_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".jfif",
}


SPLITS = (
    "train",
    "valid",
    "test",
)


# ============================================================
# 数据结构
# ============================================================

@dataclass(frozen=True)
class FileTask:
    """
    单个文件复制或移动任务。
    """

    task_id: int

    src: Path
    dst: Path

    raw_class: str
    safe_class: str

    split: str


# ============================================================
# tqdm 日志
# ============================================================

def log_info(message: str, *args) -> None:
    if args:
        message = message % args

    tqdm.write(
        f"{datetime.now().strftime('%H:%M:%S')} "
        f"INFO    | {message}"
    )


def log_warning(message: str, *args) -> None:
    if args:
        message = message % args

    tqdm.write(
        f"{datetime.now().strftime('%H:%M:%S')} "
        f"WARNING | {message}"
    )


# ============================================================
# 路径与配置检查
# ============================================================

def is_path_inside(
    path: Path,
    parent: Path,
) -> bool:
    """
    判断 path 是否位于 parent 内部。

    path 和 parent 都应当是 resolve 后的绝对路径。
    """
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_config(
    root: Path,
    out_root: Path,
) -> None:
    """
    检查配置和路径，避免误删除、错误切分等问题。
    """
    ratios = (
        TRAIN_RATIO,
        VALID_RATIO,
        TEST_RATIO,
    )

    if any(not math.isfinite(ratio) for ratio in ratios):
        raise ValueError(
            f"切分比例必须是有限数字，当前为：{ratios}"
        )

    if any(ratio < 0 for ratio in ratios):
        raise ValueError(
            f"切分比例不能为负数，当前为：{ratios}"
        )

    ratio_sum = sum(ratios)

    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            "TRAIN_RATIO、VALID_RATIO、TEST_RATIO "
            f"之和必须为 1，当前为：{ratio_sum}"
        )

    if not root.exists():
        raise FileNotFoundError(
            f"输入目录不存在：{root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"输入路径不是目录：{root}"
        )

    if COPY_WORKERS <= 0:
        raise ValueError(
            f"COPY_WORKERS 必须大于 0，当前为：{COPY_WORKERS}"
        )

    if COPY_BATCH_SIZE <= 0:
        raise ValueError(
            "COPY_BATCH_SIZE 必须大于 0，"
            f"当前为：{COPY_BATCH_SIZE}"
        )

    if MOVE_FILES and not ALLOW_MOVE_FILES:
        raise RuntimeError(
            "当前 MOVE_FILES=True，会移动并修改原始数据。\n"
            "确认需要移动后，请同时设置：\n"
            "ALLOW_MOVE_FILES = True"
        )

    root_resolved = root.resolve()
    out_resolved = out_root.resolve()

    if root_resolved == out_resolved:
        raise ValueError(
            "输入目录和输出目录不能相同："
            f"{root_resolved}"
        )

    # 输出目录位于输入目录中，会导致重复扫描或数据混乱
    if is_path_inside(out_resolved, root_resolved):
        raise ValueError(
            "输出目录不能位于输入目录内部。\n"
            f"输入目录：{root_resolved}\n"
            f"输出目录：{out_resolved}"
        )

    # 输入目录位于输出目录中，清空输出时会删除输入数据
    if is_path_inside(root_resolved, out_resolved):
        raise ValueError(
            "输入目录不能位于输出目录内部，"
            "否则清空输出目录时可能删除原始数据。\n"
            f"输入目录：{root_resolved}\n"
            f"输出目录：{out_resolved}"
        )

    # 防止误把 / 或 C:\ 等磁盘根目录作为输出目录
    if out_resolved == Path(out_resolved.anchor):
        raise ValueError(
            f"禁止将磁盘根目录作为输出目录：{out_resolved}"
        )

    # /data、/home 这一类过浅目录也不允许直接删除
    if len(out_resolved.parts) <= 2:
        raise ValueError(
            f"输出目录层级过浅，存在误删除风险：{out_resolved}"
        )

    if out_root.exists() and out_root.is_symlink():
        raise ValueError(
            f"输出目录不能是符号链接：{out_root}"
        )

    if out_root.exists() and not out_root.is_dir():
        raise NotADirectoryError(
            f"输出路径已经存在，但不是目录：{out_root}"
        )


def prepare_output_dir(
    out_root: Path,
) -> None:
    """
    清理并创建输出目录。
    """
    if out_root.exists():
        if CLEAR_EXISTING:
            log_info(
                "清空已有输出目录：%s",
                out_root,
            )

            shutil.rmtree(out_root)
        else:
            log_warning(
                "输出目录已存在且不会清空：%s",
                out_root,
            )

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 即使某个 split 没有图片，也预先创建目录
    for split in SPLITS:
        (out_root / split).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# 图片扫描
# ============================================================

def is_image_file(
    path: Path,
) -> bool:
    """
    判断路径是否为支持的图片文件。
    """
    return (
        path.is_file()
        and path.suffix.lower() in IMG_EXTS
    )


def collect_class_dirs(
    root: Path,
) -> list[Path]:
    """
    获取 ROOT 下的一级类别目录。
    """
    class_dirs = [
        path
        for path in root.iterdir()
        if path.is_dir()
    ]

    class_dirs.sort(
        key=lambda path: path.name,
    )

    if not class_dirs:
        raise ValueError(
            f"输入目录下没有找到类别目录：{root}"
        )

    return class_dirs


def collect_images(
    class_dir: Path,
    recursive: bool = False,
) -> list[Path]:
    """
    获取单个类别目录中的图片。
    """
    if recursive:
        paths = (
            path
            for path in class_dir.rglob("*")
            if is_image_file(path)
        )
    else:
        paths = (
            path
            for path in class_dir.iterdir()
            if is_image_file(path)
        )

    return sorted(
        paths,
        key=lambda path: path.as_posix(),
    )


# ============================================================
# 类别名称转换
# ============================================================

def to_pinyin_pascal(
    name: str,
) -> str:
    """
    将中文类别名称转换为拼音 PascalCase。

    示例：
        西红柿 -> XiHongShi
        土豆丝 -> TuDouSi
    """
    pinyin_parts = lazy_pinyin(
        name,
        errors="default",
    )

    safe_name = "".join(
        part[:1].upper() + part[1:]
        for part in pinyin_parts
        if part
    )

    # 只保留英文、数字、下划线和中划线
    safe_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "",
        safe_name,
    )

    safe_name = safe_name.strip("_-")

    if not safe_name:
        safe_name = "UnknownClass"

    # 防止类别名以数字开头
    if safe_name[0].isdigit():
        safe_name = f"Class_{safe_name}"

    return safe_name


def build_class_name_mapping(
    class_dirs: list[Path],
) -> tuple[dict[str, str], dict[str, str]]:
    """
    构建原始类别名和安全类别名的映射。

    当多个类别转换为相同拼音时，自动追加编号：
        XiHongShi
        XiHongShi_2
        XiHongShi_3
    """
    raw_to_safe: dict[str, str] = {}
    safe_to_raw: dict[str, str] = {}

    name_counts: dict[str, int] = {}

    for class_dir in class_dirs:
        raw_name = class_dir.name
        base_name = to_pinyin_pascal(raw_name)

        current_count = name_counts.get(
            base_name,
            0,
        ) + 1

        name_counts[base_name] = current_count

        if current_count == 1:
            safe_name = base_name
        else:
            safe_name = (
                f"{base_name}_{current_count}"
            )

        raw_to_safe[raw_name] = safe_name
        safe_to_raw[safe_name] = raw_name

    return raw_to_safe, safe_to_raw


# ============================================================
# 稳定随机切分
# ============================================================

def build_stable_class_seed(
    global_seed: int,
    class_name: str,
) -> int:
    """
    为每个类别生成独立且稳定的随机种子。

    这样某个类别增加或减少图片时，
    不会影响其他类别的切分结果。
    """
    seed_text = (
        f"{global_seed}\0{class_name}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        seed_text
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def compute_split_counts(
    total_count: int,
) -> tuple[int, int, int]:
    """
    根据比例计算 train、valid、test 数量。

    特点：
    1. 三部分数量之和始终等于 total_count。
    2. 当图片数不少于 3，并且三个比例都大于 0 时，
       尽量保证 train、valid、test 至少各有 1 张。
    """
    if total_count <= 0:
        return 0, 0, 0

    ratios = [
        TRAIN_RATIO,
        VALID_RATIO,
        TEST_RATIO,
    ]

    exact_counts = [
        total_count * ratio
        for ratio in ratios
    ]

    counts = [
        int(value)
        for value in exact_counts
    ]

    remaining = (
        total_count
        - sum(counts)
    )

    # 按小数部分从大到小分配剩余数量
    allocation_order = sorted(
        range(len(ratios)),
        key=lambda index: (
            exact_counts[index] - counts[index],
            ratios[index],
            -index,
        ),
        reverse=True,
    )

    for offset in range(remaining):
        target_index = allocation_order[
            offset % len(allocation_order)
        ]

        counts[target_index] += 1

    positive_split_indexes = [
        index
        for index, ratio in enumerate(ratios)
        if ratio > 0
    ]

    # 样本数量足够时，保证每个启用的 split 至少有 1 张
    if total_count >= len(positive_split_indexes):
        for target_index in positive_split_indexes:
            if counts[target_index] > 0:
                continue

            donor_indexes = [
                index
                for index in positive_split_indexes
                if counts[index] > 1
            ]

            if not donor_indexes:
                donor_indexes = [
                    index
                    for index in range(len(counts))
                    if counts[index] > 1
                ]

            if not donor_indexes:
                continue

            donor_index = max(
                donor_indexes,
                key=lambda index: (
                    counts[index],
                    ratios[index],
                    -index,
                ),
            )

            counts[donor_index] -= 1
            counts[target_index] += 1

    if sum(counts) != total_count:
        raise RuntimeError(
            "切分数量计算错误："
            f"总数={total_count}, counts={counts}"
        )

    return (
        counts[0],
        counts[1],
        counts[2],
    )


def split_one_class(
    images: list[Path],
    class_name: str,
) -> dict[str, list[Path]]:
    """
    对单个类别进行独立随机切分。
    """
    class_seed = build_stable_class_seed(
        SEED,
        class_name,
    )

    rng = random.Random(
        class_seed
    )

    shuffled_images = images.copy()
    rng.shuffle(shuffled_images)

    train_count, valid_count, test_count = (
        compute_split_counts(
            len(shuffled_images)
        )
    )

    train_end = train_count
    valid_end = train_count + valid_count

    split_map = {
        "train": shuffled_images[:train_end],
        "valid": shuffled_images[
            train_end:valid_end
        ],
        "test": shuffled_images[
            valid_end:
        ],
    }

    actual_count = sum(
        len(paths)
        for paths in split_map.values()
    )

    if actual_count != len(shuffled_images):
        raise RuntimeError(
            f"类别切分数量异常：{class_name}，"
            f"原始数量={len(shuffled_images)}，"
            f"切分后数量={actual_count}"
        )

    return split_map


# ============================================================
# 目标文件名生成
# ============================================================

def get_unique_destination(
    destination: Path,
    reserved_paths: set[Path],
) -> Path:
    """
    获取不会重复的目标路径。

    同时检查：
    1. 磁盘上已经存在的文件。
    2. 已经安排但还未复制的任务。

    避免并行复制时两个同名文件互相覆盖。
    """
    candidate = destination
    index = 1

    while (
        candidate in reserved_paths
        or candidate.exists()
    ):
        candidate = destination.parent / (
            f"{destination.stem}_{index}"
            f"{destination.suffix}"
        )

        index += 1

    reserved_paths.add(candidate)

    return candidate


# ============================================================
# 创建复制任务
# ============================================================

def build_file_tasks(
    root: Path,
    out_root: Path,
    class_dirs: list[Path],
    image_cache: dict[Path, list[Path]],
    raw_to_safe: dict[str, str],
) -> tuple[
    list[FileTask],
    dict[str, dict],
]:
    """
    创建所有复制或移动任务。
    """
    del root  # 保留参数语义，当前函数中暂不直接使用

    tasks: list[FileTask] = []
    report: dict[str, dict] = {}

    reserved_paths: set[Path] = set()

    task_id = 0

    for class_dir in tqdm(
        class_dirs,
        desc="拆分类别",
        unit="类",
        dynamic_ncols=True,
    ):
        raw_class = class_dir.name
        safe_class = raw_to_safe[raw_class]

        images = image_cache[class_dir]

        if not images:
            log_warning(
                "空类别：%s",
                raw_class,
            )

            report[safe_class] = {
                "raw_class": raw_class,
                "train": 0,
                "valid": 0,
                "test": 0,
            }

            continue

        if len(images) < MIN_CLASS_IMAGES_WARNING:
            log_warning(
                "类别图片较少：类别=%s，图片数=%d",
                raw_class,
                len(images),
            )

        split_map = split_one_class(
            images,
            raw_class,
        )

        report[safe_class] = {
            "raw_class": raw_class,
            "train": len(split_map["train"]),
            "valid": len(split_map["valid"]),
            "test": len(split_map["test"]),
        }

        for split_name, split_images in split_map.items():
            destination_dir = (
                out_root
                / split_name
                / safe_class
            )

            for source_path in split_images:
                destination_path = (
                    destination_dir
                    / source_path.name
                )

                destination_path = (
                    get_unique_destination(
                        destination_path,
                        reserved_paths,
                    )
                )

                tasks.append(
                    FileTask(
                        task_id=task_id,
                        src=source_path,
                        dst=destination_path,
                        raw_class=raw_class,
                        safe_class=safe_class,
                        split=split_name,
                    )
                )

                task_id += 1

    return tasks, report


# ============================================================
# 文件复制或移动
# ============================================================

def ensure_dir(
    path: Path,
) -> None:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def transfer_one_file(
    task: FileTask,
) -> Optional[str]:
    """
    执行单个文件复制或移动任务。

    成功：
        返回 None

    失败：
        返回错误字符串
    """
    temp_path: Optional[Path] = None

    try:
        ensure_dir(
            task.dst.parent
        )

        if task.dst.exists():
            raise FileExistsError(
                f"目标文件已经存在：{task.dst}"
            )

        if MOVE_FILES:
            shutil.move(
                str(task.src),
                str(task.dst),
            )

        else:
            # 先复制到临时文件，成功后再原子替换，
            # 避免程序中断后留下不完整的目标图片。
            temp_path = task.dst.with_name(
                f".{task.dst.name}"
                f".part-{task.task_id}"
            )

            if temp_path.exists():
                temp_path.unlink()

            if PRESERVE_METADATA:
                shutil.copy2(
                    str(task.src),
                    str(temp_path),
                )
            else:
                shutil.copyfile(
                    str(task.src),
                    str(temp_path),
                )

            os.replace(
                temp_path,
                task.dst,
            )

        return None

    except Exception as error:
        return (
            f"{type(error).__name__}: {error}"
        )

    finally:
        if (
            temp_path is not None
            and temp_path.exists()
        ):
            try:
                temp_path.unlink()
            except Exception:
                pass


def parallel_transfer_files(
    tasks: list[FileTask],
) -> tuple[
    int,
    int,
    list[Optional[str]],
]:
    """
    分批并行复制或移动文件。

    返回：
        success_count
        failed_count
        errors
    """
    task_count = len(tasks)

    errors: list[Optional[str]] = [
        None
    ] * task_count

    if task_count == 0:
        return 0, 0, errors

    worker_count = min(
        COPY_WORKERS,
        task_count,
    )

    success_count = 0
    failed_count = 0
    displayed_failure_count = 0

    progress_desc = (
        "移动图片"
        if MOVE_FILES
        else "复制图片"
    )

    with ThreadPoolExecutor(
        max_workers=worker_count,
    ) as executor:
        with tqdm(
            total=task_count,
            desc=progress_desc,
            unit="img",
            dynamic_ncols=True,
        ) as progress_bar:
            for batch_start in range(
                0,
                task_count,
                COPY_BATCH_SIZE,
            ):
                batch_end = min(
                    batch_start + COPY_BATCH_SIZE,
                    task_count,
                )

                batch_tasks = tasks[
                    batch_start:batch_end
                ]

                future_to_task = {
                    executor.submit(
                        transfer_one_file,
                        task,
                    ): task
                    for task in batch_tasks
                }

                for future in as_completed(
                    future_to_task
                ):
                    task = future_to_task[future]

                    try:
                        error = future.result()
                    except Exception as unexpected_error:
                        error = (
                            f"{type(unexpected_error).__name__}: "
                            f"{unexpected_error}"
                        )

                    errors[task.task_id] = error

                    if error is None:
                        success_count += 1
                    else:
                        failed_count += 1

                        if (
                            displayed_failure_count
                            < MAX_FAILURE_LOGS
                        ):
                            log_warning(
                                "文件处理失败：%s -> %s，错误：%s",
                                task.src,
                                task.dst,
                                error,
                            )

                            displayed_failure_count += 1

                    progress_bar.update(1)

    if failed_count > displayed_failure_count:
        log_warning(
            "还有 %d 条失败信息未在终端显示，"
            "请查看 copy_failed.jsonl",
            failed_count - displayed_failure_count,
        )

    return (
        success_count,
        failed_count,
        errors,
    )


# ============================================================
# 原子写文件
# ============================================================

def atomic_write_text(
    path: Path,
    content: str,
) -> None:
    """
    原子写入文本文件。
    """
    ensure_dir(path.parent)

    temp_path = path.with_name(
        f".{path.name}.tmp"
    )

    temp_path.write_text(
        content,
        encoding="utf-8",
    )

    os.replace(
        temp_path,
        path,
    )


def atomic_write_json(
    path: Path,
    data,
) -> None:
    content = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    )

    atomic_write_text(
        path,
        content,
    )


# ============================================================
# 保存类别映射
# ============================================================

def save_mapping(
    root: Path,
    out_root: Path,
    raw_to_safe: dict[str, str],
    safe_to_raw: dict[str, str],
) -> Path:
    """
    保存类别名称映射。
    """
    mapping_path = (
        out_root
        / "class_name_mapping.json"
    )

    data = {
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "input_root": str(root),
        "output_root": str(out_root),
        "seed": SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "valid": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "raw_to_safe": raw_to_safe,
        "safe_to_raw": safe_to_raw,
    }

    atomic_write_json(
        mapping_path,
        data,
    )

    log_info(
        "类别映射已保存：%s",
        mapping_path,
    )

    return mapping_path


# ============================================================
# 保存复制失败清单
# ============================================================

def save_failed_tasks(
    out_root: Path,
    tasks: list[FileTask],
    errors: list[Optional[str]],
) -> Path:
    """
    保存复制或移动失败的文件。
    """
    failed_path = (
        out_root
        / "copy_failed.jsonl"
    )

    temp_path = failed_path.with_name(
        f".{failed_path.name}.tmp"
    )

    failed_count = 0

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for task in tasks:
            error = errors[task.task_id]

            if error is None:
                continue

            failed_count += 1

            record = {
                "task_id": task.task_id,
                "source": str(task.src),
                "destination": str(task.dst),
                "raw_class": task.raw_class,
                "safe_class": task.safe_class,
                "split": task.split,
                "error": error,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            file.write("\n")

    os.replace(
        temp_path,
        failed_path,
    )

    log_info(
        "失败清单已保存：%s，失败数量=%d",
        failed_path,
        failed_count,
    )

    return failed_path


# ============================================================
# 保存完整数据清单
# ============================================================

def save_manifest(
    root: Path,
    out_root: Path,
    tasks: list[FileTask],
    errors: list[Optional[str]],
) -> Path:
    """
    保存全部图片的切分和处理结果。

    每行一个 JSON：
        source
        destination
        raw_class
        safe_class
        split
        operation
        status
        error
    """
    manifest_path = (
        out_root
        / "dataset_manifest.jsonl"
    )

    temp_path = manifest_path.with_name(
        f".{manifest_path.name}.tmp"
    )

    operation = (
        "move"
        if MOVE_FILES
        else "copy"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for task in tasks:
            error = errors[task.task_id]

            try:
                source_relative = str(
                    task.src.relative_to(root)
                )
            except ValueError:
                source_relative = ""

            record = {
                "task_id": task.task_id,
                "source": str(task.src),
                "source_relative": source_relative,
                "destination": str(task.dst),
                "raw_class": task.raw_class,
                "safe_class": task.safe_class,
                "split": task.split,
                "operation": operation,
                "status": (
                    "success"
                    if error is None
                    else "failed"
                ),
            }

            if error is not None:
                record["error"] = error

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            file.write("\n")

    os.replace(
        temp_path,
        manifest_path,
    )

    log_info(
        "数据清单已保存：%s",
        manifest_path,
    )

    return manifest_path


# ============================================================
# 保存统计报告
# ============================================================

def save_report(
    out_root: Path,
    report: dict[str, dict],
    tasks: list[FileTask],
    errors: list[Optional[str]],
    success_count: int,
    failed_count: int,
) -> Path:
    """
    保存类别切分和实际复制统计。
    """
    report_path = (
        out_root
        / "split_report.txt"
    )

    success_by_class: dict[
        str,
        dict[str, int],
    ] = {}

    failed_by_class: dict[
        str,
        dict[str, int],
    ] = {}

    for safe_class in report:
        success_by_class[safe_class] = {
            split: 0
            for split in SPLITS
        }

        failed_by_class[safe_class] = {
            split: 0
            for split in SPLITS
        }

    for task in tasks:
        error = errors[task.task_id]

        if error is None:
            success_by_class[
                task.safe_class
            ][task.split] += 1
        else:
            failed_by_class[
                task.safe_class
            ][task.split] += 1

    lines: list[str] = []

    lines.append(
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}"
    )
    lines.append(
        f"操作方式：{'移动' if MOVE_FILES else '复制'}"
    )
    lines.append(
        f"随机种子：{SEED}"
    )
    lines.append(
        "切分比例："
        f"train={TRAIN_RATIO}, "
        f"valid={VALID_RATIO}, "
        f"test={TEST_RATIO}"
    )
    lines.append("")

    planned_totals = {
        split: 0
        for split in SPLITS
    }

    success_totals = {
        split: 0
        for split in SPLITS
    }

    failed_totals = {
        split: 0
        for split in SPLITS
    }

    for safe_class in sorted(report):
        class_stat = report[safe_class]
        raw_class = class_stat["raw_class"]

        planned_train = class_stat["train"]
        planned_valid = class_stat["valid"]
        planned_test = class_stat["test"]

        planned_total = (
            planned_train
            + planned_valid
            + planned_test
        )

        success_train = success_by_class[
            safe_class
        ]["train"]

        success_valid = success_by_class[
            safe_class
        ]["valid"]

        success_test = success_by_class[
            safe_class
        ]["test"]

        success_total = (
            success_train
            + success_valid
            + success_test
        )

        class_failed = failed_by_class[
            safe_class
        ]

        class_failed_total = sum(
            class_failed.values()
        )

        lines.append(
            f"{safe_class}\t"
            f"原始类别={raw_class}\t"
            f"计划总数={planned_total}\t"
            f"train={planned_train}\t"
            f"valid={planned_valid}\t"
            f"test={planned_test}\t"
            f"成功={success_total}\t"
            f"失败={class_failed_total}"
        )

        planned_totals["train"] += planned_train
        planned_totals["valid"] += planned_valid
        planned_totals["test"] += planned_test

        success_totals["train"] += success_train
        success_totals["valid"] += success_valid
        success_totals["test"] += success_test

        for split in SPLITS:
            failed_totals[split] += (
                class_failed[split]
            )

    lines.append("")
    lines.append("=" * 100)

    planned_total_count = sum(
        planned_totals.values()
    )

    lines.append(
        "计划汇总\t"
        f"总数={planned_total_count}\t"
        f"train={planned_totals['train']}\t"
        f"valid={planned_totals['valid']}\t"
        f"test={planned_totals['test']}"
    )

    lines.append(
        "成功汇总\t"
        f"总数={success_count}\t"
        f"train={success_totals['train']}\t"
        f"valid={success_totals['valid']}\t"
        f"test={success_totals['test']}"
    )

    lines.append(
        "失败汇总\t"
        f"总数={failed_count}\t"
        f"train={failed_totals['train']}\t"
        f"valid={failed_totals['valid']}\t"
        f"test={failed_totals['test']}"
    )

    atomic_write_text(
        report_path,
        "\n".join(lines) + "\n",
    )

    log_info(
        "统计报告已保存：%s",
        report_path,
    )

    return report_path


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    root = ROOT
    out_root = OUT_ROOT

    log_info(
        "输入目录：%s",
        root,
    )

    log_info(
        "输出目录：%s",
        out_root,
    )

    log_info(
        "操作方式：%s",
        "移动" if MOVE_FILES else "复制",
    )

    # --------------------------------------------------------
    # 1. 配置检查
    # --------------------------------------------------------
    validate_config(
        root,
        out_root,
    )

    # --------------------------------------------------------
    # 2. 获取类别目录
    # --------------------------------------------------------
    class_dirs = collect_class_dirs(
        root
    )

    log_info(
        "发现类别目录：%d 个",
        len(class_dirs),
    )

    # --------------------------------------------------------
    # 3. 构建类别名映射
    # --------------------------------------------------------
    raw_to_safe, safe_to_raw = (
        build_class_name_mapping(
            class_dirs
        )
    )

    # --------------------------------------------------------
    # 4. 扫描全部图片
    #
    # 扫描成功之后再清空输出目录，
    # 避免输入目录错误时先删除原输出结果。
    # --------------------------------------------------------
    image_cache: dict[
        Path,
        list[Path],
    ] = {}

    total_images = 0
    empty_class_count = 0

    for class_dir in tqdm(
        class_dirs,
        desc="扫描类别",
        unit="类",
        dynamic_ncols=True,
    ):
        images = collect_images(
            class_dir,
            recursive=RECURSIVE,
        )

        image_cache[class_dir] = images
        total_images += len(images)

        if not images:
            empty_class_count += 1

    log_info(
        "扫描完成：类别=%d，图片=%d，空类别=%d",
        len(class_dirs),
        total_images,
        empty_class_count,
    )

    if total_images <= 0:
        raise ValueError(
            f"输入目录中没有找到支持的图片：{root}"
        )

    # --------------------------------------------------------
    # 5. 清理并创建输出目录
    # --------------------------------------------------------
    prepare_output_dir(
        out_root
    )

    # --------------------------------------------------------
    # 6. 创建切分和复制任务
    # --------------------------------------------------------
    tasks, report = build_file_tasks(
        root=root,
        out_root=out_root,
        class_dirs=class_dirs,
        image_cache=image_cache,
        raw_to_safe=raw_to_safe,
    )

    log_info(
        "切分完成，计划处理图片：%d 张",
        len(tasks),
    )

    # 保存类别映射
    save_mapping(
        root=root,
        out_root=out_root,
        raw_to_safe=raw_to_safe,
        safe_to_raw=safe_to_raw,
    )

    # --------------------------------------------------------
    # 7. 并行复制或移动
    # --------------------------------------------------------
    success_count, failed_count, errors = (
        parallel_transfer_files(
            tasks
        )
    )

    log_info(
        "文件处理完成：成功=%d，失败=%d",
        success_count,
        failed_count,
    )

    # --------------------------------------------------------
    # 8. 保存结果
    # --------------------------------------------------------
    save_report(
        out_root=out_root,
        report=report,
        tasks=tasks,
        errors=errors,
        success_count=success_count,
        failed_count=failed_count,
    )

    save_manifest(
        root=root,
        out_root=out_root,
        tasks=tasks,
        errors=errors,
    )

    save_failed_tasks(
        out_root=out_root,
        tasks=tasks,
        errors=errors,
    )

    # --------------------------------------------------------
    # 9. 最终检查
    # --------------------------------------------------------
    if failed_count > 0:
        log_warning(
            "任务完成，但有 %d 张图片处理失败，"
            "请检查：%s",
            failed_count,
            out_root / "copy_failed.jsonl",
        )
    else:
        log_info(
            "全部图片处理成功"
        )

    log_info(
        "输出结果目录：%s",
        out_root,
    )


if __name__ == "__main__":
    main()