"""使用感知哈希清理文件夹中的近似图像。

该脚本不使用命令行参数。请直接修改下方“用户参数”后运行。
默认仅预演并生成 JSON 报告，不会移动或删除任何图像。
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ============================== 用户参数 ==============================
# 待清理的图像文件夹。
IMAGE_DIR = Path(r"./data/images")

# 是否递归处理子文件夹。
RECURSIVE = True

# 感知哈希相似度阈值，范围 [0, 1]。值越低，清理力度越大：
# 0.98：非常保守；0.92：常用起点；0.85：较激进，务必先检查报告。
SIMILARITY_THRESHOLD = 0.92

# 操作方式：
# "move"        将相似图像移动到隔离目录（推荐，可恢复）
# "delete"      永久删除相似图像（不可恢复）
# "report_only" 只生成报告
ACTION = "move"

# True 时只预演，不移动或删除图像。首次运行请保持 True。
DRY_RUN = True

# None 表示自动使用“图像目录同级/<图像目录名>_similar_removed”。
# 隔离目录必须位于待清理图像目录之外。
QUARANTINE_DIR: Optional[Path] = None

# None 表示自动使用“图像目录同级/<图像目录名>_similar_report.json”。
REPORT_PATH: Optional[Path] = None

# pHash 参数。通常无需修改。
PHASH_IMAGE_SIZE = 32
PHASH_LOW_FREQ_SIZE = 8

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
# ====================================================================


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    relative_path: Path
    phash: int
    width: int
    height: int
    file_size: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def min_side(self) -> int:
        return min(self.width, self.height)


@dataclass(frozen=True)
class DuplicateMatch:
    duplicate: ImageRecord
    representative: ImageRecord
    hamming_distance: int
    similarity: float


class BKTreeNode:
    """用于按汉明距离搜索 pHash，避免对所有图像做全量两两比较。"""

    def __init__(self, hash_value: int, record_index: int) -> None:
        self.hash_value = hash_value
        self.record_indices = [record_index]
        self.children: dict[int, "BKTreeNode"] = {}

    def add(self, hash_value: int, record_index: int) -> None:
        node = self
        while True:
            distance = hamming_distance(node.hash_value, hash_value)
            if distance == 0:
                node.record_indices.append(record_index)
                return

            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKTreeNode(hash_value, record_index)
                return
            node = child

    def query(self, hash_value: int, radius: int) -> list[tuple[int, int]]:
        results: list[tuple[int, int]] = []
        pending_nodes = [self]
        while pending_nodes:
            node = pending_nodes.pop()
            distance = hamming_distance(node.hash_value, hash_value)
            if distance <= radius:
                results.extend((distance, index) for index in node.record_indices)

            lower = distance - radius
            upper = distance + radius
            pending_nodes.extend(
                child
                for edge_distance, child in node.children.items()
                if lower <= edge_distance <= upper
            )
        return results


class BKTree:
    def __init__(self) -> None:
        self.root: Optional[BKTreeNode] = None

    def add(self, hash_value: int, record_index: int) -> None:
        if self.root is None:
            self.root = BKTreeNode(hash_value, record_index)
        else:
            self.root.add(hash_value, record_index)

    def query(self, hash_value: int, radius: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        return self.root.query(hash_value, radius)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_parameters() -> tuple[Path, Optional[Path], Path, int, int]:
    image_dir = IMAGE_DIR.expanduser().resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"图像目录不存在：{image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹：{image_dir}")

    if not 0.0 <= SIMILARITY_THRESHOLD <= 1.0:
        raise ValueError("SIMILARITY_THRESHOLD 必须在 [0, 1] 范围内")
    if ACTION not in {"move", "delete", "report_only"}:
        raise ValueError("ACTION 只能是 move、delete 或 report_only")
    if PHASH_IMAGE_SIZE <= 0 or PHASH_LOW_FREQ_SIZE <= 0:
        raise ValueError("pHash 尺寸参数必须为正整数")
    if PHASH_LOW_FREQ_SIZE > PHASH_IMAGE_SIZE:
        raise ValueError("PHASH_LOW_FREQ_SIZE 不能大于 PHASH_IMAGE_SIZE")

    hash_bits = PHASH_LOW_FREQ_SIZE * PHASH_LOW_FREQ_SIZE
    max_hamming_distance = math.floor(
        (1.0 - SIMILARITY_THRESHOLD) * hash_bits + 1e-12
    )

    quarantine_dir: Optional[Path] = None
    if ACTION == "move":
        configured_quarantine = QUARANTINE_DIR or image_dir.parent / (
            image_dir.name + "_similar_removed"
        )
        quarantine_dir = configured_quarantine.expanduser().resolve()
        if quarantine_dir == image_dir or is_relative_to(quarantine_dir, image_dir):
            raise ValueError("隔离目录必须位于待清理图像目录之外")

    configured_report = REPORT_PATH or image_dir.parent / (
        image_dir.name + "_similar_report.json"
    )
    report_path = configured_report.expanduser().resolve()
    if report_path.exists() and report_path.is_dir():
        raise IsADirectoryError(f"报告路径不能是文件夹：{report_path}")

    return image_dir, quarantine_dir, report_path, hash_bits, max_hamming_distance


def read_grayscale_image(path: Path) -> np.ndarray:
    """兼容 Windows 中文路径的 OpenCV 图像读取。"""
    encoded = np.fromfile(str(path), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError("文件内容为空")
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("OpenCV 无法解码图像")
    return image


def calculate_phash(image: np.ndarray) -> int:
    resized = cv2.resize(
        image,
        (PHASH_IMAGE_SIZE, PHASH_IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    dct_values = cv2.dct(resized)
    low_frequency = dct_values[:PHASH_LOW_FREQ_SIZE, :PHASH_LOW_FREQ_SIZE]
    flattened = low_frequency.reshape(-1)

    # 排除直流分量后计算中位数，减小整体亮度变化的影响。
    values_for_median = flattened[1:] if flattened.size > 1 else flattened
    median = float(np.median(values_for_median))
    bits = flattened > median

    hash_value = 0
    for bit in bits:
        hash_value = (hash_value << 1) | int(bit)
    return hash_value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def hash_similarity(left: int, right: int, hash_bits: int) -> float:
    return 1.0 - hamming_distance(left, right) / hash_bits


def discover_images(image_dir: Path) -> list[Path]:
    iterator = image_dir.rglob("*") if RECURSIVE else image_dir.glob("*")
    paths = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(paths, key=lambda item: item.as_posix().lower())


def load_records(
    image_dir: Path, image_paths: list[Path]
) -> tuple[list[ImageRecord], list[dict[str, str]]]:
    records: list[ImageRecord] = []
    failures: list[dict[str, str]] = []

    for index, path in enumerate(image_paths, start=1):
        relative_path = path.relative_to(image_dir)
        try:
            if path.is_symlink():
                raise ValueError("为防止操作目录外文件，符号链接已跳过")
            image = read_grayscale_image(path)
            height, width = image.shape[:2]
            records.append(
                ImageRecord(
                    path=path,
                    relative_path=relative_path,
                    phash=calculate_phash(image),
                    width=width,
                    height=height,
                    file_size=path.stat().st_size,
                )
            )
        except (OSError, ValueError, cv2.error) as exc:
            failures.append({"path": relative_path.as_posix(), "error": str(exc)})
            print(f"[警告] 跳过无法读取的图像：{relative_path}，原因：{exc}")

        if index % 500 == 0 or index == len(image_paths):
            print(f"[信息] 已计算哈希：{index}/{len(image_paths)}")

    return records, failures


def quality_sort_key(record: ImageRecord) -> tuple[int, int, int, str]:
    """优先保留分辨率高、短边长、文件信息量大的图像。"""
    return (
        -record.area,
        -record.min_side,
        -record.file_size,
        record.relative_path.as_posix().lower(),
    )


def find_duplicates(
    records: list[ImageRecord], hash_bits: int, max_hamming_distance: int
) -> tuple[list[ImageRecord], list[DuplicateMatch]]:
    ordered_records = sorted(records, key=quality_sort_key)
    representatives: list[ImageRecord] = []
    duplicates: list[DuplicateMatch] = []
    tree = BKTree()

    for record in ordered_records:
        matches = tree.query(record.phash, max_hamming_distance)
        if not matches:
            representative_index = len(representatives)
            representatives.append(record)
            tree.add(record.phash, representative_index)
            continue

        # 先选哈希距离最近者；距离相同时，保留排序更靠前的高质量图像。
        distance, representative_index = min(matches, key=lambda item: (item[0], item[1]))
        representative = representatives[representative_index]
        duplicates.append(
            DuplicateMatch(
                duplicate=record,
                representative=representative,
                hamming_distance=distance,
                similarity=hash_similarity(record.phash, representative.phash, hash_bits),
            )
        )

    representatives.sort(key=lambda item: item.relative_path.as_posix().lower())
    duplicates.sort(key=lambda item: item.duplicate.relative_path.as_posix().lower())
    return representatives, duplicates


def unique_move_target(quarantine_dir: Path, relative_path: Path) -> Path:
    target = quarantine_dir / relative_path
    if not target.exists():
        return target

    counter = 1
    while True:
        candidate = target.with_name(f"{target.stem}_{counter:03d}{target.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def execute_actions(
    duplicates: list[DuplicateMatch], quarantine_dir: Optional[Path]
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    results: list[dict[str, object]] = []
    action_failures: list[dict[str, str]] = []

    for match in duplicates:
        duplicate = match.duplicate
        result: dict[str, object] = {
            "duplicate": duplicate.relative_path.as_posix(),
            "representative": match.representative.relative_path.as_posix(),
            "similarity": round(match.similarity, 6),
            "hamming_distance": match.hamming_distance,
            "duplicate_size": [duplicate.width, duplicate.height],
            "representative_size": [
                match.representative.width,
                match.representative.height,
            ],
        }

        if ACTION == "report_only":
            result["status"] = "仅报告"
            results.append(result)
            continue

        if ACTION == "move":
            assert quarantine_dir is not None
            target = unique_move_target(quarantine_dir, duplicate.relative_path)
            result["target"] = str(target)
            if DRY_RUN:
                result["status"] = "预演：计划移动"
                results.append(result)
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(duplicate.path), str(target))
                result["status"] = "已移动"
            except OSError as exc:
                result["status"] = "移动失败"
                result["error"] = str(exc)
                action_failures.append(
                    {"path": duplicate.relative_path.as_posix(), "error": str(exc)}
                )
            results.append(result)
            continue

        if DRY_RUN:
            result["status"] = "预演：计划删除"
            results.append(result)
            continue
        try:
            duplicate.path.unlink()
            result["status"] = "已删除"
        except OSError as exc:
            result["status"] = "删除失败"
            result["error"] = str(exc)
            action_failures.append(
                {"path": duplicate.relative_path.as_posix(), "error": str(exc)}
            )
        results.append(result)

    return results, action_failures


def write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            prefix=path.name + ".",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    image_dir, quarantine_dir, report_path, hash_bits, max_distance = (
        validate_parameters()
    )

    actual_min_similarity = 1.0 - max_distance / hash_bits
    print(f"[信息] 图像目录：{image_dir}")
    print(
        "[信息] 配置阈值："
        f"{SIMILARITY_THRESHOLD:.4f}，pHash 实际最低相似度：{actual_min_similarity:.4f}，"
        f"最大汉明距离：{max_distance}/{hash_bits}"
    )
    print(f"[信息] 操作方式：{ACTION}，预演模式：{DRY_RUN}")
    if ACTION == "delete" and not DRY_RUN:
        print("[警告] 当前为永久删除模式，删除后的文件无法通过本脚本恢复")

    image_paths = discover_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"目录中未找到支持的图像：{image_dir}")
    print(f"[信息] 共发现 {len(image_paths)} 张候选图像")

    records, read_failures = load_records(image_dir, image_paths)
    if not records:
        raise RuntimeError("没有可用于相似度计算的有效图像")

    representatives, duplicates = find_duplicates(records, hash_bits, max_distance)
    action_results, action_failures = execute_actions(duplicates, quarantine_dir)

    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "image_dir": str(image_dir),
        "recursive": RECURSIVE,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "actual_min_similarity": round(actual_min_similarity, 6),
        "hash_bits": hash_bits,
        "max_hamming_distance": max_distance,
        "action": ACTION,
        "dry_run": DRY_RUN,
        "quarantine_dir": str(quarantine_dir) if quarantine_dir else None,
        "counts": {
            "discovered": len(image_paths),
            "valid": len(records),
            "kept": len(representatives),
            "similar": len(duplicates),
            "read_failed": len(read_failures),
            "action_failed": len(action_failures),
        },
        "kept": [item.relative_path.as_posix() for item in representatives],
        "similar_images": action_results,
        "read_failures": read_failures,
        "action_failures": action_failures,
    }
    write_json_atomically(report_path, report)

    print(
        f"[完成] 有效图像 {len(records)} 张，保留 {len(representatives)} 张，"
        f"识别出相似图像 {len(duplicates)} 张"
    )
    print(f"[完成] 报告已写入：{report_path}")
    if DRY_RUN and ACTION != "report_only":
        print("[提示] 当前仅预演；检查报告无误后，将 DRY_RUN 改为 False 再运行")
    elif ACTION == "move" and quarantine_dir is not None:
        print(f"[完成] 相似图像隔离目录：{quarantine_dir}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[错误] {error}")
        raise SystemExit(1) from error
