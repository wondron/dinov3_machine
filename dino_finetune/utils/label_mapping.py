from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


def _to_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} 不能是布尔值")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是整数，实际为 {value!r}") from exc


def normalize_label_mapping(
    data: Any,
    expected_num_classes: int | None = None,
    model_class_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """将训练映射或 checkpoint 中的类别字段整理为统一的推理映射。"""
    if not isinstance(data, dict):
        raise TypeError(f"类别映射必须是字典，实际为 {type(data).__name__}")

    idx_to_name: dict[int, str] = {}
    idx_to_leaf_id: dict[int, int] = {}

    def add_name(index_value: Any, name_value: Any, source: str) -> None:
        index = _to_int(index_value, f"{source} 的类别索引")
        if index < 0:
            raise ValueError(f"{source} 的类别索引不能小于 0，实际为 {index}")
        if name_value is None or not str(name_value).strip():
            raise ValueError(f"{source} 中索引 {index} 的类别名称为空")
        name = str(name_value)
        if index in idx_to_name and idx_to_name[index] != name:
            raise ValueError(
                f"类别名称映射冲突：索引 {index} 同时对应 {idx_to_name[index]!r} 和 {name!r}"
            )
        idx_to_name[index] = name

    class_names = data.get("class_names")
    if class_names is not None:
        if not isinstance(class_names, (list, tuple)):
            raise TypeError("class_names 必须是列表或元组")
        for index, name in enumerate(class_names):
            add_name(index, name, "class_names")

    class_to_idx = data.get("class_to_idx")
    if class_to_idx is not None:
        if not isinstance(class_to_idx, dict):
            raise TypeError("class_to_idx 必须是字典")
        for name, index in class_to_idx.items():
            add_name(index, name, "class_to_idx")

    raw_idx_to_name = data.get("idx_to_class_name")
    if raw_idx_to_name is not None:
        if not isinstance(raw_idx_to_name, dict):
            raise TypeError("idx_to_class_name 必须是字典")
        for index, name in raw_idx_to_name.items():
            add_name(index, name, "idx_to_class_name")

    raw_to_safe = data.get("raw_to_safe")
    safe_to_raw = data.get("safe_to_raw")
    if raw_to_safe is not None or safe_to_raw is not None:
        if model_class_names is None:
            raise ValueError("raw_to_safe/safe_to_raw 映射需要 checkpoint 提供 class_names 类别顺序")
        if safe_to_raw is None:
            if not isinstance(raw_to_safe, dict):
                raise TypeError("raw_to_safe 必须是字典")
            safe_to_raw = {}
            for raw_name, safe_name in raw_to_safe.items():
                safe_name = str(safe_name)
                if safe_name in safe_to_raw:
                    raise ValueError(f"raw_to_safe 中存在重复的安全类别名：{safe_name}")
                safe_to_raw[safe_name] = str(raw_name)
        if not isinstance(safe_to_raw, dict):
            raise TypeError("safe_to_raw 必须是字典")

        missing_safe_names = [name for name in model_class_names if str(name) not in safe_to_raw]
        if missing_safe_names:
            raise ValueError(
                "名称映射缺少 checkpoint 类别："
                f"数量={len(missing_safe_names)}，前 20 个={missing_safe_names[:20]}"
            )
        for index, safe_name in enumerate(model_class_names):
            add_name(index, safe_to_raw[str(safe_name)], "safe_to_raw")

    def add_leaf_id(index_value: Any, leaf_id_value: Any, source: str) -> None:
        index = _to_int(index_value, f"{source} 的类别索引")
        leaf_id = _to_int(leaf_id_value, f"{source} 的 leaf_id")
        if index < 0:
            raise ValueError(f"{source} 的类别索引不能小于 0，实际为 {index}")
        if index in idx_to_leaf_id and idx_to_leaf_id[index] != leaf_id:
            raise ValueError(
                f"leaf_id 映射冲突：索引 {index} 同时对应 {idx_to_leaf_id[index]} 和 {leaf_id}"
            )
        idx_to_leaf_id[index] = leaf_id

    leaf_id_to_idx = data.get("leaf_id_to_idx")
    if leaf_id_to_idx is not None:
        if not isinstance(leaf_id_to_idx, dict):
            raise TypeError("leaf_id_to_idx 必须是字典")
        for leaf_id, index in leaf_id_to_idx.items():
            add_leaf_id(index, leaf_id, "leaf_id_to_idx")

    raw_idx_to_leaf_id = data.get("idx_to_leaf_id")
    if raw_idx_to_leaf_id is not None:
        if not isinstance(raw_idx_to_leaf_id, dict):
            raise TypeError("idx_to_leaf_id 必须是字典")
        for index, leaf_id in raw_idx_to_leaf_id.items():
            add_leaf_id(index, leaf_id, "idx_to_leaf_id")

    declared_num_classes: int | None = None
    if data.get("num_classes") is not None:
        declared_num_classes = _to_int(data["num_classes"], "num_classes")
        if declared_num_classes <= 0:
            raise ValueError(f"num_classes 必须大于 0，实际为 {declared_num_classes}")

    if expected_num_classes is not None:
        num_classes = _to_int(expected_num_classes, "期望类别数")
        if num_classes <= 0:
            raise ValueError(f"期望类别数必须大于 0，实际为 {num_classes}")
        if declared_num_classes is not None and declared_num_classes != num_classes:
            raise ValueError(
                f"映射类别数与模型不一致：mapping={declared_num_classes}，model={num_classes}"
            )
    elif declared_num_classes is not None:
        num_classes = declared_num_classes
    elif idx_to_name:
        num_classes = max(idx_to_name) + 1
    else:
        raise ValueError("类别映射中缺少 num_classes 和类别名称")

    expected_indices = set(range(num_classes))
    name_indices = set(idx_to_name)
    if name_indices != expected_indices:
        missing = sorted(expected_indices - name_indices)
        extra = sorted(name_indices - expected_indices)
        raise ValueError(f"类别名称映射不完整：缺失索引={missing[:20]}，越界索引={extra[:20]}")

    if not idx_to_leaf_id:
        idx_to_leaf_id = {index: index for index in range(num_classes)}
    elif set(idx_to_leaf_id) != expected_indices:
        missing = sorted(expected_indices - set(idx_to_leaf_id))
        extra = sorted(set(idx_to_leaf_id) - expected_indices)
        raise ValueError(f"leaf_id 映射不完整：缺失索引={missing[:20]}，越界索引={extra[:20]}")

    if len(set(idx_to_leaf_id.values())) != num_classes:
        raise ValueError("idx_to_leaf_id 中存在重复的 leaf_id")

    normalized_idx_to_leaf_id = {
        str(index): int(idx_to_leaf_id[index]) for index in range(num_classes)
    }
    normalized_idx_to_name = {str(index): idx_to_name[index] for index in range(num_classes)}
    normalized_leaf_id_to_idx = {
        str(idx_to_leaf_id[index]): index for index in range(num_classes)
    }
    return {
        "num_classes": num_classes,
        "leaf_id_to_idx": normalized_leaf_id_to_idx,
        "idx_to_leaf_id": normalized_idx_to_leaf_id,
        "idx_to_class_name": normalized_idx_to_name,
    }


def save_label_mapping(mapping: dict[str, Any], output_path: Path) -> None:
    """原子保存类别映射，避免导出中断时留下不完整 JSON。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(mapping, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
        temporary_path.replace(output_path)
    except Exception as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"类别映射保存失败：{output_path}，原因：{exc}") from exc
