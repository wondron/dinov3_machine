# dino_finetune/labels.py
"""
标注解析与入库校验（设计文档第 6 节）。

同时兼容两种 JSON：
  - 规范格式（v2）：顶层直接给出 is_oven / device_model / food_exist / container_type / accessory_type / rack_level；
  - 标注工具导出格式（annotation_version 1.x）：字段放在 "annotations" 下，没有 is_oven / food_exist，
    用 food_name 表示食物，列表里用 "无" 表示空，rack_level 是字符串列表（如 ["3"]、["无"]）。
解析后统一成 OvenLabel：空列表 [] 表示"无"，None 表示不适用 / 未标注（训练时由 loss 掩码排除）。
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .device import DeviceSpec

logger = logging.getLogger(__name__)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class OvenLabel:
    image_path: str
    json_path: str
    is_oven: bool
    device_model: str | None
    food_exist: bool | None
    container: list[str] | None
    accessory: list[str] | None
    rack_level: int | None

    @property
    def has_item(self) -> bool:
        """画面中是否有任何物品：食物、容器或附件。"""
        return bool(self.food_exist) or bool(self.container) or bool(self.accessory)

    @property
    def items_known(self) -> bool:
        return self.food_exist is not None and self.container is not None and self.accessory is not None


@dataclass
class LabelIssue:
    json_path: str
    level: str      # error：无法训练；warning：违反标注规范但可以训练
    code: str
    message: str


class LabelSchema:
    """类别表与"无"的写法。"""

    def __init__(self, container_classes: Sequence[str], accessory_classes: Sequence[str], none_tokens: Sequence[str]):
        self.container_classes = list(container_classes)
        self.accessory_classes = list(accessory_classes)
        self.none_tokens = {str(t).strip() for t in none_tokens}

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "LabelSchema":
        labels = cfg["labels"]
        return cls(labels["container"], labels["accessory"], labels["none_tokens"])

    def is_none(self, value: Any) -> bool:
        return value is None or (isinstance(value, str) and (not value.strip() or value.strip() in self.none_tokens))


# =========================
# 解析
# =========================
def _find_image(json_path: Path, raw: Mapping[str, Any], ann: Mapping[str, Any]) -> Path | None:
    for key in ("image", "image_name"):
        name = raw.get(key) or ann.get(key)
        if isinstance(name, str) and name.strip():
            for candidate in (json_path.parent / name.strip(), json_path.parent / Path(name.strip()).name):
                if candidate.is_file():
                    return candidate
    for ext in IMG_EXTS + tuple(e.upper() for e in IMG_EXTS):
        candidate = json_path.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


def _parse_names(
    ann: Mapping[str, Any],
    key: str,
    classes: Sequence[str],
    schema: LabelSchema,
    report: Callable[[str, str, str], None],
) -> list[str] | None:
    if key not in ann:
        report("warning", "missing_field", f"缺少 {key}，按未标注处理")
        return None
    value = ann[key]
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        report("error", "bad_type", f"{key} 必须是列表，实际为 {type(value).__name__}")
        return None

    names: list[str] = []
    saw_none = False
    for item in value:
        if schema.is_none(item):
            saw_none = True
            continue
        name = str(item).strip()
        if name not in classes:
            report("error", "unknown_class", f"{key} 中的类别 {name!r} 不在类别表中")
        elif name not in names:
            names.append(name)
    if saw_none and names:
        report("warning", "none_with_names", f"{key} 同时包含“无”和具体类别 {names}，已忽略“无”")
    return names


def _parse_rack_level(value: Any, schema: LabelSchema, report: Callable[[str, str, str], None]) -> int | None:
    if isinstance(value, list):
        values = [v for v in value if not schema.is_none(v)]
        if not values:
            return None
        if len(values) > 1:
            report("error", "bad_rack_level", f"rack_level 只能有一个层位，实际为 {value}")
            return None
        value = values[0]
    if schema.is_none(value):
        return None
    if isinstance(value, bool):
        level = None
    elif isinstance(value, int):
        level = value
    elif isinstance(value, float) and value.is_integer():
        level = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        level = int(value.strip())
    else:
        level = None
    if level is None or level < 0:
        report("error", "bad_rack_level", f"rack_level 必须是非负整数，实际为 {value!r}")
        return None
    return level


def parse_annotation(json_path: str | Path, schema: LabelSchema) -> tuple[OvenLabel | None, list[LabelIssue]]:
    """解析单个 JSON；存在硬错误时返回 (None, issues)。"""
    json_path = Path(json_path)
    issues: list[LabelIssue] = []

    def report(level: str, code: str, message: str) -> None:
        issues.append(LabelIssue(str(json_path), level, code, message))

    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        report("error", "bad_json", f"JSON 读取失败：{exc}")
        return None, issues
    if not isinstance(raw, dict):
        report("error", "bad_json", "JSON 顶层必须是对象")
        return None, issues
    ann = raw["annotations"] if isinstance(raw.get("annotations"), dict) else raw

    image_path = _find_image(json_path, raw, ann)
    if image_path is None:
        report("error", "missing_image", "找不到对应的图片文件")

    device_model = None if schema.is_none(ann.get("device_model")) else str(ann["device_model"]).strip()

    is_oven = ann.get("is_oven")
    if is_oven is None:
        is_oven = device_model is not None  # 旧格式没有 is_oven：有型号即一体机内部
    elif not isinstance(is_oven, bool):
        report("error", "bad_type", f"is_oven 必须是 true / false，实际为 {is_oven!r}")

    if "food_exist" in ann:
        food_exist = ann["food_exist"]
        if food_exist is not None and not isinstance(food_exist, bool):
            report("error", "bad_type", f"food_exist 必须是 true / false / null，实际为 {food_exist!r}")
    elif "food_name" in ann:
        food_name = ann["food_name"]
        if food_name is None:
            food_exist = None
        else:
            names = food_name if isinstance(food_name, list) else [food_name]
            food_exist = any(not schema.is_none(n) for n in names)
    else:
        food_exist = None
        report("warning", "missing_field", "缺少 food_exist / food_name，食物标签按未标注处理")

    container = _parse_names(ann, "container_type", schema.container_classes, schema, report)
    accessory = _parse_names(ann, "accessory_type", schema.accessory_classes, schema, report)
    rack_level = _parse_rack_level(ann.get("rack_level"), schema, report)

    if any(issue.level == "error" for issue in issues):
        return None, issues
    label = OvenLabel(
        image_path=str(image_path),
        json_path=str(json_path),
        is_oven=bool(is_oven),
        device_model=device_model,
        food_exist=food_exist,
        container=container,
        accessory=accessory,
        rack_level=rack_level,
    )
    return label, issues


# =========================
# 入库校验（设计文档第 6 节）
# =========================
def check_label_rules(label: OvenLabel, profile: Mapping[str, DeviceSpec]) -> list[LabelIssue]:
    issues: list[LabelIssue] = []

    def report(level: str, code: str, message: str) -> None:
        issues.append(LabelIssue(label.json_path, level, code, message))

    if label.is_oven:
        if label.device_model is None:
            report("error", "missing_device", "is_oven=true 时 device_model 必填")
        elif label.device_model not in profile:
            report("error", "unknown_device", f"型号 {label.device_model} 不在 Device Profile 中，请先补充该型号的配置")
        else:
            spec = profile[label.device_model]
            if label.rack_level is not None and label.rack_level > spec.rack_count:
                report(
                    "error",
                    "rack_out_of_range",
                    f"rack_level={label.rack_level} 超出型号 {spec.name} 的层数 rack_count={spec.rack_count}",
                )
            if label.rack_level == 0 and not spec.floor_usable:
                report("error", "rack_floor_unusable", f"型号 {spec.name} 的 floor_usable=false，rack_level 不能为 0")
            if spec.accessories is not None and label.accessory:
                unsupported = [a for a in label.accessory if a not in spec.accessories]
                if unsupported:
                    report(
                        "warning",
                        "accessory_unsupported",
                        f"附件 {unsupported} 不在型号 {spec.name} 的 Device Profile 支持列表中，推理时会被屏蔽",
                    )
        if label.rack_level is not None and label.rack_level >= 1 and label.accessory == []:
            report("warning", "rack_without_accessory", f"rack_level={label.rack_level}（导轨层）但 accessory_type 为空")
        if label.rack_level is None and label.has_item:
            report("warning", "rack_missing", "一体机内部有物品但 rack_level 为空，不参与层位 loss")
        if label.rack_level is not None and label.items_known and not label.has_item:
            report("warning", "rack_on_empty", "空腔图片的 rack_level 应为 null")
    else:
        if label.device_model is not None:
            report("error", "device_not_oven", "is_oven=false 时 device_model 必须为 null")
        if label.rack_level is not None:
            report("warning", "rack_not_oven", "is_oven=false 时 rack_level 应为 null，不参与层位 loss")
    return issues


def load_split(
    roots: Sequence[str | Path],
    split: str,
    schema: LabelSchema,
    profile: Mapping[str, DeviceSpec],
    *,
    on_error: str = "raise",
    strict: bool = False,
) -> tuple[list[OvenLabel], list[LabelIssue]]:
    """读取 <root>/<split>/*.json 并做入库校验；strict=True 时规范类警告也按错误处理。"""
    json_files: list[Path] = []
    for root in roots:
        split_dir = Path(root) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"数据目录不存在：{split_dir}")
        json_files.extend(sorted(split_dir.glob("*.json")))
    if not json_files:
        raise RuntimeError(f"{split} 划分下没有找到任何 JSON 标注：{[str(Path(r) / split) for r in roots]}")

    labels: list[OvenLabel] = []
    all_issues: list[LabelIssue] = []
    bad_samples = 0
    for json_path in json_files:
        label, issues = parse_annotation(json_path, schema)
        if label is not None:
            issues += check_label_rules(label, profile)
        if strict:
            for issue in issues:
                issue.level = "error"
        all_issues += issues
        if label is None or any(issue.level == "error" for issue in issues):
            bad_samples += 1
            continue
        labels.append(label)

    errors = [issue for issue in all_issues if issue.level == "error"]
    if errors and on_error == "raise":
        preview = "\n".join(f"  {issue.json_path}: {issue.message}" for issue in errors[:20])
        raise ValueError(
            f"{split} 划分有 {bad_samples} 个样本存在标注错误（共 {len(errors)} 条），前 20 条：\n{preview}\n"
            "请修正标注，或设置 data.on_error=skip 跳过这些样本"
        )
    if bad_samples:
        logger.warning("%s 划分跳过 %d 个标注有错误的样本（详见 label_issues.json）", split, bad_samples)

    warnings = Counter(issue.code for issue in all_issues if issue.level == "warning")
    for code, count in warnings.most_common():
        example = next(issue for issue in all_issues if issue.level == "warning" and issue.code == code)
        logger.warning("%s 划分标注规范警告 %s × %d，例如 %s：%s", split, code, count, Path(example.json_path).name, example.message)

    if not labels:
        raise RuntimeError(f"{split} 划分没有可用样本")
    return labels, all_issues


def summarize_labels(labels: Sequence[OvenLabel]) -> dict[str, Any]:
    """统计各字段分布，用于日志和数据覆盖检查。"""
    rack = Counter(
        "null" if lb.rack_level is None else str(lb.rack_level) for lb in labels if lb.is_oven
    )
    return {
        "num_samples": len(labels),
        "is_oven": sum(lb.is_oven for lb in labels),
        "non_oven": sum(not lb.is_oven for lb in labels),
        "device_model": dict(Counter(lb.device_model for lb in labels if lb.is_oven)),
        "food_exist": dict(Counter({True: "true", False: "false", None: "null"}[lb.food_exist] for lb in labels)),
        "container": dict(Counter(n for lb in labels for n in (lb.container or []))),
        "accessory": dict(Counter(n for lb in labels for n in (lb.accessory or []))),
        "rack_level": dict(sorted(rack.items())),
    }
