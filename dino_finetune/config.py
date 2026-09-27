from __future__ import annotations

from pathlib import Path
from typing import Any, Set

import yaml


_DEFAULT_CONFIG_FILES = {
    "seg": "default_seg.yaml",
    "cls": "default_cls.yaml",
    "mul": "default_mul.yaml",
}


def default_config_path(task: str = "mul") -> str:
    task_key = str(task).strip().lower()
    aliases = {
        "segment": "seg",
        "segmentation": "seg",
        "classify": "cls",
        "classification": "cls",
        "multi": "mul",
        "multitask": "mul",
    }
    task_key = aliases.get(task_key, task_key)
    if task_key not in _DEFAULT_CONFIG_FILES:
        raise ValueError(
            f"未知配置任务：{task}，仅支持：{sorted(_DEFAULT_CONFIG_FILES)}"
        )

    filename = _DEFAULT_CONFIG_FILES[task_key]
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        cand = p / "configs" / filename
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(f"未找到配置文件 {filename}（从 {here} 向上搜索）")


def load_config(path: str | None = None) -> dict[str, Any]:
    """
    读取 yaml 配置，并在此函数内部强制执行 validate_config(cfg)。
    你在 train_seg.py / tools / script 里只要 cfg = load_config(args.config) 即可。
    """
    cfg_path = Path(path or default_config_path("mul"))
    if not cfg_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误：期望 dict，实际为 {type(cfg).__name__}")

    validate_config(cfg)
    return cfg


# =========================
# 统一校验框架
# =========================
def _require(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in cfg or cfg[key] is None:
        raise ValueError(f"缺少配置模块：{key}")
    if not isinstance(cfg[key], dict):
        raise ValueError(f"配置模块 {key} 类型错误：期望 dict，实际为 {type(cfg[key]).__name__}")
    return cfg[key]


def _require_field(d: dict[str, Any], field: str, ctx: str) -> Any:
    if field not in d or d[field] is None:
        raise ValueError(f"{ctx}.{field} 为必填项")
    return d[field]


def _validate_list_len(name: str, v: Any, exp_len: int, ctx: str) -> None:
    if not isinstance(v, (list, tuple)) or len(v) != exp_len:
        raise ValueError(f"{ctx}.{name} 必须是长度为 {exp_len} 的列表")


def _validate_input(ctx: str, inp: dict[str, Any], require_mask_interp: bool) -> None:
    img_dim = _require_field(inp, "img_dim", ctx)
    mean = _require_field(inp, "mean", ctx)
    std = _require_field(inp, "std", ctx)
    img_interp = _require_field(inp, "img_interp", ctx)

    _validate_list_len("img_dim", img_dim, 2, ctx)
    _validate_list_len("mean", mean, 3, ctx)
    _validate_list_len("std", std, 3, ctx)

    if require_mask_interp:
        _require_field(inp, "mask_interp", ctx)

    # letterbox 规则：分割 input 有；分类 input_cls 没有也没关系
    if "letterbox" in inp and bool(inp.get("letterbox", False)):
        raise ValueError(f"{ctx}.letterbox 当前版本不支持，必须为 false")

    # img_interp 值域放到 resolve_interp 里统一验证
    try:
        resolve_interp(str(img_interp))
    except Exception as e:
        raise ValueError(f"{ctx}.img_interp 无效：{img_interp}，错误：{e}")


def _validate_dataset(ctx: str, ds: dict[str, Any], allow_types: Set[str]) -> None:
    root = _require_field(ds, "root", ctx)
    if not str(root).strip():
        raise ValueError(f"{ctx}.root 不能为空")

    dataset_type = str(_require_field(ds, "type", ctx)).strip().lower()
    if dataset_type not in allow_types:
        raise ValueError(
            f"{ctx}.type 不支持：{dataset_type}，期望值：{sorted(allow_types)}"
        )


def _validate_model_backbone(model: dict[str, Any]) -> None:
    for field in ("dino_type", "size"):
        value = _require_field(model, field, "model")
        if not str(value).strip():
            raise ValueError(f"model.{field} 不能为空")


def _validate_model_seg(model: dict[str, Any]) -> None:
    n_classes = _require_field(model, "n_classes", "model")
    if not isinstance(n_classes, int) or n_classes <= 0:
        raise ValueError("model.n_classes 必须是大于 0 的整数")


def _validate_model_cls(model_cls: dict[str, Any]) -> None:
    emb_dim = _require_field(model_cls, "emb_dim", "model_cls")
    if not isinstance(emb_dim, int) or emb_dim <= 0:
        raise ValueError("model_cls.emb_dim 必须是大于 0 的整数")

    num_classes = _require_field(model_cls, "num_classes", "model_cls")
    # ✅ 允许 0：表示“运行时由 dataloader 推断覆盖”
    if not isinstance(num_classes, int) or num_classes < 0:
        raise ValueError("model_cls.num_classes 必须是 >= 0 的整数（允许 0 表示运行时推断）")

    pool = _require_field(model_cls, "pool", "model_cls")
    pool = str(pool).lower().strip()
    if pool not in {"cls_token", "mean"}:
        raise ValueError("model_cls.pool 仅支持：cls_token / mean")


def _validate_postprocess(pp: dict[str, Any]) -> None:
    thr = _require_field(pp, "thr", "postprocess")
    ignore_index = _require_field(pp, "ignore_index", "postprocess")

    try:
        threshold = float(thr)
    except Exception:
        raise ValueError("postprocess.thr 必须是数字")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("postprocess.thr 必须在 0 到 1 之间")

    if not isinstance(ignore_index, int):
        raise ValueError("postprocess.ignore_index 必须是 int")

    mask_mode = str(pp.get("mask_mode", "prob_threshold")).strip().lower()
    iou_mode = str(pp.get("iou_mode", "prob_threshold")).strip().lower()

    allowed_modes = {"argmax", "prob_threshold"}
    if mask_mode not in allowed_modes:
        raise ValueError(f"postprocess.mask_mode 不支持：{mask_mode}")
    if iou_mode not in allowed_modes:
        raise ValueError(f"postprocess.iou_mode 不支持：{iou_mode}")

    if mask_mode != iou_mode:
        raise ValueError(
            f"postprocess.mask_mode（{mask_mode}）必须与 postprocess.iou_mode（{iou_mode}）一致"
        )

    if mask_mode == "argmax" and abs(threshold - 0.5) > 1e-6:
        raise ValueError("当 mask_mode=argmax 时，postprocess.thr 必须等于 0.5")


def _validate_trainparams(
    tp: dict[str, Any],
    *,
    require_seg_batch: bool,
    require_cls_batch: bool,
) -> None:
    if tp is None:
        raise ValueError("trainparams 不能为 null，必须是一个字典")
    if not isinstance(tp, dict):
        raise ValueError(f"trainparams 类型错误，期望 dict，实际为 {type(tp).__name__}")

    use_lora = tp.get("use_lora", False)
    if not isinstance(use_lora, bool):
        raise ValueError(f"trainparams.use_lora 类型错误，期望 bool，实际为 {type(use_lora).__name__}")

    use_fpn = tp.get("use_fpn", False)
    if not isinstance(use_fpn, bool):
        raise ValueError(f"trainparams.use_fpn 类型错误，期望 bool，实际为 {type(use_fpn).__name__}")

    rank_r = tp.get("rank_r", None)
    if use_lora:
        if rank_r is None:
            raise ValueError("启用 LoRA（trainparams.use_lora=true）时，必须指定 trainparams.rank_r")
        if not isinstance(rank_r, int):
            raise ValueError(f"trainparams.rank_r 类型错误，期望 int，实际为 {type(rank_r).__name__}")
        if rank_r <= 0:
            raise ValueError("trainparams.rank_r 必须是大于 0 的整数")

    required_batch_fields = []
    if require_seg_batch:
        required_batch_fields.append("batch_size_seg")
    if require_cls_batch:
        required_batch_fields.append("batch_size_cls")

    for field in ("batch_size_seg", "batch_size_cls"):
        if field not in tp and field not in required_batch_fields:
            continue
        value = tp.get(field)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"trainparams.{field} 必须是大于 0 的整数")

    steps_per_epoch = tp.get("steps_per_epoch", 0)
    if not isinstance(steps_per_epoch, int) or steps_per_epoch < 0:
        raise ValueError("trainparams.steps_per_epoch 必须是大于等于 0 的整数")

    best_score_seg_weight = tp.get("best_score_seg_weight", 0.5)
    if not isinstance(best_score_seg_weight, (int, float)):
        raise ValueError("trainparams.best_score_seg_weight 必须是数字")
    if not 0.0 <= float(best_score_seg_weight) <= 1.0:
        raise ValueError("trainparams.best_score_seg_weight 必须在 0 到 1 之间")


def validate_config(cfg: dict[str, Any]) -> None:
    has_seg = any(key in cfg for key in ("input", "dataset", "postprocess"))
    has_cls = any(key in cfg for key in ("input_cls", "dataset_cls", "model_cls"))
    if not has_seg and not has_cls:
        raise ValueError("配置中至少需要包含分割或分类任务模块")

    model = _require(cfg, "model")
    tp = _require(cfg, "trainparams")
    _validate_model_backbone(model)

    if has_seg:
        inp = _require(cfg, "input")
        ds = _require(cfg, "dataset")
        pp = _require(cfg, "postprocess")

        _validate_input("input", inp, require_mask_interp=True)
        _validate_dataset("dataset", ds, allow_types={"binary", "voc", "ade20k", "multiclass"})
        _validate_model_seg(model)
        _validate_postprocess(pp)

    if has_cls:
        inp_cls = _require(cfg, "input_cls")
        ds_cls = _require(cfg, "dataset_cls")
        model_cls = _require(cfg, "model_cls")

        _validate_input("input_cls", inp_cls, require_mask_interp=False)
        _validate_dataset("dataset_cls", ds_cls, allow_types={"local_folder"})
        _validate_model_cls(model_cls)

    _validate_trainparams(
        tp,
        require_seg_batch=has_seg,
        require_cls_batch=has_cls,
    )


def resolve_interp(name: str) -> int:
    import cv2

    key = str(name).lower().strip()
    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
    }
    if key not in mapping:
        raise ValueError(f"unknown interpolation: {name}（期望：{sorted(mapping)}）")
    return mapping[key]


def get_dino_paths(cfg: dict[str, Any]) -> tuple[str, str]:
    """
    从配置中读取 DINO 本地仓库与预训练权重路径。
    - model.dino_local_repo: DINO 本地仓库路径（默认 third_party/dinov3）
    - model.weight_path:    预训练权重路径（默认 data/model/xxx.pth）
    """
    model = _require(cfg, "model")
    dino_local_repo = str(model.get("dino_local_repo", "third_party/dinov3")).strip()
    weight_path = str(model.get("weight_path", "data/model/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")).strip()
    if not dino_local_repo:
        raise ValueError("model.dino_local_repo 不能为空")
    if not weight_path:
        raise ValueError("model.weight_path 不能为空")
    return dino_local_repo, weight_path


def print_config(cfg: dict[str, Any], indent: int = 0) -> None:
    if indent == 0:
        print("\n=== CONFIG ===")
    pad = " " * indent
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            print_config(v, indent + 2)
        else:
            print(f"{pad}{k:<15}: {v}")


def main() -> None:
    cfg = load_config(default_config_path("mul"))
    print_config(cfg)


if __name__ == "__main__":
    main()
