from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_NAME = "default_oven.yaml"

LOSS_TERMS = ("is_oven", "proj", "food", "container", "accessory", "rack")
SCORE_METRICS = {
    "is_oven_acc",
    "food_acc",
    "food_f1",
    "container_map",
    "accessory_map",
    "rack_acc",
    "rack_acc_pm1",
    "rack_floor_acc",
    "device_top1",
}

_MISSING = object()


def default_config_path() -> str:
    path = PROJECT_ROOT / "configs" / DEFAULT_CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError(f"未找到默认配置文件：{path}")
    return str(path)


def resolve_path(path: str | Path) -> Path:
    """相对路径以项目根目录为基准，绝对路径原样返回。"""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    读取 yaml 配置，并在此函数内部强制执行 validate_config(cfg)：
    校验取值、规整类型，并为可选项补上默认值，调用方可以直接按键取值。
    """
    cfg_path = Path(path) if path else Path(default_config_path())
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
def _section(cfg: dict[str, Any], key: str, *, optional: bool = False) -> dict[str, Any]:
    value = cfg.get(key)
    if value is None:
        if not optional:
            raise ValueError(f"缺少配置模块：{key}")
        cfg[key] = {}
        return cfg[key]
    if not isinstance(value, dict):
        raise ValueError(f"配置模块 {key} 类型错误：期望 dict，实际为 {type(value).__name__}")
    return value


def _fetch(d: dict[str, Any], key: str, ctx: str, default: Any) -> Any:
    if key not in d or d[key] is None:
        if default is _MISSING:
            raise ValueError(f"{ctx}.{key} 为必填项")
        d[key] = default
    return d[key]


def _check_range(
    value: float,
    name: str,
    lo: float | None,
    hi: float | None,
    lo_open: bool,
    hi_open: bool,
) -> None:
    too_low = lo is not None and (value <= lo if lo_open else value < lo)
    too_high = hi is not None and (value >= hi if hi_open else value > hi)
    if too_low or too_high:
        left = "(" if lo_open else "["
        right = ")" if hi_open else "]"
        lo_text = "-inf" if lo is None else lo
        hi_text = "inf" if hi is None else hi
        raise ValueError(f"{name} 必须在 {left}{lo_text}, {hi_text}{right} 范围内，实际为 {value}")


def _int(
    d: dict[str, Any],
    key: str,
    ctx: str,
    *,
    default: Any = _MISSING,
    lo: int | None = None,
    hi: int | None = None,
) -> int:
    value = _fetch(d, key, ctx, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{ctx}.{key} 必须是整数，实际为 {value!r}")
    _check_range(value, f"{ctx}.{key}", lo, hi, False, False)
    return value


def _float(
    d: dict[str, Any],
    key: str,
    ctx: str,
    *,
    default: Any = _MISSING,
    lo: float | None = None,
    hi: float | None = None,
    lo_open: bool = False,
    hi_open: bool = False,
) -> float:
    raw = _fetch(d, key, ctx, default)
    if isinstance(raw, bool):
        raise ValueError(f"{ctx}.{key} 必须是数字，实际为 {raw!r}")
    try:
        value = float(raw)  # 兼容 PyYAML 把 "3e-4" 解析成字符串的情况
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{ctx}.{key} 必须是数字，实际为 {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{ctx}.{key} 必须是有限数字，实际为 {raw!r}")
    _check_range(value, f"{ctx}.{key}", lo, hi, lo_open, hi_open)
    d[key] = value
    return value


def _bool(d: dict[str, Any], key: str, ctx: str, *, default: Any = _MISSING) -> bool:
    value = _fetch(d, key, ctx, default)
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        value = bool(value)  # 兼容旧配置里的 0 / 1
    if not isinstance(value, bool):
        raise ValueError(f"{ctx}.{key} 必须是 true / false，实际为 {value!r}")
    d[key] = value
    return value


def _str(d: dict[str, Any], key: str, ctx: str, *, default: Any = _MISSING) -> str:
    value = _fetch(d, key, ctx, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{ctx}.{key} 必须是非空字符串，实际为 {value!r}")
    d[key] = value.strip()
    return d[key]


def _choice(d: dict[str, Any], key: str, ctx: str, choices: tuple[str, ...], *, default: Any = _MISSING) -> str:
    value = str(_fetch(d, key, ctx, default)).strip().lower()
    if value not in choices:
        raise ValueError(f"{ctx}.{key} 不支持：{value}，可选：{list(choices)}")
    d[key] = value
    return value


def _str_list(
    d: dict[str, Any],
    key: str,
    ctx: str,
    *,
    default: Any = _MISSING,
    allow_empty: bool = False,
) -> list[str]:
    value = _fetch(d, key, ctx, default)
    if not isinstance(value, (list, tuple)) or any(not isinstance(v, str) for v in value):
        raise ValueError(f"{ctx}.{key} 必须是字符串列表，实际为 {value!r}")
    names = [v.strip() for v in value]
    if not allow_empty and not names:
        raise ValueError(f"{ctx}.{key} 不能为空")
    if len(set(names)) != len(names):
        raise ValueError(f"{ctx}.{key} 中存在重复项：{names}")
    d[key] = names
    return names


# =========================
# 各模块校验
# =========================
def _validate_model(model: dict[str, Any]) -> None:
    ctx = "model"
    _str(model, "dino_type", ctx)
    _str(model, "size", ctx)
    _str(model, "dino_local_repo", ctx, default="third_party/dinov3")
    _str(model, "weight_path", ctx, default="data/model/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    _int(model, "max_rack", ctx, default=8, lo=1)
    _int(model, "attn_pool_heads", ctx, default=8, lo=1)
    _int(model, "head_hidden_dim", ctx, default=512, lo=1)
    _float(model, "head_dropout", ctx, default=0.0, lo=0.0, hi=1.0, hi_open=True)
    _int(model, "proj_hidden_dim", ctx, default=512, lo=1)
    _int(model, "proj_dim", ctx, default=256, lo=1)
    _bool(model, "use_lora", ctx, default=True)
    _int(model, "lora_rank", ctx, default=8, lo=1)
    _int(model, "lora_last_n_blocks", ctx, default=0, lo=0)


def _validate_labels(labels: dict[str, Any]) -> None:
    ctx = "labels"
    container = _str_list(labels, "container", ctx)
    accessory = _str_list(labels, "accessory", ctx)
    safety = _str_list(labels, "safety_container", ctx, default=[], allow_empty=True)
    unknown = [name for name in safety if name not in container]
    if unknown:
        raise ValueError(f"labels.safety_container 中有不在 labels.container 里的类别：{unknown}")
    none_tokens = _str_list(labels, "none_tokens", ctx, default=["无"], allow_empty=True)
    clash = [t for t in none_tokens if t in container or t in accessory]
    if clash:
        raise ValueError(f"labels.none_tokens 不能和类别名重复：{clash}")


def _validate_data(data: dict[str, Any]) -> None:
    ctx = "data"
    root = _fetch(data, "root", ctx, _MISSING)
    roots = [root] if isinstance(root, str) else root
    if not isinstance(roots, list) or not roots or any(not isinstance(r, str) or not r.strip() for r in roots):
        raise ValueError(f"data.root 必须是非空字符串或字符串列表，实际为 {root!r}")
    data["root"] = [r.strip() for r in roots]
    _str(data, "train_split", ctx, default="train")
    _str(data, "val_split", ctx, default="val")
    test_split = data.get("test_split")
    if test_split is not None and not isinstance(test_split, str):
        raise ValueError(f"data.test_split 必须是字符串或留空，实际为 {test_split!r}")
    data["test_split"] = test_split.strip() if isinstance(test_split, str) and test_split.strip() else None
    _choice(data, "on_error", ctx, ("raise", "skip"), default="raise")
    _bool(data, "strict", ctx, default=False)
    _str_list(data, "exclude_groups", ctx, default=[], allow_empty=True)


def _validate_input(inp: dict[str, Any]) -> None:
    ctx = "input"
    img_dim = _fetch(inp, "img_dim", ctx, _MISSING)
    if (
        not isinstance(img_dim, (list, tuple))
        or len(img_dim) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in img_dim)
    ):
        raise ValueError(f"input.img_dim 必须是两个正整数 [H, W]，实际为 {img_dim!r}")
    inp["img_dim"] = [int(v) for v in img_dim]

    for key in ("mean", "std"):
        value = _fetch(inp, key, ctx, _MISSING)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"input.{key} 必须是长度为 3 的列表")
        inp[key] = [float(v) for v in value]
    if any(v <= 0 for v in inp["std"]):
        raise ValueError("input.std 必须全部大于 0")

    img_interp = _str(inp, "img_interp", ctx, default="linear")
    resolve_interp(img_interp)

    if "letterbox" in inp and bool(inp.get("letterbox", False)):
        raise ValueError("input.letterbox 当前版本不支持，必须为 false")

    aug = _section(inp, "train_aug", optional=True)
    actx = "input.train_aug"
    _bool(aug, "enabled", actx, default=True)
    _float(aug, "max_crop_frac", actx, default=0.05, lo=0.0, hi=0.5, hi_open=True)
    _float(aug, "hflip_p", actx, default=0.5, lo=0.0, hi=1.0)
    _float(aug, "color_jitter_p", actx, default=0.8, lo=0.0, hi=1.0)
    _float(aug, "brightness", actx, default=0.3, lo=0.0, hi=1.0, hi_open=True)
    _float(aug, "contrast", actx, default=0.2, lo=0.0, hi=1.0, hi_open=True)
    _float(aug, "saturation", actx, default=0.15, lo=0.0, hi=1.0)
    _float(aug, "hue", actx, default=0.02, lo=0.0, hi=0.5)


def _validate_sampler(sampler: dict[str, Any]) -> None:
    ctx = "sampler"
    _choice(sampler, "type", ctx, ("pk", "random"), default="pk")
    _int(sampler, "p_groups", ctx, default=4, lo=1)
    _float(sampler, "non_oven_ratio", ctx, default=0.25, lo=0.0, hi=1.0, hi_open=True)


def _validate_loss(loss: dict[str, Any]) -> None:
    ctx = "loss"
    weights = _section(loss, "weights")
    extra = sorted(set(weights) - set(LOSS_TERMS))
    if extra:
        raise ValueError(f"loss.weights 中有未知项：{extra}，可选：{list(LOSS_TERMS)}")
    for term in LOSS_TERMS:
        _float(weights, term, "loss.weights", lo=0.0)
    if not any(weights[term] > 0 for term in LOSS_TERMS):
        raise ValueError("loss.weights 至少要有一项大于 0")

    _choice(loss, "metric", ctx, ("supcon", "arcface"), default="supcon")
    _float(loss, "supcon_temperature", ctx, default=0.1, lo=0.0, lo_open=True)
    _float(loss, "arcface_scale", ctx, default=30.0, lo=0.0, lo_open=True)
    _float(loss, "arcface_margin", ctx, default=0.3, lo=0.0, hi=math.pi / 2, hi_open=True)
    _choice(loss, "multilabel", ctx, ("bce", "focal"), default="bce")
    _float(loss, "focal_gamma", ctx, default=2.0, lo=0.0)
    _choice(loss, "pos_weight", ctx, ("auto", "none"), default="auto")
    _float(loss, "pos_weight_max", ctx, default=10.0, lo=1.0)
    _float(loss, "rack_smoothing", ctx, default=0.0, lo=0.0, hi=1.0, hi_open=True)


def _validate_trainparams(tp: dict[str, Any]) -> None:
    ctx = "trainparams"
    _int(tp, "epochs", ctx, lo=1)
    _int(tp, "batch_size", ctx, lo=2)  # SupCon 至少需要 2 个样本
    _int(tp, "steps_per_epoch", ctx, default=0, lo=0)
    _int(tp, "num_workers", ctx, default=4, lo=0)
    _int(tp, "num_workers_eval", ctx, default=2, lo=0)
    lr = _float(tp, "lr", ctx, lo=0.0, lo_open=True)
    _float(tp, "lr_lora", ctx, default=1.0e-4, lo=0.0, lo_open=True)
    _float(tp, "weight_decay", ctx, default=0.05, lo=0.0)
    min_lr = _float(tp, "min_lr", ctx, default=0.0, lo=0.0)
    if min_lr > lr:
        raise ValueError(f"trainparams.min_lr（{min_lr}）不能大于 trainparams.lr（{lr}）")
    _int(tp, "warmup_steps", ctx, default=0, lo=0)
    _bool(tp, "use_amp", ctx, default=True)
    _float(tp, "grad_clip", ctx, default=1.0)
    _int(tp, "log_every", ctx, default=10, lo=1)
    _int(tp, "seed", ctx, default=42)


def _validate_eval(ev: dict[str, Any]) -> None:
    ctx = "eval"
    _int(ev, "gallery_max_per_model", ctx, default=200, lo=1)
    _choice(ev, "gallery_feature", ctx, ("proj", "cls"), default="proj")
    _int(ev, "knn_k", ctx, default=10, lo=1)
    _float(ev, "tau", ctx, default=0.5, lo=-1.0, hi=1.0)
    _float(ev, "default_threshold", ctx, default=0.5, lo=0.0, hi=1.0, lo_open=True, hi_open=True)
    _float(ev, "safety_recall_target", ctx, default=0.95, lo=0.0, hi=1.0, lo_open=True)
    _float(ev, "rack_target_precision", ctx, default=0.95, lo=0.0, hi=1.0, lo_open=True)

    weights = _section(ev, "score_weights", optional=True)
    if not weights:
        weights.update({"rack_acc": 1.0, "accessory_map": 1.0, "container_map": 1.0, "device_top1": 1.0})
    unknown = sorted(set(weights) - SCORE_METRICS)
    if unknown:
        raise ValueError(f"eval.score_weights 中有未知指标：{unknown}，可选：{sorted(SCORE_METRICS)}")
    for key in list(weights):
        _float(weights, key, "eval.score_weights", lo=0.0)
    if not any(v > 0 for v in weights.values()):
        raise ValueError("eval.score_weights 至少要有一项大于 0")
    _bool(ev, "run_test", ctx, default=True)
    _bool(ev, "run_stage2", ctx, default=True)


def _validate_loo(loo: dict[str, Any]) -> None:
    ctx = "loo"
    _int(loo, "epochs", ctx, default=0, lo=0)
    _float(loo, "holdout_frac", ctx, default=0.3, lo=0.0, hi=1.0, lo_open=True, hi_open=True)
    _int(loo, "min_group_images", ctx, default=4, lo=2)


def validate_config(cfg: dict[str, Any]) -> None:
    _validate_model(_section(cfg, "model"))
    _validate_labels(_section(cfg, "labels"))
    _str(cfg, "device_profile", "config", default="configs/device_profile.json")
    _validate_data(_section(cfg, "data"))
    _validate_input(_section(cfg, "input"))
    _validate_sampler(_section(cfg, "sampler", optional=True))
    _validate_loss(_section(cfg, "loss"))
    _validate_trainparams(_section(cfg, "trainparams"))
    _validate_eval(_section(cfg, "eval", optional=True))
    _validate_loo(_section(cfg, "loo", optional=True))
    _str(_section(cfg, "output", optional=True), "root", "output", default="output/oven")


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
    从配置中读取 DINO 本地仓库与预训练权重路径（相对路径以项目根目录为基准）。
    - model.dino_local_repo: DINO 本地仓库路径（默认 third_party/dinov3）
    - model.weight_path:    预训练权重路径（默认 data/model/xxx.pth）
    """
    model = _section(cfg, "model")
    dino_local_repo = _str(model, "dino_local_repo", "model", default="third_party/dinov3")
    weight_path = _str(model, "weight_path", "model", default="data/model/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    return str(resolve_path(dino_local_repo)), str(resolve_path(weight_path))


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
    cfg = load_config(default_config_path())
    print_config(cfg)


if __name__ == "__main__":
    main()
