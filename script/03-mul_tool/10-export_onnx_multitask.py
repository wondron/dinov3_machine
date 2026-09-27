from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dino_finetune.config import load_config, default_config_path, get_dino_paths
from dino_finetune.utils.ckpt_cls import build_encoder
from dino_finetune.utils.label_mapping import normalize_label_mapping, save_label_mapping
from dino_finetune import DINOEncoderLoRA
from dino_finetune.model.dino_multitask import DINOEncoderLoRA_MultiTask

logger = logging.getLogger("export_onnx_multitask")


def pick_model_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint 类型错误，期望 dict，实际为 {type(ckpt).__name__}")

    for key in ("model", "model_state_dict", "state_dict"):
        value = ckpt.get(key)
        if isinstance(value, dict) and value:
            return normalize_state_dict_keys(value)

    if ckpt and all(isinstance(value, torch.Tensor) for value in ckpt.values()):
        return normalize_state_dict_keys(ckpt)

    raise ValueError("checkpoint 中未找到模型权重，请使用多任务训练保存的 ckpt_best.pt 或 ckpt_last.pt")


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """兼容 DataParallel/DDP 保存的 module. 前缀。"""
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def infer_num_classes_cls(
    cfg: dict,
    ckpt: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
) -> int:
    candidates: dict[str, int] = {}

    weight = state_dict.get("cls_head.weight")
    if hasattr(weight, "shape") and len(weight.shape) == 2 and int(weight.shape[0]) > 0:
        candidates["cls_head.weight"] = int(weight.shape[0])

    checkpoint_value = int(ckpt.get("num_classes_cls", 0) or 0)
    if checkpoint_value > 0:
        candidates["checkpoint.num_classes_cls"] = checkpoint_value

    config_value = int((cfg.get("model_cls", {}) or {}).get("num_classes", 0) or 0)
    if config_value > 0:
        candidates["config.model_cls.num_classes"] = config_value

    if not candidates:
        raise ValueError(
            "无法确定分类类别数：checkpoint 缺少 cls_head.weight/num_classes_cls，"
            "且 config.model_cls.num_classes 未设置"
        )

    unique_values = set(candidates.values())
    if len(unique_values) != 1:
        details = "，".join(f"{key}={value}" for key, value in candidates.items())
        raise ValueError(f"分类类别数不一致：{details}")
    return next(iter(unique_values))


def get_input_hw(cfg: dict, section: str) -> tuple[int, int]:
    values = cfg[section]["img_dim"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError(f"{section}.img_dim 必须由两个正整数组成，实际为 {values}")
    height, width = values
    return height, width


def get_nested_value(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def validate_checkpoint_config(cfg: dict, ckpt: dict[str, Any]) -> None:
    saved_cfg = ckpt.get("cfg")
    if not isinstance(saved_cfg, dict):
        logger.warning("checkpoint 未保存训练配置，跳过结构配置一致性检查")
        return

    fields = (
        ("input.img_dim", ("input", "img_dim")),
        ("input_cls.img_dim", ("input_cls", "img_dim")),
        ("model.dino_type", ("model", "dino_type")),
        ("model.size", ("model", "size")),
        ("model.n_classes", ("model", "n_classes")),
        ("model_cls.pool", ("model_cls", "pool")),
        ("trainparams.rank_r", ("trainparams", "rank_r")),
        ("trainparams.use_lora", ("trainparams", "use_lora")),
        ("trainparams.use_fpn", ("trainparams", "use_fpn")),
    )
    mismatches: list[str] = []
    for label, keys in fields:
        current = get_nested_value(cfg, keys)
        saved = get_nested_value(saved_cfg, keys)
        if current is None or saved is None:
            continue
        if isinstance(current, (list, tuple)) and isinstance(saved, (list, tuple)):
            equal = tuple(current) == tuple(saved)
        else:
            equal = current == saved
        if not equal:
            mismatches.append(f"{label}：当前={current}，训练时={saved}")

    if mismatches:
        raise ValueError("当前配置与 checkpoint 训练配置不一致：\n- " + "\n- ".join(mismatches))


def prepare_label_mapping(
    ckpt: dict[str, Any],
    ckpt_path: Path,
    mapping_arg: str,
    num_classes: int,
) -> dict[str, Any]:
    mapping_data: Any = ckpt
    mapping_source = "checkpoint 内置类别字段"

    if mapping_arg:
        mapping_path = Path(mapping_arg).expanduser().resolve()
        if not mapping_path.is_file():
            raise FileNotFoundError(f"类别映射文件不存在：{mapping_path}")
        try:
            mapping_data = json.loads(mapping_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"类别映射读取失败：{mapping_path}，原因：{exc}") from exc
        mapping_source = str(mapping_path)
    elif not any(key in ckpt for key in ("class_names", "class_to_idx", "idx_to_class_name")):
        mapping_path = ckpt_path.parent / "leaf_id_map.json"
        if not mapping_path.is_file():
            raise FileNotFoundError(
                "checkpoint 中没有类别名称映射，且同目录不存在 leaf_id_map.json；"
                "请使用 --mapping 指定类别映射文件"
            )
        try:
            mapping_data = json.loads(mapping_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"类别映射读取失败：{mapping_path}，原因：{exc}") from exc
        mapping_source = str(mapping_path)

    checkpoint_class_names = ckpt.get("class_names")
    if not isinstance(checkpoint_class_names, (list, tuple)):
        checkpoint_class_names = None

    if isinstance(mapping_data, dict) and (
        "raw_to_safe" in mapping_data or "safe_to_raw" in mapping_data
    ):
        mapping_data = dict(mapping_data)
        if "leaf_id_to_idx" not in mapping_data and isinstance(ckpt.get("leaf_id_to_idx"), dict):
            mapping_data["leaf_id_to_idx"] = ckpt["leaf_id_to_idx"]

    mapping = normalize_label_mapping(
        mapping_data,
        expected_num_classes=num_classes,
        model_class_names=checkpoint_class_names,
    )
    logger.info("类别映射整理完成：来源=%s；类别数=%d", mapping_source, num_classes)
    return mapping


class SegOnly(nn.Module):
    def __init__(self, mt: DINOEncoderLoRA_MultiTask):
        super().__init__()
        self.mt = mt

    def forward(self, x640: torch.Tensor) -> torch.Tensor:
        return self.mt.forward_seg(x640)


class ClsOnly(nn.Module):
    def __init__(self, mt: DINOEncoderLoRA_MultiTask):
        super().__init__()
        self.mt = mt

    def forward(self, x224: torch.Tensor):
        logits, emb = self.mt.forward_cls(x224)
        return logits, emb


class PatchTokensOnly(nn.Module):
    def __init__(self, mt: DINOEncoderLoRA_MultiTask):
        super().__init__()
        self.mt = mt

    def forward(self, x640: torch.Tensor) -> torch.Tensor:
        # 你在 dino_multitask.py 里加的 forward_patchtokens
        return self.mt.forward_patchtokens(x640)


def build_model(
    cfg: dict,
    ckpt_path: Path,
    device: torch.device,
    mapping_arg: str,
) -> tuple[DINOEncoderLoRA_MultiTask, dict[str, Any]]:
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"checkpoint 读取失败：{ckpt_path}，原因：{exc}") from exc
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint 类型错误，期望 dict，实际为 {type(ckpt).__name__}")

    state_dict = pick_model_state_dict(ckpt)
    validate_checkpoint_config(cfg, ckpt)
    num_classes_cls = infer_num_classes_cls(cfg, ckpt, state_dict)
    label_mapping = prepare_label_mapping(ckpt, ckpt_path, mapping_arg, num_classes_cls)

    dino_local_repo, weight_path = get_dino_paths(cfg)
    encoder = build_encoder(cfg, device, dino_local_repo=dino_local_repo, weight_path=weight_path)

    train_cfg = cfg["trainparams"]
    model_cfg = cfg["model"]
    seg_img_dim = get_input_hw(cfg, "input")

    n_classes_seg = int(model_cfg["n_classes"])
    emb_dim = int(getattr(encoder, "num_features", 0) or 0)
    if emb_dim <= 0:
        raise RuntimeError("无法从 encoder 获取 num_features")

    seg_model = DINOEncoderLoRA(
        encoder=encoder,
        r=int(train_cfg["rank_r"]),
        emb_dim=emb_dim,
        img_dim=seg_img_dim,
        n_classes=n_classes_seg,
        use_lora=bool(train_cfg["use_lora"]),
        use_fpn=bool(train_cfg["use_fpn"]),
    ).to(device)

    mt = DINOEncoderLoRA_MultiTask(
        seg_model=seg_model,
        num_classes_cls=num_classes_cls,
        pool=str(cfg["model_cls"]["pool"]),
        emb_dim=emb_dim,
    ).to(device)

    try:
        missing, unexpected = mt.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(f"checkpoint 权重形状与多任务模型不匹配：{exc}") from exc
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint 权重不完整，已停止导出："
            f"缺失参数={missing[:20]}，多余参数={unexpected[:20]}"
        )

    logger.info( "多任务权重加载完成：分割类别=%d，分类类别=%d，embedding 维度=%d", n_classes_seg, num_classes_cls, emb_dim,)
    mt.eval()
    return mt, label_mapping


def export_one(
    model: nn.Module,
    out_path: Path,
    dummy_input: torch.Tensor,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None,
    opset: int = 18,
    check_onnx: bool = True,
) -> None:
    logger.info("开始导出：%s", out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{out_path.stem}.",
        suffix=".tmp.onnx",
        dir=out_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(temporary_path),
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

        if not temporary_path.is_file() or temporary_path.stat().st_size <= 0:
            raise RuntimeError(f"ONNX 导出后文件不存在或为空：{out_path}")

        if check_onnx:
            import onnx

            onnx.checker.check_model(str(temporary_path))

        temporary_path.replace(out_path)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"ONNX 导出或完整性检查失败：{out_path}，原因：{exc}") from exc

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("导出完成：%s（%.2f MB）", out_path, size_mb)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境不可用")
    return torch.device(name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ap = argparse.ArgumentParser(description="导出多任务模型的分割、分类和 patch token ONNX")
    ap.add_argument("--config", default=default_config_path("mul"), help="多任务 YAML 配置文件")
    ap.add_argument("--ckpt", required=True, help="多任务训练生成的 ckpt_best.pt 或 ckpt_last.pt")
    ap.add_argument("--mapping", default="", help="可选类别映射 JSON，默认读取 checkpoint 内置类别字段")
    ap.add_argument("--outdir", default=None, help="输出目录，默认使用 checkpoint 所在目录")
    ap.add_argument("--opset", type=int, default=18, help="ONNX opset 版本")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="导出设备")
    ap.add_argument("--skip_onnx_check", action="store_true", help="跳过导出后的 ONNX 完整性检查")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt_path}")
    if args.opset <= 0:
        raise ValueError(f"opset 必须是正整数，实际为 {args.opset}")

    outdir = ckpt_path.parent / "onnx"
    
    if outdir.exists():
        import shutil
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=False)

    device = resolve_device(args.device)
    logger.info("使用设备：%s", device)

    seg_height, seg_width = get_input_hw(cfg, "input")
    cls_height, cls_width = get_input_hw(cfg, "input_cls")
    logger.info(
        "导出输入尺寸：分割/patch token=%dx%d，分类=%dx%d",
        seg_height,
        seg_width,
        cls_height,
        cls_width,
    )
    mt, label_mapping = build_model(cfg, ckpt_path, device=device, mapping_arg=args.mapping)
    check_onnx = not args.skip_onnx_check

    # 输入名称保持兼容现有对齐脚本；实际尺寸全部来自 config。
    seg_onnx = outdir / "multitask_seg.onnx"
    seg_model = SegOnly(mt).to(device).eval()
    x640 = torch.zeros(1, 3, seg_height, seg_width, device=device)
    export_one(
        seg_model,
        seg_onnx,
        x640,
        input_names=["x640"],
        output_names=["seg_logits"],
        dynamic_axes={"x640": {0: "B"}, "seg_logits": {0: "B"}},
        opset=args.opset,
        check_onnx=check_onnx,
    )

    cls_onnx = outdir / "multitask_cls.onnx"
    cls_model = ClsOnly(mt).to(device).eval()
    x224 = torch.zeros(1, 3, cls_height, cls_width, device=device)
    export_one(
        cls_model,
        cls_onnx,
        x224,
        input_names=["x224"],
        output_names=["cls_logits", "cls_emb"],
        dynamic_axes={
            "x224": {0: "B"},
            "cls_logits": {0: "B"},
            "cls_emb": {0: "B"},
        },
        opset=args.opset,
        check_onnx=check_onnx,
    )
    mapping_output_path = cls_onnx.parent / "leaf_id_map.json"
    save_label_mapping(label_mapping, mapping_output_path)
    logger.info("类别映射已保存：%s", mapping_output_path)

    tok_onnx = outdir / "multitask_patchtokens.onnx"
    tok_model = PatchTokensOnly(mt).to(device).eval()
    export_one(
        tok_model,
        tok_onnx,
        x640,
        input_names=["x640"],
        output_names=["patch_tokens"],
        dynamic_axes={"x640": {0: "B"}, "patch_tokens": {0: "B"}},
        opset=args.opset,
        check_onnx=check_onnx,
    )

    logger.info("全部导出完成，输出目录：%s", outdir)
    logger.info("分割模型：%s", seg_onnx)
    logger.info("分类模型：%s", cls_onnx)
    logger.info("Patch token 模型：%s", tok_onnx)


if __name__ == "__main__":
    main()
