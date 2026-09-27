from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError("未安装 matplotlib，请先执行：pip install matplotlib") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dino_finetune import DINOEncoderLoRA
from dino_finetune.config import default_config_path, get_dino_paths, load_config
from dino_finetune.model.dino_multitask import DINOEncoderLoRA_MultiTask
from dino_finetune.utils.ckpt_cls import build_encoder


LOGGER = logging.getLogger("compare_pt_onnx_multitask")


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """兼容 DataParallel/DDP 保存的 module. 前缀。"""
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def pick_model_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint 类型错误，期望 dict，实际为 {type(checkpoint).__name__}")

    for key in ("model", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            return normalize_state_dict_keys(value)

    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return normalize_state_dict_keys(checkpoint)

    raise ValueError("checkpoint 中未找到模型权重，请使用多任务训练保存的 ckpt_best.pt 或 ckpt_last.pt")


def infer_num_classes_cls(
    cfg: dict,
    checkpoint: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
) -> int:
    candidates: dict[str, int] = {}

    weight = state_dict.get("cls_head.weight")
    if hasattr(weight, "shape") and len(weight.shape) == 2 and int(weight.shape[0]) > 0:
        candidates["cls_head.weight"] = int(weight.shape[0])

    checkpoint_value = int(checkpoint.get("num_classes_cls", 0) or 0)
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
    if len(set(candidates.values())) != 1:
        details = "，".join(f"{key}={value}" for key, value in candidates.items())
        raise ValueError(f"分类类别数不一致：{details}")
    return next(iter(candidates.values()))


def get_input_hw(cfg: dict, section: str) -> tuple[int, int]:
    values = cfg[section]["img_dim"]
    if (
        not isinstance(values, (list, tuple))
        or len(values) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values)
    ):
        raise ValueError(f"{section}.img_dim 必须由两个正整数组成，实际为 {values}")
    return int(values[0]), int(values[1])


def get_nested_value(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def validate_checkpoint_config(cfg: dict, checkpoint: dict[str, Any]) -> None:
    saved_cfg = checkpoint.get("cfg")
    if not isinstance(saved_cfg, dict):
        LOGGER.warning("checkpoint 未保存训练配置，跳过结构配置一致性检查")
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


def configure_torch_consistency() -> None:
    """尽量减少 PyTorch CUDA 与 ONNXRuntime CUDA 的非必要数值差异。"""
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_multitask_model(
    cfg: dict,
    checkpoint_path: Path,
    device: torch.device,
) -> DINOEncoderLoRA_MultiTask:
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"checkpoint 读取失败：{checkpoint_path}，原因：{exc}") from exc
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint 类型错误，期望 dict，实际为 {type(checkpoint).__name__}")

    state_dict = pick_model_state_dict(checkpoint)
    validate_checkpoint_config(cfg, checkpoint)
    num_classes_cls = infer_num_classes_cls(cfg, checkpoint, state_dict)

    dino_local_repo, weight_path = get_dino_paths(cfg)
    encoder = build_encoder(
        cfg,
        device,
        dino_local_repo=dino_local_repo,
        weight_path=weight_path,
    )
    emb_dim = int(getattr(encoder, "num_features", 0) or 0)
    if emb_dim <= 0:
        raise RuntimeError("无法从 encoder 获取 num_features")

    train_cfg = cfg["trainparams"]
    seg_model = DINOEncoderLoRA(
        encoder=encoder,
        r=int(train_cfg["rank_r"]),
        emb_dim=emb_dim,
        img_dim=get_input_hw(cfg, "input"),
        n_classes=int(cfg["model"]["n_classes"]),
        use_lora=bool(train_cfg["use_lora"]),
        use_fpn=bool(train_cfg["use_fpn"]),
    ).to(device)
    model = DINOEncoderLoRA_MultiTask(
        seg_model=seg_model,
        num_classes_cls=num_classes_cls,
        pool=str(cfg["model_cls"]["pool"]),
        emb_dim=emb_dim,
    ).to(device)

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(f"checkpoint 权重形状与多任务模型不匹配：{exc}") from exc
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint 权重不完整，已停止对齐验证："
            f"缺失参数={missing[:20]}，多余参数={unexpected[:20]}"
        )

    LOGGER.info(
        "多任务权重加载完成：分割类别=%d，分类类别=%d，embedding 维度=%d",
        int(cfg["model"]["n_classes"]),
        num_classes_cls,
        emb_dim,
    )
    model.eval()
    return model


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境不可用")
    return torch.device(name)


def select_providers(provider: str, pt_device: torch.device) -> list[str]:
    available = ort.get_available_providers()
    if provider == "auto":
        if pt_device.type == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CPUExecutionProvider" in available:
            return ["CPUExecutionProvider"]
        raise RuntimeError(f"没有可用的 ONNXRuntime provider：{available}")

    provider_name = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
    }[provider]
    if provider_name not in available:
        raise RuntimeError(f"指定的 ONNXRuntime provider 不可用：{provider_name}；当前可用={available}")
    providers = [provider_name]
    if provider_name != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers


def create_session(path: Path, providers: list[str]) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    except Exception as exc:
        raise RuntimeError(f"ONNXRuntime 会话创建失败：{path}，原因：{exc}") from exc
    LOGGER.info("加载 ONNX：%s；实际 providers=%s", path, session.get_providers())
    return session


def validate_session_contract(
    session: ort.InferenceSession,
    expected_hw: tuple[int, int],
    expected_outputs: tuple[str, ...],
    model_name: str,
) -> str:
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"{model_name} ONNX 输入数量错误：期望 1，实际 {len(inputs)}")
    model_input = inputs[0]
    if model_input.type != "tensor(float)":
        raise ValueError(f"{model_name} ONNX 输入类型错误：期望 tensor(float)，实际 {model_input.type}")

    shape = model_input.shape
    if len(shape) != 4:
        raise ValueError(f"{model_name} ONNX 输入维度错误：期望 NCHW，实际 {shape}")
    if isinstance(shape[1], int) and shape[1] != 3:
        raise ValueError(f"{model_name} ONNX 输入通道错误：期望 3，实际 {shape[1]}")
    for actual, expected, axis_name in zip(shape[-2:], expected_hw, ("高度", "宽度")):
        if isinstance(actual, int) and actual != expected:
            raise ValueError(
                f"{model_name} ONNX 输入{axis_name}与 config 不一致：ONNX={actual}，config={expected}"
            )

    available_outputs = {output.name for output in session.get_outputs()}
    missing_outputs = [name for name in expected_outputs if name not in available_outputs]
    if missing_outputs:
        raise ValueError(
            f"{model_name} ONNX 缺少输出：{missing_outputs}；实际输出={sorted(available_outputs)}"
        )
    return model_input.name


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def make_random_input(
    rng: np.random.Generator,
    batch: int,
    hw: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """先在 NumPy/CPU 生成，确保切换 PT 设备时仍使用完全相同的随机输入。"""
    array = rng.standard_normal((batch, 3, *hw)).astype(np.float32)
    return torch.from_numpy(array).to(device=device, dtype=torch.float32)


def compare_arrays(
    pt_value: np.ndarray,
    onnx_value: np.ndarray,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    pt_value = np.asarray(pt_value, dtype=np.float32)
    onnx_value = np.asarray(onnx_value, dtype=np.float32)
    report: dict[str, Any] = {
        "pt_shape": list(pt_value.shape),
        "onnx_shape": list(onnx_value.shape),
        "atol": float(atol),
        "rtol": float(rtol),
    }

    if pt_value.shape != onnx_value.shape:
        report.update({"passed": False, "allclose_passed": False, "reason": "shape 不一致"})
        return report
    if pt_value.size == 0:
        report.update({"passed": False, "allclose_passed": False, "reason": "输出为空"})
        return report

    pt_finite = bool(np.isfinite(pt_value).all())
    onnx_finite = bool(np.isfinite(onnx_value).all())
    if not pt_finite or not onnx_finite:
        report.update(
            {
                "passed": False,
                "allclose_passed": False,
                "reason": "输出包含 NaN 或 Inf",
                "pt_finite": pt_finite,
                "onnx_finite": onnx_finite,
            }
        )
        return report

    difference = np.abs(pt_value - onnx_value)
    scale = np.maximum(np.abs(pt_value), np.abs(onnx_value))
    relative = difference / np.maximum(scale, 1e-12)
    close = difference <= (atol + rtol * np.abs(onnx_value))
    mismatch_count = int(np.count_nonzero(~close))
    allclose_passed = mismatch_count == 0
    report.update(
        {
            "passed": allclose_passed,
            "allclose_passed": allclose_passed,
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "p99_abs": float(np.percentile(difference, 99.0)),
            "max_rel": float(relative.max()),
            "mean_rel": float(relative.mean()),
            "mismatch_count": mismatch_count,
            "mismatch_ratio": float(mismatch_count / difference.size),
        }
    )
    return report


def cosine_values(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(-1, a.shape[-1])
    b = np.asarray(b, dtype=np.float32).reshape(-1, b.shape[-1])
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denominator = np.maximum(denominator, eps)
    return np.sum(a * b, axis=1) / denominator


def build_cosine_report(
    pt_value: np.ndarray,
    onnx_value: np.ndarray,
    min_cosine: float,
) -> dict[str, Any]:
    if pt_value.shape != onnx_value.shape or pt_value.size == 0:
        return {"passed": False, "reason": "shape 不一致或输出为空"}
    cosine = cosine_values(pt_value, onnx_value)
    finite = bool(np.isfinite(cosine).all())
    minimum = float(cosine.min()) if cosine.size else float("nan")
    return {
        "passed": bool(finite and minimum >= min_cosine),
        "minimum": minimum,
        "mean": float(cosine.mean()),
        "maximum": float(cosine.max()),
        "required_minimum": float(min_cosine),
        "sample_count": int(cosine.size),
    }


def softmax_numpy(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp, axis=axis, keepdims=True), 1e-12)


def masks_from_logits(
    logits: np.ndarray,
    mask_mode: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if logits.ndim != 4:
        raise ValueError(f"分割 logits 必须为 NCHW 四维数组，实际 shape={logits.shape}")
    probs = softmax_numpy(logits, axis=1)
    n_classes = int(logits.shape[1])

    if mask_mode == "prob_threshold":
        if n_classes != 2:
            raise ValueError("postprocess.mask_mode=prob_threshold 仅支持二分类分割")
        mask = (probs[:, 1] >= threshold).astype(np.int64)
        return mask, probs
    if mask_mode == "argmax":
        return np.argmax(probs, axis=1).astype(np.int64), probs
    raise ValueError(f"不支持的 postprocess.mask_mode：{mask_mode}")


def compare_masks(
    pt_mask: np.ndarray,
    onnx_mask: np.ndarray,
    n_classes: int,
    min_pixel_agreement: float,
    min_iou: float,
    min_dice: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "pt_shape": list(pt_mask.shape),
        "onnx_shape": list(onnx_mask.shape),
        "required_min_pixel_agreement": float(min_pixel_agreement),
        "required_min_foreground_iou": float(min_iou),
        "required_min_foreground_dice": float(min_dice),
    }
    if pt_mask.shape != onnx_mask.shape or pt_mask.size == 0:
        report.update({"passed": False, "reason": "mask shape 不一致或为空"})
        return report

    pixel_agreement = float(np.mean(pt_mask == onnx_mask))
    batch_class_metrics: list[dict[str, Any]] = []
    foreground_ious: list[float] = []
    foreground_dices: list[float] = []

    for batch_index in range(pt_mask.shape[0]):
        for class_id in range(n_classes):
            pt_selected = pt_mask[batch_index] == class_id
            onnx_selected = onnx_mask[batch_index] == class_id
            intersection = int(np.logical_and(pt_selected, onnx_selected).sum())
            union = int(np.logical_or(pt_selected, onnx_selected).sum())
            pt_pixels = int(pt_selected.sum())
            onnx_pixels = int(onnx_selected.sum())
            total = pt_pixels + onnx_pixels
            iou = 1.0 if union == 0 else intersection / union
            dice = 1.0 if total == 0 else (2.0 * intersection) / total
            batch_class_metrics.append(
                {
                    "batch_index": int(batch_index),
                    "class_id": int(class_id),
                    "intersection": intersection,
                    "union": union,
                    "pt_pixels": pt_pixels,
                    "onnx_pixels": onnx_pixels,
                    "iou": float(iou),
                    "dice": float(dice),
                }
            )
            if class_id > 0:
                foreground_ious.append(float(iou))
                foreground_dices.append(float(dice))

    if not foreground_ious:
        foreground_ious = [pixel_agreement]
        foreground_dices = [pixel_agreement]

    foreground_iou_mean = float(np.mean(foreground_ious))
    foreground_iou_minimum = float(np.min(foreground_ious))
    foreground_dice_mean = float(np.mean(foreground_dices))
    foreground_dice_minimum = float(np.min(foreground_dices))
    passed = bool(
        pixel_agreement >= min_pixel_agreement
        and foreground_iou_mean >= min_iou
        and foreground_dice_mean >= min_dice
    )
    report.update(
        {
            "passed": passed,
            "pixel_agreement": pixel_agreement,
            "pixel_mismatch_count": int(np.count_nonzero(pt_mask != onnx_mask)),
            "pixel_mismatch_ratio": float(1.0 - pixel_agreement),
            "foreground_iou_mean": foreground_iou_mean,
            "foreground_iou_minimum": foreground_iou_minimum,
            "foreground_dice_mean": foreground_dice_mean,
            "foreground_dice_minimum": foreground_dice_minimum,
            "class_metrics": batch_class_metrics,
        }
    )
    return report


def build_topk_report(
    pt_logits: np.ndarray,
    onnx_logits: np.ndarray,
    max_k: int = 5,
) -> dict[str, Any]:
    if pt_logits.shape != onnx_logits.shape or pt_logits.ndim != 2 or pt_logits.size == 0:
        return {"passed": False, "reason": "分类 logits shape 不一致、不是二维或为空"}

    class_count = int(pt_logits.shape[1])
    top_k = min(max_k, class_count)
    pt_top = np.argsort(-pt_logits, axis=1)[:, :top_k]
    onnx_top = np.argsort(-onnx_logits, axis=1)[:, :top_k]
    top1_matches = pt_top[:, 0] == onnx_top[:, 0]
    topk_order_matches = np.all(pt_top == onnx_top, axis=1)
    topk_set_matches = np.asarray(
        [set(pt_row.tolist()) == set(onnx_row.tolist()) for pt_row, onnx_row in zip(pt_top, onnx_top)],
        dtype=bool,
    )

    return {
        "passed": bool(np.all(top1_matches) and np.all(topk_set_matches)),
        "top_k": int(top_k),
        "top1_all_match": bool(np.all(top1_matches)),
        "top1_match_ratio": float(np.mean(top1_matches)),
        "topk_set_all_match": bool(np.all(topk_set_matches)),
        "topk_set_match_ratio": float(np.mean(topk_set_matches)),
        "topk_order_all_match": bool(np.all(topk_order_matches)),
        "topk_order_match_ratio": float(np.mean(topk_order_matches)),
        "pt_top_indices": pt_top.tolist(),
        "onnx_top_indices": onnx_top.tolist(),
    }


def compare_segmentation(
    model: DINOEncoderLoRA_MultiTask,
    session: ort.InferenceSession,
    input_name: str,
    input_tensor: torch.Tensor,
    mask_mode: str,
    threshold: float,
    atol: float,
    rtol: float,
    max_abs_limit: float,
    mean_abs_limit: float,
    min_pixel_agreement: float,
    min_iou: float,
    min_dice: float,
) -> dict[str, Any]:
    with torch.inference_mode():
        pt_logits = tensor_to_numpy(model.forward_seg(input_tensor))
    input_numpy = tensor_to_numpy(input_tensor)
    onnx_logits = session.run(["seg_logits"], {input_name: input_numpy})[0]

    logits_report = compare_arrays(pt_logits, onnx_logits, atol=atol, rtol=rtol)
    logits_limits_passed = bool(
        "max_abs" in logits_report
        and logits_report["max_abs"] <= max_abs_limit
        and logits_report["mean_abs"] <= mean_abs_limit
    )
    logits_report["limits"] = {
        "passed": logits_limits_passed,
        "required_max_abs": float(max_abs_limit),
        "required_mean_abs": float(mean_abs_limit),
    }
    # 分割 logits 的逐元素 allclose 仅作为诊断；最终由误差上限和 mask 结果共同判定。
    logits_report["passed"] = logits_limits_passed

    if pt_logits.shape != onnx_logits.shape or pt_logits.ndim != 4:
        return {
            "passed": False,
            "mask_mode": mask_mode,
            "threshold": float(threshold),
            "logits": logits_report,
            "mask": {"passed": False, "reason": "分割 logits shape 不一致或不是 NCHW"},
        }

    pt_mask, pt_probs = masks_from_logits(pt_logits, mask_mode, threshold)
    onnx_mask, onnx_probs = masks_from_logits(onnx_logits, mask_mode, threshold)
    n_classes = int(pt_logits.shape[1])
    mask_report = compare_masks(
        pt_mask,
        onnx_mask,
        n_classes=n_classes,
        min_pixel_agreement=min_pixel_agreement,
        min_iou=min_iou,
        min_dice=min_dice,
    )
    probability_report = compare_arrays(pt_probs, onnx_probs, atol=atol, rtol=rtol)

    argmax_report: dict[str, Any] | None = None
    if mask_mode != "argmax":
        pt_argmax = np.argmax(pt_probs, axis=1).astype(np.int64)
        onnx_argmax = np.argmax(onnx_probs, axis=1).astype(np.int64)
        argmax_report = compare_masks(
            pt_argmax,
            onnx_argmax,
            n_classes=n_classes,
            min_pixel_agreement=min_pixel_agreement,
            min_iou=min_iou,
            min_dice=min_dice,
        )

    report: dict[str, Any] = {
        "passed": bool(logits_limits_passed and mask_report["passed"]),
        "mask_mode": mask_mode,
        "threshold": float(threshold),
        "logits": logits_report,
        "probabilities": probability_report,
        "mask": mask_report,
    }
    if argmax_report is not None:
        report["argmax_mask"] = argmax_report
    return report


def compare_classification(
    model: DINOEncoderLoRA_MultiTask,
    session: ort.InferenceSession,
    input_name: str,
    input_tensor: torch.Tensor,
    logits_atol: float,
    logits_rtol: float,
    logits_max_abs: float,
    emb_atol: float,
    emb_rtol: float,
    min_cosine: float,
) -> dict[str, Any]:
    with torch.inference_mode():
        pt_logits_tensor, pt_embedding_tensor = model.forward_cls(input_tensor)
    pt_logits = tensor_to_numpy(pt_logits_tensor)
    pt_embedding = tensor_to_numpy(pt_embedding_tensor)
    input_numpy = tensor_to_numpy(input_tensor)
    onnx_logits, onnx_embedding = session.run(
        ["cls_logits", "cls_emb"],
        {input_name: input_numpy},
    )

    logits_report = compare_arrays(pt_logits, onnx_logits, atol=logits_atol, rtol=logits_rtol)
    logits_limits_passed = bool(
        "max_abs" in logits_report and logits_report["max_abs"] <= logits_max_abs
    )
    logits_report["limits"] = {
        "passed": logits_limits_passed,
        "required_max_abs": float(logits_max_abs),
    }
    # 分类以最大绝对误差和 Top-K 结果为主，避免接近 0 的 logit 被相对误差放大。
    logits_report["passed"] = logits_limits_passed

    ranking_report = build_topk_report(pt_logits, onnx_logits, max_k=5)

    embedding_report = compare_arrays(pt_embedding, onnx_embedding, atol=emb_atol, rtol=emb_rtol)
    cosine_report = build_cosine_report(pt_embedding, onnx_embedding, min_cosine)
    embedding_report["cosine"] = cosine_report
    embedding_report["passed"] = bool(
        embedding_report.get("allclose_passed", False) and cosine_report["passed"]
    )

    return {
        "passed": bool(
            logits_limits_passed
            and ranking_report["passed"]
            and embedding_report["passed"]
        ),
        "logits": logits_report,
        "ranking": ranking_report,
        "emb": embedding_report,
    }


def compare_patch_tokens(
    model: DINOEncoderLoRA_MultiTask,
    session: ort.InferenceSession,
    input_name: str,
    input_tensor: torch.Tensor,
    sample_tokens: int,
    seed: int,
    atol: float,
    rtol: float,
    max_abs_limit: float,
    mean_abs_limit: float,
    min_cosine: float,
) -> dict[str, Any]:
    with torch.inference_mode():
        pt_tokens = tensor_to_numpy(model.forward_patchtokens(input_tensor))
    input_numpy = tensor_to_numpy(input_tensor)
    onnx_tokens = session.run(["patch_tokens"], {input_name: input_numpy})[0]
    report = compare_arrays(pt_tokens, onnx_tokens, atol=atol, rtol=rtol)

    if pt_tokens.shape != onnx_tokens.shape or pt_tokens.ndim != 3 or pt_tokens.size == 0:
        report["cosine"] = {"passed": False, "reason": "patch token shape 不一致或为空"}
        report["limits"] = {"passed": False, "reason": "patch token shape 不一致或为空"}
        report["passed"] = False
        return report

    token_count = int(pt_tokens.shape[1])
    take = min(sample_tokens, token_count)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(token_count, size=take, replace=False))
    sampled_pt = pt_tokens[:, indices, :]
    sampled_onnx = onnx_tokens[:, indices, :]
    cosine_report = build_cosine_report(sampled_pt, sampled_onnx, min_cosine)
    limits_passed = bool(
        "max_abs" in report
        and report["max_abs"] <= max_abs_limit
        and report["mean_abs"] <= mean_abs_limit
    )
    report["cosine"] = cosine_report
    report["limits"] = {
        "passed": limits_passed,
        "required_max_abs": float(max_abs_limit),
        "required_mean_abs": float(mean_abs_limit),
    }
    # Tokens 是高维特征，最终使用误差上限 + cosine，不要求所有元素 allclose。
    report["passed"] = bool(limits_passed and cosine_report["passed"])
    report["sample_tokens"] = int(take)
    report["sample_tokens_per_image"] = int(take)
    report["sampled_batch_size"] = int(pt_tokens.shape[0])
    report["sampled_indices"] = indices.tolist()
    return report


def log_array_report(name: str, report: dict[str, Any]) -> None:
    if "max_abs" not in report:
        LOGGER.error(
            "%s：未通过；PT shape=%s，ONNX shape=%s，原因=%s",
            name,
            report.get("pt_shape"),
            report.get("onnx_shape"),
            report.get("reason", "未知"),
        )
        return
    LOGGER.info(
        "%s：%s；shape=%s；max_abs=%.6g；mean_abs=%.6g；p99_abs=%.6g；"
        "allclose=%s；不匹配=%d（%.6f）",
        name,
        "通过" if report["passed"] else "未通过",
        report["pt_shape"],
        report["max_abs"],
        report["mean_abs"],
        report["p99_abs"],
        "通过" if report.get("allclose_passed", False) else "未通过",
        report["mismatch_count"],
        report["mismatch_ratio"],
    )
    cosine = report.get("cosine")
    if isinstance(cosine, dict) and "minimum" in cosine:
        LOGGER.info(
            "%s cosine：%s；min=%.8f；mean=%.8f；要求 min>=%.8f；样本=%d",
            name,
            "通过" if cosine["passed"] else "未通过",
            cosine["minimum"],
            cosine["mean"],
            cosine["required_minimum"],
            cosine["sample_count"],
        )


def log_segmentation_report(report: dict[str, Any]) -> None:
    log_array_report("分割 logits", report["logits"])
    mask = report.get("mask", {})
    if "pixel_agreement" in mask:
        LOGGER.info(
            "分割 mask：%s；模式=%s；阈值=%.6g；像素一致率=%.8f；"
            "前景 mean IoU=%.8f；前景 mean Dice=%.8f",
            "通过" if mask["passed"] else "未通过",
            report["mask_mode"],
            report["threshold"],
            mask["pixel_agreement"],
            mask["foreground_iou_mean"],
            mask["foreground_dice_mean"],
        )
    else:
        LOGGER.error("分割 mask：未通过；原因=%s", mask.get("reason", "未知"))


def log_classification_report(report: dict[str, Any]) -> None:
    log_array_report("分类 logits", report["logits"])
    ranking = report.get("ranking", {})
    if "top1_all_match" in ranking:
        LOGGER.info(
            "分类 Top-K：%s；Top-1 全部一致=%s；Top-%d 集合全部一致=%s；顺序全部一致=%s",
            "通过" if ranking["passed"] else "未通过",
            ranking["top1_all_match"],
            ranking["top_k"],
            ranking["topk_set_all_match"],
            ranking["topk_order_all_match"],
        )
    log_array_report("分类 embedding", report["emb"])



def configure_matplotlib() -> None:
    """配置无界面绘图，并尽量兼容中文字体。"""
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]


def add_bar_value_labels(
    ax: Any,
    bars: Any,
    *,
    value_format: str = "{:.4f}",
    padding: int = 3,
) -> None:
    """为柱形图添加数值标签，兼容较旧的 Matplotlib。"""
    if hasattr(ax, "bar_label"):
        labels = [value_format.format(float(bar.get_height())) for bar in bars]
        ax.bar_label(bars, labels=labels, padding=padding, fontsize=9)
        return

    for bar in bars:
        height = float(bar.get_height())
        ax.annotate(
            value_format.format(height),
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_plot(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    LOGGER.info("对比图表已保存：%s", path)


def safe_ratio(actual: float, limit: float) -> float:
    if limit <= 0:
        return 0.0 if actual <= 0 else float("inf")
    return float(actual / limit)


def plot_error_limit_ratios(
    report: dict[str, Any],
    output_path: Path,
    dpi: int,
) -> None:
    """绘制各项误差占允许上限的比例，1.0 表示刚好达到上限。"""
    seg_logits = report["seg"]["logits"]
    cls_logits = report["cls"]["logits"]
    tokens = report["tokens"]
    criteria = report["criteria"]

    labels = [
        "Seg max abs",
        "Seg mean abs",
        "Cls max abs",
        "Token max abs",
        "Token mean abs",
    ]
    ratios = [
        safe_ratio(
            float(seg_logits["max_abs"]),
            float(criteria["segmentation"]["max_abs"]),
        ),
        safe_ratio(
            float(seg_logits["mean_abs"]),
            float(criteria["segmentation"]["mean_abs"]),
        ),
        safe_ratio(
            float(cls_logits["max_abs"]),
            float(criteria["classification"]["max_abs"]),
        ),
        safe_ratio(
            float(tokens["max_abs"]),
            float(criteria["patch_tokens"]["max_abs"]),
        ),
        safe_ratio(
            float(tokens["mean_abs"]),
            float(criteria["patch_tokens"]["mean_abs"]),
        ),
    ]

    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    bars = ax.bar(labels, ratios)
    ax.axhline(1.0, linestyle="--", linewidth=1.5, label="Allowed limit")
    ax.set_title("Error-to-limit ratios")
    ax.set_ylabel("Actual error / allowed limit")
    ax.set_xlabel("Validation item")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    add_bar_value_labels(ax, bars, value_format="{:.3f}")
    save_plot(output_path, dpi)


def plot_segmentation_metrics(
    report: dict[str, Any],
    output_path: Path,
    dpi: int,
) -> None:
    """绘制分割 mask 的任务级一致性指标及其最低要求。"""
    mask_report = report["seg"].get("mask", {})
    required_keys = (
        "required_min_pixel_agreement",
        "required_min_foreground_iou",
        "required_min_foreground_dice",
    )
    actual_keys = (
        "pixel_agreement",
        "foreground_iou_mean",
        "foreground_dice_mean",
    )
    if not all(key in mask_report for key in (*required_keys, *actual_keys)):
        LOGGER.warning("分割 mask 指标不完整，跳过分割指标图")
        return

    labels = ["Pixel agreement", "Foreground IoU", "Foreground Dice"]
    actual = [float(mask_report[key]) for key in actual_keys]
    required = [float(mask_report[key]) for key in required_keys]
    positions = np.arange(len(labels), dtype=np.float32)
    width = 0.36

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    actual_bars = ax.bar(positions - width / 2.0, actual, width, label="Actual")
    required_bars = ax.bar(positions + width / 2.0, required, width, label="Required")
    ax.set_title("Segmentation mask consistency")
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    minimum = min(actual + required)
    ax.set_ylim(max(0.0, minimum - 0.01), 1.002)
    add_bar_value_labels(ax, actual_bars, value_format="{:.6f}")
    add_bar_value_labels(ax, required_bars, value_format="{:.6f}")
    save_plot(output_path, dpi)


def plot_classification_topk(
    report: dict[str, Any],
    output_path: Path,
    dpi: int,
) -> None:
    """绘制分类 Top-1、Top-K 集合和 Top-K 顺序的一致率。"""
    ranking = report["cls"].get("ranking", {})
    required_keys = (
        "top1_match_ratio",
        "topk_set_match_ratio",
        "topk_order_match_ratio",
    )
    if not all(key in ranking for key in required_keys):
        LOGGER.warning("分类 Top-K 指标不完整，跳过分类 Top-K 图")
        return

    top_k = int(ranking.get("top_k", 5))
    labels = ["Top-1", f"Top-{top_k} set", f"Top-{top_k} order"]
    values = [float(ranking[key]) for key in required_keys]

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    bars = ax.bar(labels, values)
    ax.axhline(1.0, linestyle="--", linewidth=1.5, label="Perfect match")
    ax.set_title("Classification ranking consistency")
    ax.set_ylabel("Match ratio")
    ax.set_xlabel("Ranking metric")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    add_bar_value_labels(ax, bars, value_format="{:.4f}")
    save_plot(output_path, dpi)


def plot_cosine_similarity(
    report: dict[str, Any],
    output_path: Path,
    dpi: int,
) -> None:
    """绘制 embedding 和 patch tokens 的 cosine 指标。"""
    emb_cosine = report["cls"].get("emb", {}).get("cosine", {})
    token_cosine = report.get("tokens", {}).get("cosine", {})
    if "minimum" not in emb_cosine or "minimum" not in token_cosine:
        LOGGER.warning("cosine 指标不完整，跳过 cosine 图")
        return

    labels = [
        "Embedding min",
        "Embedding mean",
        "Tokens min",
        "Tokens mean",
    ]
    values = [
        float(emb_cosine["minimum"]),
        float(emb_cosine["mean"]),
        float(token_cosine["minimum"]),
        float(token_cosine["mean"]),
    ]
    required = max(
        float(emb_cosine.get("required_minimum", 0.0)),
        float(token_cosine.get("required_minimum", 0.0)),
    )

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    bars = ax.bar(labels, values)
    ax.axhline(
        required,
        linestyle="--",
        linewidth=1.5,
        label=f"Required minimum={required:.6f}",
    )
    ax.set_title("Feature cosine similarity")
    ax.set_ylabel("Cosine similarity")
    ax.set_xlabel("Feature metric")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)

    minimum = min(values + [required])
    ax.set_ylim(max(-1.0, minimum - 0.0005), 1.00005)
    add_bar_value_labels(ax, bars, value_format="{:.8f}")
    save_plot(output_path, dpi)


def plot_mismatch_ratios(
    report: dict[str, Any],
    output_path: Path,
    dpi: int,
) -> None:
    """绘制严格 allclose 判定下的逐元素不匹配比例。"""
    seg = report["seg"]
    cls = report["cls"]
    tokens = report["tokens"]

    labels = [
        "Seg logits",
        "Seg probabilities",
        "Cls logits",
        "Cls embedding",
        "Patch tokens",
    ]
    sources = [
        seg.get("logits", {}),
        seg.get("probabilities", {}),
        cls.get("logits", {}),
        cls.get("emb", {}),
        tokens,
    ]
    ratios = [float(source.get("mismatch_ratio", 0.0)) * 100.0 for source in sources]

    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    bars = ax.bar(labels, ratios)
    ax.set_title("Strict allclose mismatch ratios")
    ax.set_ylabel("Mismatched elements (%)")
    ax.set_xlabel("Output")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    add_bar_value_labels(ax, bars, value_format="{:.3f}%")
    save_plot(output_path, dpi)


def plot_segmentation_class_metrics(
    report: dict[str, Any],
    output_path: Path,
    dpi: int,
) -> None:
    """按类别绘制分割 mask 的平均 IoU 和 Dice。"""
    class_metrics = report["seg"].get("mask", {}).get("class_metrics", [])
    if not isinstance(class_metrics, list) or not class_metrics:
        LOGGER.warning("分割类别指标为空，跳过类别 IoU/Dice 图")
        return

    grouped: dict[int, dict[str, list[float]]] = {}
    for item in class_metrics:
        class_id = int(item["class_id"])
        grouped.setdefault(class_id, {"iou": [], "dice": []})
        grouped[class_id]["iou"].append(float(item["iou"]))
        grouped[class_id]["dice"].append(float(item["dice"]))

    class_ids = sorted(grouped)
    labels = [f"Class {class_id}" for class_id in class_ids]
    iou_values = [float(np.mean(grouped[class_id]["iou"])) for class_id in class_ids]
    dice_values = [float(np.mean(grouped[class_id]["dice"])) for class_id in class_ids]
    positions = np.arange(len(class_ids), dtype=np.float32)
    width = 0.36

    plt.figure(figsize=(max(9, len(class_ids) * 0.8), 6))
    ax = plt.gca()
    iou_bars = ax.bar(positions - width / 2.0, iou_values, width, label="IoU")
    dice_bars = ax.bar(positions + width / 2.0, dice_values, width, label="Dice")
    ax.set_title("Segmentation class-level consistency")
    ax.set_ylabel("Score")
    ax.set_xlabel("Class")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    add_bar_value_labels(ax, iou_bars, value_format="{:.4f}")
    add_bar_value_labels(ax, dice_bars, value_format="{:.4f}")
    save_plot(output_path, dpi)


def generate_plots(
    report: dict[str, Any],
    plot_dir: Path,
    dpi: int,
) -> list[Path]:
    """生成所有 PT/ONNX 一致性图表，并返回成功保存的文件路径。"""
    configure_matplotlib()
    if dpi <= 0:
        raise ValueError(f"plot_dpi 必须大于 0，实际为 {dpi}")

    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_jobs = [
        ("01_error_limit_ratios.png", plot_error_limit_ratios),
        ("02_segmentation_metrics.png", plot_segmentation_metrics),
        ("03_classification_topk.png", plot_classification_topk),
        ("04_cosine_similarity.png", plot_cosine_similarity),
        ("05_mismatch_ratios.png", plot_mismatch_ratios),
        ("06_segmentation_class_metrics.png", plot_segmentation_class_metrics),
    ]

    generated: list[Path] = []
    for filename, plot_function in plot_jobs:
        output_path = plot_dir / filename
        if output_path.exists():
            output_path.unlink()
        try:
            plot_function(report, output_path, dpi)
        except Exception as exc:
            plt.close("all")
            raise RuntimeError(f"绘制图表失败：{filename}，原因：{exc}") from exc
        if output_path.is_file():
            generated.append(output_path)

    if not generated:
        raise RuntimeError("没有成功生成任何 Matplotlib 图表")
    return generated


def validate_ratio(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} 必须位于 [0,1]，实际为 {value}")


def validate_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} 不能为负数，实际为 {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="验证多任务 PT 与 ONNX 的数值及任务结果一致性"
    )
    parser.add_argument("--config", default=default_config_path("mul"), help="多任务 YAML 配置文件")
    parser.add_argument("--ckpt", required=True, help="多任务 ckpt_best.pt 或 ckpt_last.pt")
    parser.add_argument(
        "--onnx_dir",
        required=True,
        help=(
            "ONNX 模型目录；固定读取 multitask_seg.onnx、"
            "multitask_cls.onnx、multitask_patchtokens.onnx"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="PT 推理设备")
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto", help="ONNXRuntime provider")
    parser.add_argument("--batch", type=int, default=1, help="验证 batch size")
    parser.add_argument("--seed", type=int, default=2026, help="随机输入种子")
    parser.add_argument("--sample_tokens", type=int, default=32, help="每张图抽样的 patch token 数")

    # Embedding 保留原参数名，确保原调用兼容。
    parser.add_argument("--atol", type=float, default=1e-4, help="分类 embedding 绝对误差容限")
    parser.add_argument("--rtol", type=float, default=1e-3, help="分类 embedding 相对误差容限")
    parser.add_argument("--min_cosine", type=float, default=0.9999, help="embedding/token 最低 cosine")

    # 分割：logits 误差上限 + 最终 mask 一致性。
    parser.add_argument("--seg_atol", type=float, default=2e-2, help="分割 logits allclose 诊断 atol")
    parser.add_argument("--seg_rtol", type=float, default=1e-2, help="分割 logits allclose 诊断 rtol")
    parser.add_argument("--seg_max_abs", type=float, default=2e-2, help="分割 logits 最大绝对误差上限")
    parser.add_argument("--seg_mean_abs", type=float, default=5e-3, help="分割 logits 平均绝对误差上限")
    parser.add_argument(
        "--seg_min_pixel_agreement",
        type=float,
        default=0.999,
        help="分割 mask 最低像素一致率",
    )
    parser.add_argument("--seg_min_iou", type=float, default=0.995, help="分割前景最低 mean IoU")
    parser.add_argument("--seg_min_dice", type=float, default=0.997, help="分割前景最低 mean Dice")

    # 分类：logits 数值上限 + Top-1/Top-5。
    parser.add_argument("--cls_atol", type=float, default=1e-3, help="分类 logits allclose 诊断 atol")
    parser.add_argument("--cls_rtol", type=float, default=1e-3, help="分类 logits allclose 诊断 rtol")
    parser.add_argument("--cls_max_abs", type=float, default=1e-3, help="分类 logits 最大绝对误差上限")

    # Tokens：高维特征采用误差上限 + cosine。
    parser.add_argument("--tok_atol", type=float, default=5e-3, help="Patch tokens allclose 诊断 atol")
    parser.add_argument("--tok_rtol", type=float, default=1e-2, help="Patch tokens allclose 诊断 rtol")
    parser.add_argument("--tok_max_abs", type=float, default=5e-3, help="Patch tokens 最大绝对误差上限")
    parser.add_argument("--tok_mean_abs", type=float, default=5e-4, help="Patch tokens 平均绝对误差上限")

    parser.add_argument("--save_json", action="store_true", help="保存 JSON 验证报告")
    parser.add_argument(
        "--plot_dir",
        default=None,
        help="Matplotlib 图表输出目录，默认是 <onnx_dir>/compare_plots",
    )
    parser.add_argument("--plot_dpi", type=int, default=160, help="图表保存 DPI")
    parser.add_argument("--no_plots", action="store_true", help="不生成 Matplotlib 图表")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.batch <= 0:
        raise ValueError(f"batch 必须大于 0，实际为 {args.batch}")
    if args.sample_tokens <= 0:
        raise ValueError(f"sample_tokens 必须大于 0，实际为 {args.sample_tokens}")
    if args.plot_dpi <= 0:
        raise ValueError(f"plot_dpi 必须大于 0，实际为 {args.plot_dpi}")

    for name in (
        "atol",
        "rtol",
        "seg_atol",
        "seg_rtol",
        "seg_max_abs",
        "seg_mean_abs",
        "cls_atol",
        "cls_rtol",
        "cls_max_abs",
        "tok_atol",
        "tok_rtol",
        "tok_max_abs",
        "tok_mean_abs",
    ):
        validate_non_negative(name, float(getattr(args, name)))
    if not -1.0 <= args.min_cosine <= 1.0:
        raise ValueError(f"min_cosine 必须位于 [-1,1]，实际为 {args.min_cosine}")
    validate_ratio("seg_min_pixel_agreement", args.seg_min_pixel_agreement)
    validate_ratio("seg_min_iou", args.seg_min_iou)
    validate_ratio("seg_min_dice", args.seg_min_dice)

    configure_torch_consistency()
    cfg = load_config(args.config)
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")
    onnx_dir = Path(args.onnx_dir).expanduser().resolve()
    if not onnx_dir.is_dir():
        raise NotADirectoryError(f"ONNX 目录不存在或不是目录：{onnx_dir}")

    # 三个模型文件名固定，不再分别接受命令行路径。
    seg_onnx = onnx_dir / "multitask_seg.onnx"
    cls_onnx = onnx_dir / "multitask_cls.onnx"
    tok_onnx = onnx_dir / "multitask_patchtokens.onnx"
    for path in (seg_onnx, cls_onnx, tok_onnx):
        if not path.is_file():
            raise FileNotFoundError(f"ONNX 文件不存在：{path}")

    post_cfg = cfg.get("postprocess", {}) or {}
    mask_mode = str(post_cfg.get("mask_mode", "argmax")).strip().lower()
    mask_threshold = float(post_cfg.get("thr", 0.5))
    if mask_mode not in {"argmax", "prob_threshold"}:
        raise ValueError(f"postprocess.mask_mode 仅支持 argmax/prob_threshold，实际为 {mask_mode}")
    if not 0.0 <= mask_threshold <= 1.0:
        raise ValueError(f"postprocess.thr 必须位于 [0,1]，实际为 {mask_threshold}")

    device = resolve_device(args.device)
    providers = select_providers(args.provider, device)
    LOGGER.info("PT 使用设备：%s；ONNX 请求 providers=%s", device, providers)
    LOGGER.info(
        "一致性判定：seg(max_abs<=%.6g, mean_abs<=%.6g, pixel>=%.6f, IoU>=%.6f, Dice>=%.6f)；"
        "cls(max_abs<=%.6g + Top-1/Top-5)；tokens(max_abs<=%.6g, mean_abs<=%.6g, cosine>=%.6f)",
        args.seg_max_abs,
        args.seg_mean_abs,
        args.seg_min_pixel_agreement,
        args.seg_min_iou,
        args.seg_min_dice,
        args.cls_max_abs,
        args.tok_max_abs,
        args.tok_mean_abs,
        args.min_cosine,
    )

    model = build_multitask_model(cfg, checkpoint_path, device)
    seg_hw = get_input_hw(cfg, "input")
    cls_hw = get_input_hw(cfg, "input_cls")
    rng = np.random.default_rng(args.seed)
    seg_input = make_random_input(rng, args.batch, seg_hw, device)
    cls_input = make_random_input(rng, args.batch, cls_hw, device)

    seg_session = create_session(seg_onnx, providers)
    cls_session = create_session(cls_onnx, providers)
    tok_session = create_session(tok_onnx, providers)
    seg_input_name = validate_session_contract(seg_session, seg_hw, ("seg_logits",), "分割模型")
    cls_input_name = validate_session_contract(
        cls_session,
        cls_hw,
        ("cls_logits", "cls_emb"),
        "分类模型",
    )
    tok_input_name = validate_session_contract(
        tok_session,
        seg_hw,
        ("patch_tokens",),
        "Patch token 模型",
    )

    segmentation_report = compare_segmentation(
        model=model,
        session=seg_session,
        input_name=seg_input_name,
        input_tensor=seg_input,
        mask_mode=mask_mode,
        threshold=mask_threshold,
        atol=args.seg_atol,
        rtol=args.seg_rtol,
        max_abs_limit=args.seg_max_abs,
        mean_abs_limit=args.seg_mean_abs,
        min_pixel_agreement=args.seg_min_pixel_agreement,
        min_iou=args.seg_min_iou,
        min_dice=args.seg_min_dice,
    )
    classification_report = compare_classification(
        model=model,
        session=cls_session,
        input_name=cls_input_name,
        input_tensor=cls_input,
        logits_atol=args.cls_atol,
        logits_rtol=args.cls_rtol,
        logits_max_abs=args.cls_max_abs,
        emb_atol=args.atol,
        emb_rtol=args.rtol,
        min_cosine=args.min_cosine,
    )
    token_report = compare_patch_tokens(
        model=model,
        session=tok_session,
        input_name=tok_input_name,
        input_tensor=seg_input,
        sample_tokens=args.sample_tokens,
        seed=args.seed,
        atol=args.tok_atol,
        rtol=args.tok_rtol,
        max_abs_limit=args.tok_max_abs,
        mean_abs_limit=args.tok_mean_abs,
        min_cosine=args.min_cosine,
    )
    passed = bool(
        segmentation_report["passed"]
        and classification_report["passed"]
        and token_report["passed"]
    )

    report: dict[str, Any] = {
        "passed": passed,
        "ckpt": str(checkpoint_path),
        "onnx_dir": str(onnx_dir),
        "device_pt": str(device),
        "batch": int(args.batch),
        "seed": int(args.seed),
        "checkpoint": str(checkpoint_path),
        "onnx": {
            "segmentation": str(seg_onnx),
            "classification": str(cls_onnx),
            "patch_tokens": str(tok_onnx),
        },
        "pt_device": str(device),
        "onnx_providers": {
            "requested": providers,
            "segmentation": seg_session.get_providers(),
            "classification": cls_session.get_providers(),
            "patch_tokens": tok_session.get_providers(),
        },
        "input": {
            "batch": int(args.batch),
            "seed": int(args.seed),
            "segmentation_hw": list(seg_hw),
            "classification_hw": list(cls_hw),
            "segmentation_mean": float(tensor_to_numpy(seg_input).mean()),
            "segmentation_std": float(tensor_to_numpy(seg_input).std()),
            "classification_mean": float(tensor_to_numpy(cls_input).mean()),
            "classification_std": float(tensor_to_numpy(cls_input).std()),
        },
        "criteria": {
            "embedding": {
                "atol": float(args.atol),
                "rtol": float(args.rtol),
                "min_cosine": float(args.min_cosine),
            },
            "segmentation": {
                "diagnostic_atol": float(args.seg_atol),
                "diagnostic_rtol": float(args.seg_rtol),
                "max_abs": float(args.seg_max_abs),
                "mean_abs": float(args.seg_mean_abs),
                "min_pixel_agreement": float(args.seg_min_pixel_agreement),
                "min_foreground_iou": float(args.seg_min_iou),
                "min_foreground_dice": float(args.seg_min_dice),
                "mask_mode": mask_mode,
                "threshold": mask_threshold,
            },
            "classification": {
                "diagnostic_atol": float(args.cls_atol),
                "diagnostic_rtol": float(args.cls_rtol),
                "max_abs": float(args.cls_max_abs),
                "require_top1_match": True,
                "require_top5_set_match": True,
            },
            "patch_tokens": {
                "diagnostic_atol": float(args.tok_atol),
                "diagnostic_rtol": float(args.tok_rtol),
                "max_abs": float(args.tok_max_abs),
                "mean_abs": float(args.tok_mean_abs),
                "min_cosine": float(args.min_cosine),
            },
        },
        "seg": segmentation_report,
        "cls": classification_report,
        "tokens": token_report,
    }

    log_segmentation_report(segmentation_report)
    log_classification_report(classification_report)
    log_array_report("Patch tokens", token_report)

    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_dir = (
            Path(args.plot_dir).expanduser().resolve()
            if args.plot_dir
            else onnx_dir / "compare_plots"
        )
        plot_paths = generate_plots(
            report=report,
            plot_dir=plot_dir,
            dpi=args.plot_dpi,
        )
        report["plots"] = {
            "enabled": True,
            "directory": str(plot_dir),
            "dpi": int(args.plot_dpi),
            "files": [str(path) for path in plot_paths],
        }
        LOGGER.info("Matplotlib 图表生成完成：数量=%d，目录=%s", len(plot_paths), plot_dir)
    else:
        report["plots"] = {
            "enabled": False,
            "directory": None,
            "dpi": int(args.plot_dpi),
            "files": [],
        }
        LOGGER.info("已通过 --no_plots 跳过 Matplotlib 图表生成")

    if args.save_json:
        report_path = onnx_dir / "compare_pt_onnx_multitask.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as output_file:
            json.dump(report, output_file, ensure_ascii=False, indent=2)
        LOGGER.info("验证报告已保存：%s", report_path)

    if not passed:
        failed_parts = [
            name
            for name, part_report in (
                ("segmentation", segmentation_report),
                ("classification", classification_report),
                ("patch_tokens", token_report),
            )
            if not part_report["passed"]
        ]
        raise RuntimeError(
            "PT 与 ONNX 一致性验证未通过；失败模块="
            + ", ".join(failed_parts)
            + "。请检查日志和 compare_pt_onnx_multitask.json。"
        )
    LOGGER.info("PT 与 ONNX 数值及任务结果一致性验证通过")


if __name__ == "__main__":
    main()