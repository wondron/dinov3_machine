from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dino_finetune.config import default_config_path, load_config, resolve_interp


LOGGER = logging.getLogger("embed_with_multitask_onnx")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_images(input_path: Path, recursive: bool) -> tuple[list[Path], Path]:
    path = input_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"输入文件不是支持的图片格式：{path}")
        return [path], path.parent
    if not path.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{path}")

    iterator = path.rglob("*") if recursive else path.iterdir()
    images = sorted(
        item for item in iterator
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"输入目录中未找到图片：{path}")
    return images, path


def get_input_hw(cfg: dict, section: str) -> tuple[int, int]:
    values = cfg[section]["img_dim"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError(f"{section}.img_dim 必须由两个正整数组成，实际为 {values}")
    return int(values[0]), int(values[1])


def preprocess_image(image_bgr: np.ndarray, input_cfg: dict) -> np.ndarray:
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("输入图片为空或维度错误")

    height, width = get_input_hw({"current": input_cfg}, "current")
    interpolation = resolve_interp(str(input_cfg["img_interp"]))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(
        image_rgb,
        (width, height),
        interpolation=interpolation,
    ).astype(np.float32) / 255.0
    mean = np.asarray(input_cfg["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(input_cfg["std"], dtype=np.float32).reshape(1, 1, 3)
    normalized = (image_rgb - mean) / std
    return np.transpose(normalized, (2, 0, 1))[None].astype(np.float32)


def select_providers(provider: str) -> list[str]:
    available = ort.get_available_providers()
    if provider == "auto":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CPUExecutionProvider" in available:
            return ["CPUExecutionProvider"]
        raise RuntimeError(f"没有可用的 ONNXRuntime provider：{available}")

    provider_name = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
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
    if len(model_input.shape) != 4:
        raise ValueError(f"{model_name} ONNX 输入维度错误：期望 NCHW，实际 {model_input.shape}")
    if isinstance(model_input.shape[1], int) and model_input.shape[1] != 3:
        raise ValueError(f"{model_name} ONNX 输入通道错误：期望 3，实际 {model_input.shape[1]}")
    for actual, expected, axis_name in zip(model_input.shape[-2:], expected_hw, ("高度", "宽度")):
        if isinstance(actual, int) and actual != expected:
            raise ValueError(
                f"{model_name} ONNX 输入{axis_name}与 config 不一致：ONNX={actual}，config={expected}"
            )

    available_outputs = {output.name for output in session.get_outputs()}
    missing = [name for name in expected_outputs if name not in available_outputs]
    if missing:
        raise ValueError(f"{model_name} ONNX 缺少输出：{missing}；实际输出={sorted(available_outputs)}")
    return model_input.name


def l2_normalize(value: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(value, axis=axis, keepdims=True)
    return value / np.maximum(norm, eps)


def softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / (np.sum(exp, axis=axis, keepdims=True) + 1e-12)


def segmentation_foreground_mask(
    logits: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, float]:
    n_classes = int(cfg["model"]["n_classes"])
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != n_classes:
        raise ValueError(f"分割输出形状错误：期望 (1,{n_classes},H,W)，实际 {logits.shape}")

    input_cfg = cfg["input"]
    post_cfg = cfg["postprocess"]
    height, width = get_input_hw(cfg, "input")
    image_interpolation = resolve_interp(str(input_cfg["img_interp"]))
    mask_interpolation = resolve_interp(str(input_cfg["mask_interp"]))
    mask_mode = str(post_cfg["mask_mode"]).strip().lower()

    if mask_mode == "prob_threshold":
        if n_classes != 2:
            raise ValueError("prob_threshold 仅支持二分类分割，多分类请配置 mask_mode=argmax")
        probabilities = softmax(logits.astype(np.float32), axis=1)[0, 1]
        foreground_probability = cv2.resize(
            probabilities,
            (width, height),
            interpolation=image_interpolation,
        )
        threshold = float(post_cfg["thr"])
        foreground = foreground_probability >= threshold
        confidence = float(foreground_probability[foreground].mean()) if np.any(foreground) else 0.0
        return foreground, confidence

    if mask_mode == "argmax":
        class_mask = np.argmax(logits[0], axis=0).astype(np.uint16)
        class_mask = cv2.resize(
            class_mask,
            (width, height),
            interpolation=mask_interpolation,
        )
        foreground = class_mask > 0
        return foreground, 1.0 if np.any(foreground) else 0.0

    raise ValueError(f"不支持的分割后处理模式：{mask_mode}")


def infer_patch_grid(cfg: dict, token_count: int) -> tuple[int, int, int]:
    dino_type = str(cfg["model"]["dino_type"]).strip().lower()
    patch_size = 16 if dino_type == "dinov3" else 14
    height, width = get_input_hw(cfg, "input")
    grid_height = height // patch_size
    grid_width = width // patch_size
    if grid_height * grid_width != token_count:
        raise ValueError(
            "patch token 数量与 config 不匹配："
            f"token_count={token_count}，预期网格={grid_height}x{grid_width}，patch_size={patch_size}"
        )
    return grid_height, grid_width, patch_size


def estimate_cluster_count(token_count: int, k_min: int, k_max: int) -> int:
    if token_count <= 0:
        return 0
    estimated = int(round(np.sqrt(float(token_count))))
    estimated = max(k_min, min(k_max, estimated))
    return min(estimated, token_count)


def kmeans(
    features: np.ndarray,
    cluster_count: int,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"KMeans 输入形状错误：{features.shape}")
    if cluster_count <= 0 or cluster_count > features.shape[0]:
        raise ValueError(f"KMeans 聚类数错误：K={cluster_count}，样本数={features.shape[0]}")

    # 在单位球面上执行 KMeans++ 初始化与 cosine 聚类，和后续 cosine 召回保持一致。
    features = l2_normalize(features, axis=1)
    rng = np.random.default_rng(seed)
    centers = np.empty((cluster_count, features.shape[1]), dtype=np.float32)
    selected_indices: set[int] = set()
    first_index = int(rng.integers(0, features.shape[0]))
    centers[0] = features[first_index]
    selected_indices.add(first_index)

    nearest_distance = np.maximum(1.0 - features @ centers[0], 0.0)
    for center_id in range(1, cluster_count):
        probabilities = nearest_distance.copy()
        if selected_indices:
            probabilities[list(selected_indices)] = 0.0
        probability_sum = float(probabilities.sum())
        if probability_sum <= 1e-12:
            candidates = [index for index in range(features.shape[0]) if index not in selected_indices]
            next_index = int(candidates[0])
        else:
            probabilities /= probability_sum
            next_index = int(rng.choice(features.shape[0], p=probabilities))
        centers[center_id] = features[next_index]
        selected_indices.add(next_index)
        nearest_distance = np.minimum(
            nearest_distance,
            np.maximum(1.0 - features @ centers[center_id], 0.0),
        )

    labels = np.full(features.shape[0], -1, dtype=np.int32)

    for _ in range(iterations):
        similarities = features @ centers.T
        new_labels = np.argmax(similarities, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        new_centers = centers.copy()
        for cluster_id in range(cluster_count):
            selected = labels == cluster_id
            if np.any(selected):
                center = features[selected].mean(axis=0)
                center_norm = float(np.linalg.norm(center))
                if center_norm <= 1e-12:
                    raise ValueError(f"第 {cluster_id} 个聚类中心是零向量")
                new_centers[cluster_id] = center / center_norm
            else:
                best_similarity = np.max(similarities, axis=1)
                new_centers[cluster_id] = features[int(np.argmin(best_similarity))]
        centers = new_centers

    return labels, centers


def build_region_embeddings(
    patch_tokens: np.ndarray,
    foreground_mask: np.ndarray,
    cfg: dict,
    k_min: int,
    k_max: int,
    kmeans_iters: int,
    min_region_tokens: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray, np.ndarray]:
    if patch_tokens.ndim != 3 or patch_tokens.shape[0] != 1:
        raise ValueError(f"patch token 输出形状错误：期望 (1,N,C)，实际 {patch_tokens.shape}")
    if not np.isfinite(patch_tokens).all():
        raise ValueError("patch token 输出包含 NaN 或 Inf")

    token_count = int(patch_tokens.shape[1])
    embedding_dim = int(patch_tokens.shape[2])
    grid_height, grid_width, patch_size = infer_patch_grid(cfg, token_count)
    mask_interpolation = resolve_interp(str(cfg["input"]["mask_interp"]))
    foreground_grid = cv2.resize(
        foreground_mask.astype(np.float32),
        (grid_width, grid_height),
        interpolation=mask_interpolation,
    ) >= 0.5

    raw_tokens = patch_tokens[0].astype(np.float32)
    token_norms = np.linalg.norm(raw_tokens, axis=1)
    if np.any(token_norms <= 1e-12):
        raise ValueError("patch token 中存在零向量，无法生成可靠 embedding")
    normalized_tokens = l2_normalize(raw_tokens, axis=1)
    valid_indices = np.flatnonzero(foreground_grid.reshape(-1))
    if valid_indices.size == 0:
        metadata = {
            "token_grid_hw": [grid_height, grid_width],
            "patch_size": patch_size,
            "token_count": token_count,
            "embedding_dim": embedding_dim,
            "foreground_token_count": 0,
            "foreground_token_ratio": 0.0,
            "cluster_count": 0,
            "kept_region_count": 0,
        }
        return [], metadata, normalized_tokens, foreground_grid

    foreground_tokens = normalized_tokens[valid_indices]
    cluster_count = estimate_cluster_count(int(valid_indices.size), k_min, k_max)
    labels, _ = kmeans(foreground_tokens, cluster_count, kmeans_iters, seed)

    regions: list[dict[str, Any]] = []
    for cluster_id in range(cluster_count):
        selected = labels == cluster_id
        region_token_count = int(np.count_nonzero(selected))
        if region_token_count < min_region_tokens:
            continue

        region_indices = valid_indices[selected]
        positions_y = region_indices // grid_width
        positions_x = region_indices % grid_width
        raw_embedding = foreground_tokens[selected].mean(axis=0)
        if float(np.linalg.norm(raw_embedding)) <= 1e-12:
            raise ValueError(f"第 {cluster_id} 个 region embedding 是零向量")
        embedding = l2_normalize(raw_embedding, axis=0)
        if not np.isfinite(embedding).all():
            raise ValueError(f"第 {cluster_id} 个 region embedding 包含 NaN 或 Inf")

        regions.append(
            {
                "region_id": int(cluster_id),
                "embedding": embedding.astype(np.float32).tolist(),
                "token_count": region_token_count,
                "token_ratio": float(region_token_count / valid_indices.size),
                "bbox_token_xyxy": [
                    int(positions_x.min()),
                    int(positions_y.min()),
                    int(positions_x.max()) + 1,
                    int(positions_y.max()) + 1,
                ],
            }
        )

    regions.sort(key=lambda item: (-int(item["token_count"]), int(item["region_id"])))
    for region_id, region in enumerate(regions):
        region["region_id"] = region_id

    metadata = {
        "token_grid_hw": [grid_height, grid_width],
        "patch_size": patch_size,
        "token_count": token_count,
        "embedding_dim": embedding_dim,
        "foreground_token_count": int(valid_indices.size),
        "foreground_token_ratio": float(valid_indices.size / token_count),
        "cluster_count": cluster_count,
        "kept_region_count": len(regions),
    }
    return regions, metadata, normalized_tokens, foreground_grid


def save_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.stem}.",
        suffix=".tmp.json",
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        json.dump(data, temporary_file, ensure_ascii=False)
        temporary_path = Path(temporary_file.name)
    try:
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def extract_one(
    image_path: Path,
    relative_path: Path,
    cfg: dict,
    sessions: dict[str, ort.InferenceSession],
    input_names: dict[str, str],
    args: argparse.Namespace,
    image_seed: int,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图片读取失败：{image_path}")

    seg_input = preprocess_image(image, cfg["input"])
    cls_input = preprocess_image(image, cfg["input_cls"])
    seg_logits = sessions["seg"].run(
        ["seg_logits"],
        {input_names["seg"]: seg_input},
    )[0]
    global_embedding = sessions["cls"].run(
        ["cls_emb"],
        {input_names["cls"]: cls_input},
    )[0]
    patch_tokens = sessions["tokens"].run(
        ["patch_tokens"],
        {input_names["tokens"]: seg_input},
    )[0]

    if global_embedding.ndim != 2 or global_embedding.shape[0] != 1:
        raise ValueError(f"全局 embedding 输出形状错误：期望 (1,C)，实际 {global_embedding.shape}")
    if not np.isfinite(global_embedding).all():
        raise ValueError("全局 embedding 包含 NaN 或 Inf")
    global_embedding = global_embedding[0].astype(np.float32)
    if float(np.linalg.norm(global_embedding)) <= 1e-12:
        raise ValueError("全局 embedding 是零向量")
    global_embedding = l2_normalize(global_embedding, axis=0)

    foreground_mask, foreground_confidence = segmentation_foreground_mask(seg_logits, cfg)
    regions, token_metadata, normalized_tokens, foreground_grid = build_region_embeddings(
        patch_tokens=patch_tokens,
        foreground_mask=foreground_mask,
        cfg=cfg,
        k_min=args.k_min,
        k_max=args.k_max,
        kmeans_iters=args.kmeans_iters,
        min_region_tokens=args.min_region_tokens,
        seed=image_seed,
    )
    if int(global_embedding.shape[0]) != int(token_metadata.get("embedding_dim", global_embedding.shape[0])):
        raise ValueError(
            "全局 embedding 与 patch token 维度不一致："
            f"global={global_embedding.shape[0]}，token={token_metadata.get('embedding_dim')}"
        )

    record: dict[str, Any] = {
        "schema_version": 2,
        "record_id": relative_path.as_posix(),
        "image_path": str(image_path),
        "relative_path": relative_path.as_posix(),
        "embedding_dim": int(global_embedding.shape[0]),
        "global_emb": global_embedding.astype(np.float32).tolist(),
        "regions": regions,
        "metadata": {
            "embedding_normalized": True,
            "region_algorithm": "foreground_spherical_kmeans",
            "segmentation_input_hw": list(get_input_hw(cfg, "input")),
            "classification_input_hw": list(get_input_hw(cfg, "input_cls")),
            "foreground_pixel_ratio": float(foreground_mask.mean()),
            "foreground_confidence": foreground_confidence,
            **token_metadata,
        },
    }

    if args.save_patch_tokens:
        token_path = args.out_dir_path / "patch_tokens" / relative_path.with_suffix(".npz")
        token_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(token_path),
            patch_tokens=normalized_tokens.astype(np.float32),
            foreground_mask=foreground_grid.astype(np.uint8),
        )
        record["patch_tokens_path"] = str(token_path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="使用多任务 ONNX 提取全局和 region embedding")
    parser.add_argument("--config", default=default_config_path("mul"), help="多任务 YAML 配置文件")
    parser.add_argument("--onnx_dir", required=True, help="三个多任务 ONNX 所在目录")
    parser.add_argument("--seg_onnx", default=None, help="覆盖 multitask_seg.onnx 路径")
    parser.add_argument("--cls_onnx", default=None, help="覆盖 multitask_cls.onnx 路径")
    parser.add_argument("--tok_onnx", default=None, help="覆盖 multitask_patchtokens.onnx 路径")
    parser.add_argument("--input", required=True, help="单张图片或图片文件夹")
    parser.add_argument("--out_dir", default=None, help="输出目录，默认 onnx_dir/embeddings")
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda", "tensorrt"), default="auto")
    parser.add_argument("--recursive", action="store_true", help="递归读取输入文件夹")
    parser.add_argument("--k_min", type=int, default=1, help="最少 region 聚类数")
    parser.add_argument("--k_max", type=int, default=4, help="最多 region 聚类数")
    parser.add_argument("--kmeans_iters", type=int, default=15, help="KMeans 最大迭代次数")
    parser.add_argument("--min_region_tokens", type=int, default=4, help="region 最少 token 数")
    parser.add_argument("--seed", type=int, default=2026, help="聚类随机种子")
    parser.add_argument("--save_patch_tokens", action="store_true", help="额外保存归一化 patch token NPZ")
    parser.add_argument("--skip_errors", action="store_true", help="文件夹模式下跳过失败图片并返回成功")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.k_min <= 0 or args.k_max < args.k_min:
        raise ValueError(f"聚类数范围错误：k_min={args.k_min}，k_max={args.k_max}")
    if args.kmeans_iters <= 0:
        raise ValueError(f"kmeans_iters 必须大于 0，实际为 {args.kmeans_iters}")
    if args.min_region_tokens <= 0:
        raise ValueError(f"min_region_tokens 必须大于 0，实际为 {args.min_region_tokens}")

    cfg = load_config(args.config)
    onnx_dir = Path(args.onnx_dir).expanduser().resolve()
    if not onnx_dir.is_dir():
        raise FileNotFoundError(f"ONNX 目录不存在：{onnx_dir}")
    seg_onnx = Path(args.seg_onnx).expanduser().resolve() if args.seg_onnx else onnx_dir / "multitask_seg.onnx"
    cls_onnx = Path(args.cls_onnx).expanduser().resolve() if args.cls_onnx else onnx_dir / "multitask_cls.onnx"
    tok_onnx = Path(args.tok_onnx).expanduser().resolve() if args.tok_onnx else onnx_dir / "multitask_patchtokens.onnx"
    for path in (seg_onnx, cls_onnx, tok_onnx):
        if not path.is_file():
            raise FileNotFoundError(f"ONNX 文件不存在：{path}")

    images, input_root = resolve_images(Path(args.input), args.recursive)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else onnx_dir / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir_path = out_dir

    providers = select_providers(args.provider)
    sessions = {
        "seg": create_session(seg_onnx, providers),
        "cls": create_session(cls_onnx, providers),
        "tokens": create_session(tok_onnx, providers),
    }
    seg_hw = get_input_hw(cfg, "input")
    cls_hw = get_input_hw(cfg, "input_cls")
    input_names = {
        "seg": validate_session_contract(sessions["seg"], seg_hw, ("seg_logits",), "分割模型"),
        "cls": validate_session_contract(sessions["cls"], cls_hw, ("cls_emb",), "分类模型"),
        "tokens": validate_session_contract(sessions["tokens"], seg_hw, ("patch_tokens",), "Patch token 模型"),
    }

    success = 0
    records: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for image_path in images:
        relative_path = image_path.relative_to(input_root)
        output_path = out_dir / relative_path.with_suffix(".json")
        try:
            record = extract_one(
                image_path=image_path,
                relative_path=relative_path,
                cfg=cfg,
                sessions=sessions,
                input_names=input_names,
                args=args,
                image_seed=(
                    args.seed
                    + zlib.crc32(relative_path.as_posix().encode("utf-8"))
                ) % (2 ** 32),
            )
            save_json_atomic(output_path, record)
            success += 1
            records.append(
                {
                    "record_id": str(record["record_id"]),
                    "json_path": str(output_path),
                }
            )
            LOGGER.info(
                "Embedding 提取完成：%s；regions=%d；输出=%s",
                image_path.name,
                len(record["regions"]),
                output_path,
            )
        except Exception as exc:
            failures.append({"image_path": str(image_path), "error": str(exc)})
            LOGGER.error("Embedding 提取失败：%s；原因：%s", image_path, exc)

    summary = {
        "schema_version": 2,
        "input": str(Path(args.input).expanduser().resolve()),
        "output_dir": str(out_dir),
        "config": str(Path(args.config).expanduser().resolve()),
        "onnx": {
            "segmentation": str(seg_onnx),
            "classification": str(cls_onnx),
            "patch_tokens": str(tok_onnx),
        },
        "total": len(images),
        "success": success,
        "failed": len(failures),
        "records": records,
        "failures": failures,
    }
    summary_path = out_dir / "_metadata" / "embedding_summary.json"
    save_json_atomic(summary_path, summary)
    LOGGER.info("Embedding 提取结束：成功=%d，失败=%d，总数=%d", success, len(failures), len(images))

    if success == 0:
        raise RuntimeError("没有成功提取任何图片的 embedding")
    if failures and not args.skip_errors:
        raise RuntimeError(
            f"有 {len(failures)} 张图片提取失败，详情见 {summary_path}"
        )


if __name__ == "__main__":
    main()
