from __future__ import annotations

import argparse
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


LOGGER = logging.getLogger("embedding_recall_evaluation")


@dataclass
class EmbeddingRecord:
    record_id: str
    json_path: Path
    image_path: str
    global_embedding: np.ndarray
    region_embeddings: np.ndarray
    region_weights: np.ndarray


def normalize_record_id(value: str) -> str:
    normalized = str(value).strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def normalize_vector(value: Any, context: str, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{context} 必须是一维非空向量，实际 shape={vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{context} 包含 NaN 或 Inf")
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        raise ValueError(f"{context} 是零向量")
    return vector / norm


def load_embedding_record(json_path: Path, root: Path) -> EmbeddingRecord:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"JSON 读取失败：{json_path}，原因：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Embedding JSON 顶层必须是对象：{json_path}")
    if "global_emb" not in data or "regions" not in data:
        raise ValueError(f"Embedding JSON 缺少 global_emb/regions：{json_path}")

    fallback_id = json_path.relative_to(root).with_suffix("").as_posix()
    record_id = normalize_record_id(
        str(data.get("record_id") or data.get("relative_path") or fallback_id)
    )
    if not record_id:
        raise ValueError(f"record_id 不能为空：{json_path}")

    global_embedding = normalize_vector(data["global_emb"], f"{record_id}.global_emb")
    regions_data = data["regions"]
    if not isinstance(regions_data, list):
        raise ValueError(f"{record_id}.regions 必须是列表")

    region_embeddings: list[np.ndarray] = []
    raw_weights: list[float] = []
    for region_index, region in enumerate(regions_data):
        if not isinstance(region, dict) or "embedding" not in region:
            raise ValueError(f"{record_id}.regions[{region_index}] 缺少 embedding")
        embedding = normalize_vector(
            region["embedding"],
            f"{record_id}.regions[{region_index}].embedding",
        )
        if embedding.shape[0] != global_embedding.shape[0]:
            raise ValueError(
                f"{record_id} embedding 维度不一致："
                f"global={global_embedding.shape[0]}，region={embedding.shape[0]}"
            )
        weight_value = region.get("token_ratio", region.get("token_count", 1.0))
        try:
            weight = float(weight_value)
        except Exception as exc:
            raise ValueError(
                f"{record_id}.regions[{region_index}] 权重无法转换为数字：{weight_value}"
            ) from exc
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"{record_id}.regions[{region_index}] 权重必须是有限非负数")
        region_embeddings.append(embedding)
        raw_weights.append(weight)

    embedding_dim = int(global_embedding.shape[0])
    if region_embeddings:
        region_matrix = np.stack(region_embeddings).astype(np.float32)
        weights = np.asarray(raw_weights, dtype=np.float32)
        if float(weights.sum()) <= 1e-12:
            weights = np.ones(len(region_embeddings), dtype=np.float32)
        weights /= weights.sum()
    else:
        region_matrix = np.empty((0, embedding_dim), dtype=np.float32)
        weights = np.empty((0,), dtype=np.float32)

    return EmbeddingRecord(
        record_id=record_id,
        json_path=json_path,
        image_path=str(data.get("image_path", "")),
        global_embedding=global_embedding.astype(np.float32),
        region_embeddings=region_matrix,
        region_weights=weights,
    )


def discover_embedding_jsons(root: Path, recursive: bool) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Embedding 目录不存在：{root}")
    iterator = root.rglob("*.json") if recursive else root.glob("*.json")
    paths = sorted(
        path for path in iterator
        if path.is_file() and "_metadata" not in path.relative_to(root).parts
    )
    if not paths:
        raise FileNotFoundError(f"Embedding 目录中没有 JSON：{root}")
    return paths


def load_records(
    root: Path,
    recursive: bool,
    skip_invalid: bool,
) -> tuple[list[EmbeddingRecord], list[dict[str, str]]]:
    records: list[EmbeddingRecord] = []
    failures: list[dict[str, str]] = []
    seen_ids: dict[str, Path] = {}
    for json_path in discover_embedding_jsons(root, recursive):
        try:
            record = load_embedding_record(json_path, root)
            if record.record_id in seen_ids:
                raise ValueError(
                    f"record_id 重复：{record.record_id}，文件={seen_ids[record.record_id]} 和 {json_path}"
                )
            seen_ids[record.record_id] = json_path
            records.append(record)
        except Exception as exc:
            if not skip_invalid:
                raise
            failures.append({"json_path": str(json_path), "error": str(exc)})
            LOGGER.warning("跳过无效 embedding：%s；原因：%s", json_path, exc)

    if not records:
        raise RuntimeError(f"没有加载到有效 embedding：{root}")
    dimensions = {int(record.global_embedding.shape[0]) for record in records}
    if len(dimensions) != 1:
        raise ValueError(f"数据库包含不同 embedding 维度：{sorted(dimensions)}")
    return records, failures


def load_query_records(
    query_json: Path | None,
    query_dir: Path | None,
    recursive: bool,
    skip_invalid: bool,
) -> tuple[list[EmbeddingRecord], list[dict[str, str]]]:
    if query_json is not None:
        path = query_json.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Query JSON 不存在：{path}")
        return [load_embedding_record(path, path.parent)], []
    if query_dir is None:
        raise ValueError("必须提供 query_json 或 query_dir")
    return load_records(query_dir.expanduser().resolve(), recursive, skip_invalid)


def cosine_to_unit(value: float) -> float:
    return float(np.clip((value + 1.0) * 0.5, 0.0, 1.0))


def greedy_one_to_one_region_match(
    query: EmbeddingRecord,
    candidate: EmbeddingRecord,
    match_threshold: float,
) -> dict[str, Any]:
    query_count = int(query.region_embeddings.shape[0])
    candidate_count = int(candidate.region_embeddings.shape[0])
    if query_count == 0 or candidate_count == 0:
        return {
            "available": False,
            "query_region_count": query_count,
            "candidate_region_count": candidate_count,
            "match_count": 0,
            "region_similarity": 0.0,
            "query_coverage": 0.0,
            "candidate_coverage": 0.0,
            "coverage": 0.0,
            "matches": [],
        }

    similarity_matrix = np.clip(
        query.region_embeddings @ candidate.region_embeddings.T,
        -1.0,
        1.0,
    )
    possible_pairs = [
        (float(similarity_matrix[query_index, candidate_index]), query_index, candidate_index)
        for query_index in range(query_count)
        for candidate_index in range(candidate_count)
        if float(similarity_matrix[query_index, candidate_index]) >= match_threshold
    ]
    possible_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_query: set[int] = set()
    used_candidate: set[int] = set()
    matches: list[dict[str, Any]] = []
    weighted_similarity_sum = 0.0
    pair_weight_sum = 0.0
    for similarity, query_index, candidate_index in possible_pairs:
        if query_index in used_query or candidate_index in used_candidate:
            continue
        used_query.add(query_index)
        used_candidate.add(candidate_index)
        pair_weight = float(
            np.sqrt(
                query.region_weights[query_index]
                * candidate.region_weights[candidate_index]
            )
        )
        similarity_unit = cosine_to_unit(similarity)
        weighted_similarity_sum += pair_weight * similarity_unit
        pair_weight_sum += pair_weight
        matches.append(
            {
                "query_region": query_index,
                "candidate_region": candidate_index,
                "cosine": similarity,
                "similarity": similarity_unit,
                "pair_weight": pair_weight,
            }
        )

    query_coverage = float(query.region_weights[list(used_query)].sum()) if used_query else 0.0
    candidate_coverage = (
        float(candidate.region_weights[list(used_candidate)].sum())
        if used_candidate else 0.0
    )
    if query_coverage > 0.0 and candidate_coverage > 0.0:
        coverage = 2.0 * query_coverage * candidate_coverage / (
            query_coverage + candidate_coverage
        )
    else:
        coverage = 0.0
    region_similarity = (
        weighted_similarity_sum / pair_weight_sum if pair_weight_sum > 1e-12 else 0.0
    )
    return {
        "available": True,
        "query_region_count": query_count,
        "candidate_region_count": candidate_count,
        "match_count": len(matches),
        "region_similarity": float(region_similarity),
        "query_coverage": query_coverage,
        "candidate_coverage": candidate_coverage,
        "coverage": float(coverage),
        "matches": matches,
    }


def same_source(query: EmbeddingRecord, candidate: EmbeddingRecord) -> bool:
    if query.record_id == candidate.record_id:
        return True
    if not query.image_path or not candidate.image_path:
        return False
    try:
        return Path(query.image_path).expanduser().resolve() == Path(candidate.image_path).expanduser().resolve()
    except Exception:
        return query.image_path == candidate.image_path


def fuse_scores(
    global_score: float,
    region_report: dict[str, Any],
    global_weight: float,
    region_weight: float,
    coverage_weight: float,
) -> tuple[float, dict[str, float]]:
    # Query 本身没有 region 时只能依赖全局特征；如果仅候选缺少 region，
    # 保留局部项的零分惩罚，避免分割失败的候选获得不公平优势。
    if int(region_report["query_region_count"]) == 0:
        return global_score, {
            "global": 1.0,
            "region": 0.0,
            "coverage": 0.0,
        }

    total_weight = global_weight + region_weight + coverage_weight
    normalized_weights = {
        "global": global_weight / total_weight,
        "region": region_weight / total_weight,
        "coverage": coverage_weight / total_weight,
    }
    score = (
        normalized_weights["global"] * global_score
        + normalized_weights["region"] * float(region_report["region_similarity"])
        + normalized_weights["coverage"] * float(region_report["coverage"])
    )
    return float(score), normalized_weights


def retrieve_one(
    query: EmbeddingRecord,
    database: list[EmbeddingRecord],
    output_limit: int,
    candidate_k: int,
    match_threshold: float,
    global_weight: float,
    region_weight: float,
    coverage_weight: float,
    include_self: bool,
) -> list[dict[str, Any]]:
    eligible = [
        record for record in database
        if include_self or not same_source(query, record)
    ]
    if not eligible:
        return []
    expected_dim = int(query.global_embedding.shape[0])
    invalid_dimensions = [
        record.record_id
        for record in eligible
        if int(record.global_embedding.shape[0]) != expected_dim
    ]
    if invalid_dimensions:
        raise ValueError(
            f"Query={query.record_id} 与数据库 embedding 维度不一致：{invalid_dimensions[:10]}"
        )

    database_matrix = np.stack([record.global_embedding for record in eligible])
    global_cosines = np.clip(database_matrix @ query.global_embedding, -1.0, 1.0)
    coarse_order = sorted(
        range(len(eligible)),
        key=lambda index: (-float(global_cosines[index]), eligible[index].record_id),
    )
    if candidate_k > 0:
        coarse_order = coarse_order[:max(candidate_k, output_limit)]

    reranked: list[dict[str, Any]] = []
    for coarse_rank, database_index in enumerate(coarse_order, start=1):
        candidate = eligible[database_index]
        global_cosine = float(global_cosines[database_index])
        global_score = cosine_to_unit(global_cosine)
        region_report = greedy_one_to_one_region_match(
            query,
            candidate,
            match_threshold,
        )
        score, applied_weights = fuse_scores(
            global_score,
            region_report,
            global_weight,
            region_weight,
            coverage_weight,
        )
        reranked.append(
            {
                "record_id": candidate.record_id,
                "image_path": candidate.image_path,
                "json_path": str(candidate.json_path),
                "score": score,
                "global_cosine": global_cosine,
                "global_score": global_score,
                "coarse_rank": coarse_rank,
                "query_region_count": region_report["query_region_count"],
                "candidate_region_count": region_report["candidate_region_count"],
                "region_match_count": region_report["match_count"],
                "region_similarity": region_report["region_similarity"],
                "query_coverage": region_report["query_coverage"],
                "candidate_coverage": region_report["candidate_coverage"],
                "coverage": region_report["coverage"],
                "applied_weights": applied_weights,
                "region_matches": region_report["matches"],
            }
        )

    reranked.sort(
        key=lambda item: (
            -float(item["score"]),
            -float(item["global_cosine"]),
            str(item["record_id"]),
        )
    )
    for rank, result in enumerate(reranked, start=1):
        result["rank"] = rank
    return reranked[:output_limit]


def parse_eval_ks(value: str) -> list[int]:
    try:
        ks = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except Exception as exc:
        raise ValueError(f"eval_ks 格式错误：{value}") from exc
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"eval_ks 必须包含正整数：{value}")
    return ks


def load_ground_truth(path: Path) -> dict[str, set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth 文件不存在：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("queries"), dict):
        data = data["queries"]
    if not isinstance(data, dict):
        raise ValueError("Ground truth 顶层必须是 query_id -> relevant_ids 映射")

    ground_truth: dict[str, set[str]] = {}
    for query_id, value in data.items():
        relevant = value.get("relevant") if isinstance(value, dict) else value
        if not isinstance(relevant, list):
            raise ValueError(f"Ground truth[{query_id}] 必须是列表或包含 relevant 列表")
        normalized_relevant = {
            normalize_record_id(str(item)) for item in relevant if str(item).strip()
        }
        if normalized_relevant:
            ground_truth[normalize_record_id(str(query_id))] = normalized_relevant
    if not ground_truth:
        raise ValueError("Ground truth 中没有有效关联")
    return ground_truth


def evaluate_results(
    query_results: list[dict[str, Any]],
    ground_truth: dict[str, set[str]],
    eval_ks: list[int],
) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    for query_result in query_results:
        query_id = normalize_record_id(str(query_result["query_record_id"]))
        relevant = ground_truth.get(query_id)
        if not relevant:
            LOGGER.warning("Ground truth 缺少 query：%s，已跳过评估", query_id)
            continue
        ranked_ids = [normalize_record_id(str(item["record_id"])) for item in query_result["results"]]
        metrics: dict[str, float] = {}
        first_relevant_rank = 0
        for rank, record_id in enumerate(ranked_ids, start=1):
            if record_id in relevant:
                first_relevant_rank = rank
                break
        max_k = max(eval_ks)
        metrics[f"mrr@{max_k}"] = (
            1.0 / first_relevant_rank
            if 0 < first_relevant_rank <= max_k else 0.0
        )
        for k in eval_ks:
            retrieved = ranked_ids[:k]
            hit_count = len(set(retrieved) & relevant)
            metrics[f"recall@{k}"] = float(hit_count / len(relevant))
            metrics[f"precision@{k}"] = float(hit_count / k)
            metrics[f"hit_rate@{k}"] = 1.0 if hit_count > 0 else 0.0
        per_query.append(
            {
                "query_record_id": query_id,
                "relevant_count": len(relevant),
                **metrics,
            }
        )

    if not per_query:
        raise RuntimeError("没有 query 能够与 ground truth 匹配，无法评估")
    metric_names = [key for key in per_query[0] if key not in {"query_record_id", "relevant_count"}]
    aggregate = {
        name: float(np.mean([float(item[name]) for item in per_query]))
        for name in metric_names
    }
    return {
        "evaluated_queries": len(per_query),
        "aggregate": aggregate,
        "per_query": per_query,
    }


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
        json.dump(data, temporary_file, ensure_ascii=False, indent=2)
        temporary_path = Path(temporary_file.name)
    try:
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 embedding 两阶段召回与离线评估")
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query_json", help="单个 query embedding JSON")
    query_group.add_argument("--query_dir", help="批量 query embedding 目录")
    parser.add_argument("--db_dir", required=True, help="候选 embedding JSON 目录")
    parser.add_argument("--topk", type=int, default=5, help="每个 query 输出的结果数")
    parser.add_argument("--candidate_k", type=int, default=200, help="全局粗召回候选数，0 表示全部重排")
    parser.add_argument("--match_thr", type=float, default=0.75, help="region cosine 匹配阈值")
    parser.add_argument("--global_weight", type=float, default=0.60, help="全局相似度权重")
    parser.add_argument("--region_weight", type=float, default=0.25, help="region 相似度权重")
    parser.add_argument("--coverage_weight", type=float, default=0.15, help="双向覆盖率权重")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否递归读取 embedding 目录",
    )
    parser.add_argument("--include_self", action="store_true", help="允许返回 query 自身")
    parser.add_argument("--skip_invalid", action="store_true", help="跳过无效 embedding JSON")
    parser.add_argument("--output_json", default=None, help="保存召回与评估报告")
    parser.add_argument("--ground_truth", default=None, help="可选 query_id -> relevant_ids JSON")
    parser.add_argument("--eval_ks", default="1,5,10", help="评估 K，逗号分隔")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.topk <= 0:
        raise ValueError(f"topk 必须大于 0，实际为 {args.topk}")
    if args.candidate_k < 0:
        raise ValueError(f"candidate_k 不能为负数，实际为 {args.candidate_k}")
    if not -1.0 <= args.match_thr <= 1.0:
        raise ValueError(f"match_thr 必须位于 [-1,1]，实际为 {args.match_thr}")
    weights = (args.global_weight, args.region_weight, args.coverage_weight)
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError(f"融合权重必须为非负数且总和大于 0，实际为 {weights}")

    db_dir = Path(args.db_dir).expanduser().resolve()
    database, database_failures = load_records(db_dir, args.recursive, args.skip_invalid)
    query_json = Path(args.query_json) if args.query_json else None
    query_dir = Path(args.query_dir) if args.query_dir else None
    queries, query_failures = load_query_records(
        query_json,
        query_dir,
        args.recursive,
        args.skip_invalid,
    )
    LOGGER.info(
        "Embedding 加载完成：数据库=%d，query=%d，无效数据库=%d，无效 query=%d",
        len(database),
        len(queries),
        len(database_failures),
        len(query_failures),
    )

    eval_ks = parse_eval_ks(args.eval_ks)
    output_limit = max(args.topk, max(eval_ks) if args.ground_truth else args.topk)
    all_query_results: list[dict[str, Any]] = []
    for query in queries:
        results = retrieve_one(
            query=query,
            database=database,
            output_limit=output_limit,
            candidate_k=args.candidate_k,
            match_threshold=args.match_thr,
            global_weight=args.global_weight,
            region_weight=args.region_weight,
            coverage_weight=args.coverage_weight,
            include_self=args.include_self,
        )
        all_query_results.append(
            {
                "query_record_id": query.record_id,
                "query_image_path": query.image_path,
                "query_json_path": str(query.json_path),
                "results": results,
            }
        )
        LOGGER.info("Query=%s，召回结果=%d", query.record_id, len(results))
        for result in results[:args.topk]:
            LOGGER.info(
                "  Top-%d：%s；score=%.6f；global=%.6f；region=%.6f；coverage=%.6f",
                result["rank"],
                result["record_id"],
                result["score"],
                result["global_cosine"],
                result["region_similarity"],
                result["coverage"],
            )

    evaluation = None
    if args.ground_truth:
        ground_truth = load_ground_truth(Path(args.ground_truth).expanduser().resolve())
        evaluation = evaluate_results(all_query_results, ground_truth, eval_ks)
        LOGGER.info("离线评估完成：query=%d", evaluation["evaluated_queries"])
        for metric, value in evaluation["aggregate"].items():
            LOGGER.info("  %s=%.6f", metric, value)

    report: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": {
            "name": "global_cosine_plus_greedy_region_rerank",
            "candidate_k": int(args.candidate_k),
            "match_threshold": float(args.match_thr),
            "weights": {
                "global": float(args.global_weight),
                "region": float(args.region_weight),
                "coverage": float(args.coverage_weight),
            },
            "include_self": bool(args.include_self),
        },
        "database_dir": str(db_dir),
        "database_count": len(database),
        "query_count": len(queries),
        "invalid_database": database_failures,
        "invalid_queries": query_failures,
        "queries": all_query_results,
        "evaluation": evaluation,
    }
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        save_json_atomic(output_path, report)
        LOGGER.info("召回报告已保存：%s", output_path)


if __name__ == "__main__":
    main()
