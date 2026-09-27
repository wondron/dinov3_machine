from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from dino_finetune.model.dino_multitask import DINOEncoderLoRA_MultiTask

logger = logging.getLogger(__name__)


def _estimate_k(m_tokens: int, k_min: int, k_max: int) -> int:
    if m_tokens <= 0:
        return 0
    # 按 token 数量做极简估计，再做上下界约束。
    k = int(round(math.sqrt(float(m_tokens))))
    k = max(int(k_min), min(int(k_max), k))
    return min(k, m_tokens)


def _kmeans_torch(x: torch.Tensor, k: int, num_iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    纯 torch 的极简 KMeans。
    x: (M, C), k: 聚类数
    return:
      labels: (M,)
      centers: (k, C)
    """
    if x.ndim != 2:
        raise ValueError(f"KMeans 输入维度错误，期望 (M,C)，实际为 {tuple(x.shape)}")

    m, _ = x.shape
    if m <= 0:
        raise ValueError("KMeans 输入为空，无法聚类")
    if k <= 0 or k > m:
        raise ValueError(f"KMeans 的 k 非法：k={k}, M={m}")

    init_idx = torch.randperm(m, device=x.device)[:k]
    centers = x[init_idx].clone()

    labels = torch.zeros(m, dtype=torch.long, device=x.device)
    for _ in range(max(1, int(num_iters))):
        x2 = (x * x).sum(dim=1, keepdim=True)          # (M,1)
        c2 = (centers * centers).sum(dim=1).view(1,-1) # (1,k)
        d2 = x2 + c2 - 2.0 * (x @ centers.t())         # (M,k)
        labels = torch.argmin(d2, dim=1)

        new_centers = centers.clone()
        for cid in range(k):
            cmask = labels == cid
            if cmask.any():
                new_centers[cid] = x[cmask].mean(dim=0)
            else:
                # 空簇时随机重置中心，避免 NaN。
                ridx = int(torch.randint(0, m, (1,), device=x.device).item())
                new_centers[cid] = x[ridx]
        centers = new_centers

    return labels, centers


@torch.inference_mode()
def token_cluster_proposals(
    model: DINOEncoderLoRA_MultiTask,
    x640: torch.Tensor,
    food_mask640: torch.Tensor,
    patch: int = 16,
    k_min: int = 2,
    k_max: int = 8,
    kmeans_iters: int = 15,
    min_area640: int = 800,
) -> list[list[torch.Tensor]]:
    """
    在 food 总 mask 内做 token 聚类，生成 proposal masks。
    返回：每张图一个 list，元素为 bool 类型 640x640 mask。
    """
    if x640.ndim != 4:
        raise ValueError(f"x640 维度错误，期望 (B,3,H,W)，实际为 {tuple(x640.shape)}")

    bsz, _, h, w = x640.shape
    if (h, w) != (640, 640):
        raise ValueError(f"x640 尺寸错误，期望 (640,640)，实际为 ({h},{w})")
    if int(patch) <= 0:
        raise ValueError("patch 必须为正整数")

    if food_mask640.ndim == 4 and food_mask640.shape[1] == 1:
        food_mask640 = food_mask640[:, 0]
    if food_mask640.ndim != 3:
        raise ValueError(
            f"food_mask640 维度错误，期望 (B,640,640) 或 (B,1,640,640)，实际为 {tuple(food_mask640.shape)}"
        )
    if food_mask640.shape[0] != bsz or food_mask640.shape[1] != h or food_mask640.shape[2] != w:
        raise ValueError("food_mask640 与 x640 的 batch/空间尺寸不一致")

    tok_grid = model.forward_token_grid(x640)  # (B,C,Ht,Wt)
    if tok_grid.ndim != 4:
        raise ValueError(f"token grid 维度错误，期望 (B,C,Ht,Wt)，实际为 {tuple(tok_grid.shape)}")

    _, channels, ht, wt = tok_grid.shape
    if ht * int(patch) != h or wt * int(patch) != w:
        raise ValueError(
            f"patch 与 token 网格不匹配：patch={patch}, token_grid=({ht},{wt}), 图像=({h},{w})"
        )

    food_mask_bool = food_mask640.to(device=x640.device, dtype=torch.bool)
    food_small = F.interpolate(
        food_mask_bool.unsqueeze(1).float(),
        size=(ht, wt),
        mode="nearest",
    ).squeeze(1) > 0.5  # (B,Ht,Wt)

    tok_flat = tok_grid.permute(0, 2, 3, 1).reshape(bsz, ht * wt, channels)  # (B,N,C)
    mask_flat = food_small.reshape(bsz, ht * wt)  # (B,N)

    results: list[list[torch.Tensor]] = []
    for bid in range(bsz):
        valid_idx = torch.nonzero(mask_flat[bid], as_tuple=False).squeeze(1)
        m_tokens = int(valid_idx.numel())
        if m_tokens <= 0:
            logger.info("第 %d 张图：food 区 token 数 M=0，跳过聚类，proposal=0", bid)
            results.append([])
            continue

        feat = tok_flat[bid, valid_idx]  # (M,C)
        feat = F.normalize(feat, p=2, dim=1)

        k_eff = _estimate_k(m_tokens, k_min=k_min, k_max=k_max)
        if k_eff <= 0:
            logger.info("第 %d 张图：聚类数 K=%d 非法，proposal=0", bid, k_eff)
            results.append([])
            continue

        labels, _ = _kmeans_torch(feat, k=k_eff, num_iters=int(kmeans_iters))

        # 一次性构建 label_map（长度 N=Ht*Wt），valid_idx 处填 labels，其余为 -1
        N = ht * wt
        label_map = torch.full((N,), -1, device=x640.device, dtype=torch.long)
        label_map[valid_idx] = labels  # 0..K-1

        proposals_b: list[torch.Tensor] = []
        for cid in range(k_eff):
            grid_mask_flat = (label_map == cid)
            if not grid_mask_flat.any():
                continue

            mask_small = grid_mask_flat.view(1, 1, ht, wt).float()
            mask_big = F.interpolate(mask_small, size=(h, w), mode="nearest")[0, 0] > 0.5
            mask_big = mask_big & food_mask_bool[bid]

            area = int(mask_big.sum().item())
            if area < int(min_area640):
                continue

            proposals_b.append(mask_big)  # 保持 device，不要 cpu

        logger.info(
            "第 %d 张图：food token M=%d，K=%d，保留 proposals=%d（min_area=%d）",
            bid, m_tokens, k_eff, len(proposals_b), int(min_area640)
        )
        results.append(proposals_b)

    return results
