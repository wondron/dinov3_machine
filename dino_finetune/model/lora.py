import math

import torch
import torch.nn as nn


class LoRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

        self.in_features = qkv.in_features
        self.out_features = qkv.out_features

    def forward(self, x) -> torch.Tensor:
        # Compute the original qkv: (B, N, 3*dim)
        qkv = self.qkv(x)
        delta_q = self.linear_b_q(self.linear_a_q(x))  # (B, N, dim)
        delta_v = self.linear_b_v(self.linear_a_v(x))  # (B, N, dim)
        dim = self.dim
        q, k, v = qkv.split(dim, dim=-1)
        q = q + delta_q
        v = v + delta_v
        return torch.cat((q, k, v), dim=-1)


def inject_lora(encoder: nn.Module, r: int) -> int:
    """给 encoder 每个 block 的 attn.qkv 注入 LoRA（只作用于 Q、V），返回注入的 block 数。"""
    if not hasattr(encoder, "blocks"):
        raise ValueError("当前 encoder 不支持 LoRA 注入（缺少 blocks 属性）")

    for block in encoder.blocks:
        qkv = block.attn.qkv
        dim = qkv.in_features
        a_q, a_v = nn.Linear(dim, r, bias=False), nn.Linear(dim, r, bias=False)
        b_q, b_v = nn.Linear(r, dim, bias=False), nn.Linear(r, dim, bias=False)
        for w_a in (a_q, a_v):
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in (b_q, b_v):
            nn.init.zeros_(w_b.weight)
        block.attn.qkv = LoRA(qkv, a_q, b_q, a_v, b_v).to(qkv.weight.device)
    return len(encoder.blocks)
