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
