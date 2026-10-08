"""PyTorch SDPA fallback for HRM's expected flash_attn module."""

import torch
import torch.nn.functional as F


def flash_attn_func(q, k, v, causal=False):
    # HRM uses tensors shaped [batch, seq, heads, head_dim].
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2).contiguous()
