import torch
import torch.nn as nn
from torch import Tensor
    
class RotaryEmbedding(nn.Module):
    """
    Generate cos and sin given positional id (not fully understand)
    """
    def __init__(self, dim, max_seq_length, base=10_000):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(x.shape[2], device=x.device, dtype=torch.int64).unsqueeze(0).expand(x.shape[0], -1)
        
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates hald the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 : ]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_embed(q, k, cos, sin, positional_ids=None, unsqueeze_dim=1):
    """
    To better understand vectorized equation, rotate_half(x) = (-b, a) when x = (a, b)
    embed = x * cos + (rotate_half(x) * sin) = (a, b) * cos + (-b, a) * sin = (acos - bsin, asin + bcos)
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)
    
    else:
        q_embed = None
    
    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    
    else:
        k_embed = None
    return q_embed, k_embed
