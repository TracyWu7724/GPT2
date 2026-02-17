"""
Improved GPT2 Implementation with Modern Techniques.

This module provides a state-of-the-art GPT2 implementation incorporating
recent advances in transformer architecture design.

Improvements over vanilla GPT2:
- RoPE (Rotary Position Embeddings)
- RMSNorm (instead of LayerNorm)
- SwiGLU FFN
- GQA (Grouped Query Attention): K/V use fewer heads than Q, reducing KV-cache size
- Flash Attention 2 via F.scaled_dot_product_attention (fused CUDA kernel, O(N) memory)

"""


import os
import math
from typing import Dict, Any, Optional, List


import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rotary_pos_embed

from transformers import AutoTokenizer



class GPTEmbedding(nn.Module):
    """
    GPT embedding layer. Convert token ids into continousous vectors.
    """
    def __init__(self, vocab_size: int, emb_dim: int = 768, context_length: int = 512):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
    
    def forward(self, token_ids):
        return self.token_embeddings(token_ids)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False) # Gate
        self.w2 = nn.Linear(in_features, hidden_features, bias=False) # Up
        self.w3 = nn.Linear(hidden_features, out_features, bias=False) # Down

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class VanillaMHA(nn.Module):
    """
    This vanilla attention follows the formula: Softmax((q @ k^T) / sqrt(d_model // num_heads)) @ V (batch_size, num_heads, seq_length, seq_length).
    It is only for reference and will not be used in the main GPTBlock.

    Dim check:
        d_head = d_model // num_heads
        x: (B, L, d_model)
        q, k, v: (B, H, L, d_head)
        k^T: (B, H, d_head, L)
        attn scores: (B, H, L, L)
        final_out: (B, L, d_model)

    """
    def __init__(self, d_in, num_heads, bias_qkv=False):
        super().__init__()
        d_out = d_in
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.d_head = d_in // num_heads

        self.q_linear = nn.Linear(in_features=d_in, out_features=d_out, bias=bias_qkv)
        self.k_linear = nn.Linear(in_features=d_in, out_features=d_out, bias=bias_qkv)
        self.v_linear = nn.Linear(in_features=d_in, out_features=d_out, bias=bias_qkv)

        self.out = nn.Linear(in_features=d_in, out_features=d_out, bias=bias_qkv)

    def forward(self, x, mask=None):
        batch_size, L, _ = x.shape
        
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head) # (B, H, L, d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        output = attn @ v

        concat = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_out) # (B, L, d_model)
        output = self.out(concat)

        return output, attn


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads to match the number of query heads in GQA.

    When n_rep=1 (i.e. num_heads == num_kv_heads, standard MHA), this is a no-op.
    Otherwise, each KV head is duplicated n_rep times along the head dimension
    so that every query head group has its own copy of the shared KV head.

    Args:
        x: (B, num_kv_heads, L, d_head)
        n_rep: num_heads // num_kv_heads (number of query heads per KV group)
    Returns:
        (B, num_heads, L, d_head)
    """
    if n_rep == 1:
        return x
    B, num_kv_heads, L, d_head = x.shape
    # (B, num_kv_heads, 1, L, d_head) -> (B, num_kv_heads, n_rep, L, d_head) -> merge
    return (
        x[:, :, None, :, :]
        .expand(B, num_kv_heads, n_rep, L, d_head)
        .reshape(B, num_kv_heads * n_rep, L, d_head)
    )


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Grouped Query Attention (GQA) and Flash Attention.

    GQA: Q has `num_heads` heads, while K and V share `num_kv_heads` heads
    (num_kv_heads <= num_heads). Each group of (num_heads // num_kv_heads) query
    heads attends to the same KV head. This reduces KV-cache memory by the group
    factor without significant quality loss.

    Special cases:
        num_kv_heads == num_heads  -> standard Multi-Head Attention (MHA)
        num_kv_heads == 1          -> Multi-Query Attention (MQA)

    Flash Attention: uses F.scaled_dot_product_attention, which dispatches to
    FlashAttention-2 or memory-efficient attention fused CUDA kernels when available,
    giving O(N) memory and better wall-clock speed than the manual softmax path.
    """
    def __init__(self, d_in, num_heads, num_kv_heads, max_seq_length, dropout, bias_qkv=False):
        super().__init__()
        d_out = d_in
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads  # how many Q heads share one KV head
        self.d_head = d_out // num_heads

        # Q projection: full num_heads
        self.W_query = nn.Linear(d_in, num_heads * self.d_head, bias=bias_qkv)
        # K, V projections: only num_kv_heads (smaller when GQA is active)
        self.W_key = nn.Linear(d_in, num_kv_heads * self.d_head, bias=bias_qkv)
        self.W_value = nn.Linear(d_in, num_kv_heads * self.d_head, bias=bias_qkv)
        self.out = nn.Linear(d_out, d_out)

        self.dropout_p = dropout

        self.rope = RotaryEmbedding(self.d_head, max_seq_length)

    def forward(self, x, kv_cache=None, use_cache=None):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V and reshape into multi-head layout
        q = self.W_query(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_key(x).view(batch_size, seq_len, self.num_kv_heads, self.d_head).transpose(1, 2)
        v = self.W_value(x).view(batch_size, seq_len, self.num_kv_heads, self.d_head).transpose(1, 2)
        # q: (B, num_heads, L, d_head), k/v: (B, num_kv_heads, L, d_head)

        # Apply RoPE (works per-head independently, head count doesn't matter)
        rope_cos, rope_sin = self.rope(q)
        q, _ = apply_rotary_pos_embed(q, None, rope_cos, rope_sin)
        # RoPE cos/sin are generated from q's seq_len; recompute for k if num_kv_heads differs
        rope_cos_k, rope_sin_k = self.rope(k)
        _, k = apply_rotary_pos_embed(None, k, rope_cos_k, rope_sin_k)

        # Handle KV cache for autoregressive generation
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # GQA: expand KV heads to match Q heads by repeating each KV head n_rep times
        k_expanded = repeat_kv(k, self.n_rep)  # (B, num_heads, kv_len, d_head)
        v_expanded = repeat_kv(v, self.n_rep)  # (B, num_heads, kv_len, d_head)

        # Flash Attention via PyTorch's SDPA (dispatches to FlashAttention-2 / memory-efficient kernels)
        dropout_p = self.dropout_p if self.training else 0.0
        output = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )
        # output: (B, num_heads, L, d_head)

        # Concat all heads back into (B, L, d_out)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)
        output = self.out(output)

        return output, new_cache

class FeedForward(nn.Module):
    """
    FFN(x) = (Swish(xW1) ⊙ xW2) W3
    """
    def __init__(self, emb_dim, expansion=8/3):
        super().__init__()

        d_ff = int(round(emb_dim * expansion))

        self.fc1 = nn.Linear(emb_dim, 2 * d_ff) # fused gate and up into one layer
        self.fc2 = nn.Linear(d_ff, emb_dim)
    
    def forward(self, x):
        gate, up = self.fc1(x).chunk(2, dim=-1)
        output = F.silu(gate) * up
        return self.fc2(output)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_in=cfg["emb_dim"],
                                            num_heads=cfg["n_heads"],
                                            num_kv_heads=cfg.get("n_kv_heads", cfg["n_heads"]),
                                            max_seq_length=cfg["context_length"],
                                            dropout=cfg["drop_rate"])
        self.ffn = FeedForward(cfg["emb_dim"])
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])
        self.dropout_p = cfg["drop_rate"]
    
    def maybe_dropout(self, x):
        if self.dropout_p > 0:
            return nn.functional.dropout(x, p=self.dropout_p, training=self.training)
        
        elif self.dropout_p == 0:
            return x

    def forward(self, x, kv_cache=None, use_cache=False):
        """
        Forward pass through the transformer block.

        Args:
            x: Input hidden states of shape [batch_size, seq_len, emb_dim]
        Returns:
            Output hidden states of shape [batch_size, seq_len, emb_dim]
        """
        attn_out, new_cache = self.self_attn(self.norm1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = self.maybe_dropout(attn_out) + x
        x = self.maybe_dropout(self.ffn(self.norm2(x))) + x

        return x, new_cache

class GPT(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.embedding = GPTEmbedding(cfg["vocab_size"], cfg["emb_dim"], context_length=cfg["context_length"])
        
        self.dropout = nn.Dropout(p=cfg["drop_rate"])

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg["n_layers"])
        ])

        self.final_norm = RMSNorm(cfg["emb_dim"])

        self.lm_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Weight tying: share weights between embedding and lm_head
        self.lm_head.weight = self.embedding.token_embeddings.weight

    def forward(self, token_ids, kv_cache=None, use_cache=False):
        """
        Args:
            token_ids: (B, T) input token indices
            kv_cache: list of per-layer (k, v) tuples from previous forward pass
            use_cache: whether to return new kv_cache for autoregressive generation

        Returns:
            logits: (B, T, vocab_size)
            new_caches: list of per-layer (k, v) tuples, or None
        """
        x = self.embedding(token_ids)  # (B, T, emb_dim)
        x = self.dropout(x)

        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, cache = block(x, kv_cache=layer_cache, use_cache=use_cache)
            new_caches.append(cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits, new_caches if use_cache else None

# =============================================================================
# Text Generation Functions
# =============================================================================

def generate_new_tokens(model, idx, max_new_tokens, context_size, temperature=1.0):
    """
    Autoregressively generates `max_new_tokens` tokens from the model.

    Args:
        model: The language model
        idx: Starting tensor of shape (batch, seq)
        max_new_tokens: Number of tokens to generate
        context_size: Context window size for the model input
        temperature: Softmax temperature (>0). Lower = more greedy, higher = more random
    Returns:
        idx: The resulting sequence with new tokens appended
    """
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:].to(device)

        with torch.no_grad():
            logits, _ = model(idx_cond)

        logits = logits[:, -1, :]  # Final token in the sequence
        logits = logits / temperature  # Apply temperature
        
        probas = torch.softmax(logits, dim=-1)
        # Sample from the distribution rather than argmax for more natural randomness
        idx_next = torch.multinomial(probas, num_samples=1)
        # Keep new token on the same device as the running sequence to avoid device mismatch
        idx_next = idx_next.to(idx.device)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_text(start_context: str, tokenizer, model, max_new_tokens, context_size):
    """
    Generate text from a starting context.

    Args:
        start_context: Starting text prompt
        tokenizer: Tokenizer to use for encoding/decoding
        model: GPT model
        max_new_tokens: Number of tokens to generate
        context_size: Context window size
    Returns:
        Generated text string
    """
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    model.eval()
    out = generate_new_tokens(model=model, idx=encoded_tensor,
                              max_new_tokens=max_new_tokens,
                              context_size=context_size)
    print("Output:", out)
    print("Output length:", len(out[0]))
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    return decoded_text

