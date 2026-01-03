# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    Optimized flash attention implementation with variable-length sequence support.

    Args:
        q:              (B, L_q, num_heads, head_dim) - query tensor
        k:              (B, L_k, num_heads, head_dim) - key tensor
        v:              (B, L_k, num_heads, head_dim) - value tensor
        q_lens:         (B,) or None - actual query sequence lengths per sample
        k_lens:         (B,) or None - actual key/value sequence lengths per sample
        dropout_p:      float. Dropout probability.
        softmax_scale:  float. The scaling of QK^T before applying softmax.
        q_scale:        float or None. Optional scaling factor for queries.
        causal:         bool. Whether to apply causal attention mask.
        window_size:    (left, right). If not (-1, -1), apply sliding window local attention.
        deterministic:  bool. If True, slightly slower and uses more memory.
        dtype:          torch.dtype. Target dtype when q/k/v is not float16/bfloat16.
        version:        int or None. Flash attention version (2 or 3).

    Returns:
        (B, L_q, num_heads, head_dim) - attention output
    """
    # q: (B, L_q, num_heads, head_dim)
    # k: (B, L_k, num_heads, head_dim)
    # v: (B, L_k, num_heads, head_dim)
    # q_lens: (B,) or None
    # k_lens: (B,) or None
    # Returns: (B, L_q, num_heads, head_dim)

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))  # q: (B*L_q, num_heads, head_dim)
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)  # q_lens: (B,) all equal to L_q
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))  # q: (sum(q_lens), num_heads, head_dim) - variable length packed

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))  # k: (B*L_k, num_heads, head_dim)
        v = half(v.flatten(0, 1))  # v: (B*L_k, num_heads, head_dim)
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)  # k_lens: (B,) all equal to L_k
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))  # k: (sum(k_lens), num_heads, head_dim)
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))  # v: (sum(k_lens), num_heads, head_dim)

    q = q.to(v.dtype)  # q: same shape, dtype matched
    k = k.to(v.dtype)  # k: same shape, dtype matched

    if q_scale is not None:
        q = q * q_scale  # q: scaled, same shape

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,  # q: (sum(q_lens), num_heads, head_dim)
            k=k,  # k: (sum(k_lens), num_heads, head_dim)
            v=v,  # v: (sum(k_lens), num_heads, head_dim)
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),  # cumulative sequence lengths: (B+1,)
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),  # cumulative sequence lengths: (B+1,)
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))  # x: (B, L_q, num_heads, head_dim)
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,  # q: (sum(q_lens), num_heads, head_dim)
            k=k,  # k: (sum(k_lens), num_heads, head_dim)
            v=v,  # v: (sum(k_lens), num_heads, head_dim)
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),  # cumulative sequence lengths: (B+1,)
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),  # cumulative sequence lengths: (B+1,)
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))  # x: (B, L_q, num_heads, head_dim)

    # output
    return x.type(out_dtype)  # -> (B, L_q, num_heads, head_dim)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """
    Wrapper for attention that uses flash attention if available, otherwise falls back to PyTorch's implementation.

    Args:
        q:              (B, L_q, num_heads, head_dim) - query tensor
        k:              (B, L_k, num_heads, head_dim) - key tensor
        v:              (B, L_k, num_heads, head_dim) - value tensor
        q_lens:         (B,) or None - actual query sequence lengths
        k_lens:         (B,) or None - actual key/value sequence lengths
        dropout_p:      float - dropout probability
        softmax_scale:  float or None - scaling for QK^T
        q_scale:        float or None - scaling for queries
        causal:         bool - whether to use causal masking
        window_size:    tuple - sliding window size
        deterministic:  bool - deterministic mode
        dtype:          torch.dtype - target dtype
        fa_version:     int or None - flash attention version

    Returns:
        (B, L_q, num_heads, head_dim) - attention output
    """
    # q: (B, L_q, num_heads, head_dim)
    # k: (B, L_k, num_heads, head_dim)
    # v: (B, L_k, num_heads, head_dim)
    # Returns: (B, L_q, num_heads, head_dim)

    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,  # (B, L_q, num_heads, head_dim)
            k=k,  # (B, L_k, num_heads, head_dim)
            v=v,  # (B, L_k, num_heads, head_dim)
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )  # -> (B, L_q, num_heads, head_dim)
    else:
        # Fallback to PyTorch's scaled_dot_product_attention
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)  # q: (B, num_heads, L_q, head_dim)
        k = k.transpose(1, 2).to(dtype)  # k: (B, num_heads, L_k, head_dim)
        v = v.transpose(1, 2).to(dtype)  # v: (B, num_heads, L_k, head_dim)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)  # out: (B, num_heads, L_q, head_dim)

        out = out.transpose(1, 2).contiguous()  # out: (B, L_q, num_heads, head_dim)
        return out  # -> (B, L_q, num_heads, head_dim)
