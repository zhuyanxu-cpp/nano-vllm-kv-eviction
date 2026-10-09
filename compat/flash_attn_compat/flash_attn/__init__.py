"""Pure-PyTorch stand-in for the subset of flash-attn that nano-vllm imports.

Why this exists
---------------
``nanovllm/layers/attention.py`` does::

    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

The real package ships no Windows wheel, and its CUDA kernels require Ampere
(sm_80) or newer, so it cannot run on a Turing GPU such as the GTX 1650 Ti
(sm_75). This module reimplements the two used entry points on top of
``torch.nn.functional.scaled_dot_product_attention``, which falls back to the
memory-efficient kernel on sm_75.

Supported / intentionally unsupported
-------------------------------------
Supported: the exact call patterns used by nano-vllm --

* ``flash_attn_varlen_func`` with dense (unpaged) q/k/v, plus ``causal``;
* ``flash_attn_varlen_func`` with a paged KV cache and ``block_table``
  (nano-vllm's prefix-caching prefill path);
* ``flash_attn_with_kvcache`` single-query decode with ``cache_seqlens`` +
  ``block_table``;
* grouped-query attention (``num_heads > num_kv_heads``).

Unsupported (raise ``NotImplementedError``): dropout, alibi slopes, sliding
window, softcap, ``return_attn_probs``, left padding, local attention. nano-vllm
uses none of them.
"""

import math

import torch
import torch.nn.functional as F

__all__ = ["flash_attn_varlen_func", "flash_attn_with_kvcache"]

_UNSUPPORTED = (
    "alibi_slopes",
    "window_size",
    "softcap",
    "return_attn_probs",
    "leftpad_k",
)


def _check_kwargs(kwargs, fn):
    if kwargs.get("dropout_p", 0.0):
        raise NotImplementedError(f"flash_attn shim: dropout is not supported ({fn})")
    for name in _UNSUPPORTED:
        value = kwargs.get(name)
        if value is None:
            continue
        if name == "window_size" and tuple(value) == (-1, -1):
            continue
        if name == "return_attn_probs" and value is False:
            continue
        raise NotImplementedError(f"flash_attn shim: {name} is not supported ({fn})")


def _expand_kv(x, n_rep):
    """Repeat KV heads to match the query head count. (..., n_kv_heads, D)."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=-2)


def _attend(q, k, v, scale, causal, key_lens=None):
    """Scaled dot-product attention on (B, H, Tq, D) / (B, H, Tk, D) tensors.

    ``key_lens`` marks how many leading keys of each batch row are real (the
    rest are padding produced by the per-sequence gather). When ``causal`` is
    set, query ``i`` may attend key ``j`` only while ``j <= i + (key_lens - Tq)``
    -- i.e. the causal mask is aligned to the *bottom right*, which is what
    nano-vllm needs when a query chunk is shorter than its key history.
    """
    tq, tk = q.shape[-2], k.shape[-2]
    attn_mask = None
    is_causal = False
    if key_lens is not None:
        idx = torch.arange(tk, device=q.device)
        mask = (idx[None, :] < key_lens[:, None])[:, None, None, :].expand(-1, -1, tq, -1)
        if causal:
            qidx = torch.arange(tq, device=q.device)[None, :, None]
            offset = (key_lens - tq)[:, None, None]
            mask = mask & (idx[None, None, :] <= qidx + offset)[:, None]
        attn_mask = mask
    elif causal:
        if tq == tk:
            is_causal = True
        else:
            idx = torch.arange(tk, device=q.device)
            qidx = torch.arange(tq, device=q.device)[:, None]
            attn_mask = (idx[None, :] <= qidx + (tk - tq))[None, None]
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=scale, is_causal=is_causal
    ).to(q.dtype)


def _heads(q, k, v, scale, causal):
    """One sequence: q (Tq, H, D), k/v (Tk, KvH, D) -> (Tq, H, D)."""
    n_heads, n_kv_heads = q.shape[1], k.shape[1]
    q_ = q.transpose(0, 1).unsqueeze(0)
    k_ = _expand_kv(k, n_heads // n_kv_heads).transpose(0, 1).unsqueeze(0)
    v_ = _expand_kv(v, n_heads // n_kv_heads).transpose(0, 1).unsqueeze(0)
    return _attend(q_, k_, v_, scale, causal).squeeze(0).transpose(0, 1).contiguous()


def _dequant_k(k_i8, k_scale, group):
    """(T, KvH, D) int8 + (T, KvH, D // group) fp16 -> fp16.

    K is quantised in groups along the head dimension, i.e. each token gets one
    scale per channel group. Symmetric int8: ``q = round(x / s)`` with
    ``s = max|x| / 127`` within the group.

    The multiply is written as int8 * fp16 on purpose: PyTorch promotes that to
    fp16 in one kernel, whereas ``.to(torch.float16) * scale`` is two passes over
    the whole cache. This path is the reason int8 KV costs decode time at all, so
    the number of elementwise passes is the thing to keep an eye on.
    """
    t, h, d = k_i8.shape
    k = k_i8.reshape(t, h, d // group, group)
    return (k * k_scale.unsqueeze(-1)).reshape(t, h, d)


def _dequant_v(v_i8, v_scale):
    """(T, KvH, D) int8 + (T, KvH) fp16 -> fp16. One scale per token per head."""
    return v_i8 * v_scale.unsqueeze(-1)


def _maybe_dequant(k, v, k_scale, v_scale, group, dtype):
    if k_scale is None:
        return k, v
    return (_dequant_k(k, k_scale, group).to(dtype),
            _dequant_v(v, v_scale).to(dtype))


def _gather_paged(cache, block_table_row, length, block_size, tail):
    """Materialise the first ``length`` tokens of one sequence out of the paged cache.

    ``tail`` is the trailing shape of one slot, so the same gather serves the
    quantised caches and their per-slot scales.
    """
    n_blocks = (length + block_size - 1) // block_size
    if block_table_row.numel() < n_blocks:
        raise ValueError("block_table row is shorter than the sequence needs")
    blocks = block_table_row[:n_blocks].to(torch.long)
    k = cache.index_select(0, blocks)                     # (n_blocks, block_size, *tail)
    return k.reshape(-1, *tail)[:length].contiguous()


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    block_table=None,
    k_scale=None,
    v_scale=None,
    quant_group=32,
    **kwargs,
):
    """Variable-length attention; ``k``/``v`` are either dense or a paged cache.

    ``k_scale``/``v_scale`` are a local extension for an int8 KV cache (upstream
    flash-attn takes fp8 descales instead): when given, K and V are dequantised
    after the gather, in the one place that knows the paged address maths.
    """
    _check_kwargs({**kwargs, "dropout_p": dropout_p}, "flash_attn_varlen_func")
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    k_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist()
    q_offsets = cu_seqlens_q.tolist()
    paged = k.dim() == 4
    if paged and block_table is None:
        raise NotImplementedError("flash_attn shim: paged k/v requires block_table")

    outs = []
    for i, (q_len, k_len) in enumerate(zip(q_lens, k_lens)):
        if q_len == 0:
            continue
        q_i = q[q_offsets[i]:q_offsets[i] + q_len]
        if paged:
            kvh, hd = k.shape[-2], k.shape[-1]
            k_i = _gather_paged(k, block_table[i], k_len, k.shape[1], (kvh, hd))
            v_i = _gather_paged(v, block_table[i], k_len, v.shape[1], (kvh, hd))
            if k_scale is not None:
                ks = _gather_paged(k_scale, block_table[i], k_len, k.shape[1],
                                   (kvh, hd // quant_group))
                vs = _gather_paged(v_scale, block_table[i], k_len, v.shape[1], (kvh,))
                k_i, v_i = _maybe_dequant(k_i, v_i, ks, vs, quant_group, q.dtype)
        else:
            k_off = cu_seqlens_k[i].item()
            k_i = k[k_off:k_off + k_len].contiguous()
            v_i = v[k_off:k_off + k_len].contiguous()
            k_i, v_i = _maybe_dequant(k_i, v_i, k_scale, v_scale, quant_group, q.dtype)
        outs.append(_heads(q_i, k_i, v_i, scale, causal))
    if not outs:
        return q.new_zeros((0, q.shape[1], q.shape[2]))
    return torch.cat(outs, dim=0)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    k_scale=None,
    v_scale=None,
    quant_group=32,
    **kwargs,
):
    """Single-query (decode) attention against a paged KV cache.

    ``q`` is (B, 1, H, D) in nano-vllm's usage; the return keeps that shape.
    ``k_scale``/``v_scale`` are the local int8-KV extension (see
    ``flash_attn_varlen_func``).
    """
    _check_kwargs(kwargs, "flash_attn_with_kvcache")
    if k is not None or v is not None or rotary_cos is not None or cache_leftpad is not None:
        raise NotImplementedError("flash_attn shim: only cache-only decoding is supported")
    squeeze = q.dim() == 3
    if squeeze:
        q = q.unsqueeze(1)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    batch, tq = q.shape[0], q.shape[1]
    block_size = k_cache.shape[1]
    n_kv_heads, head_dim = k_cache.shape[-2], k_cache.shape[-1]

    if cache_seqlens is None:
        raise NotImplementedError("flash_attn shim: cache_seqlens is required")
    lens = cache_seqlens.reshape(-1).to(torch.long)
    max_len = int(lens.max().item())
    if max_len == 0:
        return q.new_zeros((batch, tq, q.shape[2], head_dim)) if squeeze else q.new_zeros(q.shape)

    n_blocks = (max_len + block_size - 1) // block_size
    if block_table is None:
        # Contiguous cache: every block belongs to its own row.
        assert k_cache.shape[0] % batch == 0
        blocks = (torch.arange(batch, device=lens.device)[:, None] * n_blocks
                  + torch.arange(n_blocks, device=lens.device)[None, :])
    else:
        if block_table.shape[1] < n_blocks:
            raise ValueError("block_table has fewer columns than the longest sequence needs")
        # Rows shorter than the longest one are padded with -1; those slots are
        # masked out below, but they still have to be a legal index.
        blocks = block_table[:, :n_blocks].to(torch.long).clamp(min=0)
    slot = (blocks[:, :, None] * block_size
            + torch.arange(block_size, device=lens.device)[None, None, :]).reshape(batch, -1)
    slot = slot[:, :max_len]

    flat = slot.reshape(-1)
    k = k_cache.reshape(-1, n_kv_heads, head_dim).index_select(0, flat).reshape(batch, max_len, n_kv_heads, head_dim)
    v = v_cache.reshape(-1, n_kv_heads, head_dim).index_select(0, flat).reshape(batch, max_len, n_kv_heads, head_dim)
    if k_scale is not None:
        # Same index math as the data: the scales live in the same paged layout,
        # just with a shorter trailing shape. That is what keeps quantisation
        # from leaking into the addressing.
        groups = head_dim // quant_group
        ks = k_scale.reshape(-1, n_kv_heads, groups).index_select(0, flat)
        vs = v_scale.reshape(-1, n_kv_heads).index_select(0, flat)
        k = _dequant_k(k.reshape(-1, n_kv_heads, head_dim), ks, quant_group)
        v = _dequant_v(v.reshape(-1, n_kv_heads, head_dim), vs)
        k = k.reshape(batch, max_len, n_kv_heads, head_dim).to(q.dtype)
        v = v.reshape(batch, max_len, n_kv_heads, head_dim).to(q.dtype)

    n_rep = q.shape[2] // n_kv_heads
    out = _attend(
        q.transpose(1, 2),
        _expand_kv(k, n_rep).transpose(1, 2),
        _expand_kv(v, n_rep).transpose(1, 2),
        scale,
        causal,
        key_lens=lens,
    ).transpose(1, 2).contiguous()
    return out.squeeze(1) if squeeze else out
