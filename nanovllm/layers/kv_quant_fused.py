"""The int8 KV write as a single Triton kernel.

``quantize_kvcache`` in ``attention.py`` is ~20 unfused PyTorch ops, so the fp32
promote, the per-group max, the scale, the divide, the round and the clamp each
round-trip through HBM. Priced against the plain bf16 write (``store_kvcache``,
one read + one write) on a 4 GiB Turing card, that costs **13-20x**; fusing the
whole thing into one kernel costs **1.1-1.2x** - i.e. within ~15 % of a pure
memory copy. Per 1000 prompt tokens across 28 layers: 74 ms -> 4 ms.

Numerics: the scales come out bit-for-bit identical, and the int8 output differs
on ~0.0009 % of entries - those land exactly on a ``.5`` tie and round the other
way under FP reassociation - by at most one quantisation step.

**Why K and V arrive as uint16.** Turing has no bf16 instructions at all:
``ptxas`` rejects even ``cvt.f32.bf16`` with "requires .target sm_80 or higher".
bf16 *is* the top half of fp32, so a ``uint16`` bit pattern shifted left 16 and
reinterpreted as fp32 is exactly the same value - no bf16 op anywhere.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


@triton.jit
def _quantize_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    k_scale_ptr, v_scale_ptr,
    slot_mapping_ptr,
    NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    GROUP: tl.constexpr, GROUPS: tl.constexpr,
):
    """One program per token: read, quantise in registers, store int8.

    Offsets are 3-D so the (H, GROUPS, GROUP) view flattens onto the same
    (H, HEAD_DIM) layout with no reshape. ``key_stride`` is passed in because at
    the call site K is a slice of the fused qkv projection, so its row stride is
    the full qkv width rather than ``NUM_HEADS * HEAD_DIM``.
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return

    head = tl.arange(0, NUM_HEADS)[:, None, None]        # (H, 1, 1)
    grp = tl.arange(0, GROUPS)[None, :, None]            # (1, G, 1)
    ele = tl.arange(0, GROUP)[None, None, :]             # (1, 1, GROUP)
    off3 = head * HEAD_DIM + grp * GROUP + ele           # inside one token

    slot_base = slot * (NUM_HEADS * HEAD_DIM)

    # ---- K: one scale per (head, channel group) ----
    k = (tl.load(key_ptr + idx * key_stride + off3).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    k_s = tl.maximum(tl.max(tl.abs(k), axis=2) / 127.0, 1e-8)            # (H, G)
    k_q = tl.minimum(tl.maximum(libdevice.rint(k / k_s[:, :, None]), -127.0), 127.0)
    tl.store(k_cache_ptr + slot_base + off3, k_q.to(tl.int8))
    tl.store(k_scale_ptr + slot * (NUM_HEADS * GROUPS)
             + tl.arange(0, NUM_HEADS)[:, None] * GROUPS
             + tl.arange(0, GROUPS)[None, :],
             k_s.to(tl.float16))

    # ---- V: one scale per (head) over the whole head vector ----
    h2 = tl.arange(0, NUM_HEADS)[:, None]
    d2 = tl.arange(0, HEAD_DIM)[None, :]
    off2 = h2 * HEAD_DIM + d2

    v = (tl.load(value_ptr + idx * value_stride + off2).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    v_s = tl.maximum(tl.max(tl.abs(v), axis=1) / 127.0, 1e-8)            # (H,)
    v_q = tl.minimum(tl.maximum(libdevice.rint(v / v_s[:, None]), -127.0), 127.0)
    tl.store(v_cache_ptr + slot_base + off2, v_q.to(tl.int8))
    tl.store(v_scale_ptr + slot * NUM_HEADS + tl.arange(0, NUM_HEADS), v_s.to(tl.float16))


def quantize_kvcache_fused(key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping, group):
    """Drop-in replacement for ``attention.quantize_kvcache`` (same signature)."""
    n, num_heads, head_dim = key.shape
    _quantize_kvcache_kernel[(n,)](
        key.view(torch.uint16), key.stride(0),
        value.view(torch.uint16), value.stride(0),
        k_cache, v_cache, k_scale, v_scale, slot_mapping,
        NUM_HEADS=num_heads, HEAD_DIM=head_dim,
        GROUP=group, GROUPS=head_dim // group,
    )
