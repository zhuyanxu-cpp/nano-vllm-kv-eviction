"""Can the int8 KV write be fused into one kernel, and does it actually help?

``quantize_kvcache`` is ~20 unfused PyTorch ops, so every intermediate (the fp32
promote, the per-group max, the scale, the divide, the round, the clamp) is
written to and read back from HBM. ``store_kvcache`` next to it is a single
Triton kernel that just reads once and writes once, and measured 12.56x faster.

This script asks the obvious follow-up: write the same quantisation as one
Triton kernel (intermediates stay in registers) and check both

* **correctness** - same input, are the int8 caches and fp16 scales the same?
* **speed** - how close does it get to the bf16 write?

Everything here is additive: ``nanovllm`` is untouched.

Run: .venv\\Scripts\\python.exe bench\\micro_kv_write_fused.py
"""

import os
import sys
import time

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.layers.attention import quantize_kvcache, store_kvcache  # noqa: E402

LAYERS, BLOCKS, BLOCK, KVH, HD, GROUP = 28, 93, 256, 8, 128, 32
GROUPS = HD // GROUP


@triton.jit
def quantize_kvcache_kernel(
    key_ptr, value_ptr,
    k_cache_ptr, v_cache_ptr,
    k_scale_ptr, v_scale_ptr,
    slot_mapping_ptr,
    NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    GROUP: tl.constexpr, GROUPS: tl.constexpr,
):
    """One program per token: read bf16 K/V, quantise in registers, store int8.

    K gets one scale per (head, group of GROUP channels); V one scale per (head),
    matching ``quantize_kvcache``. Offsets are written as 3-D so the (H, G, GRP)
    view flattens onto the same (H, HEAD_DIM) layout with no reshape needed.

    **bf16 workaround.** Turing's PTX has no bf16 instructions at all - ``ptxas``
    rejects even ``cvt.f32.bf16`` with "requires .target sm_80 or higher". The
    key/value tensors are therefore passed in already viewed as ``uint16`` and
    widened by hand: bf16 *is* the top 16 bits of fp32, so ``bits << 16``
    reinterpreted as fp32 is exactly the value, with no bf16 op involved.
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return

    head = tl.arange(0, NUM_HEADS)[:, None, None]        # (H, 1, 1)
    grp = tl.arange(0, GROUPS)[None, :, None]            # (1, G, 1)
    ele = tl.arange(0, GROUP)[None, None, :]             # (1, 1, GROUP)
    off3 = head * HEAD_DIM + grp * GROUP + ele           # (H, G, GROUP) inside a token

    tok_base = idx * (NUM_HEADS * HEAD_DIM)
    slot_base = slot * (NUM_HEADS * HEAD_DIM)

    # ---- K: one scale per (head, channel group) ----
    k = (tl.load(key_ptr + tok_base + off3).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    k_s = tl.max(tl.abs(k), axis=2) / 127.0              # (H, G)
    k_s = tl.maximum(k_s, 1e-8)
    k_q = tl.minimum(tl.maximum(libdevice.rint(k / k_s[:, :, None]), -127.0), 127.0)
    tl.store(k_cache_ptr + slot_base + off3, k_q.to(tl.int8))
    tl.store(k_scale_ptr + slot * (NUM_HEADS * GROUPS)
             + tl.arange(0, NUM_HEADS)[:, None] * GROUPS
             + tl.arange(0, GROUPS)[None, :],
             k_s.to(tl.float16))

    # ---- V: one scale per (head) over the whole head vector ----
    h2 = tl.arange(0, NUM_HEADS)[:, None]                # (H, 1)
    d2 = tl.arange(0, HEAD_DIM)[None, :]                 # (1, HEAD_DIM)
    off2 = h2 * HEAD_DIM + d2

    v = (tl.load(value_ptr + tok_base + off2).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    v_s = tl.max(tl.abs(v), axis=1) / 127.0              # (H,)
    v_s = tl.maximum(v_s, 1e-8)
    v_q = tl.minimum(tl.maximum(libdevice.rint(v / v_s[:, None]), -127.0), 127.0)
    tl.store(v_cache_ptr + slot_base + off2, v_q.to(tl.int8))
    tl.store(v_scale_ptr + slot * NUM_HEADS + tl.arange(0, NUM_HEADS), v_s.to(tl.float16))


def fused_quantize_kvcache(key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping, group):
    n = key.shape[0]
    quantize_kvcache_kernel[(n,)](
        # bf16 is handed to the kernel as raw uint16 bits: Turing's PTX has no
        # bf16 instructions (see the kernel docstring).
        key.view(torch.uint16), value.view(torch.uint16),
        k_cache, v_cache, k_scale, v_scale, slot_mapping,
        NUM_HEADS=key.shape[1], HEAD_DIM=key.shape[2],
        GROUP=group, GROUPS=key.shape[2] // group,
    )


def timeit(fn, iters=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def new_caches(dtype, dev="cuda"):
    """A fresh set of paged caches, zeroed, so the two paths write identical slots."""
    kc = torch.zeros(BLOCKS, BLOCK, KVH, HD, dtype=dtype, device=dev)
    vc = torch.zeros_like(kc)
    ks = torch.zeros(BLOCKS, BLOCK, KVH, GROUPS, dtype=torch.float16, device=dev)
    vs = torch.zeros(BLOCKS, BLOCK, KVH, dtype=torch.float16, device=dev)
    return kc, vc, ks, vs


def main():
    dev = "cuda"
    n = 512
    torch.manual_seed(0)
    k = torch.randn(n, KVH, HD, dtype=torch.bfloat16, device=dev)
    v = torch.randn(n, KVH, HD, dtype=torch.bfloat16, device=dev)
    slots = torch.arange(n, dtype=torch.int32, device=dev)

    # ---------------------------------------------------------------- correctness
    kf, vf, _, _ = new_caches(torch.bfloat16)
    ref = new_caches(torch.int8)
    got = new_caches(torch.int8)
    quantize_kvcache(k, v, *ref[:2], *ref[2:], slots, GROUP)
    fused_quantize_kvcache(k, v, *got[:2], *got[2:], slots, GROUP)
    torch.cuda.synchronize()

    print("correctness (same input, 512 tokens):")
    for name, a, b in [("k_cache int8", ref[0], got[0]), ("v_cache int8", ref[1], got[1]),
                       ("k_scale fp16", ref[2], got[2]), ("v_scale fp16", ref[3], got[3])]:
        ai, bi = a.float(), b.float()
        diff = (ai - bi).abs().max().item()
        nz = int((ai != bi).sum().item())
        print(f"  {name:<14} max|diff|={diff:8.3e}  differing={nz}/{ai.numel()}")

    # Dequantised values are what the attention actually sees; compare those too.
    dq_ref = ref[0].float() * ref[2].float().repeat_interleave(GROUP, dim=-1)
    dq_got = got[0].float() * got[2].float().repeat_interleave(GROUP, dim=-1)
    rel = ((dq_ref - dq_got).abs() / dq_ref.abs().clamp(min=1e-6)).max().item()
    print(f"  dequantised K  max relative diff = {rel:.3e}")

    # --------------------------------------------------------------------- speed
    kc, vc, ks, vs = new_caches(torch.int8)
    t_bf16 = timeit(lambda: store_kvcache(k, v, kf, vf, slots))
    t_torch = timeit(lambda: quantize_kvcache(k, v, kc, vc, ks, vs, slots, GROUP))
    t_fused = timeit(lambda: fused_quantize_kvcache(k, v, kc, vc, ks, vs, slots, GROUP))

    print()
    print("speed: write one 512-token chunk into the paged cache, per layer")
    print(f"  bf16 write, Triton (store_kvcache)   {t_bf16:7.3f} ms   (1.00x)")
    print(f"  int8 write, ~20 torch ops            {t_torch:7.3f} ms   ({t_torch / t_bf16:.2f}x)")
    print(f"  int8 write, fused Triton kernel      {t_fused:7.3f} ms   ({t_fused / t_bf16:.2f}x)")
    print(f"  fused vs unfused                     {t_torch / t_fused:7.2f}x faster "
          f"({t_torch - t_fused:+.3f} ms per chunk)")
    print()
    print(f"per 1000 prompt tokens, x{LAYERS} layers:")
    print(f"  int8 unfused   {LAYERS * t_torch * 1000 / n:7.2f} ms")
    print(f"  int8 fused     {LAYERS * t_fused * 1000 / n:7.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
