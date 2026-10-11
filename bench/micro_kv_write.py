"""Does the int8 write cost more than a plain bf16 write?

``micro_kv_quant.py`` prices the *read* paths (int8+dequant vs bf16) but never
asks what the *write* side costs without quantisation, so its "the bottleneck is
the read path, not the write" conclusion rests on comparing magnitudes rather
than on a like-for-like pair. This closes that gap.

Both paths write the same 512-token chunk into the same-shaped paged cache and
are timed the same way (3 warm-up calls, 30 timed calls, ``cuda.synchronize``
around the loop):

* bf16 write - ``store_kvcache``, the upstream Triton kernel, unchanged;
* int8 write - ``quantize_kvcache``: per-channel-group max for K, per-token max
  for V, round to int8, and store the fp16 scales.

Engine-level context (from results/q_*.json): int8 also slows *prefill* down -
1229 -> 1108 tokens/s between q_fp16 and q_int8 - and prefill is the larger half
of the wall clock, so the write path is where that shows up.

Run: .venv\\Scripts\\python.exe bench\\micro_kv_write.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.layers.attention import quantize_kvcache, store_kvcache  # noqa: E402

LAYERS, BLOCKS, BLOCK, KVH, HD, GROUP = 28, 93, 256, 8, 128, 32


def timeit(fn, iters=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3     # ms per call


def main():
    dev = "cuda"

    # bf16 cache: what store_kvcache writes into.
    kf = torch.zeros(BLOCKS, BLOCK, KVH, HD, dtype=torch.bfloat16, device=dev)
    vf = torch.zeros_like(kf)
    # int8 cache + fp16 per-group (K) / per-token (V) scales: what quantize_kvcache writes into.
    kc = torch.zeros(BLOCKS, BLOCK, KVH, HD, dtype=torch.int8, device=dev)
    vc = torch.zeros_like(kc)
    ks = torch.zeros(BLOCKS, BLOCK, KVH, HD // GROUP, dtype=torch.float16, device=dev)
    vs = torch.zeros(BLOCKS, BLOCK, KVH, dtype=torch.float16, device=dev)

    n = 512
    k = torch.randn(n, KVH, HD, dtype=torch.bfloat16, device=dev)
    v = torch.randn(n, KVH, HD, dtype=torch.bfloat16, device=dev)
    slots = torch.arange(n, dtype=torch.int32, device=dev)

    t_bf16 = timeit(lambda: store_kvcache(k, v, kf, vf, slots))
    t_i8 = timeit(lambda: quantize_kvcache(k, v, kc, vc, ks, vs, slots, GROUP))

    print("write one 512-token chunk into the paged cache, per layer:")
    print(f"  bf16 write (store_kvcache)      {t_bf16:7.3f} ms")
    print(f"  int8 write (quantize_kvcache)   {t_i8:7.3f} ms")
    print(f"  int8 write costs                {t_i8 - t_bf16:+7.3f} ms "
          f"({t_i8 / t_bf16:.2f}x the bf16 write)")
    print()
    print(f"per 1000 prompt tokens, x{LAYERS} layers:")
    print(f"  bf16 write   {LAYERS * t_bf16 * 1000 / n:7.2f} ms")
    print(f"  int8 write   {LAYERS * t_i8 * 1000 / n:7.2f} ms "
          f"-> {LAYERS * (t_i8 - t_bf16) * 1000 / n:+.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
