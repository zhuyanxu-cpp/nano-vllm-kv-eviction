"""Where int8 KV actually costs time: per-layer microbenchmark of the three paths.

The engine-level number ("decode +92 %") says *that* it is slower, not *why*.
This isolates the three operations that quantisation touches and prices them per
layer, so the per-step cost can be reconstructed instead of guessed:

* **write** - ``quantize_kvcache`` for one prefill chunk (512 tokens);
* **dequant read** - gather + dequantise a decode context (12 sequences x 1050
  tokens) the way ``flash_attn_with_kvcache`` does;
* **fp16 read** - the same gather from a bf16 cache, i.e. what quantisation
  replaces.

Run: .venv\\Scripts\\python.exe bench\\micro_kv_quant.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.layers.attention import quantize_kvcache  # noqa: E402

LAYERS, BLOCKS, BLOCK, KVH, HD, GROUP = 28, 93, 256, 8, 128, 32
DIM = KVH * HD


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
    kc = torch.zeros(BLOCKS, BLOCK, KVH, HD, dtype=torch.int8, device=dev)
    vc = torch.zeros_like(kc)
    ks = torch.zeros(BLOCKS, BLOCK, KVH, HD // GROUP, dtype=torch.float16, device=dev)
    vs = torch.zeros(BLOCKS, BLOCK, KVH, dtype=torch.float16, device=dev)
    kf = torch.zeros(BLOCKS, BLOCK, KVH, HD, dtype=torch.bfloat16, device=dev)
    vf = torch.zeros_like(kf)

    # --- write path: a 512-token prefill chunk ---
    n = 512
    k = torch.randn(n, KVH, HD, dtype=torch.bfloat16, device=dev)
    v = torch.randn(n, KVH, HD, dtype=torch.bfloat16, device=dev)
    slots = torch.arange(n, device=dev)
    t_write = timeit(lambda: quantize_kvcache(k, v, kc, vc, ks, vs, slots, GROUP))

    # --- read path: one decode step, 12 sequences x 1050 tokens ---
    ctx = 12 * 1050
    idx = torch.randint(0, BLOCKS * BLOCK, (ctx,), device=dev)
    groups = HD // GROUP

    def read_quant():
        kk = kc.reshape(-1, KVH, HD).index_select(0, idx)
        kk = (kk.reshape(-1, KVH, groups, GROUP)
              * ks.reshape(-1, KVH, groups).index_select(0, idx).unsqueeze(-1)).reshape(-1, KVH, HD)
        vv = vc.reshape(-1, KVH, HD).index_select(0, idx)
        vv = vv * vs.reshape(-1, KVH).index_select(0, idx).unsqueeze(-1)
        return kk.to(torch.bfloat16), vv.to(torch.bfloat16)

    def read_fp16():
        return (kf.reshape(-1, KVH, HD).index_select(0, idx),
                vf.reshape(-1, KVH, HD).index_select(0, idx))

    t_read_q = timeit(read_quant)
    t_read_f = timeit(read_fp16)

    print(f"per layer, per call:")
    print(f"  write  (512 tokens)          {t_write:7.3f} ms")
    print(f"  read   int8 + dequant (12x1050) {t_read_q:7.3f} ms")
    print(f"  read   bf16 gather   (12x1050) {t_read_f:7.3f} ms")
    print(f"  dequant adds                 {t_read_q - t_read_f:7.3f} ms "
          f"({t_read_q / t_read_f:.2f}x the fp16 read)")
    print(f"\nreconstructed per decode step (x{LAYERS} layers):")
    print(f"  fp16 read   {LAYERS * t_read_f:7.1f} ms")
    print(f"  int8 read   {LAYERS * t_read_q:7.1f} ms  -> +{LAYERS * (t_read_q - t_read_f):.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
