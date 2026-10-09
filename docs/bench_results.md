# KV cache eviction — measured results

All numbers below are from the real engine (`LLM` → `Scheduler` → `BlockManager` →
model forward) on one machine. Nothing here comes from a mock or a simulation.
Every row can be regenerated with `bench/run_all.py`; the raw JSON of each run is
in `results/`.

| | |
| --- | --- |
| GPU | NVIDIA GeForce GTX 1650 Ti, 4 GiB, Turing sm_75 |
| Model | Qwen3-0.6B, bf16 (28 layers, 8 KV heads, head_dim 128) |
| Engine | nano-vllm, `enforce_eager=True`, `tensor_parallel_size=1`, `eviction_window_tokens=128`, `eviction_sink_tokens=4` |
| KV block | 256 tokens = **28.0 MiB** (`2 × 28 × 256 × 8 × 128 × 2 B`) |
| Python / torch | 3.12.10 / 2.6.0+cu124, pure-PyTorch attention (flash-attn shim) |

## 1. Why eviction is needed here (engine shape)

`num_kvcache_blocks` is what the engine can afford after the model and the
*warmup peak* are accounted for, so the KV budget depends on
`max_num_batched_tokens` (mbt) far more than on the memory-utilization knob:

| mbt | gpu_mem_util | blocks | KV token capacity | vs. `max_model_len=4096` |
| --- | --- | --- | --- | --- |
| 2048 | 0.75 | 10 | 2 560 | **0.6×** — cannot hold one full-length sequence |
| 2048 | 0.88 | 29 | 7 424 | 1.8× |
| 512 | 0.75 | 33 | 8 448 | 2.1× |
| 512 | 0.88 | 48 | 12 288 | 3.0× |

All experiments below use **48 blocks = 12 288 tokens**.

## 2. Throughput and latency under 2.67× oversubscription

32 requests × 1024 prompt tokens (32 768 prompt tokens wanted, 12 288 available),
`max_tokens=8`, all fired at once.

| run | wall | prefill / decode | re-prefilled tokens | preemptions | evicted tokens | out tok/s | TTFT max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (no eviction) | 39.49 s | 31.97 / 7.52 s | 38 153 | **9** | 0 | 6.5 | 37.76 s |
| eviction, LRU r=0.5 | **37.84 s** | 30.07 / 7.77 s | 36 360 | **4** | 3 351 | **6.8** | 36.52 s |
| eviction, H2O r=0.5 | 38.35 s | 29.83 / 8.52 s | 36 360 | **4** | 3 351 | 6.7 | 36.95 s |

* Preemptions **−56 %** (9 → 4). Avoiding a preemption removes a *repeated*
  prefill, which shows up directly as `prefill_tokens` **−4.7 %**
  (38 153 → 36 360).
* Wall clock only improves **4.2 %** (39.49 → 37.84 s), because **prefill is 81 %
  of the wall time** on this card (31.97 s of 39.49 s) and eviction does not
  reduce prompt work — it only avoids repeating it. On a GPU where decode
  dominates, the same preemption reduction would be worth much more.
* The cost side is small: **10 % of the prompt tokens** were dropped
  (3 351 of 32 768), and the KV move itself is negligible (see §4).
* **Policy does not matter for throughput, but collecting the signal does.**
  The drop count comes from block arithmetic, not from the policy, so LRU,
  H2O and random evict the same 3 351 tokens and differ only in *which* ones.
  The one systematic difference in the table is the probe: H2O reads the keys a
  second time on 4 of 28 layers to accumulate attention mass, which costs
  **+9.7 % decode time** (8.52 s vs LRU's 7.77 s; +3.7 % on the needle workload).
  The cleanest measurement of that cost is a workload where **nothing is ever
  evicted** (8×1024, 0.67× oversubscribed, `bench/run_all.py --quick
  --only throughput`, archived as `results/raq_*.json`): there enabling the probe
  alone costs **+11.2 % decode** (1.97 s → 2.19 s). An earlier run of the same
  command measured 2.00 → 2.25 s (**+12.5 %**), but that artifact was overwritten
  before the quick/full output prefixes were separated, so quote **~+11–13 %**
  and treat the `raq_*` pair as the evidence. Either way the cost is isolated
  from eviction dynamics: **the importance signal is not free** — §3 is what it
  buys.

## 3. Accuracy: needle-in-a-haystack

24 requests, each an exactly **1024-token** prompt (4 blocks) with a unique
passphrase buried at 40 % depth — inside the evictable region, outside the
4-token sink and the 128-token window. 12 requests fit in the pool at once, so
eviction is forced during decode. Thinking disabled, 16 output tokens, scored by
exact substring match.

| config | needle hits | accuracy | preemptions | evicted tokens |
| --- | --- | --- | --- | --- |
| baseline (no eviction) | 22/24 | 92 % | 6 | 0 |
| LRU r=0.5 | 23/24 | 96 % | 3 | 2 580 |
| **H2O r=0.5** | 23/24 | 96 % | 3 | 2 580 |
| **random r=0.5** | **17/24** | **71 %** | 3 | 2 580 |
| H2O r=0.25 | 23/24 | 96 % | 3 | 2 580 |

* **The control condition is the result.** Dropping the *same number* of tokens,
  but choosing them at random, costs **21 points against the baseline and 25
  against LRU/H2O**, while recency (LRU) and accumulated attention mass (H2O)
  hold the baseline level. The importance signal is doing real work — this is the
  experiment that a "−95 % KV usage" slide cannot replace.
* At n=24 one request is ±4 %, so **LRU 96 % vs H2O 96 % (and 92 % baseline) is
  noise**; only the random gap is significant. Reading a winner out of the top
  three rows would be over-reading.
* `r=0.25` matches `r=0.50` because eviction is pressure-driven: the retention
  floor only binds when the engine must squeeze a sequence below it, and at this
  oversubscription the *need* (free one block) sets the drop, not the ratio. The
  ratio is a floor, not a target — see `plan_drop` / `resident_budget`.

## 4. Designs that were measured and rejected

Each of these was implemented, run, and then removed or defaulted off. The paths
are kept (behind `eviction_on_admission` / `eviction_bulk_kv_move`, or in
comments) so the measurement stays reproducible.

| design | measured effect | decision |
| --- | --- | --- |
| Evict to **admit** new requests (`--on-admission`) | 16×1024 workload: wall 19.02 s → **22.42 s** (+18 %) with the same 26 evictions / 6 959 evicted tokens, for no throughput gain | **off by default**; admission stays first-come-first-served |
| KV move as one tensor op (`--bulk-kv-move`) | same workload: decode **12.33 s** vs **8.81 s** with the per-layer loop (+40 %), wall 25.94 s vs 22.42 s; ~1 GB transient per 900-token eviction | **per-layer copy**, transient < 2 MB |
| `plan_drop` clamped down to the retention budget (`min` instead of `max`) | one 1-block request evicted ~510 tokens instead of 257; run measured 18.8 s → 28.7 s — **confounded**: that run also had admission eviction on, so the two effects were not separated | changed anyway: dropping 510 tokens where 257 suffice is strictly worse; the budget is a floor |
| "A new block is needed when `length % block_size == 1`" | wrong once the table holds spare capacity: allocates a block that already exists | `needs_new_block` derives it from the block table |
| Score buffer not cleared when a block is recycled | a new sequence inherited the previous occupant's accumulation — policy quality only, never addressing | fixed: `clear_block_scores` on every freed block, covered by a test |

## 5. Correctness evidence

* `tests/test_eviction_scheduler.py` — drives the **real** `Scheduler` on CPU with
  two injected callables standing in for the GPU: pressure fires eviction instead
  of preemption, all three policies run, evicted sequences leave the prefix-cache
  index, shared blocks are never rewritten, recycled blocks have their scores
  cleared, preemption resets the split, and no pressure means no eviction.
* `tests/test_eviction_policy.py` — the policy maths (head/tail never dropped,
  exact drop count, ascending order, floor respected, LRU vs H2O behaviour,
  determinism of the random arm).
* **Control experiment** (`tests/test_control_identity.py`, opt-in): eviction
  *enabled* on a workload that never triggers it produces **bit-identical token
  ids** to the baseline. That is what proves the physical/logical split, the
  `can_append`/`may_append` change and the prefix-cache change are no-ops when
  nothing is evicted.
* `compat/tests/test_flash_attn_shim.py` — the attention implementation is checked
  against a naive reference in all three call patterns, fp32 and bf16.

```bash
python -m pytest tests/ -q                            # 17 CPU tests, ~9 s
$env:NANOVLLM_GPU_TESTS=1; python -m pytest tests/ -q # + the control experiment
python bench/run_all.py                               # every table above
```

## 6. Honest limits

* One 0.6B model on one 4 GiB Turing card. The *mechanism* (preemption avoided,
  accuracy lost per policy) should transfer; the absolute numbers should not be
  quoted as if they came from a serving cluster.
* No shared-prefix workload: a sequence whose blocks are shared through the prefix
  cache is never evicted, because rewriting a shared block in place would corrupt
  every other reader. Copy-on-write would lift this and is not implemented.
* `tensor_parallel_size == 1` only. The sequence serialisation does carry
  `num_evicted_tokens` (all ranks derive slot addresses from the physical
  length), but the eviction decision itself is single-rank.
* Every experiment runs with `enforce_eager=True`: the CUDA-graph decode path
  (`capture_cudagraph`) has never been exercised together with eviction.
* Accuracy is measured by retrieval (needle), not perplexity, and with n=24.
* The ratio knob does not bind at the oversubscription levels tested.
* Single runs per configuration; run-to-run spread on this 4 GiB card is on the
  order of ±1 s of wall clock, so differences below ~3 % should not be read as
  effects. The probe cost in §2 was measured twice for exactly that reason.
* Reproducing the tables one configuration at a time is safe; running two
  benchmarks *concurrently* is not. The engine rendezvouses on a hard-coded
  `tcp://localhost:2333`, so two engine processes collide (and the loser exits
  non-zero), which reads like a flaky benchmark but is a port clash.
* `results/raq_*.json` (quick) and `results/ra_*.json` (full) are separate
  prefixes on purpose: the quick workload creates no pressure, so mixing the two
  sets would leave a results directory that no single command reproduces.

## 7. int8 KV cache: the second way to buy memory

Eviction trades *information* for memory. Quantisation trades *precision* for it,
and the JD for this kind of role lists both. Same card, same config, same
workload (24 × 1024-token needle requests, `--kv-quant int8`):

| KV dtype | eviction | needle | preemptions | per-sequence decode step | wall |
| --- | --- | --- | --- | --- | --- |
| fp16 | off | 22/24 (92 %) | 6 | 34.6 ms | 35.4 s |
| fp16 | on (r=0.5) | 23/24 (96 %) | 3 | 39.1 ms | 36.7 s |
| **int8** | off | 23/24 (96 %) | 5 | **66.6 ms** | 49.8 s |
| **int8** | **on (r=0.5)** | 22/24 (92 %) | **2** | **44.3 ms** | 39.0 s |

**The memory claim, measured.** Block bytes go from **28.00 MiB to 14.55 MiB**
(the fp16 scale planes are 8 % of the int8 payload, not free), so at the same
`gpu_memory_utilization` the pool grows from **48 to 93 blocks** and KV capacity
from **12 288 to 23 808 tokens (×1.94)**. Predicted 1.92, measured 1.94.

**The accuracy claim, measured.** All four cells land in 92–96 %, i.e. **within one
request of each other at n=24 (±4 %)**. On this task int8 KV costs nothing
measurable — the honest phrasing is "no measurable loss", not "lossless".

**The cost claim, measured, and it is the interesting part.** int8 alone nearly
*doubles* the per-sequence decode step (34.6 → 66.6 ms). It is not concurrency:
total decode work is identical (354 vs 355 sequence-steps), so the comparison is
like-for-like. `bench/micro_kv_quant.py` prices the three paths per layer:

| per layer, per call | |
| --- | --- |
| write, 512-token chunk (`quantize_kvcache`) | 1.193 ms |
| read, 12 × 1050 context, bf16 gather | 1.682 ms |
| read, same context, int8 gather + dequant | **4.628 ms (2.75×)** |
| reconstructed per decode step (× 28 layers) | 47.1 ms → 129.6 ms (**+82.5 ms**) |

The dequant has to *materialise* fp16 K/V (int8 → fp16 multiply → cast to bf16),
so it pays back the bandwidth it saved and then some. **This is why production
quantised KV needs a fused dequant-and-attend kernel** (the direction vLLM takes
with fp8 descales): with the dequant inside the kernel, the fp16 K/V is never
written to memory. Removing one elementwise pass by multiplying int8 × fp16
directly (letting type promotion do the cast) changed 67.5 → 66.6 ms, i.e. the
per-pass cost is not the issue — the materialisation is.

**And the interaction is the useful finding.** Eviction *pays back* the
quantisation cost: int8 alone is 66.6 ms/step, int8 **with** eviction is 44.3 ms,
because dequant cost is linear in context length and eviction is exactly the thing
that shortens context. Combining them also gives the fewest preemptions of any
configuration (2 vs 6 for fp16-alone). So the two techniques are not alternatives:
eviction buys headroom, quantisation buys capacity, and the second one's runtime
cost is partly paid by the first.

Limits of this section: one model, one card, n=24 accuracy (single-hit noise of
±4 %), no fused kernel implemented, and K is quantised per channel *group within a
token* rather than per channel *across tokens* as in KIVI — a streaming paged
cache cannot revisit tokens to recompute a stale scale, and that is the price.

