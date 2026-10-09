<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

> **About this repository.** This is a derivative of
> [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (MIT), used as a
> platform for studying KV-cache management. **The upstream engine is not my
> work.** Everything under **KV Cache Eviction** below is: the eviction policies,
> int8 KV quantisation, the benchmark suite, and the three consistency fixes that
> eviction needs inside a paged-attention + prefix-caching engine. Every number
> in this README comes from a real run — the raw artifacts behind the tables are
> in [`results/`](results/) and [`docs/bench_results.md`](docs/bench_results.md).

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## KV Cache Eviction — what this repository adds

Everything below is measured on the author's machine: **GTX 1650 Ti (4 GiB, Turing
sm_75) + Qwen3-0.6B in bf16**, real engine, no mocks. Full numbers and the
rejected designs: [`docs/bench_results.md`](docs/bench_results.md).

**Why it is needed.** One KV block is 256 tokens = 28.0 MiB, and the engine only
gets ~48 blocks (12 288 tokens) on this card, i.e. **less than three 4 k
sequences**. When the pool runs dry, upstream `Scheduler` calls `preempt()` and
throws the whole sequence's KV away to be recomputed later.

**What this adds.** Under exactly that pressure, the engine now *compacts* a
running sequence's KV instead of discarding it: keep a small attention-sink head
and a recent window, drop a chosen set of middle tokens, and re-pack the survivors
into the first blocks. Policies: `lru` (recency), `h2o` (accumulated attention
mass, collected by a probe on a few layers), `random` (control condition).

```bash
python -m pytest tests/ -q                                     # 21 CPU tests
$env:NANOVLLM_GPU_TESTS=1; python -m pytest tests/ -q          # + control experiment (GPU)
python bench/run_all.py                                        # every number below
python bench/run_case.py --out results/base.json  --requests 32 --prompt-len 1024
python bench/run_case.py --out results/evict.json --requests 32 --prompt-len 1024 \
    --eviction on --policy h2o --ratio 0.5
python bench/bench_needle.py --out results/needle.json --requests 24 --eviction on
python bench/trace_schedule.py --eviction off                  # per-step log (results/trace_*_steps.txt)
python bench/trace_schedule.py --eviction on --policy lru
```

| measurement (32×1024 tokens on a 12 288-token pool) | baseline | eviction |
| --- | --- | --- |
| preemptions | 9 | **4** |
| re-prefilled tokens | 38 153 | **36 360** |
| wall clock | 39.49 s | 37.84 s (prefill is 81 % of it) |
| context dropped | – | 10 % of prompt tokens |

| needle-in-a-haystack, 1024-token prompt, 24 requests | accuracy |
| --- | --- |
| baseline (no eviction) | 92 % |
| H2O / LRU r=0.5 | 96 % / 96 % (within noise of each other) |
| **random r=0.5** | **71 %** |

The random arm is the point: dropping the *same number* of tokens without an
importance signal costs **21 accuracy points**, so the signal earns its keep. It
is not free, though — H2O reads the keys a second time on 4 of 28 layers, which
costs **~+12 % decode** measured on a workload where nothing is evicted at all.

**The hard part is not the policy, it is three invariants** that only exist once
eviction touches a real paged-attention + prefix-caching engine:

1. slot arithmetic must switch to the *physical* length while RoPE positions stay
   *logical* (RoPE is relative, so cached keys keep the phase they were written
   with);
2. a compacted block no longer holds the tokens its content hash claims, so an
   evicted sequence must leave the prefix cache permanently - and a block with
   `ref_count > 1` must never be rewritten in place at all;
3. preemption must reset the split, because the sequence is recomputed from
   scratch.

Known limits: shared-prefix sequences are not evictable (no copy-on-write yet),
`tensor_parallel_size == 1` only, and eviction is bounded - once the evictable
history is exhausted the engine preempts as before.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)