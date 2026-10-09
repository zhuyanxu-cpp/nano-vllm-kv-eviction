"""Throughput / latency benchmark: one configuration per process, JSON out.

    python bench/run_case.py --out results/base.json  --requests 32 --prompt-len 1024
    python bench/run_case.py --out results/evict.json --requests 32 --prompt-len 1024 \
        --eviction on --policy lru --ratio 0.5

Prompts are synthetic token-id lists of an exact length, so runs are comparable
and no tokenizer is needed. See bench/bench_needle.py for the accuracy side.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.harness import add_common_args, build_llm, drive, write_json  # noqa: E402


def make_prompts(requests: int, prompt_len: int, seed: int):
    """Deterministic token ids, distinct per request (no prefix sharing)."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1000, 20000, (prompt_len,), generator=g).tolist()
            for _ in range(requests)]


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    llm = build_llm(args)
    result = drive(llm, make_prompts(args.requests, args.prompt_len, args.seed),
                   args.max_tokens, args.temperature, args.seed)
    result["args"] = vars(args)
    write_json(args.out, result)

    m, e, w = result["metrics"], result["eviction"], result["workload"]
    print(f"[{os.path.basename(args.out)}] blocks={m['block_pool_size']} "
          f"cap={result['engine']['kv_token_capacity']} oversub={w['oversubscription']:.2f}x "
          f"wall={m['wall_seconds']:.2f}s (prefill {m['prefill_seconds']:.2f} / decode {m['decode_seconds']:.2f}) "
          f"ttft_mean={m['ttft_mean_s']:.2f}s ttft_max={m['ttft_max_s']:.2f}s "
          f"out_tok/s={m['output_tokens_per_s']:.1f} preempt={e['preemptions']} "
          f"evict={e['events']} evicted_tokens={e['tokens_evicted']} "
          f"prefill_tokens={m['prefill_tokens']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
