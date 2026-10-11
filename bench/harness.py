"""Shared driver for the real-engine benchmarks.

Both benchmarks (throughput/latency and accuracy) must exercise the *same* code
path: the real ``LLM``/``Scheduler``/``BlockManager`` and the real model, with
only the configuration differing. So the engine construction and the step loop
live here, once.

Each configuration runs in its own process, because ``torch.distributed`` can
only be initialised once per process - hence the JSON-in/JSON-out CLI shape of
the scripts that use this.
"""

import os
import time

import torch

from nanovllm import LLM, SamplingParams

DEFAULT_MODEL = os.environ.get("NANOVLLM_MODEL", r"D:\huggingface\Qwen3-0.6B")


def build_llm(args) -> LLM:
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_batched_tokens=args.mbt,
        gpu_memory_utilization=args.util,
        kv_eviction=args.eviction == "on",
        eviction_policy=args.policy,
        eviction_keep_ratio=args.ratio,
        eviction_sink_tokens=args.sink,
        eviction_window_tokens=args.window,
        eviction_probe_layers=args.probe_layers,
        eviction_on_admission=args.on_admission,
        eviction_bulk_kv_move=args.bulk_kv_move,
        kv_quant=args.kv_quant,
        kv_quant_group=args.kv_quant_group,
        kv_quant_fused=args.kv_quant_fused,
    )
    llm.scheduler.stats.record_history = args.record_history
    return llm


def drive(llm: LLM, prompts, max_tokens: int, temperature: float, seed: int) -> dict:
    """Run all prompts, sampling metrics every step.

    Prefill and decode are timed separately: they scale with different things
    (prompt tokens vs. context length x concurrency), and separating them is what
    made an early eviction slowdown explainable instead of mysterious.
    ``LLMEngine.step`` reports a negative token count for a decode batch, which
    is how the two are told apart from outside.
    """
    bm = llm.scheduler.block_manager
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, ignore_eos=True)
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for p in prompts:
        llm.add_request(p, params)

    first_token_at, finished, occupancy, per_step = {}, {}, [], []
    phase = dict(prefill_seconds=0.0, prefill_steps=0, prefill_tokens=0,
                 decode_seconds=0.0, decode_steps=0, decode_batch_seqs=0)
    steps = 0
    while not llm.is_finished():
        t_step = time.perf_counter()
        outputs, num_tokens = llm.step()
        t_step_end = time.perf_counter()
        steps += 1
        dt = t_step_end - t_step
        occupancy.append(len(bm.used_block_ids))
        per_step.append(dt)
        if num_tokens < 0:
            phase["decode_seconds"] += dt
            phase["decode_steps"] += 1
            phase["decode_batch_seqs"] += -num_tokens
        else:
            phase["prefill_seconds"] += dt
            phase["prefill_steps"] += 1
            phase["prefill_tokens"] += num_tokens
        for seq in list(llm.scheduler.running):
            if seq.num_completion_tokens >= 1 and seq.seq_id not in first_token_at:
                first_token_at[seq.seq_id] = t_step_end
        for seq_id, token_ids in outputs:
            first_token_at.setdefault(seq_id, t_step_end)
            finished[seq_id] = token_ids
    wall = time.perf_counter() - t0

    total_completion = sum(len(v) for v in finished.values())
    ttfts = [first_token_at[s] - t0 for s in sorted(finished)]
    tpots = [(wall - (first_token_at[s] - t0)) / (len(finished[s]) - 1)
             for s in sorted(finished) if len(finished[s]) > 1]
    total_prompt = sum(len(p) for p in prompts)
    return {
        "engine": {
            "num_kvcache_blocks": len(bm.blocks),
            "block_size": bm.block_size,
            "kv_token_capacity": len(bm.blocks) * bm.block_size,
            "max_model_len": llm.model_runner.config.max_model_len,
        },
        "workload": {
            "requests": len(prompts),
            "prompt_tokens": total_prompt,
            "max_tokens": max_tokens,
            "total_completion_tokens": total_completion,
            "oversubscription": total_prompt / (len(bm.blocks) * bm.block_size),
        },
        "metrics": {
            "wall_seconds": wall,
            "steps": steps,
            "ttft_mean_s": sum(ttfts) / len(ttfts) if ttfts else None,
            "ttft_p90_s": sorted(ttfts)[int(0.9 * (len(ttfts) - 1))] if ttfts else None,
            "ttft_max_s": max(ttfts) if ttfts else None,
            "tpot_mean_s": sum(tpots) / len(tpots) if tpots else None,
            "output_tokens_per_s": total_completion / wall,
            "step_mean_s": sum(per_step) / len(per_step) if per_step else None,
            "block_occupancy_mean": sum(occupancy) / len(occupancy) if occupancy else None,
            "block_occupancy_max": max(occupancy) if occupancy else None,
            "block_pool_size": len(bm.blocks),
            **phase,
        },
        "eviction": llm.eviction_stats(),
        "outputs": {s: finished[s] for s in sorted(finished)},
    }


def add_common_args(ap):
    """The knobs every benchmark shares (engine shape + eviction configuration)."""
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mbt", type=int, default=512, help="max_num_batched_tokens")
    ap.add_argument("--util", type=float, default=0.85, help="gpu_memory_utilization")
    ap.add_argument("--eviction", choices=["on", "off"], default="off")
    ap.add_argument("--policy", choices=["lru", "h2o", "random"], default="h2o")
    ap.add_argument("--ratio", type=float, default=0.5,
                    help="least share of the prompt kept resident")
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--probe-layers", type=int, default=4)
    ap.add_argument("--on-admission", action="store_true",
                    help="also evict to admit new requests (measured to be slower)")
    ap.add_argument("--bulk-kv-move", action="store_true",
                    help="reproduce the single-tensor KV copy the per-layer loop replaced")
    ap.add_argument("--kv-quant", choices=["none", "int8"], default="none")
    ap.add_argument("--kv-quant-fused", action="store_true",
                    help="write the int8 KV cache with one fused Triton kernel")
    ap.add_argument("--kv-quant-group", type=int, default=32,
                    help="channel group for K; V is quantised per token")
    ap.add_argument("--record-history", action="store_true")
    return ap


def write_json(path: str, result: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
