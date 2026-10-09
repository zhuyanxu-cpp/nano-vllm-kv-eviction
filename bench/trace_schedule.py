"""Step-by-step trace of what the scheduler actually does.

The aggregate counters in results/*.json say *how many* preemptions happened;
they do not say *when*. This prints the per-step timeline so claims like
"3 preemptions per wave" can be checked against data instead of derived by
reading the loop.

    .venv\\Scripts\\python.exe bench\\trace_schedule.py --eviction off
    .venv\\Scripts\\python.exe bench\\trace_schedule.py --eviction on --policy lru --ratio 0.5
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.harness import add_common_args, build_llm  # noqa: E402
from bench.run_case import make_prompts  # noqa: E402
from nanovllm import SamplingParams  # noqa: E402


def main():
    ap = add_common_args(argparse.ArgumentParser())
    # This script prints its trace instead of writing a result file; --out is
    # only accepted because add_common_args() declares it for the other benches.
    for _action in ap._actions:
        if _action.dest == "out":
            _action.required = False
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--all-steps", action="store_true",
                    help="print idle steps too (default: only steps with events)")
    args = ap.parse_args()

    llm = build_llm(args)
    sched = llm.scheduler
    bm = sched.block_manager
    params = SamplingParams(temperature=args.temperature,
                            max_tokens=args.max_tokens, ignore_eos=True)
    torch.manual_seed(args.seed)
    for p in make_prompts(args.requests, args.prompt_len, args.seed):
        llm.add_request(p, params)

    s = sched.stats
    prev = (s.preemptions, s.events, s.tokens_evicted, s.blocks_freed)

    print(f"# eviction={args.eviction} policy={args.policy} ratio={args.ratio} "
          f"mbt={args.mbt} util={args.util} requests={args.requests} "
          f"prompt_len={args.prompt_len} max_tokens={args.max_tokens}", flush=True)
    print(f"{'step':>4} {'kind':>7} {'bs':>4} {'fin':>3} {'preempt':>7} {'ev':>3} "
          f"{'ev_tok':>6} {'ev_blk':>6} {'wait':>4} {'run':>4} {'used':>4} {'free':>4}",
          flush=True)

    step = 0
    while not llm.is_finished():
        outputs, num_tokens = llm.step()
        step += 1
        cur = (s.preemptions, s.events, s.tokens_evicted, s.blocks_freed)
        d_pre, d_ev, d_tok, d_blk = (c - p for c, p in zip(cur, prev))
        prev = cur
        kind = "prefill" if num_tokens >= 0 else "decode"
        bs = num_tokens if num_tokens >= 0 else -num_tokens
        if args.all_steps or d_pre or d_ev or d_blk:
            print(f"{step:>4} {kind:>7} {bs:>4} {len(outputs):>3} {d_pre:>7} {d_ev:>3} "
                  f"{d_tok:>6} {d_blk:>6} {len(sched.waiting):>4} {len(sched.running):>4} "
                  f"{len(bm.used_block_ids):>4} {len(bm.free_block_ids):>4}", flush=True)

    print(f"# TOTAL steps={step} preemptions={s.preemptions} evictions={s.events} "
          f"evicted_tokens={s.tokens_evicted} blocks_freed={s.blocks_freed}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
