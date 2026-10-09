"""End-to-end smoke test: load Qwen3-0.6B through nano-vllm and generate text.

Tuned for a 4GB Turing GPU: the default ``max_num_batched_tokens=16384`` makes
``compute_logits`` materialise a (16384, 151936) tensor (~5GB in bf16), which
cannot fit. The prompt is generated twice so the second run exercises the
prefix-cache path (paged k/v + block_table).

Run: .venv\\Scripts\\python.exe compat\\tests\\smoke_generate.py
"""

import os
import sys
import time

import torch
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = os.environ.get("NANOVLLM_MODEL", r"D:\huggingface\Qwen3-0.6B")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    t0 = time.perf_counter()
    llm = LLM(
        MODEL,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.75,
    )
    print(f"[smoke] engine ready in {time.perf_counter() - t0:.1f}s", flush=True)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2? Answer with one number."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sampling = SamplingParams(temperature=0.6, max_tokens=32)

    for run in (1, 2):
        torch.manual_seed(0)  # nano-vllm rejects greedy sampling, so fix the seed
        t = time.perf_counter()
        out = llm.generate([prompt], sampling)
        dt = time.perf_counter() - t
        print(f"[smoke] run {run} ({dt:.1f}s): {out[0]['text']!r}", flush=True)

    # LLMEngine registers its own atexit hook; calling exit() again would raise,
    # so leave teardown to the hook.
    return 0


if __name__ == "__main__":
    sys.exit(main())
