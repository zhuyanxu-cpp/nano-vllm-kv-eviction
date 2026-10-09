"""Report the real KV-cache capacity this machine gives nano-vllm.

This is the denominator of every honest memory claim: how many tokens of KV the
engine can hold at a given ``gpu_memory_utilization``.

Run: .venv\\Scripts\\python.exe compat\\tests\\kv_capacity.py
"""

import os
import sys

import torch

from nanovllm import LLM

MODEL = os.environ.get("NANOVLLM_MODEL",
                       os.path.expanduser("~/huggingface/Qwen3-0.6B"))
MAX_BATCHED = int(os.environ.get("NANOVLLM_MBT", "2048"))
UTIL = float(os.environ.get("NANOVLLM_UTIL", "0.75"))


def main():
    llm = LLM(
        MODEL,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_batched_tokens=MAX_BATCHED,
        gpu_memory_utilization=UTIL,
    )
    bm = llm.scheduler.block_manager
    cfg = llm.model_runner.config
    num_blocks = len(bm.blocks)
    capacity = num_blocks * bm.block_size
    free, total = torch.cuda.mem_get_info()
    block_bytes = 2 * cfg.hf_config.num_hidden_layers * bm.block_size \
        * (cfg.hf_config.num_key_value_heads // cfg.tensor_parallel_size) \
        * getattr(cfg.hf_config, "head_dim",
                  cfg.hf_config.hidden_size // cfg.hf_config.num_attention_heads) \
        * cfg.hf_config.dtype.itemsize

    print(f"gpu                     : {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]})")
    print(f"vram total / free now   : {total / 2**30:.2f} GiB / {free / 2**30:.2f} GiB")
    print(f"block_size              : {bm.block_size} tokens")
    print(f"bytes per block         : {block_bytes / 2**20:.2f} MiB")
    print(f"num_kvcache_blocks      : {num_blocks}")
    print(f"-> KV token capacity    : {capacity} tokens "
          f"({capacity / cfg.max_model_len:.1f} x max_model_len={cfg.max_model_len})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
