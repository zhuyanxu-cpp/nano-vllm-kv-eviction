# Local Windows / Turing compatibility layer

> This folder is only about *making nano-vllm run here*. The feature built on top
> of it (KV cache eviction, its measurements and its interview notes) lives in
> `nanovllm/engine/kv_eviction.py`, `bench/`, `tests/` and `docs/`.

This folder is **not** part of upstream nano-vllm. It holds the local shims that
make the project run on this machine: Windows 10/11 + GTX 1650 Ti (Turing, sm_75,
4GB), with no MSVC build tools installed.

## Environment layout (everything on D:, nothing on C:)

| Component | Path |
| --- | --- |
| Python 3.12.10 | `D:\Python312` |
| Virtualenv | `D:\nano-Vllm\.venv` |
| pip cache | `D:\pip-cache` |
| Qwen3-0.6B weights | `D:\huggingface\Qwen3-0.6B` (junction: `C:\Users\<you>\huggingface` → `D:\huggingface`, so upstream `example.py` works unchanged) |

Installed: `torch 2.6.0+cu124`, `transformers`, `triton-windows 3.2.0.post21`
(the 3.8 line breaks `torch.compile` in `nanovllm/layers/activation.py`, because
`torch 2.6`'s inductor imports `AttrsDescriptor`, removed in Triton ≥ 3.4),
`xxhash`, `numpy`, `tqdm`, `safetensors`, plus the two editable packages.

## Why a shim is needed

| Upstream dependency | Problem on this machine |
| --- | --- |
| `flash-attn` | No Windows wheel exists on PyPI, and its CUDA kernels require Ampere (sm_80+). Turing can never run them, even from source. Replaced by `flash_attn_compat`. |
| `triton` | PyPI `triton` is Linux-only; the community `triton-windows` fork works and is used as-is, so `store_kvcache` keeps its real Triton kernel. |
| `dist.init_process_group("nccl")` | Windows PyTorch ships no NCCL. Patched (see below). |
| `dist` TCPStore | This torch build has no libuv support, so the rendezvous uses `?use_libuv=0`. |

## `flash_attn_compat/` — pure-PyTorch `flash_attn`

Installed into the venv as the `flash_attn` package, so
`nanovllm/layers/attention.py` imports it unchanged. It reimplements the two
entry points nano-vllm calls on top of `torch.nn.functional.scaled_dot_product_attention`
(Turing takes the memory-efficient kernel path):

* `flash_attn_varlen_func` — dense varlen prefill, and paged k/v with
  `block_table` (the prefix-cache path);
* `flash_attn_with_kvcache` — single-query decode against a paged cache;
* grouped-query attention (`num_heads > num_kv_heads`), bottom-right-aligned
  causal masking when the query chunk is shorter than its key history.

Slower than the fused CUDA kernels; functionally equivalent.

```powershell
D:\nano-Vllm\.venv\Scripts\python.exe -m pip install -e D:\nano-Vllm\compat\flash_attn_compat
```

## Code changes in `nanovllm/`

Exactly one, in `engine/model_runner.py` — pick the distributed backend at
runtime and disable libuv for the gloo rendezvous:

```python
backend = "nccl" if dist.is_nccl_available() else "gloo"
init_method = "tcp://localhost:2333" + ("" if backend == "nccl" else "?use_libuv=0")
dist.init_process_group(backend, init_method, world_size=self.world_size, rank=rank)
```

On Linux with NCCL this behaves exactly like upstream. Revert with
`git checkout -- nanovllm`.

## Tests

```powershell
cd D:\nano-Vllm
$py = D:\nano-Vllm\.venv\Scripts\python.exe

& $py compat\tests\triton_probe.py            # Triton JIT can compile a kernel here
& $py compat\tests\test_flash_attn_shim.py    # shim vs naive reference, f32 + bf16
& $py compat\tests\smoke_generate.py          # real Qwen3-0.6B generation, twice
```

## Running upstream code

`example.py` uses `max_num_batched_tokens=16384` by default, which makes
`compute_logits` materialise a `(16384, 151936)` bf16 tensor (~5GB) and cannot
fit a 4GB card. For this GPU, pass a smaller budget:

```python
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1,
          max_num_batched_tokens=2048, gpu_memory_utilization=0.75)
```

Observed on the GTX 1650 Ti: engine init ~11-15s, decode ~8 tokens/s.
