"""Correctness checks for the local flash_attn shim against a naive reference.

Covers the three call patterns nano-vllm uses:
  A. varlen prefill, dense k/v (no prefix cache)
  B. varlen prefill, paged k/v + block_table (prefix cache, seqlen_k > seqlen_q)
  C. single-query decode against a paged cache with cache_seqlens

Run: .venv\\Scripts\\python.exe compat\\tests\\test_flash_attn_shim.py
"""

import sys

import torch

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def ref_attention(q, k, v, scale, q_offset=0):
    """q (Tq,H,D), k/v (Tk,KvH,D) dense; query i (0-based within q) attends
    keys j <= i + q_offset."""
    n_rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)
    scores = torch.einsum("thd,shd->hts", q.float(), k.float()) * scale
    idx_q = torch.arange(q.shape[0], device=q.device)[:, None]
    idx_k = torch.arange(k.shape[0], device=q.device)[None, :]
    scores = scores.masked_fill(idx_k > idx_q + q_offset, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hts,shd->thd", probs, v.float()).to(q.dtype)


def check(name, actual, expected, tol=2e-3):
    err = (actual.float() - expected.float()).abs().max().item()
    status = "ok" if err <= tol else "FAIL"
    print(f"  {name}: max_err={err:.2e} tol={tol:.0e} {status}")
    return err <= tol


def case_dense(dtype, H=4, KVH=2, D=8):
    lens_q = [5, 3]
    lens_k = [5, 3]
    total_q, total_k = sum(lens_q), sum(lens_k)
    q = torch.randn(total_q, H, D, device=DEV, dtype=dtype)
    k = torch.randn(total_k, KVH, D, device=DEV, dtype=dtype)
    v = torch.randn(total_k, KVH, D, device=DEV, dtype=dtype)
    cu_q = torch.tensor([0, 5, 8], dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, 5, 8], dtype=torch.int32, device=DEV)
    scale = D ** -0.5
    out = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=5,
        max_seqlen_k=5, softmax_scale=scale, causal=True,
    )
    ref = torch.cat([
        ref_attention(q[0:5], k[0:5], v[0:5], scale),
        ref_attention(q[5:8], k[5:8], v[5:8], scale),
    ])
    return check(f"dense varlen {dtype}", out, ref)


def case_paged_prefill(dtype, H=4, KVH=2, D=8, block_size=4):
    # seq 0: 3 cached + 2 new (chunked prefill with prefix cache)
    # seq 1: 5 new, no cache
    lens_q, lens_k = [2, 5], [5, 5]
    n_blocks = 8
    k_cache = torch.randn(n_blocks, block_size, KVH, D, device=DEV, dtype=dtype)
    v_cache = torch.randn(n_blocks, block_size, KVH, D, device=DEV, dtype=dtype)
    block_table = torch.tensor([[0, 1, -1], [2, 3, -1]], dtype=torch.int32, device=DEV)

    q = torch.randn(sum(lens_q), H, D, device=DEV, dtype=dtype)
    cu_q = torch.tensor([0, 2, 7], dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, 5, 10], dtype=torch.int32, device=DEV)
    scale = D ** -0.5
    out = flash_attn_varlen_func(
        q, k_cache, v_cache, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=5, max_seqlen_k=5, softmax_scale=scale, causal=True,
        block_table=block_table,
    )

    refs = []
    for s, (lq, lk) in enumerate(zip(lens_q, lens_k)):
        nb = (lk + block_size - 1) // block_size
        ks = k_cache[block_table[s, :nb].long()].reshape(-1, KVH, D)[:lk]
        vs = v_cache[block_table[s, :nb].long()].reshape(-1, KVH, D)[:lk]
        qs = q[cu_q[s]:cu_q[s] + lq]
        refs.append(ref_attention(qs, ks, vs, scale, q_offset=lk - lq))
    return check(f"paged varlen (prefix cache) {dtype}", out, torch.cat(refs))


def case_decode(dtype, H=4, KVH=2, D=8, block_size=4):
    lens = [6, 3]
    n_blocks = 8
    k_cache = torch.randn(n_blocks, block_size, KVH, D, device=DEV, dtype=dtype)
    v_cache = torch.randn(n_blocks, block_size, KVH, D, device=DEV, dtype=dtype)
    block_table = torch.tensor([[0, 1, -1], [2, -1, -1]], dtype=torch.int32, device=DEV)
    cache_seqlens = torch.tensor(lens, dtype=torch.int32, device=DEV)
    q = torch.randn(len(lens), 1, H, D, device=DEV, dtype=dtype)
    scale = D ** -0.5
    out = flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=cache_seqlens,
        block_table=block_table, softmax_scale=scale, causal=True,
    )
    assert out.shape == q.shape, out.shape

    refs = []
    for s, length in enumerate(lens):
        nb = (length + block_size - 1) // block_size
        ks = k_cache[block_table[s, :nb].long()].reshape(-1, KVH, D)[:length]
        vs = v_cache[block_table[s, :nb].long()].reshape(-1, KVH, D)[:length]
        refs.append(ref_attention(q[s], ks, vs, scale, q_offset=length - 1))
    return check(f"decode with kvcache {dtype}", out, torch.stack(refs))


def main():
    print(f"device={DEV} ({torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu'})")
    results = []
    for dtype in (torch.float32, torch.bfloat16):
        results.append(case_dense(dtype))
        results.append(case_paged_prefill(dtype))
        results.append(case_decode(dtype))
    print("ALL PASS" if all(results) else "FAILURES PRESENT")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
