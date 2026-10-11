import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.kv_quant_fused import quantize_kvcache_fused
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def quantize_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor,
                     v_cache: torch.Tensor, k_scale: torch.Tensor, v_scale: torch.Tensor,
                     slot_mapping: torch.Tensor, group: int):
    """int8 KV write: quantise on the way into the paged cache.

    Two different granularities, and the asymmetry is the point:

    * **K per channel group.** Outlier magnitudes in a key vector sit in the same
      few channels across tokens (the KIVI observation), so one scale per group of
      channels costs almost nothing in accuracy, while a single scale for the whole
      head vector would clip those channels.
    * **V per token.** V is the weight in the output sum, so its magnitude is what
      has to be preserved per token; one scale per head vector is enough.

    Quantising per token rather than per filled block is what keeps this
    streaming-safe: no token is ever revisited to fix up a stale scale, which is
    the price of not being able to compute a block-wide statistic mid-decode.
    """
    block_size = k_cache.shape[1]
    num_heads = key.shape[1]
    head_dim = key.shape[2]
    groups = head_dim // group
    slots = slot_mapping.long()
    valid = slots >= 0
    slots = slots[valid]
    if slots.numel() == 0:
        return
    blocks, offsets = slots // block_size, slots % block_size
    num = slots.numel()

    k = key[valid].float().view(num, num_heads, groups, group)
    k_s = (k.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=1e-8)
    k_q = torch.round(k / k_s).clamp(-127, 127).to(torch.int8).view(num, num_heads, head_dim)
    v = value[valid].float()
    v_s = (v.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=1e-8)
    v_q = torch.round(v / v_s).clamp(-127, 127).to(torch.int8)

    k_cache[blocks, offsets] = k_q
    v_cache[blocks, offsets] = v_q
    k_scale[blocks, offsets] = k_s.view(num, num_heads, groups).to(torch.float16)
    v_scale[blocks, offsets] = v_s.view(num, num_heads).to(torch.float16)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        # Set by ModelRunner when this layer is one of the probed layers of the
        # h2o eviction policy; None means "do not collect importance scores".
        self.eviction_scores = None
        # Set by ModelRunner when the KV cache is int8; None means fp16/bf16.
        self.k_scale = self.v_scale = None
        self.quant_group = 32
        # Set by ModelRunner: write the int8 cache with the fused Triton kernel.
        self.quant_fused = False

    def _dequant_keys(self, keys_i8, blocks, length):
        """(n_blocks, block_size, KvH, D) int8 -> (length, KvH, D) float."""
        groups = self.head_dim // self.quant_group
        scales = self.k_scale.index_select(0, blocks)
        scales = scales.reshape(-1, self.num_kv_heads, groups)[:length].unsqueeze(-1)
        keys = keys_i8.reshape(-1, self.num_kv_heads, groups, self.quant_group)[:length]
        # int8 * fp16 promotes to fp16 in one kernel (see the shim's _dequant_k).
        return (keys * scales).reshape(length, self.num_kv_heads, self.head_dim)

    @torch.no_grad()
    def _accumulate_importance(self, q: torch.Tensor, context):
        """Add this decode step's attention mass into the per-slot score buffer.

        This is a *probe*: it recomputes attention probabilities for a few layers
        so the eviction policy has an importance signal (the H2O heavy-hitter
        idea). A kernel that emits attention weights would fuse this into the
        attention op for free; doing it here costs one extra read of the keys per
        probed layer, and that cost is measured rather than assumed away.
        """
        block_size = self.k_cache.shape[1]
        n_rep = self.num_heads // self.num_kv_heads
        for b in range(q.shape[0]):
            length = int(context.context_lens[b])
            if length == 0:
                continue
            n_blocks = (length + block_size - 1) // block_size
            blocks = context.block_tables[b, :n_blocks].to(torch.long)
            keys = self.k_cache.index_select(0, blocks)
            if self.k_scale is None:
                keys = keys.reshape(-1, self.num_kv_heads, self.head_dim)[:length].float()
            else:
                keys = self._dequant_keys(keys, blocks, length).float()
            # Grouped-query attention without materialising the expanded heads.
            query = q[b].float().view(self.num_kv_heads, n_rep, self.head_dim)
            logits = torch.einsum("grd,lgd->grl", query, keys) * self.scale
            mass = torch.softmax(logits, dim=-1).sum(dim=(0, 1))
            slots = (blocks[:, None] * block_size
                     + torch.arange(block_size, device=keys.device)[None, :]).reshape(-1)[:length]
            self.eviction_scores.index_add_(0, slots, mass)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            if self.k_scale is None:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            elif self.quant_fused:
                quantize_kvcache_fused(k, v, k_cache, v_cache, self.k_scale, self.v_scale,
                                       context.slot_mapping, self.quant_group)
            else:
                quantize_kvcache(k, v, k_cache, v_cache, self.k_scale, self.v_scale,
                                 context.slot_mapping, self.quant_group)
        if context.is_prefill:
            # Scales only apply when the paged cache is the source. A plain
            # prefill attends the K/V computed for this batch, which is dense and
            # already in the model dtype - handing the shim int8 descales there
            # would reinterpret fp16 bits as int8.
            quant = {}
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
                quant = dict(k_scale=self.k_scale, v_scale=self.v_scale,
                             quant_group=self.quant_group)
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables,
                                       **quant)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True,
                                        k_scale=self.k_scale, v_scale=self.v_scale,
                                        quant_group=self.quant_group)
            if self.eviction_scores is not None:
                self._accumulate_importance(q, context)
        return o
