import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # ---- KV cache eviction (local feature, default off => upstream behaviour) ----
    kv_eviction: bool = False
    eviction_policy: str = "h2o"          # "lru" | "h2o" | "random"
    eviction_keep_ratio: float = 0.5      # fraction of the evictable region to keep
    eviction_sink_tokens: int = 4         # always keep the first N tokens (attention sink)
    eviction_window_tokens: int = 128     # always keep the most recent N tokens
    eviction_probe_layers: int = 4        # how many layers feed the importance scores
    # Off by default: measured to be a net loss (see docs/bench_results.md).
    eviction_on_admission: bool = False
    # Reproduces the bulk (single-tensor) KV copy that the per-layer loop
    # replaced; kept so the measurement behind that change stays reproducible.
    eviction_bulk_kv_move: bool = False

    # ---- KV cache quantisation (local feature, default off) ----
    kv_quant: str = "none"                # "none" | "int8"
    kv_quant_group: int = 32              # K is quantised per channel group; V per token

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.kv_quant in ("none", "int8"), self.kv_quant
        if self.kv_quant == "int8":
            assert self.kv_quant_group in (16, 32, 64, 128), self.kv_quant_group
        if self.kv_eviction:
            assert self.eviction_policy in ("lru", "h2o", "random"), self.eviction_policy
            assert 0.0 < self.eviction_keep_ratio < 1.0, self.eviction_keep_ratio
            assert self.eviction_sink_tokens >= 0
            assert self.eviction_window_tokens >= self.eviction_sink_tokens
            assert self.eviction_probe_layers >= 0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
