from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.num_evicted_tokens = 0
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # ---- KV eviction: logical length (num_tokens) vs physical length -----------
    # num_tokens is the logical sequence length: what the model sees, what RoPE
    # positions are derived from, and what the prefix cache hashes. After an
    # eviction some of those tokens no longer have KV resident in the block table,
    # so every *slot index* must be derived from the physical length instead.
    # num_evicted_tokens is the single source of truth for the difference; the
    # rest are derived, so they cannot drift apart.

    @property
    def physical_num_tokens(self) -> int:
        """Number of tokens whose KV the model accounts for."""
        return self.num_tokens - self.num_evicted_tokens

    @property
    def resident_num_tokens(self) -> int:
        """Tokens whose KV *slot* exists in the block table right now.

        Normally equal to `physical_num_tokens`. It is one less for exactly one
        step per block boundary: `append_token` grows the sequence immediately,
        while the block for the new token is only allocated by `may_append` when
        the sequence is scheduled again. Eviction must use this number, because
        that boundary step is precisely when memory pressure shows up - so it
        cannot simply refuse to evict sequences in this state.
        """
        return min(self.physical_num_tokens, len(self.block_table) * self.block_size)

    @property
    def physical_num_blocks(self) -> int:
        return (self.physical_num_tokens + self.block_size - 1) // self.block_size

    @property
    def physical_last_block_num_tokens(self) -> int:
        return self.physical_num_tokens - (self.physical_num_blocks - 1) * self.block_size

    @property
    def has_evicted_kv(self) -> bool:
        """True once KV has been evicted: the block table no longer holds a
        faithful copy of the first `num_tokens` tokens, so this sequence must
        stay out of the content-hashed prefix cache."""
        return self.num_evicted_tokens > 0

    def reset_eviction_state(self):
        """Called when KV is thrown away entirely (preemption -> full recompute)."""
        self.num_evicted_tokens = 0

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        # num_evicted_tokens has to travel: the worker ranks compute their own
        # slot_mapping and cache lengths from the physical length, so dropping it
        # here would put every rank except 0 on a different address space.
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state, self.num_evicted_tokens)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state, self.num_evicted_tokens = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
