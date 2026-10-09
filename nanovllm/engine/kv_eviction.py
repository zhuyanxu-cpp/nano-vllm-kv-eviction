"""KV cache eviction: deciding which resident keys to drop when blocks run short.

This is the piece that turns memory pressure into a *choice* instead of a
`preempt()` (which throws away a whole sequence's KV and recomputes it later).

Four constraints shaped the design; each one is load-bearing, and each one is a
good interview answer:

1. **Decisions are made in physical index space**, i.e. in slot order inside the
   block table, never in logical token space. After an eviction the block table
   is a dense copy of the *surviving* tokens only, so "physical index i" is the
   only thing that can be mapped to a slot without carrying a position table.
   The single quantity that crosses between the two spaces is the *count* of
   evicted tokens (`Sequence.num_evicted_tokens`); everything else is derived.

2. **Only prefill-finished sequences are evictable.** Every future query of such
   a sequence is a single token that must attend all surviving keys, so the
   attention mask is "attend everything" regardless of which keys were dropped.
   A chunked prefill would instead rely on bottom-right causal alignment, which
   silently assumes the cached keys are a contiguous prefix - not true after
   eviction. Restricting eviction to decode is what keeps the mask trivially
   correct.

3. **The head is never evicted** (`sink` tokens). Attention sinks accumulate
   disproportionate mass and dropping them is known to wreck quality
   (StreamingLLM). The tail (`window`) is never evicted either: it is what the
   next few queries attend to most.

4. **Survivors are kept in ascending order**, so compaction produces a dense
   layout from slot 0 and slot arithmetic stays `block_table[i // bs] * bs +
   i % bs`.
"""

from dataclasses import dataclass, field

import numpy as np

POLICIES = ("lru", "h2o", "random")


def select_keep_indices(
    policy: str,
    num_resident: int,
    sink: int,
    window: int,
    drop_count: int,
    scores: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Pick which resident physical indices survive, in ascending order.

    Only the *middle* region ``[sink, num_resident - window)`` is eligible; the
    head is the attention sink and the tail is what the next queries attend to.
    ``drop_count`` tokens are removed from the middle, chosen by the policy:

    * ``lru``    - drop the oldest (keep the most recent); no scores needed.
    * ``h2o``    - drop the smallest accumulated attention mass (keep the heavy
                   hitters).
    * ``random`` - drop a uniform subset; the control condition that tells you
                   whether the importance signal is worth anything at all.

    Note what ``drop_count`` is *not*: it is not "a fraction of what is left".
    Applying a ratio to the current resident set compounds - each eviction halves
    what the previous one left - so a sequence under sustained pressure is
    destroyed geometrically. The caller derives ``drop_count`` from a stable
    per-sequence budget instead (see ``resident_budget``).
    """
    assert policy in POLICIES, policy
    num_resident = int(num_resident)
    head_end = min(sink, num_resident)
    tail_start = max(head_end, num_resident - window)
    head = np.arange(0, head_end, dtype=np.int64)
    tail = np.arange(tail_start, num_resident, dtype=np.int64)
    middle = np.arange(head_end, tail_start, dtype=np.int64)

    drop_count = int(min(max(drop_count, 0), len(middle)))
    keep_middle = len(middle) - drop_count
    if keep_middle <= 0:
        chosen = middle[:0]
    elif keep_middle >= len(middle):
        chosen = middle
    elif policy == "lru":
        chosen = middle[-keep_middle:]
    elif policy == "random":
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(middle, size=keep_middle, replace=False))
    else:
        if scores is None:
            raise ValueError("the h2o policy needs attention scores")
        s = np.asarray(scores, dtype=np.float64)
        assert s.shape[0] == num_resident, (s.shape, num_resident)
        # Stable ordering: ties fall back to the more recent key.
        order = np.lexsort((middle, s[middle]))[::-1]
        chosen = np.sort(middle[order[:keep_middle]])

    return np.concatenate([head, chosen, tail])


def resident_budget(num_prompt_tokens: int, keep_ratio: float, floor: int) -> int:
    """The least number of KV entries a sequence is guaranteed to keep.

    This is a *floor on retention*, not a target to jump to. Eviction is
    pressure-driven, so the question it answers is "how far may this sequence be
    squeezed", and the answer is a fraction of its prompt. Two other readings of
    the same knob were tried and rejected:

    * "past a fraction of the prompt, evict on sight" - that is proactive
      eviction, a different feature, and it drops context nothing asked for;
    * "drop a fraction of whatever is resident" - compounds: every eviction
      halves what the previous one left, so sustained pressure destroys a
      sequence geometrically instead of asymptotically.
    """
    return max(int(floor), int(np.ceil(num_prompt_tokens * keep_ratio)))


def plan_drop(physical: int, table_blocks: int, blocks_needed: int,
              block_size: int, min_keep: int) -> int:
    """Tokens to drop so that `blocks_needed` whole blocks are actually released.

    Derived from the block table, not from the token count: a block is released
    when the sequence stops needing it, i.e. when the retained tokens fit in
    `table_blocks - blocks_needed` blocks. Working from the token count instead
    gets this wrong in exactly the state that matters - right after a token was
    appended but before its block was allocated, where the table already has
    spare capacity and a "block boundary" is crossed without any block being
    needed.

    The result is clamped so the post-eviction *physical* length never falls
    below `min_keep`. That invariant (``physical_num_tokens >= min_keep`` after
    every eviction) is what makes the retention floor checkable rather than
    approximate: the token that was appended but has no KV yet still counts
    toward the length the model will see, so leaving it out of the floor made
    the clamp miss by exactly one token and refuse evictions that were possible.
    """
    drop = physical - (table_blocks - blocks_needed) * block_size
    if drop <= 0:
        return 0
    return int(min(drop, max(0, physical - min_keep)))



@dataclass
class EvictionStats:
    """Counters the benchmark reads; also the raw material for the write-up."""

    events: int = 0
    tokens_evicted: int = 0
    blocks_freed: int = 0
    preemptions: int = 0
    select_seconds: float = 0.0
    move_seconds: float = 0.0
    score_read_seconds: float = 0.0
    kv_private_sequences: int = 0
    history: list[dict] = field(default_factory=list)
    record_history: bool = False

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "history"}
        return d
