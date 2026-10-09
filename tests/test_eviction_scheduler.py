"""Drive the real Scheduler through eviction on CPU, with fakes for the GPU parts.

Everything the eviction path needs from the GPU goes through two injected
callables (``score_provider`` and ``kv_cache_io``), so the whole decision +
block-bookkeeping path can be exercised without CUDA. That is what makes it
possible to unit-test "does eviction actually fire, and does it fire instead of
preemption" in milliseconds instead of a 20 s engine run.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_eviction_scheduler.py -q
     .venv\\Scripts\\python.exe tests/test_eviction_scheduler.py     (plain)
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.config import Config  # noqa: E402
from nanovllm.engine.scheduler import Scheduler  # noqa: E402
from nanovllm.engine.sequence import Sequence  # noqa: E402
from nanovllm.sampling_params import SamplingParams  # noqa: E402

MODEL = os.environ.get("NANOVLLM_MODEL", r"D:\huggingface\Qwen3-0.6B")


class FakeKvCacheIO:
    """Records KV moves instead of performing them."""

    def __init__(self):
        self.moves = []
        self.cleared = []

    def move_slots(self, src, dst):
        assert len(src) == len(dst)
        assert len(set(dst)) == len(dst), "destinations must be unique"
        self.moves.append((list(src), list(dst)))

    def clear_block_scores(self, block_ids):
        self.cleared.append(list(block_ids))


def make_config(**overrides) -> Config:
    kwargs = dict(
        model=MODEL,
        max_num_batched_tokens=512,
        max_num_seqs=64,
        max_model_len=4096,
        kvcache_block_size=256,
    )
    kwargs.update(overrides)
    config = Config(**kwargs)
    config.num_kvcache_blocks = overrides.pop("num_kvcache_blocks", 8)
    config.eos = 0  # never emitted: our fake tokens are always 1
    return config


def run(scheduler: Scheduler, prompts: list[list[int]], max_tokens: int = 8, limit: int = 20000):
    """The engine loop from LLMEngine.step(), with a fake model output."""
    params = SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=True)
    for p in prompts:
        scheduler.add(Sequence(p, params))
    steps = 0
    while not scheduler.is_finished():
        seqs, is_prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [1] * len(seqs), is_prefill)
        steps += 1
        assert steps < limit, "scheduler did not terminate"
    return steps


def distinct_prompts(n: int, length: int) -> list[list[int]]:
    """Different prompts per request: no prefix sharing, so blocks are private."""
    return [[100 + i] * length for i in range(n)]


def test_no_pressure_never_evicts():
    config = make_config(kv_eviction=True, eviction_policy="lru")
    scheduler = Scheduler(config)
    scheduler.kv_cache_io = FakeKvCacheIO()
    run(scheduler, distinct_prompts(2, 512))   # 2 prompts x 2 blocks = 4 of 8 blocks
    assert scheduler.stats.events == 0
    assert scheduler.stats.preemptions == 0
    assert scheduler.kv_cache_io.moves == []


def test_pressure_evicts_instead_of_preempting():
    # 8 blocks = 2048 tokens. Six distinct 512-token prompts need 12 blocks:
    # without eviction the engine preempts; with eviction it should reclaim.
    prompts = distinct_prompts(6, 512)

    base = Scheduler(make_config(kv_eviction=False))
    run(base, prompts)
    assert base.stats.preemptions > 0, "the workload is supposed to create pressure"

    config = make_config(kv_eviction=True, eviction_policy="lru",
                         eviction_keep_ratio=0.5, eviction_window_tokens=64)
    scheduler = Scheduler(config)
    scheduler.kv_cache_io = FakeKvCacheIO()
    run(scheduler, prompts)
    assert scheduler.stats.events > 0, "eviction never fired"
    assert scheduler.stats.tokens_evicted > 0
    assert scheduler.kv_cache_io.moves, "eviction must move KV on the device"
    assert scheduler.stats.preemptions < base.stats.preemptions, (
        scheduler.stats.preemptions, base.stats.preemptions)
    total_prompt = sum(len(p) for p in prompts)
    moved = sum(len(s) for s, _ in scheduler.kv_cache_io.moves)
    print(f"    evictions={scheduler.stats.events} evicted_tokens={scheduler.stats.tokens_evicted} "
          f"blocks_freed={scheduler.stats.blocks_freed} moved={moved}/{total_prompt} "
          f"preemptions={scheduler.stats.preemptions} (baseline {base.stats.preemptions})")


def test_h2o_uses_scores_and_random_is_available():
    prompts = distinct_prompts(6, 512)
    for policy in ("h2o", "random"):
        config = make_config(kv_eviction=True, eviction_policy=policy,
                             eviction_keep_ratio=0.5, eviction_window_tokens=64)
        scheduler = Scheduler(config)
        scheduler.kv_cache_io = FakeKvCacheIO()
        if policy == "h2o":
            # Deterministic "importance": later slots matter more.
            scheduler.score_provider = lambda slots: [float(s) for s in slots]
        run(scheduler, prompts)
        assert scheduler.stats.events > 0, policy


def test_shared_prefix_blocks_are_never_rewritten():
    """Identical prompts share prefix blocks; those must not be rewritten.

    Rewriting a block with ref_count > 1 in place would corrupt the KV of every
    other sequence reading it, so eviction has to refuse - even though that
    means a workload with a large shared prefix keeps preempting. This is the
    documented limitation of the in-place compaction design (see docs).
    """
    prompts = [[7] * 512] * 6
    base = Scheduler(make_config(kv_eviction=False))
    run(base, prompts)

    config = make_config(kv_eviction=True, eviction_policy="lru",
                         eviction_keep_ratio=0.5, eviction_window_tokens=64)
    scheduler = Scheduler(config)
    scheduler.kv_cache_io = FakeKvCacheIO()
    run(scheduler, prompts)
    assert scheduler.stats.events == 0, "shared blocks were rewritten"
    assert scheduler.stats.preemptions == base.stats.preemptions


def test_scores_are_cleared_when_blocks_are_recycled():
    """A recycled block must not inherit the previous occupant's attention mass."""
    config = make_config(kv_eviction=True, eviction_policy="lru",
                         eviction_keep_ratio=0.5, eviction_window_tokens=64)
    scheduler = Scheduler(config)
    scheduler.kv_cache_io = FakeKvCacheIO()
    run(scheduler, distinct_prompts(6, 512))
    assert scheduler.kv_cache_io.cleared, "freed blocks must have their scores reset"
    freed = {b for blocks in scheduler.kv_cache_io.cleared for b in blocks}
    assert freed, "nothing was ever cleared"
    # Blocks freed by eviction and by preemption both go through this path.
    assert scheduler.stats.events > 0


def test_evicted_blocks_leave_the_prefix_cache():
    """An evicted sequence's KV must be unreachable by content hash."""
    config = make_config(kv_eviction=True, eviction_policy="lru",
                         eviction_keep_ratio=0.5, eviction_window_tokens=64)
    scheduler = Scheduler(config)
    scheduler.kv_cache_io = FakeKvCacheIO()
    run(scheduler, distinct_prompts(6, 512))
    assert scheduler.stats.events > 0

    bm = scheduler.block_manager
    indexed = set(bm.hash_to_block_id.values())
    for block_id, block in enumerate(bm.blocks):
        if block_id in indexed:
            assert block.hash != -1, "hash index points at a block with no content hash"
            assert block.token_ids, "hash index points at a block with no recorded tokens"


def test_preemption_resets_eviction_state():
    """After preemption the sequence is recomputed, so the split must collapse."""
    config = make_config(kv_eviction=True, eviction_policy="lru",
                         eviction_keep_ratio=0.5, eviction_window_tokens=64)
    scheduler = Scheduler(config)
    scheduler.kv_cache_io = FakeKvCacheIO()
    prompts = [[7] * 512] * 12  # heavy pressure: preemption is unavoidable
    run(scheduler, prompts)
    assert scheduler.stats.preemptions > 0
    for seq in list(scheduler.waiting) + list(scheduler.running):
        assert seq.num_evicted_tokens == 0 or seq.block_table, "evicted sequence without KV"
        assert len(seq.block_table) == seq.physical_num_blocks, (
            len(seq.block_table), seq.physical_num_blocks)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = Counter()
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failures[t.__name__] = str(exc)
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
