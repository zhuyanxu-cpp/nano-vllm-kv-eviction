"""Unit tests for the eviction policy layer - pure functions, no GPU, no engine.

These pin down the properties the rest of the system relies on: the head and tail
are never dropped, the drop count is respected exactly, and the retention floor is
a floor (not a target that gets jumped to).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.engine.kv_eviction import (  # noqa: E402
    plan_drop, resident_budget, select_keep_indices,
)


def test_head_and_tail_are_never_dropped():
    keep = select_keep_indices("lru", 1000, sink=4, window=128, drop_count=500)
    assert list(keep[:4]) == [0, 1, 2, 3]
    assert list(keep[-128:]) == list(range(872, 1000))
    assert len(keep) == 1000 - 500


def test_drop_count_is_exact_and_order_is_ascending():
    for drop in (0, 1, 137, 500, 868):
        keep = select_keep_indices("h2o", 1000, 4, 128, drop,
                                   scores=np.random.default_rng(0).random(1000))
        assert len(keep) == 1000 - drop, drop
        assert np.all(np.diff(keep) > 0), "compaction needs ascending physical indices"


def test_drop_count_is_clamped_to_the_evictable_region():
    """Asking to drop more than the middle holds must not touch sink or window."""
    keep = select_keep_indices("lru", 200, sink=4, window=128, drop_count=10_000)
    assert list(keep) == list(range(4)) + list(range(72, 200))


def test_lru_keeps_the_most_recent_and_h2o_keeps_the_scored():
    scores = np.zeros(1000)
    scores[500] = 100.0                      # one obvious heavy hitter
    # Drop 600 of the 868-token middle, so the oldest middle tokens are gone but
    # the retained recent part still ends at the window.
    lru = select_keep_indices("lru", 1000, 4, 128, 600, scores=scores)
    h2o = select_keep_indices("h2o", 1000, 4, 128, 600, scores=scores)
    assert 500 not in lru, "lru drops the oldest middle tokens, so 500 should be gone"
    assert 500 in h2o, "h2o should have kept the highest-scoring token"
    assert lru[-128:].tolist() == h2o[-128:].tolist() == list(range(872, 1000))


def test_random_is_deterministic_per_seed():
    a = select_keep_indices("random", 1000, 4, 128, 300, seed=7)
    b = select_keep_indices("random", 1000, 4, 128, 300, seed=7)
    c = select_keep_indices("random", 1000, 4, 128, 300, seed=8)
    assert np.array_equal(a, b), "a benchmark run must be reproducible"
    assert not np.array_equal(a, c), "different sequences should evict differently"


def test_h2o_without_scores_is_an_error():
    try:
        select_keep_indices("h2o", 100, 4, 8, 10)
    except ValueError:
        return
    raise AssertionError("h2o must not silently fall back to another policy")


def test_resident_budget_is_a_floor():
    # floor wins when the ratio would keep less than sink + window
    assert resident_budget(100, 0.5, floor=132) == 132
    assert resident_budget(4000, 0.5, floor=132) == 2000
    assert resident_budget(1000, 1.0, floor=132) == 1000


def test_plan_drop_frees_exactly_the_blocks_asked_for():
    # 4 blocks of 256 hold 1024 tokens; 1025 physical means a 5th block is needed
    for table_blocks, physical, needed in ((4, 1025, 1), (4, 1025, 2), (8, 2049, 1)):
        drop = plan_drop(physical, table_blocks, needed, 256, min_keep=0)
        after = physical - drop
        blocks_after = (after + 255) // 256
        assert table_blocks - blocks_after >= needed, (physical, drop, blocks_after)


def test_plan_drop_respects_the_retention_floor():
    # 1025 tokens in 4 blocks; freeing 2 blocks needs a big drop, floor forbids it
    drop = plan_drop(1025, 4, 2, 256, min_keep=1000)
    assert drop == 25, drop
    assert 1025 - drop >= 1000


def test_plan_drop_is_zero_when_nothing_can_be_gained():
    # 256 tokens already fit in the 2 allocated blocks, so "free one block" is
    # already satisfied and there is nothing to do.
    assert plan_drop(256, 2, 1, 256, min_keep=0) == 0, "table already has spare capacity"
    assert plan_drop(1000, 4, 1, 256, min_keep=1000) == 0, "floor blocks the drop"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
