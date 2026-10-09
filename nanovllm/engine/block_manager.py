from collections import deque
from dataclasses import dataclass
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


@dataclass
class EvictionPlan:
    """What an eviction will do, computed before anything is mutated.

    `src_slots`/`dst_slots` drive the GPU-side KV copy; the block bookkeeping is
    only committed after that copy succeeded.
    """

    src_slots: list[int]
    dst_slots: list[int]
    freed_block_ids: list[int]
    num_kept: int
    num_evicted: int
    old_num_blocks: int
    new_num_blocks: int


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def lookup_blocks(self, seq: Sequence) -> tuple[int, int]:
        """(num_cached_blocks, num_new_blocks) a fresh sequence would need.

        Split out of `can_allocate` so the scheduler can ask *how many* blocks are
        missing and evict exactly that many, instead of guessing.
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        return num_cached_blocks, num_new_blocks

    def can_allocate(self, seq: Sequence) -> int:
        num_cached_blocks, num_new_blocks = self.lookup_blocks(seq)
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def needs_new_block(self, seq: Sequence) -> bool:
        """Does the token about to be written land outside the allocated blocks?

        Upstream spells this "len(seq) % block_size == 1", which is equivalent
        only while the block table holds exactly ceil(length / block_size)
        blocks. After an eviction that is no longer true - the table can hold
        spare capacity - and the length-based test would then allocate a block
        that already exists. Deriving it from the table itself is the general
        form of the same rule.
        """
        return (seq.physical_num_tokens - 1) // self.block_size >= len(seq.block_table)

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= self.needs_new_block(seq)

    def may_append(self, seq: Sequence):
        if self.needs_new_block(seq):
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        # An evicted sequence is excluded from the content-hashed prefix cache
        # for good. After compaction, physical block i no longer holds
        # seq.block(i), and `can_allocate` validates a cache hit by comparing the
        # block's recorded token_ids -- metadata against metadata -- so a block
        # whose KV was rewritten underneath would still be handed out as a valid
        # prefix and silently produce wrong output.
        if seq.has_evicted_kv:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

    # ---- KV cache eviction -------------------------------------------------

    def slot_of(self, seq: Sequence, physical_index: int) -> int:
        """Slot holding the `physical_index`-th resident token of `seq`."""
        return seq.block_table[physical_index // self.block_size] * self.block_size \
            + physical_index % self.block_size

    def _deregister_hash(self, block_id: int):
        """Remove a block from the content-hash index.

        Clearing `block.hash` alone is *not* enough: `_allocate_block` only
        deletes the dict entry while it still points at that block, so a stale
        entry would survive and `can_allocate` would go on to hand out a block
        whose KV no longer matches the token_ids recorded for it. Both sides of
        that check are metadata, so nothing else would catch the mismatch.
        """
        block = self.blocks[block_id]
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.hash = -1
        block.token_ids = []

    def plan_eviction(self, seq: Sequence, keep_indices: np.ndarray) -> EvictionPlan | None:
        """Work out how to compact `seq`'s KV down to `keep_indices`.

        `keep_indices` are *physical* indices (ascending) of the resident tokens
        to keep. Returns None when the sequence cannot or should not be evicted:

        * any block it owns is shared (`ref_count > 1`) - rewriting a shared
          block in place would corrupt the other sequence reading it, and
          deallocating it would only drop one reference;
        * compaction would not free a whole block, i.e. it cannot relieve
          pressure and would only cost work.
        """
        expected_blocks = (seq.resident_num_tokens + self.block_size - 1) // self.block_size
        assert len(seq.block_table) == expected_blocks, \
            (len(seq.block_table), expected_blocks, seq.resident_num_tokens, seq.num_evicted_tokens)
        num_keep = len(keep_indices)
        if num_keep == 0 or num_keep >= seq.resident_num_tokens:
            return None
        if any(self.blocks[b].ref_count > 1 for b in seq.block_table):
            return None
        # The token appended but not yet given a block rides along right behind
        # the survivors, so it is part of the post-eviction length even though it
        # has no KV to move.
        num_evicted = seq.resident_num_tokens - num_keep
        physical_after = seq.physical_num_tokens - num_evicted
        new_num_blocks = (physical_after + self.block_size - 1) // self.block_size
        if new_num_blocks >= len(seq.block_table):
            return None

        old_table = list(seq.block_table)
        src = [self.slot_of(seq, int(i)) for i in keep_indices]
        dst = [old_table[j // self.block_size] * self.block_size + j % self.block_size
               for j in range(num_keep)]
        return EvictionPlan(
            src_slots=src,
            dst_slots=dst,
            freed_block_ids=old_table[new_num_blocks:],
            num_kept=num_keep,
            num_evicted=num_evicted,
            old_num_blocks=len(old_table),
            new_num_blocks=new_num_blocks,
        )

    def commit_eviction(self, seq: Sequence, plan: EvictionPlan):
        """Apply the block bookkeeping. Call only after the KV copy succeeded."""
        for block_id in seq.block_table:
            self._deregister_hash(block_id)
        for block_id in plan.freed_block_ids:
            block = self.blocks[block_id]
            # Deliberately not block.reset(): that sets ref_count = 1, which is
            # the "handed to a new owner" semantic. A freed block must go back to
            # the free list with ref_count == 0, or the next _allocate_block
            # trips over its own assertion.
            block.ref_count = 0
            self._deallocate_block(block_id)
        seq.block_table = seq.block_table[:plan.new_num_blocks]
        seq.num_evicted_tokens += plan.num_evicted
