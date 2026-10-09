from collections import deque
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.kv_eviction import EvictionStats, plan_drop, resident_budget, select_keep_indices


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        # ---- KV cache eviction --------------------------------------------
        # Purely pressure-driven: eviction is only attempted when the block pool
        # cannot satisfy an allocation, i.e. exactly where upstream would have
        # called preempt() and thrown a whole sequence's KV away.
        self.kv_eviction = config.kv_eviction
        self.eviction_policy = config.eviction_policy
        self.eviction_keep_ratio = config.eviction_keep_ratio
        self.eviction_sink = config.eviction_sink_tokens
        self.eviction_window = config.eviction_window_tokens
        self.eviction_on_admission = config.eviction_on_admission
        self.stats = EvictionStats()
        # Injected by LLMEngine: reads accumulated attention mass for given
        # physical slots, and copies KV between slots on the GPU.
        self.score_provider = None
        self.kv_cache_io = None

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # Measured decision: eviction is NOT used to admit new
                    # requests here. Doing so drops the context of sequences that
                    # are already running in order to start more prefill work,
                    # and on the 16 x 1024-token workload that made the run 12%
                    # slower (19.0 s -> 21.4 s, see docs/bench_results.md). The
                    # 53% figure belongs to the plan_drop clamping experiment.
                    # Admission control stays first-come-first-served; the queued
                    # prompt simply waits.
                    if self.eviction_on_admission:
                        _, num_new_blocks = self.block_manager.lookup_blocks(seq)
                        shortfall = num_new_blocks - len(self.block_manager.free_block_ids)
                        batch_ids = {s.seq_id for s in scheduled_seqs}
                        if self._evict_for_blocks(max(1, shortfall), batch_ids):
                            num_cached_blocks = self.block_manager.can_allocate(seq)
                    if num_cached_blocks == -1:
                        break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self._evict_for_blocks(1, {s.seq_id for s in scheduled_seqs}):
                    continue
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        blocks = list(seq.block_table)
        self.block_manager.deallocate(seq)
        self._release_block_scores(blocks)
        # All KV is gone, so the physical/logical split collapses back to zero;
        # the sequence will be re-prefilled from scratch on the next schedule().
        seq.reset_eviction_state()
        self.stats.preemptions += 1
        self.waiting.appendleft(seq)

    def _release_block_scores(self, block_ids: list[int]):
        """Drop stale importance scores when blocks go back to the free pool."""
        clear = getattr(self.kv_cache_io, "clear_block_scores", None) if self.kv_cache_io else None
        if clear is not None and block_ids:
            clear(block_ids)

    # ---- KV cache eviction -------------------------------------------------

    def retention_floor(self, seq: Sequence) -> int:
        """The least KV a sequence keeps: sink + window, and a share of its prompt."""
        return resident_budget(seq.num_prompt_tokens, self.eviction_keep_ratio,
                               self.eviction_sink + self.eviction_window)

    def evictable_tokens(self, seq: Sequence) -> int:
        """How many resident tokens this sequence could still give up.

        Only RUNNING sequences are eligible. In this engine that is exactly the
        "prefill finished" state (the prefill branch sets RUNNING only when the
        whole prompt has been processed), which is what keeps the attention mask
        trivially correct after eviction. Two tempting-looking alternatives are
        both wrong:

        * ``num_cached_tokens == num_tokens`` is never true once generation
          starts, because ``postprocess`` bumps ``num_cached_tokens`` *before*
          ``append_token`` bumps ``num_tokens``, leaving a permanent gap of 1;
        * "has blocks allocated" is true for a sequence that is halfway through
          a chunked prefill.
        """
        if seq.status is not SequenceStatus.RUNNING or len(seq.block_table) == 0:
            return 0
        return max(0, seq.resident_num_tokens - self.retention_floor(seq))

    def planned_drop(self, seq: Sequence, blocks_needed: int) -> int:
        return plan_drop(seq.physical_num_tokens, len(seq.block_table), blocks_needed,
                         self.block_size, self.retention_floor(seq))

    def _pick_victim(self, blocks_needed: int, exclude_ids: set[int]) -> Sequence | None:
        """The sequence that can give up the requested blocks with the least loss."""
        victim, most = None, 0
        for seq in self.running:
            if seq.seq_id in exclude_ids:
                # Part of the batch being scheduled right now: either its KV for
                # this step has not been written yet (fresh prefill), or it is
                # the sequence that is already being served.
                continue
            if seq.status is not SequenceStatus.RUNNING or len(seq.block_table) == 0:
                continue
            if self.planned_drop(seq, blocks_needed) <= 0:
                continue
            headroom = self.evictable_tokens(seq)
            if headroom > most:
                victim, most = seq, headroom
        return victim

    def _evict_for_blocks(self, blocks_needed: int = 1, exclude_ids: set[int] = frozenset()) -> bool:
        """Free at least one block by evicting KV. True iff a block was freed."""
        if not self.kv_eviction:
            return False
        victim = self._pick_victim(blocks_needed, exclude_ids)
        if victim is None:
            return False
        return self.evict(victim, blocks_needed)

    def evict(self, seq: Sequence, blocks_needed: int = 1) -> bool:
        """Compact `seq`'s KV down to the tokens the policy wants to keep."""
        num_resident = seq.resident_num_tokens
        drop = self.planned_drop(seq, blocks_needed)
        if drop <= 0:
            return False

        scores = None
        if self.eviction_policy == "h2o":
            if self.score_provider is None:
                raise RuntimeError("the h2o eviction policy needs an attention-score provider")
            t0 = perf_counter()
            slots = [self.block_manager.slot_of(seq, i) for i in range(num_resident)]
            scores = self.score_provider(slots)
            self.stats.score_read_seconds += perf_counter() - t0

        t0 = perf_counter()
        keep = select_keep_indices(
            self.eviction_policy, num_resident, self.eviction_sink,
            self.eviction_window, drop, scores, seed=seq.seq_id,
        )
        self.stats.select_seconds += perf_counter() - t0

        plan = self.block_manager.plan_eviction(seq, keep)
        if plan is None:
            return False

        t0 = perf_counter()
        if self.kv_cache_io is not None:
            self.kv_cache_io.move_slots(plan.src_slots, plan.dst_slots)
        self.stats.move_seconds += perf_counter() - t0

        self.block_manager.commit_eviction(seq, plan)
        self._release_block_scores(plan.freed_block_ids)
        self.stats.events += 1
        self.stats.tokens_evicted += plan.num_evicted
        self.stats.blocks_freed += len(plan.freed_block_ids)
        if self.stats.record_history:
            self.stats.history.append(dict(
                seq_id=seq.seq_id,
                num_tokens=seq.num_tokens,
                kept=plan.num_kept,
                evicted=plan.num_evicted,
                blocks=plan.old_num_blocks - plan.new_num_blocks,
                policy=self.eviction_policy,
            ))
        return len(plan.freed_block_ids) > 0

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                blocks = list(seq.block_table)
                self.block_manager.deallocate(seq)
                self._release_block_scores(blocks)
                self.running.remove(seq)
