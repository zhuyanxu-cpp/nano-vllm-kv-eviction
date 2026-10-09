"""Accuracy under eviction: needle-in-a-haystack plus generation agreement.

Throughput turned out to be *policy invariant* (the three policies drop the same
number of tokens; only the choice of which ones differs), so a policy can only be
judged by what it costs in quality. This script measures that:

* **needle retrieval** - one unique passphrase is hidden at a fixed depth inside a
  long prompt, and the model is asked for it. Exact-match rate per configuration.
* **generation agreement** - the token ids from each run are saved, so agreement
  with the uncompressed baseline can be computed afterwards (see compare in
  docs/bench_results.md).

Every request carries a different passphrase and different filler, so no prefix
sharing occurs - shared blocks must never be rewritten (see the tests), and
identical prompts would silently disable eviction.

    python bench/bench_needle.py --out results/needle_base.json  --requests 16
    python bench/bench_needle.py --out results/needle_r50.json --requests 16 \
        --eviction on --policy h2o --ratio 0.5
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.harness import add_common_args, build_llm, drive, write_json  # noqa: E402

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
         "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
         "xray yankee zulu river mountain forest harbour bridge engine factor signal "
         "window candle meadow orbit matrix puzzle lantern marble copper velvet").split()


def filler(rng: random.Random, n_sentences: int) -> str:
    return "\n".join(
        f"Note {i}: the {rng.choice(WORDS)} near the {rng.choice(WORDS)} was recorded "
        f"in log {rng.randrange(1000, 9999)} after the {rng.choice(WORDS)}."
        for i in range(n_sentences))


MARKER = "@@@@@"


def build_ids(tokenizer, rng: random.Random, n_sentences: int, depth_fraction: float,
              needle: str, target_tokens: int = 1024):
    """Assemble the prompt in *token* space to hit an exact length.

    Sentence granularity is ~25 tokens, so no sentence count can land in the few
    tokens below a block boundary - and that is the whole point of the
    calibration: memory pressure only appears when a sequence has to move into a
    new block, and with a 16-token output budget that must happen within the
    first couple of decode steps. Building head + body + tail as ids (the engine
    accepts token-id prompts directly) gives exact control and keeps the needle
    at its depth, because truncation only ever trims trailing filler.

    Thinking is switched off as well: Qwen3 otherwise emits a long `` thinking``
    block first, and a short token budget would then measure the reasoning budget
    instead of what the model retrieved.
    """
    lines = filler(rng, n_sentences).split("\n")
    lines.insert(min(int(n_sentences * depth_fraction), len(lines)), needle)
    body = "\n".join(lines)
    question = ("\n\nWhat is the secret passphrase? Reply with the passphrase only. /no_think")
    messages = [{"role": "user", "content": MARKER + body + MARKER + question}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # The marker brackets the body, so the chat template's own prefix and suffix
    # can be measured exactly and the body can be truncated to fill the rest.
    prefix, _body, suffix = text.split(MARKER)
    head = tokenizer.encode(prefix, add_special_tokens=False)
    tail = tokenizer.encode(suffix, add_special_tokens=False)
    body_ids = tokenizer.encode(body + question, add_special_tokens=False)
    budget = max(16, target_tokens - len(head) - len(tail))
    return head + body_ids[:budget] + tail


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--sentences", type=int, default=60, help="filler sentences before truncation")
    ap.add_argument("--prompt-tokens", type=int, default=1024,
                    help="exact prompt length; 1024 = 4 blocks, so the 2nd generated "
                         "token already needs a 5th block and creates pressure")
    ap.add_argument("--depth-fraction", type=float, default=0.4,
                    help="where the needle sits; 0.4 keeps it out of the sink/window")
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    rng = random.Random(args.seed)
    needles, ids = [], []
    for i in range(args.requests):
        needle = f"The secret passphrase is {rng.choice(WORDS).upper()}-{rng.randrange(1000, 9999)}-{i}."
        needles.append(needle)
        ids.append(build_ids(tokenizer, rng, args.sentences, args.depth_fraction,
                             needle, args.prompt_tokens))
    lengths = [len(x) for x in ids]

    llm = build_llm(args)
    result = drive(llm, ids, args.max_tokens, 0.6, args.seed)
    result["args"] = vars(args)

    # Score: does the answer contain the passphrase (ignoring spacing/case)?
    # Sequence ids are not 0-based indices (the warmup pass consumes a few), but
    # requests are added in order and ids increase monotonically, so the n-th
    # sorted id belongs to the n-th prompt.
    hits, answers = 0, {}
    for i, seq_id in enumerate(sorted(result["outputs"])):
        token_ids = result["outputs"][seq_id]
        text = tokenizer.decode(token_ids)
        needle = needles[i].split("is ")[-1].rstrip(".")
        ok = needle.lower() in text.lower()
        hits += ok
        answers[str(seq_id)] = {"text": text, "needle": needle, "hit": bool(ok)}
    result["needle"] = {
        "hits": hits,
        "requests": len(answers),
        "accuracy": hits / max(1, len(answers)),
        "prompt_tokens_mean": sum(lengths) / len(lengths),
        "answers": answers,
    }
    write_json(args.out, result)

    m, e, n = result["metrics"], result["eviction"], result["needle"]
    print(f"[{os.path.basename(args.out)}] prompt_tokens~{n['prompt_tokens_mean']:.0f} "
          f"needle={n['hits']}/{n['requests']}={n['accuracy']:.0%} "
          f"wall={m['wall_seconds']:.1f}s preempt={e['preemptions']} "
          f"evicted_tokens={e['tokens_evicted']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
