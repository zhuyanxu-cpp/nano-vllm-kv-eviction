"""Reproduce every headline number in docs/bench_results.md with one command.

Each configuration must run in its own process (torch.distributed initialises
once per process), so this is a small driver around the two benchmark scripts
rather than a library.

    python bench/run_all.py                 # full matrix (~6 min on a 1650 Ti)
    python bench/run_all.py --quick         # smaller, for a smoke check
    python bench/run_all.py --only needle

``--quick`` shrinks the workload below the point where the block pool runs out, so
it checks the plumbing only: with no memory pressure nothing is ever evicted and
every configuration scores the same. The findings in docs/bench_results.md need
the full runs.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(script, out, extra, quick=False):
    args = [sys.executable, os.path.join(HERE, script), "--out", os.path.join(ROOT, out)] + extra
    if quick:
        args += ["--requests", "8", "--mbt", "512", "--util", "0.85"]
    proc = subprocess.run(args, cwd=ROOT)
    if proc.returncode != 0:
        print(f"!! {out} failed with exit code {proc.returncode}")
    return os.path.join(ROOT, out)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def throughput_table(paths, quick):
    head = "| run | wall | prefill/decode | re-prefilled | preempt | evicted | out tok/s |"
    print("\n### Throughput and latency"
          f" ({'quick' if quick else '32 requests x 1024 tokens, 2.67x oversubscribed'})\n")
    print(head)
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for label, path in paths:
        d = load(path)
        m, e = d["metrics"], d["eviction"]
        print(f"| {label} | {m['wall_seconds']:.2f} s | "
              f"{m['prefill_seconds']:.2f} / {m['decode_seconds']:.2f} s | "
              f"{m['prefill_tokens']} | {e['preemptions']} | {e['tokens_evicted']} | "
              f"{m['output_tokens_per_s']:.1f} |")


def accuracy_table(paths):
    print("\n### Needle-in-a-haystack\n")
    print("| config | hits | accuracy | preempt | evicted |")
    print("| --- | --- | --- | --- | --- |")
    for label, path in paths:
        d = load(path)
        n, e = d["needle"], d["eviction"]
        print(f"| {label} | {n['hits']}/{n['requests']} | {n['accuracy']:.0%} | "
              f"{e['preemptions']} | {e['tokens_evicted']} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["throughput", "needle"], default=None)
    args = ap.parse_args()
    # Quick and full runs must never share file names: the quick workload has no
    # memory pressure, so a full run overwriting a quick artifact (or the other
    # way round) silently produces a mixed, unreproducible results directory.
    pre = "raq_" if args.quick else "ra_"

    req = "8" if args.quick else "32"
    nreq = "8" if args.quick else "24"
    tp = [("baseline", f"results/{pre}base.json"),
          ("LRU r=0.5", f"results/{pre}lru.json"),
          ("H2O r=0.5", f"results/{pre}h2o.json")]
    nd = [("baseline", f"results/{pre}n_base.json"),
          ("LRU r=0.5", f"results/{pre}n_lru.json"),
          ("H2O r=0.5", f"results/{pre}n_h2o.json"),
          ("random r=0.5", f"results/{pre}n_rand.json"),
          ("H2O r=0.25", f"results/{pre}n_r25.json")]

    if args.only != "needle":
        base = ["--requests", req, "--prompt-len", "1024", "--max-tokens", "8"]
        run("run_case.py", tp[0][1], base + ["--eviction", "off"])
        for label, path in tp[1:]:
            policy = "lru" if "LRU" in label else "h2o"
            run("run_case.py", path, base + ["--eviction", "on", "--policy", policy, "--ratio", "0.5"])
        throughput_table(tp, args.quick)

    if args.only != "throughput":
        base = ["--requests", nreq]
        run("bench_needle.py", nd[0][1], base + ["--eviction", "off"])
        for label, path in nd[1:]:
            extra = ["--eviction", "on", "--ratio", "0.25" if "0.25" in label else "0.5"]
            extra += ["--policy", label.split()[0].lower()]
            run("bench_needle.py", path, base + extra)
        accuracy_table(nd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
