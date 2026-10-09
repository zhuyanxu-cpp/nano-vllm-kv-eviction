"""Control experiment: eviction enabled must be a no-op when it never fires.

This is the strongest correctness evidence in the project. It runs the real
engine twice - once with eviction off, once with eviction on but a workload that
is far below the block pool - and requires the two runs to produce **identical
token ids**. If the physical/logical split, the `can_append`/`may_append` change
or the prefix-cache change had any effect when nothing is evicted, this fails.

It needs a GPU and ~40 s (two engine initialisations), so it is opt-in:

    $env:NANOVLLM_GPU_TESTS=1; python -m pytest tests/test_control_identity.py -q

The two configurations must run in separate processes: torch.distributed can only
be initialised once per process, which is also why bench/run_case.py is a CLI.
"""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("NANOVLLM_MODEL", r"D:\huggingface\Qwen3-0.6B")

pytestmark = pytest.mark.skipif(
    not os.environ.get("NANOVLLM_GPU_TESTS"),
    reason="needs a GPU; set NANOVLLM_GPU_TESTS=1 to run",
)


def run_config(out: str, eviction: str):
    cmd = [sys.executable, os.path.join(ROOT, "bench", "run_case.py"),
           "--out", out, "--model", MODEL, "--requests", "2", "--prompt-len", "256",
           "--max-tokens", "8", "--mbt", "512", "--util", "0.85",
           "--eviction", eviction, "--policy", "h2o", "--ratio", "0.5"]
    subprocess.run(cmd, cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def test_eviction_enabled_is_bit_identical_when_it_never_fires():
    base = run_config(os.path.join(ROOT, "results", "ctl_test_base.json"), "off")
    evict = run_config(os.path.join(ROOT, "results", "ctl_test_on.json"), "on")

    assert evict["eviction"]["events"] == 0, "the workload must not create pressure"
    assert evict["eviction"]["preemptions"] == 0
    assert base["outputs"] == evict["outputs"], "eviction changed the output without firing"
