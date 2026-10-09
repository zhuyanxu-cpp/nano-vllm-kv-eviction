"""Probe whether Triton can actually compile and launch a kernel on this machine.

Triton's Windows build still needs MSVC (``cl.exe``) to compile the launcher it
generates, so ``import triton`` succeeding does not mean a kernel will run.
"""

import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _double(x_ptr, o_ptr):
    i = tl.program_id(0)
    tl.store(o_ptr + i, tl.load(x_ptr + i) * 2)


def main():
    x = torch.arange(8, dtype=torch.float32, device="cuda")
    o = torch.empty_like(x)
    _double[(8,)](x, o)
    torch.cuda.synchronize()
    ok = torch.equal(o, x * 2)
    print(f"triton {triton.__version__}: kernel launch ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
