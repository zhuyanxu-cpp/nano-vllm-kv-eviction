"""What the int8 KV layout costs numerically - and why the K/V asymmetry is right.

These run on CPU and call the *production* `quantize_kvcache`, so the numbers
below describe the code that actually runs, not a model of it.

The one thing worth being precise about: KIVI quantises **K per channel across
tokens**. A streaming paged cache cannot do that - tokens arrive one at a time
and no token is ever revisited to fix up a stale scale - so the only choice left
is the granularity *along the head dimension*: one scale for the whole head
vector, or one per channel group. `test_channel_outliers_...` measures that
choice. V goes the other way (one scale per token), and
`test_v_token_outliers_...` shows why.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.layers.attention import quantize_kvcache  # noqa: E402

BLOCK = 256


def quantize_roundtrip(k, v, group):
    """Run the real writer, then dequantise the way the attention path does."""
    n, kvh, hd = k.shape
    nb = (n + BLOCK - 1) // BLOCK
    kc = torch.zeros(nb, BLOCK, kvh, hd, dtype=torch.int8)
    vc = torch.zeros_like(kc)
    ks = torch.zeros(nb, BLOCK, kvh, hd // group, dtype=torch.float16)
    vs = torch.zeros(nb, BLOCK, kvh, dtype=torch.float16)
    quantize_kvcache(k, v, kc, vc, ks, vs, torch.arange(n), group)
    groups = hd // group
    kd = (kc.reshape(-1, kvh, groups, group).float()
          * ks.reshape(-1, kvh, groups)[:n].unsqueeze(-1)).reshape(n, kvh, hd)
    vd = vc.reshape(-1, kvh, hd).float()[:n] * vs.reshape(-1, kvh)[:n].unsqueeze(-1)
    return kd, vd


def one_scale_per_head(x):
    """The arm we rejected for K: a single scale for the whole head vector."""
    s = (x.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=1e-8)
    return torch.round(x / s).clamp(-127, 127) * s


def rel_err(x, y):
    return (x - y).norm() / x.norm()


def test_int8_roundtrip_error_is_small():
    g = torch.Generator().manual_seed(0)
    k = torch.randn(512, 8, 128, generator=g)
    v = torch.randn(512, 8, 128, generator=g)
    kd, vd = quantize_roundtrip(k, v, group=32)
    ek, ev = rel_err(k, kd).item(), rel_err(v, vd).item()
    print(f"    random K rel_err={ek:.4f}  V rel_err={ev:.4f}")
    assert ek < 0.02, ek          # ~1/127 per level, averaged over 32 channels
    assert ev < 0.02, ev


def test_channel_outliers_justify_grouping_k():
    """K's outliers live in the same few channels for every token."""
    g = torch.Generator().manual_seed(1)
    k = torch.randn(512, 8, 128, generator=g)
    k[..., :4] *= 12.0                      # four persistent outlier channels
    grouped, _ = quantize_roundtrip(k, k, group=32)
    flat = one_scale_per_head(k)
    eg, ef = rel_err(k, grouped).item(), rel_err(k, flat).item()
    print(f"    K with channel outliers: group-32 rel_err={eg:.4f}  "
          f"one-scale-per-head rel_err={ef:.4f}  ({ef / eg:.1f}x worse)")
    # Measured ~2.0x on this construction; the claim is "clearly better", not a
    # fabricated factor. With group=32 and 128 channels only three of the four
    # groups are outlier-free, which caps the gain - smaller groups help accuracy
    # and cost scale bytes (see test_scale_storage_overhead...).
    assert eg < ef * 0.6, (eg, ef)


def test_v_token_outliers_justify_per_token_v():
    """V's outliers are per token, so per-token scaling is the right granularity."""
    g = torch.Generator().manual_seed(2)
    v = torch.randn(512, 8, 128, generator=g)
    v[::7] *= 12.0                          # one loud token in seven
    _, per_token = quantize_roundtrip(v, v, group=32)
    # Rejected alternative: scale each channel across the whole batch (this is
    # what "per channel" would mean for V, and it is unavailable when streaming).
    s = (v.abs().amax(dim=0, keepdim=True) / 127).clamp(min=1e-8)
    per_channel = torch.round(v / s).clamp(-127, 127) * s
    et, ec = rel_err(v, per_token).item(), rel_err(v, per_channel).item()
    print(f"    V with token outliers: per-token rel_err={et:.4f}  "
          f"per-channel rel_err={ec:.4f}")
    assert et < 0.03, et
    assert et < ec * 0.6, (et, ec)


def test_scale_storage_overhead_matches_the_accounting():
    """The block-size arithmetic in model_runner must match reality."""
    layers, block, kvh, hd, group = 28, 256, 8, 128, 32
    fp16 = 2 * layers * block * kvh * hd * 2
    int8 = 2 * layers * block * kvh * hd * 1 + layers * block * kvh * (hd // group + 1) * 2
    ratio = int8 / fp16
    print(f"    block bytes: fp16={fp16 / 2**20:.2f} MiB  int8={int8 / 2**20:.2f} MiB  "
          f"ratio={ratio:.3f}  -> blocks x{1 / ratio:.2f}")
    assert 0.50 < ratio < 0.56, ratio
    assert 1.8 < 1 / ratio < 2.0


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
