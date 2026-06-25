"""Regression test for hashmap capacity on the on-the-fly downsample path.

The ``hashmap_on_the_fly`` kernel-map builder inserts the OUTPUT coords into the
GPU hashtable. With ``downsample_mode="spconv"`` a strided conv on a scattered
sparse cloud expands the point count (up to ``input * kernel_volume``), so a
table sized on the input count (``hash_rsv_ratio * input``) overflows. Upstream
this is a silent illegal memory access; with the overflow check it surfaces as a
clean ValueError. The fix sizes the table on ``input * kernel_volume`` for the
downsample path, so the default ``hash_rsv_ratio=2`` no longer overflows.

Run standalone:  python tests/python/test_hashmap_capacity.py
"""

import torch

import torchsparse
import torchsparse.backends
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.nn import functional as F

__all__ = ["test_on_the_fly_downsample_capacity"]


def test_on_the_fly_downsample_capacity(device="cuda:0"):
    """A strided on-the-fly conv on a scattered cloud must not overflow the
    hashtable at the default hash_rsv_ratio (it expands ~3x at stride 2)."""
    results = {}
    torchsparse.backends.hash_rsv_ratio = 2  # default; pre-fix this overflowed
    cfg = F.conv_config.get_default_conv_config()
    cfg.kmap_mode = "hashmap_on_the_fly"
    F.conv_config.set_global_conv_config(cfg)

    for ks in (3, 5):  # larger kernel -> larger expansion
        ok = True
        for seed in range(4):
            g = torch.Generator().manual_seed(seed)
            n = 2000
            xyz = torch.randint(0, 128, (n, 3), generator=g, dtype=torch.int32)
            coords = torch.cat([torch.zeros(n, 1, dtype=torch.int32), xyz], dim=1).to(device)
            feats = torch.randn(n, 16, generator=g).half().to(device)
            conv = spnn.Conv3d(16, 32, kernel_size=ks, stride=2).to(device).half()
            try:
                y = conv(SparseTensor(coords=coords, feats=feats))
                torch.cuda.synchronize()
                ok &= y.feats.shape[0] > n  # spconv downsample expands
                ok &= bool(torch.isfinite(y.feats.float()).all().item())
            except Exception:
                ok = False
        results[f"kernel_{ks}_no_overflow"] = ok
    return results


if __name__ == "__main__":
    res = test_on_the_fly_downsample_capacity()
    ok = all(res.values())
    print(f"[{'PASS' if ok else 'FAIL'}] test_on_the_fly_downsample_capacity")
    for check, val in res.items():
        print(f"    {'ok ' if val else 'FAIL'} {check}")
    print("ALL PASS" if ok else "SOME FAILED")
