"""Regression tests for crashes triggered by empty / degenerate kernel maps.

Two upstream bugs surfaced on real (sparse, bs=1) point clouds and are pinned here:

1. Empty kernel map (this file's focus). When an encoder-decoder bottleneck is
   downsampled until a frame has 0 output voxels, ``out_in_map`` has shape
   ``(0, K)``. Every backend launcher computes ``grid = (work + tile - 1) / tile``
   and would launch with a 0-block grid -> ``cudaErrorInvalidValue``, which
   poisons the CUDA context for the whole process (no catch-and-continue). The
   fix guards each launcher; these tests assert no crash, correct empty shapes,
   and -- crucially -- that the CUDA context survives.

2. GPU hashmap crash (commit a6ba56a). murmur3 signed-overflow produced negative
   table indices -> illegal memory access. ``test_hashmap_large_coords`` stresses
   the hashmap with large coordinates and asserts a correct, crash-free result.

Run standalone:  python tests/python/test_empty_kmap.py
"""

import torch

import torchsparse
import torchsparse.backend
import torchsparse.backends
from torchsparse import SparseTensor
from torchsparse import nn as spnn

__all__ = [
    "test_empty_backend_launchers",
    "test_empty_conv3d_forward",
    "test_thin_input_preserved",
    "test_hashmap_large_coords",
]


def _cuda_context_alive(device) -> bool:
    """A 0-block launch poisons the context: any later op then raises. A
    successful op proves the context is still intact."""
    x = torch.ones(8, device=device)
    torch.cuda.synchronize()
    return bool((x + 1).sum().item() == 16.0)


def test_empty_backend_launchers(device="cuda:0"):
    """Each guarded backend launcher must handle a (0, K) map without launching
    a 0-block grid, returning the correctly-shaped empty/identity tensor."""
    K = 27  # 3x3x3 kernel
    results = {}
    empty_map = torch.empty((0, K), dtype=torch.int32, device=device)

    # convert_transposed_out_in_map: out tensor is pre-filled with -1 (the
    # transpose of an empty map); guard must leave it untouched.
    out_t = torch.full((128, K), -1, dtype=torch.int32, device=device)
    torchsparse.backend.convert_transposed_out_in_map(empty_map, out_t)
    torch.cuda.synchronize()
    results["convert_transposed/shape"] = tuple(out_t.shape) == (128, K)
    results["convert_transposed/all_-1"] = bool((out_t == -1).all().item())

    # derive_bitmask_from_out_in_map -> (split_mask_num, 0)
    bitmask = torchsparse.backend.derive_bitmask_from_out_in_map(empty_map, 1, 0)
    torch.cuda.synchronize()
    results["derive_bitmask/shape"] = tuple(bitmask.shape) == (1, 0)

    # reorder_out_in_map_cuda -> (0, K)
    reorder_loc = torch.zeros((1, 0), dtype=torch.int32, device=device)
    reordered = torchsparse.backend.reorder_out_in_map_cuda(empty_map, reorder_loc)
    torch.cuda.synchronize()
    results["reorder/shape"] = tuple(reordered.shape) == (0, K)

    # reduce_bitmask_cuda on an empty bitmask must not read out of bounds
    empty_mask = torch.zeros((1, 0), dtype=torch.int32, device=device)
    torchsparse.backend.reduce_bitmask_cuda(empty_mask, 128)
    torch.cuda.synchronize()
    results["reduce_bitmask/no_crash"] = True

    # conv_forward_implicit_gemm_cuda with 0 output voxels -> (0, OC)
    in_feats = torch.randn(0, 32, device=device, dtype=torch.float16)
    weight = torch.randn(K, 32, 64, device=device, dtype=torch.float16)
    out = torchsparse.backend.conv_forward_implicit_gemm_cuda(
        in_feats, weight, empty_map, 0, 64,
        torchsparse.backends.allow_tf32, torchsparse.backends.allow_fp16,
    )
    torch.cuda.synchronize()
    results["gemm/shape"] = tuple(out.shape) == (0, 64)

    # The whole point: the context must still be usable afterwards.
    results["cuda_context_alive"] = _cuda_context_alive(device)
    return results


def test_empty_conv3d_forward(device="cuda:0"):
    """End-to-end: a strided Conv3d on an empty SparseTensor must produce an
    empty output without crashing, and leave the CUDA context alive -- the
    minimal reproduction of the fully-downsampled-bottleneck failure."""
    results = {}
    coords = torch.zeros((0, 4), dtype=torch.int32, device=device)
    feats = torch.zeros((0, 16), dtype=torch.float16, device=device)
    x = SparseTensor(coords=coords, feats=feats)

    conv = spnn.Conv3d(16, 32, kernel_size=3, stride=2).to(device).half()
    y = conv(x)
    torch.cuda.synchronize()

    results["output_empty"] = y.feats.shape[0] == 0
    results["output_channels"] = y.feats.shape[1] == 32
    results["cuda_context_alive"] = _cuda_context_alive(device)
    return results


def test_thin_input_preserved(device="cuda:0"):
    """Thin / planar inputs (a single voxel, a flat ground plane) must survive
    even-kernel (ks=2) stride-2 downsampling instead of collapsing to 0 voxels.

    ks=2 shrinks each axis' extent by (ks-1) before // stride, so a thin axis'
    coords_max dropped below coords_min and the frame collapsed to empty -- then
    the on-the-fly kmap builder launched a 0-block grid (cudaErrorInvalidValue),
    and even when guarded the whole scan returned empty (data loss). The
    coords_max>=coords_min clamp preserves a thin layer; this is common in real
    LiDAR (ground/walls) so it must not vanish."""
    results = {}

    # single voxel: used to collapse to 0, now preserved
    x1 = SparseTensor(
        coords=torch.tensor([[0, 0, 0, 0]], dtype=torch.int32, device=device),
        feats=torch.randn(1, 4, device=device),
    )
    y1 = spnn.Conv3d(4, 8, kernel_size=2, stride=2).to(device)(x1)
    torch.cuda.synchronize()
    results["single_voxel_preserved"] = y1.feats.shape[0] > 0

    # flat ground plane (z constant) through 4 stride-2 downsamples (stride 16)
    g = torch.Generator().manual_seed(0)
    n = 3000
    xy = torch.randint(0, 150, (n, 2), generator=g)
    coords = torch.cat(
        [torch.zeros(n, 1, dtype=torch.int64), xy, torch.zeros(n, 1, dtype=torch.int64)],
        dim=1,
    ).int().to(device)
    x2 = SparseTensor(coords=coords, feats=torch.randn(n, 4, device=device))
    enc = torch.nn.Sequential(
        *[spnn.Conv3d(c0, c1, kernel_size=2, stride=2)
          for c0, c1 in [(4, 8), (8, 8), (8, 8), (8, 8)]]
    ).to(device)
    y2 = enc(x2)
    torch.cuda.synchronize()
    results["ground_plane_not_collapsed"] = y2.feats.shape[0] > 0
    results["cuda_context_alive"] = _cuda_context_alive(device)
    return results


def test_hashmap_large_coords(device="cuda:0"):
    """Regression for the murmur3 signed-overflow hashmap crash (a6ba56a):
    large coordinates must not produce negative table indices / illegal access.
    A downsampling conv exercises insert + lookup; we assert it completes."""
    results = {}
    g = torch.Generator(device="cpu").manual_seed(0)
    n = 4096
    # large spatial extent -> large hash keys, the regime that overflowed
    xyz = torch.randint(0, 2000, (n, 3), generator=g, dtype=torch.int32)
    batch = torch.zeros((n, 1), dtype=torch.int32)
    coords = torch.cat([batch, xyz], dim=1).to(device)
    feats = torch.randn(n, 16, generator=g).half().to(device)
    x = SparseTensor(coords=coords, feats=feats)

    conv = spnn.Conv3d(16, 16, kernel_size=2, stride=2).to(device).half()
    y = conv(x)
    torch.cuda.synchronize()

    results["produced_output"] = y.feats.shape[0] > 0
    results["finite"] = bool(torch.isfinite(y.feats.float()).all().item())
    results["cuda_context_alive"] = _cuda_context_alive(device)
    return results


if __name__ == "__main__":
    dev = "cuda:0"
    all_ok = True
    for fn in (
        test_empty_backend_launchers,
        test_empty_conv3d_forward,
        test_thin_input_preserved,
        test_hashmap_large_coords,
    ):
        res = fn(device=dev)
        ok = all(res.values())
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {fn.__name__}")
        for check, val in res.items():
            print(f"    {'ok ' if val else 'FAIL'} {check}")
    print("=" * 40)
    print("ALL PASS" if all_ok else "SOME FAILED")
