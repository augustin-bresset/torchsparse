"""
Tests for TorchScript support (ScriptableConv3d + SparseTensorImpl).

What is tested:
  1. make_scriptable()  → creates a SparseTensorImpl from raw tensors
  2. ScriptableConv3d forward  → correct feats shape, same result as direct call
  3. torch.jit.script()        → model is scriptable without errors
  4. torch.jit.save/load       → round-trip through a file, no Python class needed
  5. Scripted inference        → identical output to non-scripted path
  6. Submanifold (stride=1) and strided (stride=2) cases
"""

import os
import tempfile
from typing import List

import torch
import torch.nn as nn

import torchsparse
from torchsparse.script import ScriptableConv3d, make_scriptable

__all__ = ["test_torchscript_basics", "test_torchscript_save_load"]


def _make_coords_feats(n: int, c: int, device: str
                       ) -> tuple:
    torch.manual_seed(0)
    coords = torch.stack([
        torch.zeros(n, dtype=torch.int32),          # batch index
        torch.randint(0, 16, (n,), dtype=torch.int32),
        torch.randint(0, 16, (n,), dtype=torch.int32),
        torch.randint(0, 16, (n,), dtype=torch.int32),
    ], dim=1).to(device)
    feats = torch.randn(n, c, device=device)
    return coords, feats


class _TwoLayerNet(nn.Module):
    """Simple scriptable sparse conv model for testing."""
    def __init__(self, c_in: int, c_mid: int, c_out: int) -> None:
        super().__init__()
        self.conv1 = ScriptableConv3d(c_in,  c_mid, kernel_size=3, stride=1)
        self.conv2 = ScriptableConv3d(c_mid, c_out, kernel_size=2, stride=2)

    def forward(
        self,
        x: "torch.classes.torchsparse.SparseTensorImpl",
    ) -> "torch.classes.torchsparse.SparseTensorImpl":
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def test_torchscript_basics(device: str = "cuda:0") -> dict:
    """
    Tests:
      - SparseTensorImpl construction and accessors
      - ScriptableConv3d subm forward (stride=1)
      - ScriptableConv3d strided forward (stride=2)
      - torch.jit.script on a 2-layer model
    """
    results = {}

    coords, feats = _make_coords_feats(256, 8, device)

    # --- 1. SparseTensorImpl ---
    st = make_scriptable(feats, coords, [1, 1, 1])
    assert torch.equal(st.feats(), feats),  "feats accessor broken"
    assert torch.equal(st.coords(), coords), "coords accessor broken"
    assert st.stride() == [1, 1, 1],        "stride accessor broken"
    assert st.spatial_range() is None,      "spatial_range accessor broken"
    results["SparseTensorImpl_accessors"] = True

    # --- 2. Submanifold conv (stride=1) ---
    conv_subm = ScriptableConv3d(8, 16, kernel_size=3, stride=1).to(device).eval()
    with torch.no_grad():
        out_subm = conv_subm(st)
    assert out_subm.feats().shape == (256, 16), \
        f"subm out shape {out_subm.feats().shape}"
    assert torch.equal(out_subm.coords(), coords), \
        "subm must preserve coordinates"
    results["subm_forward"] = True

    # --- 3. Strided conv (stride=2) ---
    conv_stride = ScriptableConv3d(8, 16, kernel_size=2, stride=2).to(device).eval()
    with torch.no_grad():
        out_stride = conv_stride(st)
    assert out_stride.feats().shape[1] == 16, "strided out channels wrong"
    assert out_stride.feats().shape[0] <= 256, "strided output can only shrink"
    assert out_stride.stride() == [2, 2, 2], \
        f"output stride wrong: {out_stride.stride()}"
    results["strided_forward"] = True

    # --- 4. torch.jit.script ---
    net = _TwoLayerNet(8, 16, 32).to(device).eval()
    scripted = torch.jit.script(net)
    with torch.no_grad():
        out_scripted = scripted(st)
    assert out_scripted.feats().shape[1] == 32, "scripted output channels wrong"
    results["jit_script"] = True

    # --- 5. Scripted == non-scripted ---
    with torch.no_grad():
        out_eager = net(st)
    max_adiff = (out_eager.feats() - out_scripted.feats()).abs().max().item()
    assert max_adiff == 0.0, f"scripted vs eager mismatch: {max_adiff}"
    results["scripted_eq_eager"] = True

    return results


def test_torchscript_save_load(device: str = "cuda:0") -> float:
    """
    Tests torch.jit.save → torch.jit.load round-trip.
    Returns max absolute difference between saved-and-loaded output
    and the original scripted output (should be 0.0).
    """
    coords, feats = _make_coords_feats(256, 8, device)
    st = make_scriptable(feats, coords, [1, 1, 1])

    net = _TwoLayerNet(8, 16, 32).to(device).eval()
    scripted = torch.jit.script(net)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sparse_model.pt")
        torch.jit.save(scripted, path)

        # Load without any torchsparse Python class in scope
        loaded = torch.jit.load(path, map_location=device)

    with torch.no_grad():
        out_orig   = scripted(st).feats()
        out_loaded = loaded(st).feats()

    return (out_orig - out_loaded).abs().max().item()
