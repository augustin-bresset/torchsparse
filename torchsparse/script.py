"""
Scriptable sparse convolution for TorchScript export.

Usage:
    model = MySparseModel()          # uses ScriptableConv3d
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, "model.pt")
    loaded = torch.jit.load("model.pt")

The scriptable path uses GatherScatter in inference mode (no TensorCache).
It is slower than the normal path for repeated inference on the same input
shape, but produces identical numerical results.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchsparse.backend  # triggers TORCH_LIBRARY → registers SparseTensorImpl + ops

__all__ = ["ScriptableConv3d", "make_scriptable"]


def make_scriptable(feats: torch.Tensor,
                    coords: torch.Tensor,
                    stride: List[int],
                    spatial_range: Optional[List[int]] = None
                    ) -> "torch.classes.torchsparse.SparseTensorImpl":
    """Wrap raw tensors into a ScriptableSparseTensor."""
    return torch.classes.torchsparse.SparseTensorImpl(
        feats, coords, stride, spatial_range
    )


# ─────────────────────────────────────────────────────────────────────────────
# ScriptableConv3d
# ─────────────────────────────────────────────────────────────────────────────
class ScriptableConv3d(nn.Module):
    """
    Drop-in replacement for spnn.Conv3d that supports torch.jit.script.

    Restrictions vs the full Conv3d:
      - Inference only (no autograd through the sparse op)
      - GatherScatter dataflow (no ImplicitGEMM / FetchOnDemand)
      - No dilation
      - No transposed / generative mode
      - No TensorCache (kmap recomputed every call)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels: int  = in_channels
        self.out_channels: int = out_channels

        ks = kernel_size
        st = stride
        self.kernel_size: List[int] = [ks, ks, ks]
        self.stride: List[int]      = [st, st, st]

        # Auto-padding like spnn.Conv3d: (k-1)//2 for odd k with stride=1
        if ks % 2 == 1 and st == 1:
            pad = (ks - 1) // 2
        else:
            pad = padding
        self.padding: List[int] = [pad, pad, pad]

        self.subm: bool = (st == 1)

        kernel_vol = ks * ks * ks
        if kernel_vol > 1 or st != 1:
            self.weight = nn.Parameter(
                torch.zeros(kernel_vol, in_channels, out_channels)
            )
        else:
            self.weight = nn.Parameter(torch.zeros(in_channels, out_channels))

        # TorchScript requires Optional[Tensor], not Optional[nn.Parameter]
        if bias:
            self.bias: Optional[torch.Tensor] = nn.Parameter(
                torch.zeros(out_channels)
            )
        else:
            self.bias: Optional[torch.Tensor] = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.in_channels * int(self.weight.shape[0])
                               if self.weight.dim() == 3
                               else self.in_channels)
        nn.init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -std, std)

    def forward(
        self,
        x: "torch.classes.torchsparse.SparseTensorImpl",
    ) -> "torch.classes.torchsparse.SparseTensorImpl":

        feats  = x.feats()
        coords = x.coords()
        in_stride: List[int] = x.stride()

        # 1x1x1 conv with stride=1: pure linear, no kmap needed
        w = self.weight
        if self.kernel_size == [1, 1, 1] and self.stride == [1, 1, 1]:
            out_feats = feats.matmul(w)
            if self.bias is not None:
                out_feats = out_feats + self.bias
            return torch.classes.torchsparse.SparseTensorImpl(
                out_feats, coords, in_stride, x.spatial_range()
            )

        out_feats, out_coords = torch.ops.torchsparse.scatter_conv_forward(
            feats, coords, w,
            self.kernel_size, self.stride, self.padding,
            self.subm
        )

        if self.bias is not None:
            out_feats = out_feats + self.bias

        out_stride = [in_stride[i] * self.stride[i] for i in range(3)]
        return torch.classes.torchsparse.SparseTensorImpl(
            out_feats, out_coords, out_stride, None
        )
