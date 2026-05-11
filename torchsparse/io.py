import json
import struct
from typing import Optional, Union

import numpy as np
import torch

from torchsparse.tensor import SparseTensor

__all__ = ['save', 'load']

_MAGIC = b'\x93TSPTS'
_VERSION_MAJOR = 1
_VERSION_MINOR = 0
_HEADER_ALIGNMENT = 64
# Bytes before the header data: magic(6) + version(2) + header_len(4)
_PREFIX_LEN = len(_MAGIC) + 2 + 4


def _torch_to_descr(dtype: torch.dtype) -> str:
    """Return a numpy dtype descriptor string (e.g. '<f4') for a torch dtype."""
    return torch.empty(0, dtype=dtype).numpy().dtype.str


def _descr_to_torch(descr: str) -> torch.dtype:
    return torch.from_numpy(np.empty(0, dtype=np.dtype(descr))).dtype


def _build_header(header_dict: dict) -> bytes:
    """Serialise header as JSON and pad so that prefix + header is 64-byte aligned."""
    raw = json.dumps(header_dict, separators=(',', ':')).encode('utf-8')
    # We want (_PREFIX_LEN + len(header_bytes)) % _HEADER_ALIGNMENT == 0
    # header_bytes ends with \n (1 byte), the rest is raw + spaces
    target_mod = (-_PREFIX_LEN) % _HEADER_ALIGNMENT
    current_len = len(raw) + 1  # +1 for the terminating \n
    padding = (target_mod - current_len % _HEADER_ALIGNMENT) % _HEADER_ALIGNMENT
    return raw + b' ' * padding + b'\n'


def save(sparse_tensor: SparseTensor, path: str) -> None:
    """Save a SparseTensor to a .pts file."""
    coords = sparse_tensor.coords.detach().cpu().contiguous()
    feats = sparse_tensor.feats.detach().cpu().contiguous()

    header_dict = {
        'stride': list(sparse_tensor.stride),
        'spatial_range': list(sparse_tensor.spatial_range) if sparse_tensor.spatial_range is not None else None,
        'coord_shape': list(coords.shape),
        'coord_dtype': _torch_to_descr(coords.dtype),
        'feat_shape': list(feats.shape),
        'feat_dtype': _torch_to_descr(feats.dtype),
    }
    header_bytes = _build_header(header_dict)

    with open(path, 'wb') as f:
        f.write(_MAGIC)
        f.write(bytes([_VERSION_MAJOR, _VERSION_MINOR]))
        f.write(struct.pack('<I', len(header_bytes)))
        f.write(header_bytes)
        f.write(coords.numpy().tobytes(order='C'))
        f.write(feats.numpy().tobytes(order='C'))


def load(path: str, device: Optional[Union[str, torch.device]] = None) -> SparseTensor:
    """Load a SparseTensor from a .pts file."""
    with open(path, 'rb') as f:
        magic = f.read(len(_MAGIC))
        if magic != _MAGIC:
            raise ValueError(f'{path!r} is not a valid .pts file (bad magic)')

        version = f.read(2)
        version_major, version_minor = version[0], version[1]
        if version_major != _VERSION_MAJOR:
            raise ValueError(
                f'Unsupported .pts version {version_major}.{version_minor} '
                f'(this build supports {_VERSION_MAJOR}.x)'
            )

        (header_len,) = struct.unpack('<I', f.read(4))
        header = json.loads(f.read(header_len).decode('utf-8'))

        coord_dtype = np.dtype(header['coord_dtype'])
        feat_dtype = np.dtype(header['feat_dtype'])
        coord_shape = header['coord_shape']
        feat_shape = header['feat_shape']

        coord_buf = f.read(int(np.prod(coord_shape)) * coord_dtype.itemsize)
        feat_buf = f.read(int(np.prod(feat_shape)) * feat_dtype.itemsize)

    coords = torch.from_numpy(
        np.frombuffer(coord_buf, dtype=coord_dtype).reshape(coord_shape).copy()
    )
    feats = torch.from_numpy(
        np.frombuffer(feat_buf, dtype=feat_dtype).reshape(feat_shape).copy()
    )

    spatial_range = tuple(header['spatial_range']) if header['spatial_range'] is not None else None

    tensor = SparseTensor(
        feats=feats,
        coords=coords,
        stride=tuple(header['stride']),
        spatial_range=spatial_range,
    )

    if device is not None:
        tensor = tensor.to(device)

    return tensor
