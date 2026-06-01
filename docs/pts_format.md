# TorchSparse `.pts` Binary Format

The `.pts` format stores a `SparseTensor` as a self-describing binary file. It is inspired by NumPy's [`.npy` format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html): a compact fixed header followed by raw tensor data. The format does **not** use pickle, so it can be loaded without any model class definition -- only TorchSparse is required.

## File Layout

```
Offset        Size        Field
─────────────────────────────────────────────────────────────
0             6           Magic bytes: \x93TSPTS
6             1           Version major (uint8)   -- currently 1
7             1           Version minor (uint8)   -- currently 0
8             4           Header length H (uint32, little-endian)
12            H           JSON header (UTF-8, space-padded, ends with \n)
12 + H        C           Coords data  (C-contiguous raw bytes)
12 + H + C    F           Feats data   (C-contiguous raw bytes)
─────────────────────────────────────────────────────────────
```

The header is padded with spaces so that `12 + H` is a multiple of **64 bytes**. This aligns tensor data to a 64-byte boundary, which is a requirement for safe memory-mapped access.

## JSON Header Fields

| Field | Type | Description |
|---|---|---|
| `stride` | `[int, int, int]` | Spatial stride of the tensor |
| `spatial_range` | `[int, ...]` or `null` | Spatial extent used by `.dense()`, or `null` |
| `coord_shape` | `[N, D]` | Shape of the coordinate tensor |
| `coord_dtype` | string | NumPy dtype descriptor, e.g. `"<i4"` (int32 LE) |
| `feat_shape` | `[N, C]` | Shape of the feature tensor |
| `feat_dtype` | string | NumPy dtype descriptor, e.g. `"<f4"` (float32 LE) |

Dtype descriptors follow the [NumPy array protocol](https://numpy.org/doc/stable/reference/arrays.interface.html) (`<` = little-endian, `i4` = int32, `f4` = float32, `f2` = float16, etc.).

### Example header

```json
{"stride":[1,2,4],"spatial_range":[2,32,32,32],"coord_shape":[1000,4],"coord_dtype":"<i4","feat_shape":[1000,32],"feat_dtype":"<f4"}
```

## Data Sections

Both tensors are stored in **C-contiguous order** (row-major) with no padding between them.

- **Coords** -- integer tensor of shape `[N, D]`, typically `int32`.  
  For 3D point clouds with a batch dimension: `D = 4` (`[batch, x, y, z]`).
- **Feats** -- floating-point tensor of shape `[N, C]`.

The byte length of each section can be computed from the header:

```python
import numpy as np
coord_bytes = int(np.prod(header['coord_shape'])) * np.dtype(header['coord_dtype']).itemsize
feat_bytes  = int(np.prod(header['feat_shape']))  * np.dtype(header['feat_dtype']).itemsize
```

## Versioning

The major version byte is bumped on any backward-incompatible change. A reader should reject files whose major version does not match its own. The minor version may be incremented for backward-compatible additions (new optional header fields, etc.) and can be ignored by older readers.

## Reading Without TorchSparse

The format is simple enough to read with NumPy alone:

```python
import json, struct, numpy as np

MAGIC = b'\x93TSPTS'

with open('pointcloud.pts', 'rb') as f:
    assert f.read(6) == MAGIC, "not a .pts file"
    major, minor = f.read(1)[0], f.read(1)[0]
    (hlen,) = struct.unpack('<I', f.read(4))
    header = json.loads(f.read(hlen))

    coords = np.frombuffer(f.read(
        int(np.prod(header['coord_shape'])) * np.dtype(header['coord_dtype']).itemsize
    ), dtype=header['coord_dtype']).reshape(header['coord_shape']).copy()

    feats = np.frombuffer(f.read(
        int(np.prod(header['feat_shape'])) * np.dtype(header['feat_dtype']).itemsize
    ), dtype=header['feat_dtype']).reshape(header['feat_shape']).copy()
```
