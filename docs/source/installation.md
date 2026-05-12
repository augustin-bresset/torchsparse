# Installation

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.8 |
| PyTorch | ≥ 1.13 |
| CUDA toolkit | ≥ 11.3 (matching PyTorch's CUDA version) |
| numpy | any recent |

TorchSparse compiles CUDA extensions at install time. Make sure `nvcc` is on your
`PATH` and that `nvcc --version` matches `python -c "import torch; print(torch.version.cuda)"`.

## From source

```bash
git clone https://github.com/mit-han-lab/torchsparse.git
cd torchsparse
pip install -r requirements.txt
pip install -e .
```

For a non-editable install:

```bash
pip install .
```

If compilation is slow, limit the number of parallel jobs to avoid running out of memory:

```bash
MAX_JOBS=4 pip install .
```

To force CUDA compilation even when no GPU is available at build time (e.g. in a CI
container or during a Docker image build):

```bash
FORCE_CUDA=1 pip install .
```

To target specific GPU architectures:

```bash
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" pip install .
```

Use [this chart](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
to find the compute capability of your GPU.

## Docker

A ready-to-use image is available:

```bash
docker pull blopausore/torchsparse:1.0
docker run --rm --gpus all blopausore/torchsparse:1.0 python3 -c "import torchsparse; print(torchsparse.__version__)"
```

## Verifying the installation

```python
import torch
import torchsparse
from torchsparse import SparseTensor

print(torchsparse.__version__)

coords = torch.randint(0, 16, (128, 4), dtype=torch.int32).cuda()
coords[:, 0] = 0          # batch index
feats  = torch.randn(128, 16).cuda()
st = SparseTensor(feats=feats, coords=coords)
print("SparseTensor OK:", st.feats.shape)
```

## Troubleshooting

See `docs/FAQ.md` for common issues (CUDA version mismatch, out-of-memory during
compilation, cross-compilation for a different GPU, etc.).
