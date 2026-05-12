import glob
import os

import torch
import torch.cuda
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

# from torchsparse import __version__

version_file = open("./torchsparse/version.py")
version = version_file.read().split("'")[1]
print("torchsparse version:", version)

if (torch.cuda.is_available() and CUDA_HOME is not None) or (
    os.getenv("FORCE_CUDA", "0") == "1"
):
    device = "cuda"
    pybind_fn = f"pybind_{device}.cu"
else:
    device = "cpu"
    pybind_fn = f"pybind_{device}.cpp"

sources = [os.path.join("torchsparse", "backend", pybind_fn)]
for fpath in glob.glob(os.path.join("torchsparse", "backend", "**", "*")):
    if (fpath.endswith("_cpu.cpp") and device in ["cpu", "cuda"]) or (
        fpath.endswith("_cuda.cu") and device == "cuda"
    ):
        sources.append(fpath)

if device == "cuda":
    sources.append(os.path.join("torchsparse", "backend", "torch_library.cu"))

extension_type = CUDAExtension if device == "cuda" else CppExtension
extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp"],
    "nvcc": [
        "-O3", "-std=c++17",
        "-gencode=arch=compute_86,code=sm_86",   # Ampere consumer (RTX 3090/3080)
        "-gencode=arch=compute_89,code=sm_89",   # Ada Lovelace (RTX 4090 Ti / RTX 4090)
        "-gencode=arch=compute_90,code=sm_90",   # Hopper (H100/H200)
        "-gencode=arch=compute_120,code=sm_120", # Blackwell (B100/B200)
    ],
}

setup(
    name="torchsparse",
    version=version,
    packages=find_packages(),
    ext_modules=[
        extension_type(
            "torchsparse.backend", sources, extra_compile_args=extra_compile_args
        )
    ],
    url="https://github.com/mit-han-lab/torchsparse",
    install_requires=[
        "numpy",
        "backports.cached_property",
        "tqdm",
        "typing-extensions",
        "wheel",
        "rootpath",
        "torch",
        "torchvision"
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cu128'
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    
)
