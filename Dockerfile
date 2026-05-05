FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.6 8.9 9.0 12.0"

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    ninja-build \
    cmake \
    libopenblas-dev \
    libsparsehash-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --upgrade pip

RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

COPY . /workspace/torchsparse
WORKDIR /workspace/torchsparse

RUN FORCE_CUDA=1 pip install --no-build-isolation -v .

CMD ["/bin/bash"]
