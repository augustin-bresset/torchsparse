FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --upgrade pip

RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

COPY . /workspace/torchsparse
WORKDIR /workspace/torchsparse

RUN FORCE_CUDA=1 pip install -v .

CMD ["/bin/bash"]
