# SAM3 Segmentation Container
# Requires Python 3.12+, PyTorch 2.7+, CUDA 12.6+
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3-pip \
        python3.12-venv \
        python3.12-dev \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
        libavutil-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libffi-dev \
        sudo \
        unzip \
        wget \
        && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Create working directory
WORKDIR /workspace/sam3

# Upgrade pip and install base dependencies
RUN python3 -m pip install --upgrade --ignore-installed pip setuptools wheel --break-system-packages && \
    pip install --no-cache-dir --break-system-packages numpy cython

# Copy setup files first for better layer caching
COPY pyproject.toml /workspace/sam3/pyproject.toml
COPY README.md /workspace/sam3/README.md

# Install PyTorch 2.7 with CUDA 12.6 support (SAM3 requirement)
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install additional dependencies for video processing
RUN pip install --no-cache-dir --break-system-packages \
    opencv-python \
    opencv-python-headless \
    tqdm \
    Pillow \
    gcsfs \
    google-cloud-storage \
    flytekit>=1.13.0 \
    "transformers>=4.46.0" \
    "accelerate>=0.30.0" \
    safetensors \
    sentencepiece \
    psutil

# Copy HLX workflow wheel (reuse from project)
COPY hlx_wf-2.4.2.dev9-py3-none-any.whl /workspace/sam3/

# Install the HLX workflow wheel
RUN python3 -m pip install --break-system-packages /workspace/sam3/hlx_wf-2.4.2.dev9-py3-none-any.whl

# Copy entire project
COPY . /workspace/sam3

# Install SAM3 in editable mode
RUN pip install --break-system-packages -e . --no-build-isolation

# Default command
CMD ["/bin/bash"]
