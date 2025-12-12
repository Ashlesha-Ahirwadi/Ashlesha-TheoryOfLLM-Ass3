# Dockerfile (cleaned)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    python3-distutils \
    build-essential \
    ca-certificates \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Ensure 'python' and 'pip' commands are available and point to python3
RUN ln -sfn /usr/bin/python3 /usr/local/bin/python \
 && ln -sfn /usr/bin/pip3 /usr/local/bin/pip \
 && python --version \
 && pip --version

WORKDIR /workspace

# Upgrade pip (using python -m pip to be explicit)
RUN python -m pip install --upgrade pip

# Install PyTorch matching CUDA 11.8
RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core Python dependencies (pin versions as needed)
RUN python -m pip install --no-cache-dir \
    transformers==4.36.0 \
    datasets==2.16.0 \
    accelerate==0.25.0 \
    peft==0.7.1 \
    trl==0.7.10 \
    bitsandbytes==0.41.3 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    pandas==2.1.4 \
    numpy==1.24.3 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    tqdm==4.66.1 \
    wandb==0.16.1 \
    tensorboard==2.15.1

# Copy project files
COPY . /workspace/

# Initialize PYTHONPATH properly (avoid referencing an undefined variable)
ENV PYTHONPATH=/workspace

# Default command
CMD ["/bin/bash"]
