# RunPod Serverless Dockerfile for SAM Audio
# Based on PyTorch with CUDA support

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install torchaudio to match PyTorch version in base image
RUN pip install --no-cache-dir torchaudio

# Install core dependencies
RUN pip install --no-cache-dir \
    runpod \
    transformers \
    scipy \
    soundfile \
    requests \
    einops \
    sentencepiece \
    protobuf \
    timm \
    xformers \
    torchcodec \
    torchdiffeq \
    descript-audiotools

# Install additional dependencies needed by Facebook packages first
RUN pip install --no-cache-dir \
    iopath \
    decord \
    ftfy \
    regex \
    inflect \
    unidecode \
    pytorch-lightning

# Install Facebook Research dependencies one by one
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f

RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/dacvae.git

RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/ImageBind.git

RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/perception_models.git@unpin-deps

RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/sam-audio.git

# Copy handler code
COPY handler.py /app/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Model loading options (in order of priority):
# 1. RunPod Cached Models: Set endpoint "Cached Models" to the HF model URL
#    (e.g., https://huggingface.co/facebook/sam-audio-large)
#    The handler auto-detects and uses /runpod-volume/huggingface-cache/hub
# 2. HF_TOKEN: Set at runtime for downloading gated models on-demand

# Start the handler
CMD ["python", "-u", "handler.py"]

