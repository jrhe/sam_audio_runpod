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
ENV HF_HOME=/app/huggingface_cache
# HF_TOKEN should be set at runtime via RunPod environment variables
# This allows the model to download gated weights from Hugging Face

# Pre-download the model weights (optional - makes cold start faster but image larger)
# Uncomment the following lines if you want to bake the model into the image
# RUN python -c "from sam_audio import SAMAudio, SAMAudioProcessor; \
#     SAMAudio.from_pretrained('facebook/sam-audio-large'); \
#     SAMAudioProcessor.from_pretrained('facebook/sam-audio-large')"

# Start the handler
CMD ["python", "-u", "handler.py"]

