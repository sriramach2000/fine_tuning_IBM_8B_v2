# ============================================================================
# Dockerfile for Granite-8B QLoRA Fine-Tuning on AWS SageMaker
# ============================================================================
# Optimized for embedded automotive code generation (AVB/TSN)
# ============================================================================

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies - CRITICAL VERSION LOCKS
RUN pip install --no-cache-dir \
    pyarrow==14.0.1 \
    "transformers>=4.45.2" \
    peft>=0.12.0 \
    trl>=0.9.6 \
    bitsandbytes>=0.43.0 \
    accelerate>=0.31.0 \
    datasets==2.14.5 \
    sentencepiece>=0.1.99 \
    protobuf>=3.20.0 \
    scipy>=1.11.0 \
    rich \
    tqdm \
    colorama \
    boto3>=1.34.0 \
    python-dotenv \
    pyyaml \
    jsonlines

# Copy training scripts
COPY training/ /opt/ml/code/

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/opt/ml/code/.cache/huggingface
ENV TRANSFORMERS_CACHE=/opt/ml/code/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

# CUDA optimizations
ENV CUDA_LAUNCH_BLOCKING=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Transformers optimizations
ENV TRANSFORMERS_NO_CUDA_SDPA=1
ENV TORCH_USE_CUDA_DSA=0

# Set entrypoint for SageMaker
ENTRYPOINT ["python", "train_granite_qlora.py"]
