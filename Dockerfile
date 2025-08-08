# =============================================================================
# Root Dockerfile for Cloud Build - Future Vision Transport API
# =============================================================================

# Multi-stage build for production optimization
FROM python:3.9-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Production stage
FROM python:3.9-slim

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libhdf5-103 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app directory and user
RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app

# Copy application code from backend directory
COPY --chown=app:app backend/main_cpu_optimized.py ./
COPY --chown=app:app backend/static/ ./static/
COPY --chown=app:app backend/segmentation_test_interface.html ./

# Create models directory for GCS downloads
RUN mkdir -p models && chown app:app models

# Switch to app user
USER app

# Environment variables for Google Cloud
ENV GOOGLE_CLOUD_PROJECT=py2nb-production
ENV GCS_BUCKET=cityscapes_data2
ENV MODEL_BASE_PATH=gs://cityscapes_data2/models/tf_2_15_compatible/
ENV UNET_MINI_PATH=gs://cityscapes_data2/models/tf_2_15_compatible/unet_mini_tf_2_15_final_20250805_132446.keras
ENV VGG16_UNET_PATH=gs://cityscapes_data2/models/tf_2_15_compatible/vgg16_unet_tf_2_15_final_20250805_142633.keras

# TensorFlow CPU optimization
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_FORCE_GPU_ALLOW_GROWTH=false
ENV TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
ENV TF_DISABLE_SEGMENT_REDUCTION_OP=1
ENV NVIDIA_TF32_OVERRIDE=0

# Cloud Run optimization
ENV PORT=8000
ENV WORKERS=1
ENV TIMEOUT=300

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Start command optimized for Cloud Run
CMD exec uvicorn main_cpu_optimized:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers ${WORKERS} \
    --timeout-keep-alive ${TIMEOUT} \
    --access-log \
    --no-use-colors