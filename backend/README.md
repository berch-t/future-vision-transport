# Future Vision Transport - Segmentation API

Production-ready FastAPI backend for autonomous vehicle image segmentation using Cityscapes-trained models.

## 🎯 Overview

This API serves two segmentation models:
- **UNet Mini** (1.9M parameters) - Fast inference, champion model
- **VGG16 UNet** (25.9M parameters) - Higher accuracy alternative

**Performance Targets:**
- Inference time: <100ms per image
- Input resolution: 512x1024
- Output: 8 Cityscapes classes with color mapping

## 🚀 API Variants

This repository provides **two production-ready API implementations** optimized for different deployment environments:

### 📱 `main.py` - GPU-Optimized Version
- **Target Environment**: Google Cloud Run with GPU allocation
- **TensorFlow**: 2.18.0 with GPU acceleration
- **Features**: Mixed precision, GPU memory growth, OneDNN optimizations
- **Performance**: ~45ms inference (UNet Mini), ~75ms (VGG16 UNet)
- **Best for**: Production deployments with GPU resources

### 🖥️ `main_final_cpu_optimized.py` - CPU-Safe Version
- **Target Environment**: CPU-only Cloud Run instances or local development
- **TensorFlow**: 2.18.0 with CPU-specific optimizations
- **Features**: Disabled OneDNN, CPU threading limits, memory management
- **Performance**: ~200-400ms inference (CPU-dependent)
- **Best for**: Cost-effective deployments, development, testing

## 🚀 Quick Start

### 1. GPU-Optimized Launch (Recommended for Production)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Start GPU-optimized API (automatically handles dependencies)
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Or use the predefined script
uv run start-api
```

### 2. CPU-Safe Launch (Development & Testing)

```bash
# Start CPU-optimized API for safe development
uv run uvicorn main_final_cpu_optimized:app --host 0.0.0.0 --port 8000 --reload

# Alternative with environment optimization
TF_ENABLE_ONEDNN_OPTS=0 uv run uvicorn main_final_cpu_optimized:app --host 0.0.0.0 --port 8000
```

### 3. Test the API

```bash
# Run comprehensive test suite with uv
uv run python test_production_api.py

# Or use the predefined script
uv run test-api

# Simple test runner
uv run run-tests
```

### 4. Alternative: Traditional Setup

```bash
# Create virtual environment (if not using uv)
conda create -n fastapi_segmentation python=3.9
conda activate fastapi_segmentation

# Install requirements
pip install -r requirements.txt

# Start API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

### Core Endpoints
- `GET /` - API information and status
- `GET /health` - Health check and model status
- `GET /models` - Available model information
- `GET /classes` - Cityscapes class definitions

### Prediction Endpoints
- `POST /predict` - JSON response with segmentation metadata
- `POST /predict/mask` - Returns color segmentation mask as image
- `POST /predict/overlay` - Returns original + mask overlay

### Documentation
- `GET /docs` - Swagger UI documentation
- `GET /openapi.json` - OpenAPI schema

## 🧪 Testing

The comprehensive test suite validates:

### ✅ **Endpoint Testing**
- All API endpoints functional
- Proper HTTP status codes
- Response schema validation
- Error handling

### ✅ **Model Testing** 
- Both UNet Mini and VGG16 UNet models
- Real Cityscapes image processing
- Inference time benchmarking
- Confidence score validation

### ✅ **Performance Testing**
- Sequential request performance
- Concurrent load testing (10 simultaneous requests)
- Memory usage monitoring
- Response time statistics

### ✅ **Production Readiness**
- CORS configuration for Next.js frontend
- Error handling and validation
- API documentation availability
- System resource monitoring

### Test Results
```bash
# Example successful test output
📊 Endpoint Success Rate: 100.0%
🤖 Model Success Rate: 100.0%
⚡ Performance Acceptable: True
🏭 Production Ready: True
✅ ALL TESTS PASSED - API IS PRODUCTION READY!
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Enable GPU memory growth
TF_FORCE_GPU_ALLOW_GROWTH=true

# Optional: Reduce TensorFlow logging
TF_CPP_MIN_LOG_LEVEL=2
```

### Model Files
Models are automatically loaded from `models/` directory:
- `models_milestone_training_unet_mini_milestone_20250719_124201.h5`
- `models_milestone_training_vgg16_unet_milestone_20250719_132046.h5`

## 🖼️ Usage Examples

### Python Client
```python
import httpx

# Upload image for prediction
with open("test_image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    response = httpx.post("http://localhost:8000/predict", files=files)
    result = response.json()
    
print(f"Model: {result['model_name']}")
print(f"Inference time: {result['inference_time_ms']}ms")
print(f"Confidence: {result['overall_confidence']}")
```

### cURL
```bash
# Get API health
curl http://localhost:8000/health

# Predict segmentation
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"

# Get color mask
curl -X POST "http://localhost:8000/predict/mask" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg" \
  --output mask.png
```

## 🏗️ Architecture

### Model Management
- Automatic loading of both models at startup
- GPU memory growth configuration
- Fallback error handling
- Performance benchmarking

### Image Processing Pipeline
1. **Input validation** - File type and size checks
2. **Preprocessing** - Resize to 512x1024, ImageNet normalization
3. **Inference** - Model prediction with timing
4. **Postprocessing** - Argmax, color mapping, statistics

### Response Formats
- **JSON metadata** - Inference time, confidence, class statistics
- **Image responses** - PNG color masks and overlays
- **Error handling** - Proper HTTP status codes and messages

## 🌐 Next.js Integration

The API is designed for seamless integration with Next.js frontend:

```typescript
// TypeScript interface for API response
interface SegmentationResponse {
  model_name: string;
  inference_time_ms: number;
  overall_confidence: number;
  class_statistics: Record<string, {
    pixel_count: number;
    percentage: number;
    avg_confidence: number;
  }>;
  timestamp: string;
}

// React component usage
const uploadImage = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('/api/predict', {
    method: 'POST',
    body: formData,
  });
  
  const result: SegmentationResponse = await response.json();
  return result;
};
```

## 🐳 Docker & Cloud Run Deployment

### Google Cloud Run - GPU Version (Production)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

# GPU-optimized API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Cloud Run Deployment:**
```bash
# Build and deploy GPU-optimized version
gcloud run deploy future-vision-gpu \
  --source . \
  --platform managed \
  --region europe-west1 \
  --cpu 4 \
  --memory 8Gi \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --allow-unauthenticated \
  --max-instances 10
```

### Google Cloud Run - CPU Version (Cost-Effective)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# CPU optimization environment variables
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_DISABLE_SEGMENT_REDUCTION_OP=1
ENV CUDA_VISIBLE_DEVICES=""

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

# CPU-optimized API
CMD ["uvicorn", "main_final_cpu_optimized:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Cloud Run Deployment:**
```bash
# Build and deploy CPU-optimized version  
gcloud run deploy future-vision-cpu \
  --source . \
  --platform managed \
  --region europe-west1 \
  --cpu 4 \
  --memory 8Gi \
  --allow-unauthenticated \
  --max-instances 20
```

### Deployment Comparison

| Feature | GPU Version (`main.py`) | CPU Version (`main_final_cpu_optimized.py`) |
|---------|-------------------------|---------------------------------------------|
| **Cost** | ~$0.50/hour | ~$0.10/hour |
| **Performance** | 45-75ms | 200-400ms |
| **Concurrency** | 10 instances | 20 instances |
| **Reliability** | High GPU dependency | Very stable |
| **Best For** | Production traffic | Development, testing, budget-conscious |

## 📊 Performance Metrics

### GPU Performance (`main.py`)
**RTX 4080 / Cloud Run GPU:**
- UNet Mini: ~45ms inference time
- VGG16 UNet: ~75ms inference time  
- Concurrent throughput: ~15-20 requests/second
- Memory usage: ~2GB for both models loaded

### CPU Performance (`main_final_cpu_optimized.py`)
**Intel i9-14900K / Cloud Run CPU (4 vCPU):**
- UNet Mini: ~200-300ms inference time
- VGG16 UNet: ~350-450ms inference time
- Concurrent throughput: ~5-8 requests/second
- Memory usage: ~1.5GB for both models loaded

### Environment-Specific Optimizations

**GPU Version Features:**
- Mixed precision training for faster inference
- GPU memory growth to prevent allocation issues
- OneDNN optimizations enabled
- CUDA-specific TensorFlow operations

**CPU Version Features:**
- OneDNN optimizations disabled (prevents segfaults)
- CPU thread limiting for stability
- Segment reduction operations disabled
- Fallback error handling for edge cases

## 🔍 Monitoring

The API includes built-in monitoring:
- Request/response timing
- Model performance tracking
- System resource usage
- Error rate monitoring

## 📝 Development

### Choosing the Right API Version

**Use `main.py` when:**
- Deploying to Cloud Run with GPU allocation
- Performance is critical (<100ms target)
- Production environment with sufficient budget
- GPU resources are reliably available

**Use `main_final_cpu_optimized.py` when:**
- Development and testing environments
- Cost optimization is a priority
- GPU resources are unavailable or unreliable
- CPU-only Cloud Run deployment
- Local development on machines without GPU

### Adding New Models
1. Place `.h5` model file in `models/` directory
2. Update `model_files` dict in `SegmentationModelManager` (both API versions)
3. Test with the production test suite on both versions

### Custom Preprocessing
Modify `preprocess_image()` method in `SegmentationModelManager` class for custom preprocessing pipelines.

### Environment Variables

**GPU Version (`main.py`):**
```bash
TF_FORCE_GPU_ALLOW_GROWTH=true
TF_CPP_MIN_LOG_LEVEL=2
```

**CPU Version (`main_final_cpu_optimized.py`):**
```bash
TF_ENABLE_ONEDNN_OPTS=0
TF_DISABLE_SEGMENT_REDUCTION_OP=1
CUDA_VISIBLE_DEVICES=""
TF_CPP_MIN_LOG_LEVEL=2
```

---

## 🤝 Contributing

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Ensure production test suite passes
4. Update documentation

## 📄 License

Part of the Future Vision Transport project for autonomous vehicle computer vision.