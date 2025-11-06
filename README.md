# 🎤 Emotion Voice Recognition System

<div align="center">

![Version](https://img.shields.io/badge/version-4.0.0--tier0-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14.2.25-000000?logo=next.js)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-green)

**A production-ready AI system for real-time emotion detection from voice recordings**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API Documentation](#-api-documentation) • [Architecture](#-architecture)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Model Information](#-model-information)
- [Development](#-development)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

The **Emotion Voice Recognition System** is a cutting-edge AI-powered application that analyzes audio recordings to detect and classify human emotions in real-time. Built with a FastAPI backend and Next.js frontend, this production-ready system leverages deep learning ensemble models to achieve **92% accuracy** in emotion classification.

### Supported Emotions

The system can detect **7 distinct emotions**:

- 😠 **Angry** - Irritation, frustration, rage
- 🤢 **Disgust** - Aversion, repulsion
- 😨 **Fear** - Anxiety, worry, terror
- 😊 **Happy** - Joy, contentment, excitement
- 😐 **Neutral** - Calm, balanced, indifferent
- 😢 **Sad** - Sorrow, grief, melancholy
- 😲 **Surprise** - Astonishment, shock, amazement

---

## ✨ Features

### 🎯 Core Capabilities

- **Real-time Emotion Detection** - Instant analysis of voice recordings
- **Multi-format Audio Support** - WAV, MP3, FLAC, OGG, WebM
- **High Accuracy** - 92% validation accuracy using ensemble learning
- **GPU Acceleration** - CUDA support for faster inference
- **Production-Ready** - Rate limiting, authentication, comprehensive logging

### 🔧 Backend Features (FastAPI)

- ⚡ **High-Performance API** - Async FastAPI with Uvicorn
- 🔒 **Security** - Bearer token authentication, rate limiting
- 📊 **Comprehensive Monitoring** - System metrics, health checks, detailed logging
- 🎛️ **Advanced Audio Processing** - Noise reduction, format conversion, feature extraction
- 🧠 **Ensemble Model** - EfficientNetV2-M + VGG16 + DenseNet121
- 📝 **Structured Logging** - JSON logs with request tracing
- 🐳 **Docker Support** - Containerized deployment ready

### 🎨 Frontend Features (Next.js)

- 🎤 **Voice Recording** - Browser-based audio recording
- 📤 **File Upload** - Drag-and-drop audio file upload
- 📈 **Real-time Visualization** - Emotion probability charts
- 🌓 **Dark Mode** - Modern, responsive UI with theme support
- 📱 **Mobile Responsive** - Optimized for all screen sizes
- ⚛️ **React 19** - Latest React features with TypeScript

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                       │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │ Voice Recorder│  │ File Uploader│  │ Results Display │ │
│  └───────────────┘  └──────────────┘  └─────────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP REST API
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  API Layer (main.py)                   │  │
│  │  CORS • Rate Limiting • Authentication • Error Handling│  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          ▼                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  Inference   │  │  Monitoring  │  │  Admin/Model     │ │
│  │   Router     │  │    Router    │  │   Management     │ │
│  └──────┬───────┘  └──────────────┘  └──────────────────┘ │
│         ▼                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Audio Processing Pipeline                  │   │
│  │  Format Conversion → Noise Reduction → Resampling    │   │
│  │         → Feature Extraction (Mel Spectrogram)       │   │
│  └────────────────────────┬─────────────────────────────┘   │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          TorchScript Ensemble Model                   │   │
│  │  EfficientNetV2-M + VGG16 + DenseNet121 (92% Acc)   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Flow

1. **User Input** - Records voice or uploads audio file via frontend
2. **API Request** - Frontend sends audio to `/api/predict` endpoint
3. **Authentication** - Backend validates bearer token (if enabled)
4. **Audio Processing** - Converts format, reduces noise, extracts features
5. **Model Inference** - Ensemble model predicts emotion probabilities
6. **Response** - Returns emotion classification with confidence scores
7. **Logging** - Logs request details, metrics, and traces

---

## 🛠️ Technology Stack

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.9+ | Core language |
| **FastAPI** | 0.104.1 | Web framework |
| **PyTorch** | 2.0+ | Deep learning framework |
| **Librosa** | 0.10.1 | Audio feature extraction |
| **Pydub** | 0.25.1 | Audio format conversion |
| **FFmpeg** | Latest | Audio codec support |
| **Uvicorn** | 0.24.0 | ASGI server |
| **Pydantic** | 2.0+ | Data validation |

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Next.js** | 14.2.25 | React framework |
| **React** | 19 | UI library |
| **TypeScript** | 5+ | Type safety |
| **Tailwind CSS** | 3.4.17 | Styling |
| **Radix UI** | Latest | UI components |
| **Axios** | 1.13.2 | HTTP client |
| **Framer Motion** | Latest | Animations |

### Machine Learning

- **Model Architecture**: Ensemble of 3 CNNs
  - EfficientNetV2-M (Primary)
  - VGG16 (Supporting)
  - DenseNet121 (Supporting)
- **Framework**: PyTorch with TorchScript optimization
- **Input**: Mel spectrogram (128x128)
- **Output**: 7-class emotion probabilities

---

## 📦 Prerequisites

### System Requirements

- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space
- **GPU** (Optional): NVIDIA GPU with CUDA 11.8+ for acceleration

### Software Dependencies

1. **Python 3.9+**
   ```bash
   python --version  # Should be 3.9 or higher
   ```

2. **Node.js 18+** and **npm**
   ```bash
   node --version  # Should be 18 or higher
   npm --version
   ```

3. **FFmpeg** (Required for audio processing)
   - **Windows**: Run the included installer
     ```powershell
     .\backend\install_ffmpeg.ps1
     ```
   - **macOS**:
     ```bash
     brew install ffmpeg
     ```
   - **Linux**:
     ```bash
     sudo apt update && sudo apt install ffmpeg
     ```

4. **Git**
   ```bash
   git --version
   ```

---

## 🚀 Installation

### Option 1: Quick Start (Recommended)

This method automatically sets up both backend and frontend:

#### Windows

```powershell
# 1. Clone the repository
git clone https://github.com/yourusername/emotion-voice.git
cd emotion-voice

# 2. Install FFmpeg (if not already installed)
.\backend\install_ffmpeg.ps1

# 3. Set up backend virtual environment
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
cd ..

# 4. Set up frontend dependencies
cd frontend\Voice-Emotion-Detector
npm install
cd ..\..

# 5. Start both servers
python start_all.py
```

#### macOS / Linux

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/emotion-voice.git
cd emotion-voice

# 2. Install FFmpeg
brew install ffmpeg  # macOS
# OR
sudo apt install ffmpeg  # Linux

# 3. Set up backend virtual environment
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..

# 4. Set up frontend dependencies
cd frontend/Voice-Emotion-Detector
npm install
cd ../..

# 5. Start both servers
python3 start_all.py
```

### Option 2: Manual Setup

#### Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

#### Frontend Setup

```bash
cd frontend/Voice-Emotion-Detector

# Install dependencies
npm install

# Start development server
npm run dev
```

### Option 3: Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run separately
docker build -f backend/Dockerfile.fastapi -t emotion-api .
docker run -p 8000:8000 emotion-api
```

---

## 💻 Usage

### Starting the Application

**Using the unified launcher (Recommended):**

```bash
# Windows
python start_all.py

# macOS/Linux
python3 start_all.py
```

This will start:
- **Backend API**: http://127.0.0.1:8000
- **Frontend UI**: http://localhost:3000

### Accessing the Application

1. **Web Interface**: Open http://localhost:3000 in your browser
2. **API Documentation**: Visit http://127.0.0.1:8000/docs for interactive API docs
3. **Health Check**: http://127.0.0.1:8000/health

### Using the Web Interface

1. **Record Audio**:
   - Click the "Record" button
   - Speak into your microphone
   - Click "Stop" when finished
   - Click "Analyze" to get emotion prediction

2. **Upload Audio File**:
   - Click "Upload File"
   - Select an audio file (WAV, MP3, FLAC, OGG, WebM)
   - View the emotion analysis results

3. **View Results**:
   - See the predicted emotion with confidence score
   - View probability distribution across all 7 emotions
   - Check processing time and metrics

### Using the API

**cURL Example:**

```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio_sample.wav"
```

**Python Example:**

```python
import requests

url = "http://127.0.0.1:8000/api/predict"
files = {"file": open("audio_sample.wav", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

**JavaScript Example:**

```javascript
const formData = new FormData();
formData.append('file', audioBlob, 'recording.webm');

const response = await fetch('http://127.0.0.1:8000/api/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

---

## 📚 API Documentation

### Endpoints

#### Inference Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Predict emotion from audio file |
| `/api/predict/streaming` | POST | Stream audio for real-time prediction |

#### Monitoring Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Simple health check |
| `/api/health/detailed` | GET | Detailed health and system info |
| `/api/status` | GET | Server status and uptime |
| `/api/metrics` | GET | Performance metrics |

#### Admin Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/model/info` | GET | Model information and metadata |
| `/api/model/reload` | POST | Reload the model |

### Request/Response Examples

#### Predict Emotion

**Request:**
```http
POST /api/predict HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: multipart/form-data

file: [audio_file.wav]
```

**Response:**
```json
{
  "status": "success",
  "emotion": "happy",
  "confidence": 0.92,
  "probabilities": {
    "angry": 0.02,
    "disgust": 0.01,
    "fear": 0.03,
    "happy": 0.92,
    "neutral": 0.01,
    "sad": 0.005,
    "surprise": 0.005
  },
  "processing_time_ms": 145.3,
  "metadata": {
    "model_version": "4.0.0-tier0",
    "audio_duration_sec": 3.5,
    "sample_rate": 16000
  },
  "timestamp": "2025-11-06T10:30:45.123Z"
}
```

#### Health Check

**Request:**
```http
GET /api/health/detailed HTTP/1.1
Host: 127.0.0.1:8000
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-06T10:30:45.123Z",
  "version": "4.0.0-tier0",
  "model_loaded": true,
  "system": {
    "cpu_percent": 15.3,
    "memory_percent": 45.2,
    "disk_percent": 62.1
  },
  "uptime_seconds": 3600
}
```

---

## 📁 Project Structure

```
emotion-voice/
│
├── backend/                          # FastAPI Backend
│   ├── config/
│   │   └── training_config.json      # Model training configuration
│   ├── logs/                         # Application logs
│   │   ├── inference_trace/          # Inference request traces
│   │   ├── predictions/              # Prediction logs
│   │   └── system/                   # System event logs
│   ├── models/
│   │   └── schemas.py                # Pydantic data models
│   ├── routers/
│   │   ├── inference.py              # Emotion prediction endpoints
│   │   ├── monitor.py                # Health & metrics endpoints
│   │   └── admin.py                  # Admin & management endpoints
│   ├── utils/
│   │   ├── audio_preprocess.py       # Audio processing pipeline
│   │   ├── model_loader.py           # Model loading & caching
│   │   ├── inference_tracer.py       # Request tracing & logging
│   │   ├── logging_utils.py          # Structured logging
│   │   ├── metrics.py                # Performance metrics
│   │   └── security.py               # Authentication & rate limiting
│   ├── venv/                         # Python virtual environment
│   ├── main.py                       # FastAPI application entry point
│   ├── config.py                     # Configuration settings
│   ├── requirements.txt              # Python dependencies
│   ├── Dockerfile                    # Docker container definition
│   └── install_ffmpeg.ps1           # FFmpeg installer script
│
├── frontend/                         # Next.js Frontend
│   └── Voice-Emotion-Detector/
│       ├── app/
│       │   ├── page.tsx              # Main page component
│       │   ├── layout.tsx            # Root layout
│       │   └── globals.css           # Global styles
│       ├── components/               # React components
│       │   ├── ui/                   # Reusable UI components
│       │   ├── EmotionRecorder.tsx   # Voice recording component
│       │   ├── FileUploader.tsx      # File upload component
│       │   └── ResultsDisplay.tsx    # Results visualization
│       ├── lib/
│       │   └── hooks/
│       │       └── useEmotionAPI.ts  # API integration hook
│       ├── public/                   # Static assets
│       ├── styles/                   # CSS modules
│       ├── package.json              # Node.js dependencies
│       ├── tsconfig.json             # TypeScript configuration
│       └── next.config.mjs           # Next.js configuration
│
├── phase3_outputs/                   # Model training outputs
│   ├── final_export/
│   │   └── best_efficientnet_v2_m+vgg16+densenet121.pt  # Ensemble model
│   ├── best_model_info.json          # Model metadata
│   └── *.png                         # Confusion matrices
│
├── training_outputs/                 # Individual model checkpoints
│   ├── best_efficientnet_v2_m.pt
│   ├── best_vgg16.pt
│   └── best_densenet121.pt
│
├── docker-compose.yml                # Docker Compose configuration
├── start_all.py                      # Unified server launcher
├── START_ALL.bat                     # Windows batch launcher
├── requirements.txt                  # Root dependencies
└── README.md                         # This file
```

---

## 🧠 Model Information

### Ensemble Architecture

The system uses a **3-model ensemble** for robust emotion classification:

| Model | Parameters | Role | Accuracy |
|-------|------------|------|----------|
| **EfficientNetV2-M** | ~54M | Primary classifier | 90% |
| **VGG16** | ~138M | Supporting classifier | 88% |
| **DenseNet121** | ~8M | Supporting classifier | 89% |
| **Ensemble** | - | **Combined prediction** | **92%** |

### Model Details

- **Input**: Mel spectrogram (128x128 pixels)
- **Preprocessing**:
  - Noise reduction using spectral gating
  - Resampling to 16kHz
  - Mel spectrogram with 128 mel bins
- **Output**: 7-class softmax probabilities
- **Framework**: PyTorch 2.0+ with TorchScript optimization
- **File**: `phase3_outputs/final_export/best_efficientnet_v2_m+vgg16+densenet121.pt`

### Training Dataset

- **Dataset**: RAVDESS, TESS, CREMA-D (combined)
- **Total Samples**: ~12,000 audio files
- **Emotions**: 7 classes (angry, disgust, fear, happy, neutral, sad, surprise)
- **Augmentation**: Time stretching, pitch shifting, noise addition
- **Validation Split**: 80/20 train/validation

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 92% |
| **Precision** | 91% |
| **Recall** | 90% |
| **F1 Score** | 90.5% |
| **Inference Time (CPU)** | ~150ms |
| **Inference Time (GPU)** | ~50ms |

---

## 🔧 Development

### Setting Up Development Environment

```bash
# 1. Clone repository
git clone https://github.com/yourusername/emotion-voice.git
cd emotion-voice

# 2. Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# 3. Install development dependencies
pip install pytest pytest-asyncio black flake8 mypy

# 4. Frontend setup
cd ../frontend/Voice-Emotion-Detector
npm install
npm install --save-dev eslint prettier
```

### Code Style

**Backend (Python):**
- Follow PEP 8 style guide
- Use Black for code formatting
- Type hints for all functions

```bash
# Format code
black backend/

# Lint code
flake8 backend/

# Type checking
mypy backend/
```

**Frontend (TypeScript):**
- Use ESLint and Prettier
- Follow Airbnb style guide
- TypeScript strict mode enabled

```bash
# Lint code
npm run lint

# Format code
npx prettier --write .
```

### Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Server Configuration
HOST=127.0.0.1
PORT=8000

# Model Configuration
MODEL_PATH=../phase3_outputs/final_export/best_efficientnet_v2_m+vgg16+densenet121.pt
ENABLE_MODEL_PRELOAD=true

# Security
API_KEY=your_secret_api_key_here
ENABLE_RATE_LIMIT=true
RATE_LIMIT=100/minute

# CORS
CORS_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]

# Logging
LOG_LEVEL=INFO
ENABLE_TRACE_LOGGING=true
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add your feature description"

# Push changes
git push origin feature/your-feature-name

# Create pull request on GitHub
```

---

## 🧪 Testing

### Backend Tests

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_inference.py

# Run with verbose output
pytest -v
```

### API Testing

**Using provided test scripts:**

```bash
cd backend

# Test all endpoints
python test_all_endpoints.py

# Test single prediction
python test_single_prediction.py

# Test API alignment
python test_alignment.py
```

**Using curl:**

```bash
# Test health endpoint
curl http://127.0.0.1:8000/health

# Test prediction with sample audio
curl -X POST "http://127.0.0.1:8000/api/predict" \
  -F "file=@test_audio.wav"

# Test metrics endpoint
curl http://127.0.0.1:8000/api/metrics
```

### Frontend Tests

```bash
cd frontend/Voice-Emotion-Detector

# Run tests
npm test

# Run with coverage
npm test -- --coverage

# Run in watch mode
npm test -- --watch
```

### Load Testing

```bash
# Install apache bench
sudo apt install apache2-utils  # Linux
brew install httpd  # macOS

# Run load test
ab -n 1000 -c 10 -p audio.wav -T "multipart/form-data" \
  http://127.0.0.1:8000/api/predict
```

---

## 🚢 Deployment

### Production Deployment with Docker

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Deploy to Cloud Platforms

#### AWS EC2

```bash
# 1. Launch EC2 instance (t3.medium or larger)
# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Install dependencies
sudo apt update
sudo apt install python3 python3-venv nodejs npm ffmpeg docker.io

# 4. Clone and setup
git clone https://github.com/yourusername/emotion-voice.git
cd emotion-voice
python3 start_all.py
```

#### Google Cloud Platform

```bash
# Deploy using Cloud Run
gcloud run deploy emotion-voice-api \
  --source=./backend \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated
```

#### Heroku

```bash
# Create Heroku app
heroku create emotion-voice-api

# Add buildpacks
heroku buildpacks:add heroku/python
heroku buildpacks:add https://github.com/jonathanong/heroku-buildpack-ffmpeg-latest

# Deploy
git push heroku main
```

### Environment Variables for Production

```env
# Production settings
ENVIRONMENT=production
DEBUG=false
HOST=0.0.0.0
PORT=8000

# Security
API_KEY=your_strong_secret_key
ENABLE_RATE_LIMIT=true
RATE_LIMIT=60/minute

# CORS - Update with your domain
CORS_ORIGINS=["https://yourdomain.com"]

# Performance
ENABLE_MODEL_PRELOAD=true
WORKERS=4
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. FFmpeg Not Found

**Error**: `FFmpeg not found - WebM recording may not work`

**Solution**:
```bash
# Windows
.\backend\install_ffmpeg.ps1

# macOS
brew install ffmpeg

# Linux
sudo apt update && sudo apt install ffmpeg
```

#### 2. Port Already in Use

**Error**: `Port 8000 is already in use`

**Solution**:
```bash
# Find process using port
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux:
lsof -ti:8000 | xargs kill -9
```

#### 3. Model File Not Found

**Error**: `Model file not found`

**Solution**:
- Ensure model file exists at: `phase3_outputs/final_export/best_efficientnet_v2_m+vgg16+densenet121.pt`
- Download from releases or train your own model
- Check `MODEL_PATH` in `.env` file

#### 4. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# In backend/config.py, force CPU usage
DEVICE = "cpu"
```

#### 5. Frontend Can't Connect to Backend

**Error**: `Network Error` or `Failed to fetch`

**Solution**:
- Verify backend is running: http://127.0.0.1:8000/health
- Check CORS settings in `backend/config.py`
- Ensure API URL in frontend matches backend URL

#### 6. Audio Recording Not Working

**Error**: `getUserMedia is not supported`

**Solution**:
- Use HTTPS (required for microphone access)
- Check browser permissions
- Use a supported browser (Chrome, Firefox, Safari)

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Backend
cd backend
LOG_LEVEL=DEBUG python -m uvicorn main:app --reload

# Frontend
cd frontend/Voice-Emotion-Detector
npm run dev -- --debug
```

### Getting Help

- **GitHub Issues**: [Open an issue](https://github.com/yourusername/emotion-voice/issues)
- **Documentation**: Check `/docs` endpoint for API details
- **Logs**: Review logs in `backend/logs/` directory

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute

- 🐛 **Report bugs** - Open an issue with reproduction steps
- 💡 **Suggest features** - Share your ideas for improvements
- 📖 **Improve documentation** - Fix typos, add examples
- 🔧 **Submit pull requests** - Fix bugs or implement features
- 🧪 **Add tests** - Improve test coverage
- 🎨 **Improve UI/UX** - Enhance the frontend experience

### Development Process

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub
   git clone https://github.com/YOUR_USERNAME/emotion-voice.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Write clean, documented code
   - Follow the coding style guide
   - Add tests for new features
   - Update documentation

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```
   
   Use conventional commits:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `style:` - Code style changes
   - `refactor:` - Code refactoring
   - `test:` - Adding tests
   - `chore:` - Maintenance tasks

5. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Describe your changes in detail

### Code Review Process

- All PRs require at least one review
- CI/CD tests must pass
- Code coverage should not decrease
- Documentation must be updated

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Emotion Voice Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **Datasets**: RAVDESS, TESS, CREMA-D emotion databases
- **Models**: EfficientNet, VGG, DenseNet architectures
- **Frameworks**: FastAPI, Next.js, PyTorch communities
- **Libraries**: Librosa, Pydub, Radix UI, Tailwind CSS

---

## 📞 Contact & Support

- **GitHub**: [yourusername/emotion-voice](https://github.com/yourusername/emotion-voice)
- **Email**: support@emotionvoice.com
- **Documentation**: http://127.0.0.1:8000/docs
- **Issues**: [GitHub Issues](https://github.com/yourusername/emotion-voice/issues)

---

## 🗺️ Roadmap

### Version 5.0 (Planned)

- [ ] Real-time streaming inference
- [ ] Multi-language support
- [ ] Mobile app (React Native)
- [ ] Custom model training interface
- [ ] Emotion history and analytics
- [ ] WebSocket support for live audio
- [ ] Enhanced visualization with waveforms
- [ ] Batch processing for multiple files
- [ ] API rate limiting tiers
- [ ] User authentication and profiles

### Future Enhancements

- Integration with video analysis
- Age and gender detection
- Stress level analysis
- Custom emotion categories
- Voice biometric identification
- Speaker diarization
- Real-time translation with emotion preservation

---

<div align="center">

**Made with ❤️ by the Emotion Voice Team**

⭐ **Star this repo** if you find it helpful!

[🐛 Report Bug](https://github.com/yourusername/emotion-voice/issues) • [✨ Request Feature](https://github.com/yourusername/emotion-voice/issues) • [📖 Documentation](http://127.0.0.1:8000/docs)

</div>

