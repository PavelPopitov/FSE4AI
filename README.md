# TinyVision

**TinyVision** is a lightweight image classification web application powered by MobileNetV3-Small with Grad-CAM visualization support. The project combines FastAPI for the backend API and Gradio for an interactive web interface, enabling users to classify images and visualize model attention using Gradient-weighted Class Activation Mapping (Grad-CAM).

## Features

- 🚀 **Fast Image Classification**: Uses pretrained MobileNetV3-Small model on ImageNet-1K (1000 classes)
- 🔍 **Top-K Predictions**: Returns top-5 most likely classes with confidence scores
- 🎨 **Grad-CAM Visualization**: Highlights regions of the image that influenced the model's decision
- 🌐 **Dual Interface**: RESTful API (FastAPI) + Interactive Web UI (Gradio)
- 🐳 **Docker Support**: Fully containerized deployment
- ✅ **Comprehensive Testing**: Unit tests with pytest
- ⚡ **Lightweight**: Optimized for CPU inference with minimal dependencies

## Tech Stack

- **Deep Learning**: PyTorch, TorchVision
- **Web Framework**: FastAPI, Uvicorn
- **UI**: Gradio 4.44+
- **Image Processing**: Pillow, NumPy
- **Testing**: pytest, httpx
- **Build System**: setuptools, pyproject.toml
- **Containerization**: Docker

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Docker (optional, for containerized deployment)

## Installation

### Local Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd FSE4AI
```

2. **Install dependencies**:
```bash
make setup
```

Or manually:
```bash
python -m pip install --upgrade pip
pip install -e .[dev]
```

This will install the project in editable mode along with development dependencies (pytest, httpx, black, ruff).

### Docker Setup

Build the Docker image:
```bash
make docker-build
```

Or manually:
```bash
docker build -t tinyvision:latest .
```

## Usage

### Running Locally

Start the server:
```bash
make run
```

Or manually:
```bash
uvicorn tinyvision.app:app --host 0.0.0.0 --port 8000 --reload
```

The application will be available at:
- **API**: http://localhost:8000
- **Interactive UI**: http://localhost:8000/ui
- **API Documentation**: http://localhost:8000/docs

### Running with Docker

```bash
make docker-run
```

Or manually:
```bash
docker run -p 8000:8000 tinyvision:latest
```

### Using the Command Line Entry Point

After installation, you can also run:
```bash
tinyvision
```

## API Endpoints

### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "ok"
}
```

### Get Available Labels
```http
GET /labels
```

Returns all 1000 ImageNet class labels.

### Image Classification
```http
POST /predict
```

**Parameters**:
- `file`: Image file (multipart/form-data)
- `gradcam`: Boolean query parameter (optional, default: false)

**Example with curl**:
```bash
curl -X POST "http://localhost:8000/predict?gradcam=true" \
  -F "file=@/path/to/image.jpg"
```

**Response** (without Grad-CAM):
```json
{
  "predictions": [
    {"label": "golden retriever", "prob": 0.892},
    {"label": "Labrador retriever", "prob": 0.054},
    {"label": "cocker spaniel", "prob": 0.023},
    {"label": "English setter", "prob": 0.012},
    {"label": "clumber", "prob": 0.008}
  ]
}
```

**Response** (with Grad-CAM):
```json
{
  "predictions": [...],
  "gradcam_png_b64": "<base64-encoded-png>",
  "gradcam_for_class": 207
}
```

The Grad-CAM overlay is returned as a base64-encoded PNG image with RGBA heatmap showing which regions of the image contributed most to the prediction.

## Web Interface

Navigate to http://localhost:8000/ui to access the interactive Gradio interface:

1. Upload an image
2. Check "Показать Grad-CAM" (Show Grad-CAM) if you want visualization
3. Click "Классифицировать" (Classify)
4. View top-5 predictions and Grad-CAM overlay

## Testing

Run all tests:
```bash
make test
```

Or manually:
```bash
pytest -q
```

### Test Coverage

The project includes comprehensive tests for:
- **API endpoints** (`tests/test_api.py`): Health check, prediction endpoint, Grad-CAM integration
- **Model inference** (`tests/test_model.py`): Top-K predictions, output types
- **Preprocessing** (`tests/test_preprocess.py`): Image transformation and tensor shape validation
- **Grad-CAM** (`tests/test_gradcam.py`): Activation map generation and normalization

## Project Structure

```
FSE4AI/
├── tinyvision/              # Main application package
│   ├── init.py              # Package initialization
│   ├── app.py               # FastAPI application and endpoints
│   ├── model.py             # MobileNetV3 model and inference logic
│   ├── gradcam.py           # Grad-CAM implementation
│   ├── ui.py                # Gradio web interface
│   ├── main.py              # CLI entry point
│   └── assets/              # Sample images for testing
│       ├── dog.jpg
│       ├── goldfish.JPG
│       ├── horse.jpeg
│       └── absolute_cinema.png
├── tests/                   # Test suite
│   ├── test_api.py          # API endpoint tests
│   ├── test_model.py        # Model inference tests
│   ├── test_gradcam.py      # Grad-CAM tests
│   └── test_preprocess.py   # Preprocessing tests
├── .github/
│   └── workflows/
│       └── ci.yml           # CI/CD configuration (placeholder)
├── pyproject.toml           # Project metadata and dependencies
├── Makefile                 # Build automation
├── Dockerfile               # Container definition
├── .gitignore               # Git ignore patterns
└── README.md                # This file
```

## Development

### Code Formatting

Format code with Black:
```bash
make fmt
```

### Linting

Check code quality with Ruff:
```bash
make lint
```

### Available Make Commands

- `make setup` - Install dependencies
- `make test` - Run tests
- `make run` - Start development server
- `make docker-build` - Build Docker image
- `make docker-run` - Run Docker container
- `make lint` - Lint code
- `make fmt` - Format code

## Model Details

- **Architecture**: MobileNetV3-Small
- **Weights**: IMAGENET1K_V1 (pretrained on ImageNet)
- **Input Size**: 224×224 RGB
- **Output**: 1000 classes (ImageNet categories)
- **Device Support**: CUDA (if available) or CPU

The model is automatically downloaded from torchvision on first run.

## Grad-CAM Explanation

Gradient-weighted Class Activation Mapping (Grad-CAM) is a technique that produces visual explanations for decisions from CNN-based models. It uses the gradients flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.

In TinyVision:
- Target layer: `model.features[-1]` (last convolutional layer of MobileNetV3)
- The heatmap shows areas with high attention in red/orange
- The overlay is composited with the original image for easy interpretation

## License

MIT License

## Contributing

This project was developed as part of the FSE4AI course final project. Contributions, issues, and feature requests are welcome!

## Authors

FSE4AI Course Project Team

## Acknowledgments

- PyTorch and TorchVision teams for pretrained models
- FastAPI and Gradio communities for excellent frameworks
- Original Grad-CAM paper: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"

