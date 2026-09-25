# Face Matching API

A FastAPI-based face matching service that compares faces between two images with multi-face support and advanced preprocessing capabilities.

## Features

- **Multi-face detection**: Detects and compares multiple faces in images
- **Rotation testing**: Tests multiple image orientations for better matching accuracy
- **Noise reduction**: Automatic image denoising for improved face detection
- **High accuracy**: Uses AdaFace model for robust face recognition
- **RESTful API**: Simple HTTP endpoints for easy integration

## Installation

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (optional, for super faster processing)

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd face-matching
```

2. Install dependencies using uv:
```bash
uv sync
```

Or with pip:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server
```bash
cd src
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Or use the provided scripts:
- Windows: `run.bat`
- Linux/Mac: `run.sh`

### API Endpoints

#### Health Check
```http
GET /
```
Returns server status.

#### Compare Faces
```http
POST /compare
```
Compares faces between two uploaded images.

**Parameters:**
- `image1`: First image file (multipart/form-data)
- `image2`: Second image file (multipart/form-data)

**Response:**
```json
{
  "result": "match",
  "match_percentage": 85.42
}
```

### Example Usage

#### Using curl
```bash
curl -X POST "http://localhost:8000/compare" \
  -F "image1=@path/to/image1.jpg" \
  -F "image2=@path/to/image2.jpg"
```

#### Using Python requests
```python
import requests

url = "http://localhost:8000/compare"
files = {
    'image1': open('image1.jpg', 'rb'),
    'image2': open('image2.jpg', 'rb')
}

response = requests.post(url, files=files)
result = response.json()
print(f"Match result: {result['result']}")
print(f"Confidence: {result['match_percentage']}%")
```

## Technical Details

### Model
- **Face Detection**: InsightFace Buffalo_L model
- **Face Recognition**: AdaFace IR-50 trained on MS1MV2
- **Similarity Threshold**: 0.3 (configurable)

### Processing Pipeline
1. Image upload and validation
2. Noise detection and denoising (if needed)
3. Face detection with padding fallback
4. Multi-rotation testing (0°, 180° combinations)
5. Face alignment and embedding extraction
6. Similarity calculation and matching

### Supported Formats
- JPEG, PNG, BMP, TIFF
- RGB and grayscale images
- Various resolutions

## Configuration

Key parameters in `helper_functions.py`:
- `DEFAULT_THRESHOLD`: Similarity threshold (default: 0.3)
- `FACE_SIZE`: Target face size for alignment (112x112)
- `PADDING_RATIO`: Padding for difficult face detection (0.3)

## Logging

Comprehensive logging is available in:
- Console output
- `app.log` file

Log levels include face detection results, similarity scores, and processing times.

## Performance

- **CPU**: Works on CPU-only systems
- **GPU**: CUDA acceleration supported for faster processing
- **Memory**: Automatic cleanup of temporary files
- **Concurrency**: FastAPI handles multiple concurrent requests

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
