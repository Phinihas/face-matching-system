# Face Matching System

A production-oriented **Face Matching and Face Recognition API** built with **Python, FastAPI, OpenCV YuNet, AdaFace, Qdrant, MySQL, Docker, Prometheus, and OpenTelemetry**.

The system provides face detection, face alignment, face embedding generation, 1:1 face comparison, 1:N face search, duplicate face detection, single-image upload, bulk image processing, batch progress tracking, vector similarity search, and monitoring.

---

## 🚀 Features

- Face detection
- Face alignment
- Face embedding generation using AdaFace
- 1:1 face comparison
- 1:N face search
- Cosine similarity matching
- Configurable matching threshold
- Qdrant vector database integration
- Single image upload
- Bulk image processing
- Concurrent image processing
- Existing-dataset duplicate detection
- Same-batch duplicate detection
- Batch progress tracking
- Batch history
- Search result ZIP generation
- Temporary file cleanup
- FastAPI REST API
- Swagger/OpenAPI documentation
- Prometheus metrics
- OpenTelemetry tracing
- Docker support
- LFW-based model evaluation
- Performance benchmarking

---

# 📌 Project Overview

The Face Matching System is designed to identify whether two faces belong to the same person and to search a large face dataset for similar faces.

The system converts a detected face into a numerical **face embedding** and compares that embedding with stored embeddings using vector similarity search.

It supports two major use cases:

### 1. Face Verification

Compare two images and determine whether the faces match.

```text
Image 1
   │
   ▼
Face Detection
   │
   ▼
Face Alignment
   │
   ▼
AdaFace Embedding
   │
   ├──────────────┐
   │              │
   ▼              ▼
Embedding 1   Embedding 2
   │              │
   └──────┬───────┘
          ▼
   Similarity Score
          │
          ▼
    Match / No Match
```

### 2. Face Identification / Search

Upload one image and search the stored face dataset for the most similar faces.

```text
Query Image
     │
     ▼
Face Detection
     │
     ▼
Face Alignment
     │
     ▼
AdaFace Embedding
     │
     ▼
Qdrant Vector Search
     │
     ▼
Top-K Similar Faces
```

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │       Client         │
                         │ Web / Frontend / API │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │      REST API        │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
          /compare               /search           Upload APIs
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Image Preprocessing  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Face Detection     │
                         │      YuNet           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Face Alignment     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      AdaFace         │
                         │ Face Embedding Model │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Face Embedding     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Similarity Matching  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       Qdrant         │
                         │   Vector Database    │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                      Match                No Match
```

---

# 🔄 Face Matching Pipeline

The complete pipeline is:

```text
Input Image
     │
     ▼
Image Validation
     │
     ▼
Face Detection
     │
     ▼
Face Alignment
     │
     ▼
Image Preprocessing
     │
     ▼
AdaFace Embedding
     │
     ▼
Embedding Vector
     │
     ▼
Cosine Similarity
     │
     ▼
Threshold Comparison
     │
     ├───────────────┐
     │               │
     ▼               ▼
   Match          No Match
```

---

# 👤 Face Detection

The face detection stage identifies faces and facial landmarks from the input image.

The project uses **OpenCV YuNet** for the face detection stage.

The detector provides information such as:

```text
Face Bounding Box
        +
Facial Landmarks
        │
        ├── Left Eye
        ├── Right Eye
        ├── Nose
        ├── Left Mouth
        └── Right Mouth
```

These landmarks are used for face alignment.

---

# 🎯 Face Alignment

After detecting a face, the system aligns it into a normalized orientation.

```text
Detected Face
      │
      ▼
Facial Landmarks
      │
      ▼
Alignment
      │
      ▼
Normalized Face
```

Face alignment helps reduce variations caused by:

- Face rotation
- Face position
- Different image dimensions
- Facial landmark positioning

---

# 🧠 AdaFace Embedding

The aligned face is passed to the AdaFace model.

```text
Normalized Face
      │
      ▼
Preprocessing
      │
      ▼
AdaFace
      │
      ▼
Face Embedding
```

The face is represented as a numerical embedding vector.

The embedding is then used for:

- Face comparison
- Duplicate detection
- Vector search
- Identity matching

---

# 🔍 Similarity Matching

The system uses cosine similarity to compare face embeddings.

The similarity between two embeddings is calculated conceptually as:

```text
              A · B
Similarity = ─────────────
             ||A|| ||B||
```

The resulting score is compared with the configured matching threshold.

```text
Similarity Score
       │
       ▼
Threshold Check
       │
   ┌───┴────┐
   │        │
   ▼        ▼
 Match   No Match
```

---

# 🗄️ Qdrant Vector Database

Qdrant is used to store and search face embeddings.

A stored face record contains:

```text
Vector
 │
 ├── Vector ID
 │
 └── Payload
      │
      └── image_path
```

During face search:

```text
Query Image
     │
     ▼
Face Embedding
     │
     ▼
Qdrant
     │
     ▼
Nearest Vectors
     │
     ▼
Similarity Scores
```

Qdrant allows the system to efficiently search the stored face embeddings.

---

# ♻️ Duplicate Face Detection

The system prevents duplicate faces from being inserted into the dataset.

Duplicate detection is performed in two stages.

## Existing Dataset Duplicate

```text
New Image
    │
    ▼
Generate Embedding
    │
    ▼
Search Existing Faces
    │
    ▼
Similarity Check
    │
 ┌──┴───────────┐
 │              │
 ▼              ▼
Duplicate      Unique
 │              │
 ▼              ▼
Reject         Insert
```

## Same-Batch Duplicate

The system also prevents duplicates within the same bulk-upload batch.

Example:

```text
Batch

Image 1 → Unique → Insert
Image 2 → Unique → Insert
Image 3 → Duplicate of Image 1 → Reject
Image 4 → Unique → Insert
Image 5 → Duplicate of Image 2 → Reject
```

This prevents the same face from being inserted multiple times during a single bulk operation.

---

# 📡 REST API

The application exposes the following REST endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API health/status |
| `POST` | `/compare` | 1:1 face comparison |
| `POST` | `/search` | 1:N face search |
| `POST` | `/single_upload` | Upload one face image |
| `POST` | `/bulk_upload` | Start bulk processing |
| `GET` | `/batch/{batch_id}/status` | Get batch progress |
| `GET` | `/batches` | Get all batch jobs |
| `GET` | `/metrics` | Prometheus metrics |

---

# 1. GET `/`

## Description

Health/status endpoint used to verify that the API is running.

## Request

```http
GET /
```

## Example

```bash
curl http://localhost:9000/
```

## Response

```json
{
  "message": "Face Matching API - Multi-Face Support",
  "status": "running"
}
```

---

# 2. POST `/compare`

## Description

Performs **1:1 face comparison** between two uploaded images.

Use this endpoint when you want to determine whether the faces in two images match.

## Request

```http
POST /compare
Content-Type: multipart/form-data
```

### Form Data

| Field | Type | Required | Description |
|---|---|---|---|
| `image1` | File | Yes | First face image |
| `image2` | File | Yes | Second face image |

## cURL Example

```bash
curl -X POST "http://localhost:9000/compare" \
  -F "image1=@person1.jpg" \
  -F "image2=@person2.jpg"
```

## Match Response

```json
{
  "result": "match",
  "match_percentage": 98.72
}
```

## No Match Response

```json
{
  "result": "no match",
  "match_percentage": 61.43
}
```

## Processing Flow

```text
Image 1 ──┐
          ├──> Face Detection
Image 2 ──┘
                │
                ▼
          Face Alignment
                │
                ▼
           AdaFace
                │
                ▼
        Similarity Matrix
                │
                ▼
          Threshold Check
                │
          ┌─────┴─────┐
          ▼           ▼
        Match      No Match
```

---

# 3. POST `/search`

## Description

Performs **1:N face matching** against the stored face dataset.

The endpoint accepts a query image and searches Qdrant for the most similar stored face embeddings.

## Request

```http
POST /search
Content-Type: multipart/form-data
```

### Form Data

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | File | Yes | Query face image |

### Query Parameter

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `top_k` | Integer | `3` | Number of matching faces to return |

## cURL Example

```bash
curl -X POST "http://localhost:9000/search?top_k=5" \
  -F "image=@query.jpg"
```

## Processing Flow

```text
Query Image
     │
     ▼
Face Detection
     │
     ▼
Face Alignment
     │
     ▼
AdaFace
     │
     ▼
Embedding
     │
     ▼
Qdrant Search
     │
     ▼
Top-K Similar Faces
```

## Response

The endpoint returns a ZIP archive containing the matching images.

Example result filenames:

```text
1_score_0.982_person_01.jpg
2_score_0.961_person_27.jpg
3_score_0.934_person_14.jpg
```

The response also includes headers such as:

```text
X-Search-Time
X-Results-Count
```

## No Results

HTTP:

```text
404 Not Found
```

Response:

```json
{
  "message": "No similar faces found",
  "status": "no results"
}
```

---

# 4. POST `/single_upload`

## Description

Processes and uploads a single face image.

The endpoint performs:

1. Image upload
2. Face detection
3. Face alignment
4. Face embedding
5. Duplicate detection
6. Qdrant insertion
7. Image storage
8. Temporary-file cleanup

## Request

```http
POST /single_upload
Content-Type: multipart/form-data
```

### Form Data

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | File | Yes | Face image |

## cURL Example

```bash
curl -X POST "http://localhost:9000/single_upload" \
  -F "image=@person.jpg"
```

## Successful Response

```json
{
  "message": "Image processed and uploaded successfully",
  "request_id": "a8f21c9d",
  "status": "completed"
}
```

## Processing Flow

```text
Image
  │
  ▼
Face Detection
  │
  ▼
Face Embedding
  │
  ▼
Duplicate Check
  │
 ┌┴──────────────┐
 │               │
 ▼               ▼
Duplicate       Unique
 │               │
 ▼               ▼
Reject          Qdrant
                 │
                 ▼
             Completed
```

---

# 5. POST `/bulk_upload`

## Description

Starts processing a folder of images in the background.

The endpoint immediately returns a `batch_id`, allowing the client to monitor the processing job without waiting for all images to finish.

## Request

```http
POST /bulk_upload
```

### Query Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---:|---|
| `folder_path` | String | Yes | - | Folder containing images |
| `max_workers` | Integer | No | `4` | Number of concurrent workers |

## cURL Example

```bash
curl -X POST \
  "http://localhost:9000/bulk_upload?folder_path=/data/faces&max_workers=4"
```

## Response

```json
{
  "message": "Upload started in background",
  "batch_id": "a81f29c3",
  "status": "pending"
}
```

The `batch_id` is used to track the job.

---

# 6. GET `/batch/{batch_id}/status`

## Description

Returns the current status of a bulk-processing job.

## Request

```http
GET /batch/{batch_id}/status
```

Example:

```bash
curl http://localhost:9000/batch/a81f29c3/status
```

## Response

```json
{
  "batch_id": "a81f29c3",
  "status": "processing",
  "folder_path": "/data/faces",
  "created_at": "2026-09-25T10:30:00",
  "started_at": "2026-09-25T10:30:03",
  "total_images": 10000,
  "processed_images": 6500,
  "progress_percentage": 65.0
}
```

Possible states:

```text
pending
processing
completed
failed
```

## Batch Not Found

HTTP:

```text
404 Not Found
```

Response:

```json
{
  "error": "Batch not found",
  "batch_id": "a81f29c3"
}
```

---

# 7. GET `/batches`

## Description

Returns all bulk-processing jobs.

## Request

```http
GET /batches
```

Example:

```bash
curl http://localhost:9000/batches
```

## Response

```json
{
  "batches": [
    {
      "batch_id": "a81f29c3",
      "status": "completed",
      "folder_path": "/data/faces",
      "max_workers": 4,
      "created_at": "2026-09-25T10:30:00",
      "started_at": "2026-09-25T10:30:02",
      "completed_at": "2026-09-25T11:12:15",
      "total_images": 10000,
      "processed_images": 10000,
      "progress_percentage": 100.0
    }
  ],
  "total_count": 1
}
```

---

# 8. GET `/metrics`

## Description

Exposes application metrics for monitoring.

The application uses Prometheus FastAPI Instrumentator.

## Request

```http
GET /metrics
```

Example:

```bash
curl http://localhost:9000/metrics
```

The endpoint can be connected to monitoring systems such as Prometheus and Grafana.

---

# 📦 Bulk Processing

Bulk processing is designed for large image datasets.

```text
                    Folder
                      │
                      ▼
                /bulk_upload
                      │
                      ▼
                  Batch ID
                      │
                      ▼
            Background Processing
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
       Worker 1    Worker 2    Worker 3
          │           │           │
          └───────────┼───────────┘
                      │
                      ▼
             Face Processing
                      │
                      ▼
            Duplicate Detection
                      │
                      ▼
                   Qdrant
                      │
                      ▼
                 Completed
```

---

# 🗃️ Batch Job Tracking

Bulk-processing jobs are tracked in MySQL.

A batch contains information such as:

```text
batch_id
status
folder_path
max_workers
created_at
started_at
completed_at
total_images
processed_images
error_message
```

Progress is calculated using:

```text
progress_percentage =
(processed_images / total_images) × 100
```

---

# 📊 Example Bulk Processing

Suppose a folder contains:

```text
10,000 images
```

The processing pipeline can look like:

```text
10,000 Images
      │
      ▼
Concurrent Processing
      │
      ├── 8,500 Unique
      ├── 1,000 Existing Duplicates
      └── 500 Same-Batch Duplicates
      │
      ▼
8,500 Unique Faces Stored
```

---

# 🧰 Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| API Framework | FastAPI |
| API Server | Uvicorn |
| Face Detection | OpenCV YuNet |
| Face Embedding | AdaFace |
| Vector Database | Qdrant |
| Database | MySQL |
| Image Processing | OpenCV |
| Numerical Processing | NumPy |
| Deep Learning | PyTorch |
| Containerization | Docker |
| Monitoring | Prometheus |
| Distributed Tracing | OpenTelemetry |
| Evaluation Dataset | LFW |

---

# 📁 Project Structure

```text
face-matching/
│
├── backend/
│   │
│   ├── src/
│   │   └── face_matcher/
│   │       │
│   │       ├── main.py
│   │       │
│   │       ├── components/
│   │       │   ├── adaface_config.py
│   │       │   ├── qdrant_config.py
│   │       │   └── scrfd_config.py
│   │       │
│   │       ├── database/
│   │       │   └── database.py
│   │       │
│   │       └── helpers/
│   │           ├── constants.py
│   │           ├── helpers.py
│   │           ├── net.py
│   │           └── pre_process.py
│   │
│   ├── pretrained/
│   ├── uploads/
│   ├── temp_uploads/
│   ├── monitoring/
│   ├── requirements.txt
│   └── pyproject.toml
│
├── evaluation/
│   ├── cli.py
│   ├── evaluator.py
│   └── reports/
│
├── docker_services/
│
├── analysis/
│
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>

cd face-matching
```

---

# 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

---

# 3. Install Dependencies

Using pip:

```bash
pip install -r backend/requirements.txt
```

Or using uv:

```bash
uv sync
```

---

# 🗄️ Start Qdrant

Run Qdrant using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Qdrant will be available at:

```text
http://localhost:6333
```

---

# 🗄️ Configure MySQL

Create the required MySQL database and configure the connection.

Example:

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=face_matching
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
```

Do not commit database credentials to GitHub.

---

# 🔐 Environment Configuration

Create a `.env` file:

```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=my_collection

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=face_matching
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password

MATCH_THRESHOLD=0.20

MODEL_PATH=pretrained/adaface_ir50_ms1mv2.ckpt
YUNET_MODEL_PATH=pretrained/face_detection_yunet.onnx
```

Use the actual environment variable names required by your project configuration.

---

# ▶️ Run the Application

Navigate to the backend:

```bash
cd backend
```

Run using uv:

```bash
uv run uvicorn app.main:app --reload --port 9000
```

Or:

```bash
uvicorn app.main:app --reload --port 9000
```

The API will be available at:

```text
http://localhost:9000
```

---

# 📖 API Documentation

After starting the server:

### Swagger UI

```text
http://localhost:9000/docs
```

### ReDoc

```text
http://localhost:9000/redoc
```

Swagger allows you to test all available API endpoints directly from the browser.

---

# 🐳 Docker

Build the application:

```bash
docker build -t face-matching .
```

Run:

```bash
docker run -p 9000:9000 face-matching
```

If Qdrant is running in another Docker container, configure:

```env
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

---

# 📊 Model Evaluation

The face matching pipeline was evaluated using the **Labeled Faces in the Wild (LFW)** dataset.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- Specificity
- F1 Score
- ROC AUC
- Equal Error Rate (EER)
- Optimal Threshold
- Face Detection Latency
- Embedding Latency
- Similarity Latency
- End-to-End Pipeline Latency
- Throughput

---

# 📈 Evaluation Results

Evaluation on a 6,000-pair test dataset:

| Metric | Result |
|---|---:|
| Accuracy | 99.83% |
| Precision | 100.00% |
| Recall | 99.67% |
| Specificity | 100.00% |
| F1 Score | 99.83% |
| ROC AUC | 0.9986 |
| EER | 0.0040 |
| Optimal Threshold | 0.20 |

### Performance

| Component | Time |
|---|---:|
| Face Detection | 16.97 ms |
| Embedding | 522.29 ms |
| Similarity | 0.53 ms |
| Total Pipeline | 1085.65 ms |
| Throughput | 3.68 pairs/sec |
| Image Throughput | 7.36 images/sec |

> Performance depends on hardware, CPU/GPU configuration, model implementation, image resolution, and runtime configuration.

---

# ⚡ Performance Optimization

## PyTorch Inference Optimization

Inference can be performed without gradient tracking:

```python
with torch.no_grad():
    embedding = model(face)
```

This avoids unnecessary autograd graph creation during inference.

---

## CPU Thread Optimization

When multiple images are processed concurrently, CPU thread oversubscription can reduce performance.

The system can use:

```python
import torch
import cv2

torch.set_num_threads(1)
cv2.setNumThreads(1)
```

This is particularly useful when using `ThreadPoolExecutor` for bulk image processing.

---

# 📊 Monitoring

The project supports application monitoring through:

### Prometheus

```text
GET /metrics
```

### OpenTelemetry

FastAPI requests can be instrumented for distributed tracing.

This allows the system to be integrated with monitoring and observability platforms.

---

# 🧹 Temporary File Management

Uploaded files are temporarily stored during processing.

```text
Upload
   │
   ▼
Temporary File
   │
   ▼
Face Processing
   │
   ▼
Vector Storage
   │
   ▼
Temporary File Cleanup
```

This prevents unnecessary accumulation of temporary files.

---

# 🔄 Model Migration

The face detection component was migrated from an InsightFace/SCRFD-based implementation toward OpenCV YuNet.

### Previous Pipeline

```text
Image
  │
  ▼
InsightFace / SCRFD
  │
  ▼
Face Alignment
  │
  ▼
AdaFace
  │
  ▼
Qdrant
```

### Current Pipeline

```text
Image
  │
  ▼
OpenCV YuNet
  │
  ▼
Face Alignment
  │
  ▼
AdaFace
  │
  ▼
Qdrant
```

The modular architecture allows the face detector to be replaced without redesigning the complete face embedding and vector-search pipeline.

---

# 🔐 Security Considerations

Because this system processes biometric information, production deployments should implement appropriate security controls.

Recommended practices:

- Authentication
- Authorization
- HTTPS
- API rate limiting
- Upload size limits
- File type validation
- Filename sanitization
- Secure image storage
- Qdrant network protection
- Database credential protection
- Secret management
- Audit logging
- Access control
- Dataset protection

Do not commit the following to a public repository:

```text
.env
API keys
Database passwords
Private model URLs
Real face images
Private biometric datasets
Pretrained model weights unless licensing permits redistribution
```

---

# ⚠️ Model and Dataset Licensing

This project may use third-party libraries, pretrained models, model weights, and datasets.

Each component can have a separate license.

Before commercial deployment, verify the licensing terms for:

- Face detection models
- Face embedding models
- Pretrained model weights
- Training datasets
- Evaluation datasets
- Third-party Python packages
- Other downloaded artifacts

The license of a Python library does not automatically determine the license of its pretrained model weights.

Use only models, weights, and datasets whose licenses permit the intended use.

---

# 🧪 Testing

Run all tests:

```bash
pytest
```

Verbose testing:

```bash
pytest -v
```

Run a specific test:

```bash
pytest tests/test_face_matching.py
```

---

# 🛠️ Troubleshooting

## Qdrant Connection Error

Check whether Qdrant is running:

```bash
docker ps
```

Then open:

```text
http://localhost:6333
```

---

## Model Not Found

Check the configured model directory:

```text
pretrained/
```

Make sure the required model files are available.

---

## Google Drive Model Download Error

If a pretrained model is downloaded from Google Drive during application startup, startup can fail when the Drive file reaches its download quota.

Example:

```text
Too many users have viewed or downloaded this file recently.
```

For production deployments, model artifacts should preferably be packaged or retrieved from a controlled model/artifact repository instead of depending on a personal Google Drive download link.

---

# 📋 API Quick Reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/compare` | Compare two face images |
| `POST` | `/search` | Search stored face dataset |
| `POST` | `/single_upload` | Upload one image |
| `POST` | `/bulk_upload` | Start bulk processing |
| `GET` | `/batch/{batch_id}/status` | Monitor batch |
| `GET` | `/batches` | View batch history |
| `GET` | `/metrics` | Application metrics |

---

# 📌 Project Status

## Completed

- [x] Face detection
- [x] Face alignment
- [x] AdaFace face embeddings
- [x] 1:1 face comparison
- [x] 1:N face search
- [x] Qdrant integration
- [x] Single image upload
- [x] Bulk image processing
- [x] Existing dataset duplicate detection
- [x] Same-batch duplicate detection
- [x] Batch status tracking
- [x] Batch history
- [x] Search result ZIP generation
- [x] Temporary file cleanup
- [x] Prometheus metrics
- [x] OpenTelemetry instrumentation
- [x] LFW evaluation
- [x] Performance benchmarking
- [x] YuNet migration

## Future Improvements

- [ ] Authentication and authorization
- [ ] Role-based access control
- [ ] API rate limiting
- [ ] GPU inference optimization
- [ ] Distributed batch processing
- [ ] Advanced monitoring dashboards
- [ ] CI/CD automation
- [ ] Model version management
- [ ] Automated model artifact management
- [ ] Comprehensive integration testing
- [ ] Advanced audit logging

---

# 📚 Evaluation Dataset

The system was evaluated using the:

**Labeled Faces in the Wild (LFW)** dataset.

The dataset should be obtained from its official distribution source and used according to its applicable terms.

Do not upload restricted datasets or real biometric information to this repository.

---

# 🤝 Contribution

Contributions are welcome.

```bash
git clone <repository-url>

cd face-matching

git checkout -b feature/my-feature

# Make changes

git add .

git commit -m "Add my feature"

git push origin feature/my-feature
```

Then create a Pull Request.

---

# 📄 License

Add the actual license applicable to this repository.

For example:

```text
MIT License
```

Do not use the MIT License unless the project source code and included components are compatible with it.

Model weights, datasets, and third-party components may have separate licensing requirements.

---

# 👨‍💻 Author

## Phinihas Gandi

AI / Machine Learning Developer

### Areas of Expertise

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Face Recognition
- Generative AI
- Python
- FastAPI
- Vector Databases
- AI Application Development

---

# ⭐ Project Summary

This project provides an end-to-end face matching platform combining:

```text
FastAPI
   +
OpenCV YuNet
   +
AdaFace
   +
Qdrant
   +
MySQL
   +
Docker
   +
Prometheus
   +
OpenTelemetry
```

The system supports both **real-time face comparison/search** and **large-scale batch face ingestion**, with duplicate prevention, batch tracking, monitoring, and model evaluation.

---

## 🔗 API Documentation

Once the application is running:

**Swagger UI**

```text
http://localhost:9000/docs
```

**ReDoc**

```text
http://localhost:9000/redoc
```

---
