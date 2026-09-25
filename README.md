# Face Matching System

A production-oriented face matching system for **face detection, face embedding generation, similarity matching, duplicate detection, and vector search**.

The system uses **OpenCV YuNet** for face detection, **AdaFace** for face embeddings, and **Qdrant** for high-performance vector similarity search. It also supports single-image and bulk-image processing with duplicate prevention.

---

## 🚀 Features

- Face detection using OpenCV YuNet
- Face alignment and preprocessing
- Face embedding generation using AdaFace
- Cosine similarity-based face matching
- Qdrant vector database integration
- Single image face upload
- Bulk image upload
- Duplicate face detection
- Same-batch duplicate prevention
- Existing-dataset duplicate prevention
- Face search using an uploaded image
- Delete face from dataset using image search
- REST API using FastAPI
- Concurrent bulk image processing
- Evaluation using LFW dataset
- Performance and accuracy evaluation
- Configurable similarity threshold
- Docker support
- Health-check endpoints
- Structured service-oriented backend architecture

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │      Client         │
                    │ Web / API / App     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │      REST API       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Image Preprocessing │
                    │ Resize / Normalize  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   YuNet Detector    │
                    │   Face Detection     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Face Alignment      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      AdaFace        │
                    │ Face Embeddings     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Duplicate Check    │
                    │ Cosine Similarity   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Qdrant         │
                    │ Vector Database     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Matching Result     │
                    │ Score / Identity    │
                    └─────────────────────┘
