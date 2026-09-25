import os
import time
import uuid
import shutil
import logging
import numpy as np
import cv2
import zipfile
import torch

# --- CPU Optimization ---
# Restrict PyTorch and OpenCV internal threads to 1.
# Since we use ThreadPoolExecutor to process multiple faces concurrently,
# allowing PyTorch/OpenCV to spawn their own threads per-worker causes
# severe CPU thrashing (100% CPU lockup and 3x slower execution).
torch.set_num_threads(1)
cv2.setNumThreads(1)
import io
import base64
import asyncio
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from face_matcher.helpers.helpers import compare_two_images, cleanup_file, process_and_upload, search_vector, process_single_image, compare_two_images_eval
from face_matcher.database.database import (
    create_batch_job, get_batch_job, get_all_batch_jobs,
    cancel_batch_job, create_single_upload_job, complete_single_upload_job,
    _db_insert_compare_pending, _db_insert_compare_success, _db_insert_compare_error
)
from face_matcher.helpers.pre_process import denoise_image_if_needed
from face_matcher.components.adaface_config import download_pretrained_model
from face_matcher.components.qdrant_config import get_qdrant_client, init_collection
from face_matcher.helpers.constants import TEMP_DIR, LOG_DIR, MATCH_THRESHOLD, COLLECTION_NAME, MODEL_FILE_ID, MODEL_PATH
from face_matcher.database.request_id import generate_request_id  # adjust path as needed
from face_matcher import settings
# Monitoring
from prometheus_fastapi_instrumentator import Instrumentator

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

SEARCH_RESULTS_DIR = settings.SEARCH_RESULTS_DIR
log_file = os.path.join(settings.LOG_DIR, "app.log")

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ],
)
logger = logging.getLogger(__name__)

resource = Resource(attributes={"service.name": "fastapi"})

OTEL_ENDPOINT = settings.OTEL_ENDPOINT

otlp_exporter = OTLPSpanExporter(
    endpoint=OTEL_ENDPOINT,
    insecure=True
)

trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== FACE MATCHING API STARTUP ===")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Temp directory: {TEMP_DIR}")
    logger.info(f"Log directory: {LOG_DIR}")
    logger.info(f"Match threshold: {MATCH_THRESHOLD}")
    logger.info(f"OTEL endpoint: {OTEL_ENDPOINT}")

    try:
        download_pretrained_model(MODEL_FILE_ID, MODEL_PATH)
        logger.info("Model download/verification completed")

        # ✅ FIX 2: Qdrant init moved here from module import time.
        #           By the time lifespan runs, Qdrant is healthy (docker-compose
        #           healthcheck ensures service_healthy before backend starts).
        global client
        client = get_qdrant_client()
        init_collection(client)
        logger.info("Qdrant collection initialized")

        instrumentator.expose(app)
        logger.info("Metrics endpoint exposed at /metrics")
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Application startup failed: {e}", exc_info=True)
        raise

    yield

    logger.info("=== FACE MATCHING API SHUTDOWN ===")


app = FastAPI(
    title="Face Matching API",
    description="Matches faces from 2 images with multi-face support",
    root_path=settings.ROOT_PATH,
    lifespan=lifespan
)

instrumentator = Instrumentator().instrument(app)
FastAPIInstrumentor.instrument_app(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Search-Time", "X-Results-Count"]
)

# ✅ FIX 3: client is created here (module level) for use in endpoints,
#           but collection init (the call that was crashing) is now in lifespan above.
from qdrant_client import QdrantClient
client: QdrantClient = None


@app.get('/')
def root():
    logger.debug("Health check endpoint accessed")
    return {"message": "Face Matching API - Multi-Face Support", "status": "running"}

@app.get('/supported-formats')
def supported_formats():
    """Return list of image formats supported by this application."""
    formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic']
    return JSONResponse({
        "supported_formats": formats,
        "total": len(formats)
    })

@app.get('/stats')
def get_stats():
    """Returns overall face matching statistics."""
    from face_matcher.database.db import get_db
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
        SELECT
            COUNT(*)                                                              AS total_requests,
            SUM(status = 's')                                                    AS success_requests,
            SUM(status = 'p')                                                    AS pending_requests,
            SUM(status = 'e')                                                    AS error_requests,
            SUM(status = 'c')                                                    AS cancelled_requests,
            SUM(total_count)                                                     AS total_files,
            SUM(CASE WHEN status = 's' THEN total_count ELSE 0 END)             AS success_files,
            SUM(CASE WHEN status = 'p' THEN total_count - processed_count ELSE 0 END) AS pending_files,
            SUM(CASE WHEN status = 'e' THEN total_count ELSE 0 END)             AS error_files,
            SUM(CASE WHEN status = 'c' THEN total_count ELSE 0 END)             AS cancelled_files
        FROM pii_status
        WHERE type = 'FaceMatching'
    """)
    row = cur.fetchone()
    return {
        "total_requests":     int(row["total_requests"]     or 0),
        "success_requests":   int(row["success_requests"]   or 0),
        "pending_requests":   int(row["pending_requests"]   or 0),
        "error_requests":     int(row["error_requests"]     or 0),
        "cancelled_requests": int(row["cancelled_requests"] or 0),
        "total_files":        int(row["total_files"]        or 0),
        "success_files":      int(row["success_files"]      or 0),
        "pending_files":      int(row["pending_files"]      or 0),
        "error_files":        int(row["error_files"]        or 0),
        "cancelled_files":    int(row["cancelled_files"]    or 0),
    }

# --- evaluation --- #
async def compare_faces_eval(image1_path_input, image2_path_input):
    """Compare faces between two existing image paths (no file upload objects)."""
    request_id = uuid.uuid4().hex[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] === FACE COMPARISON REQUEST START ===")
    logger.info(f"[{request_id}] Images: '{image1_path_input}' vs '{image2_path_input}'")

    image1_filename = os.path.basename(image1_path_input)
    image2_filename = os.path.basename(image2_path_input)

    image1_path = TEMP_DIR / f"img1_{request_id}_{image1_filename}"
    image2_path = TEMP_DIR / f"img2_{request_id}_{image2_filename}"

    try:
        logger.debug(f"[{request_id}] Copying given image files to temp directory")
        shutil.copy(image1_path_input, image1_path)
        file1_size = image1_path.stat().st_size

        shutil.copy(image2_path_input, image2_path)
        file2_size = image2_path.stat().st_size

        logger.info(f"[{request_id}] Files copied - Image1: {file1_size:,} bytes, Image2: {file2_size:,} bytes")

        logger.info(f"[{request_id}] Starting face comparison (threshold={MATCH_THRESHOLD})")
        comparison_start = time.time()

        matches, status_msg, faces1, faces2, similarity_matrix = await asyncio.get_event_loop().run_in_executor(
            None, compare_two_images_eval, str(image1_path), str(image2_path), request_id
        )

        comparison_time = time.time() - comparison_start
        logger.info(f"[{request_id}] Face comparison completed in {comparison_time:.2f}s")

        total_faces1 = len(faces1) if faces1 else 0
        total_faces2 = len(faces2) if faces2 else 0
        total_time = time.time() - start_time

        logger.info(f"[{request_id}] Detection summary: {total_faces1} faces in image1, {total_faces2} faces in image2")

        if matches and len(matches) > 0:
            best_match = matches[0]
            match_percentage = best_match['match_percentage']
            similarity_score = best_match['similarity_score']
            face1_id = best_match['face1_id']
            face2_id = best_match['face2_id']

            logger.info(f"[{request_id}] MATCH FOUND: {match_percentage:.2f}% confidence")
            logger.info(f"[{request_id}] Best match: face1[{face1_id}] ↔ face2[{face2_id}] (similarity={similarity_score:.4f})")

            response = {
                "result": "match",
                "match_percentage": round(float(match_percentage), 2)
            }
            logger.info(f"[{request_id}] SUCCESS - Match: {match_percentage:.2f}% | Time: {total_time:.2f}s")
            return JSONResponse(response)
        else:
            if similarity_matrix is not None and similarity_matrix.size > 0:
                max_similarity = float(np.max(similarity_matrix))
                match_percentage = round(((max_similarity + 1) / 2) * 100, 2)
            else:
                max_similarity = -1.0
                match_percentage = 0.0

            logger.info(f"[{request_id}] NO MATCH: best similarity={max_similarity:.4f} ({match_percentage:.2f}%)")

            response = {
                "result": "no match",
                "match_percentage": round(match_percentage, 2)
            }
            logger.info(f"[{request_id}] NO MATCH - Best: {match_percentage:.2f}% | Time: {total_time:.2f}s")
            return JSONResponse(response)

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] PROCESSING ERROR after {total_time:.2f}s: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing error: {str(e)}", "request_id": request_id}
        )

    finally:
        await asyncio.to_thread(cleanup_file, image1_path)
        await asyncio.to_thread(cleanup_file, image2_path)
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] === REQUEST COMPLETED in {total_time:.2f}s ===")


# --- 1:1 --- #
@app.post('/compare')
async def compare_faces_endpoint(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    """Compare faces between two uploaded images."""
    
    # ── Use consistent request_id format ─────────────────────────
    request_id = generate_request_id()
    _db_insert_compare_pending(request_id, image1.filename, image2.filename)
    start_time = time.time()

    logger.info(f"[{request_id}] === FACE COMPARISON REQUEST START ===")
    logger.info(f"[{request_id}] Images: '{image1.filename}' vs '{image2.filename}'")

    image1_path = TEMP_DIR / f"img1_{request_id}_{image1.filename}"
    image2_path = TEMP_DIR / f"img2_{request_id}_{image2.filename}"

    try:
        content1 = await image1.read()
        with open(image1_path, "wb") as f:
            f.write(content1)

        content2 = await image2.read()
        with open(image2_path, "wb") as f:
            f.write(content2)

        denoise_task1 = asyncio.to_thread(denoise_image_if_needed, str(image1_path), request_id)
        denoise_task2 = asyncio.to_thread(denoise_image_if_needed, str(image2_path), request_id)
        denoised_img1, denoised_img2 = await asyncio.gather(denoise_task1, denoise_task2)

        save_tasks = []
        if denoised_img1 is not None:
            image1_path = Path(str(image1_path).rsplit('.', 1)[0] + '.jpg')
            save_tasks.append(asyncio.to_thread(cv2.imwrite, str(image1_path), denoised_img1))
        if denoised_img2 is not None:
            image2_path = Path(str(image2_path).rsplit('.', 1)[0] + '.jpg')
            save_tasks.append(asyncio.to_thread(cv2.imwrite, str(image2_path), denoised_img2))
        if save_tasks:
            await asyncio.gather(*save_tasks)

        matches, status_msg, faces1, faces2, similarity_matrix = await compare_two_images(
            str(image1_path), str(image2_path), request_id
        )

        total_time = time.time() - start_time

        if matches and len(matches) > 0:
            best_match = matches[0]
            match_percentage = best_match['match_percentage']
            
            # ── DB: insert success ────────────────────────────────
            _db_insert_compare_success(
                request_id    = request_id,
                image1        = image1.filename,
                image2        = image2.filename,
                result        = "match",
                match_percentage = round(float(match_percentage), 2),
                execution_time   = round(total_time, 3),
            )

            return JSONResponse({
                "request_id":       str(request_id),   # ← added
                "result":           "match",
                "match_percentage": round(float(match_percentage), 2),
            })
        else:
            if similarity_matrix is not None and similarity_matrix.size > 0:
                max_similarity   = float(np.max(similarity_matrix))
                match_percentage = round(((max_similarity + 1) / 2) * 100, 2)
            else:
                match_percentage = 0.0

            # ── DB: insert success (no match) ─────────────────────
            _db_insert_compare_success(
                request_id       = request_id,
                image1           = image1.filename,
                image2           = image2.filename,
                result           = "no match",
                match_percentage = match_percentage,
                execution_time   = round(total_time, 3),
            )

            return JSONResponse({
                "request_id":       str(request_id),   # ← added
                "result":           "no match",
                "match_percentage": match_percentage,
            })

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] PROCESSING ERROR after {total_time:.2f}s: {str(e)}", exc_info=True)
        
        # ── DB: insert error ──────────────────────────────────────
        _db_insert_compare_error(request_id, image1.filename, image2.filename, str(e))
        
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing error: {str(e)}", "request_id": str(request_id)}
        )

    finally:
        await asyncio.to_thread(cleanup_file, image1_path)
        await asyncio.to_thread(cleanup_file, image2_path)

        
# --- 1:N --- #
@app.post('/search')
async def search(image: UploadFile = File(...), top_k: int = 3):
    """Search for similar faces and return actual image files as a zip."""
    request_id = uuid.uuid4().hex[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] === FACE SEARCH REQUEST START ===")
    logger.info(f"[{request_id}] Query image: '{image.filename}' | Top-K: {top_k}")

    try:
        results = await asyncio.get_event_loop().run_in_executor(
            None, search_vector, image, request_id, top_k
        )
        search_time = time.time() - start_time

        if not results or len(results) == 0:
            return JSONResponse(status_code=200, content={"message": "No similar faces found", "status": "no results"})

        zip_buffer = io.BytesIO()
        files_added = 0
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, result in enumerate(results):
                image_path = result['image_path']
                if os.path.exists(image_path):
                    filename = f"{i+1}_score_{result['score']:.3f}_{os.path.basename(image_path)}"
                    zip_file.write(image_path, filename)
                    files_added += 1
                else:
                    logger.warning(f"[{request_id}] Missing file: {image_path}")

        # Save zip to disk for preview endpoint
        zip_path = SEARCH_RESULTS_DIR / f"{request_id}.zip"
        with open(zip_path, "wb") as f:
            f.write(zip_buffer.getvalue())
        logger.info(f"[{request_id}] Zip saved to disk: {zip_path}")

        # Save result metadata (image paths + scores)
        import json
        meta_path = SEARCH_RESULTS_DIR / f"{request_id}.json"
        with open(meta_path, "w") as f:
            json.dump(results, f)

        zip_buffer.seek(0)
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] SEARCH COMPLETED - {files_added}/{len(results)} files | Total: {total_time:.2f}s")

        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=search_results_{request_id}.zip",
                "X-Search-Time": f"{search_time:.2f}s",
                "X-Results-Count": str(len(results)),
                "X-Search-ID": request_id
            }
        )
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] SEARCH ERROR after {total_time:.2f}s: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e), "status": "failed"})

@app.post('/search/preview')
async def search_preview(search_id: str, top_k: int = 3):
    """Return base64 images from a previous search result."""
    request_id = uuid.uuid4().hex[:8]
    import json, time as _time

    logger.info(f"[{request_id}] === SEARCH PREVIEW REQUEST === search_id={search_id} | top_k={top_k}")

    # Cleanup zips older than configuration TTL
    now = _time.time()
    for f in SEARCH_RESULTS_DIR.iterdir():
        if now - f.stat().st_mtime > settings.SEARCH_ID_TTL_SECONDS:
            f.unlink()
            logger.info(f"Deleted expired file: {f.name}")

    meta_path = SEARCH_RESULTS_DIR / f"{search_id}.json"
    if not meta_path.exists():
        return JSONResponse(status_code=404, content={"error": "Search ID not found or expired.", "search_id": search_id})

    with open(meta_path, "r") as f:
        results = json.load(f)

    available = len(results)
    to_show = min(top_k, available)

    response_results = []
    for i, result in enumerate(results[:to_show]):
        image_path = result['image_path']
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            response_results.append({
                "rank": i + 1,
                "score": round(result['score'], 3),
                "filename": os.path.basename(image_path),
                "image_base64": img_base64
            })
        else:
            logger.warning(f"[{request_id}] Missing file: {image_path}")

    return JSONResponse({
        "search_id": search_id,
        "requested": top_k,
        "available": available,
        "returned": len(response_results),
        "message": f"Requested {top_k}, available {available}, returning {len(response_results)}.",
        "results": response_results
    })

# --- Upload --- #
@app.post('/bulk_upload')
def bulk_upload(background_tasks: BackgroundTasks, folder_path: str = Form(...), max_workers: int = Form(4)):
    """Start background processing of images and return batch_id immediately."""

    logger.info(f"=== BULK UPLOAD REQUEST START === Folder: '{folder_path}' | Workers: {max_workers}")

    try:
        batch_id = create_batch_job(folder_path=folder_path, max_workers=max_workers)
        logger.info(f"[{batch_id}] Batch job created successfully")

        background_tasks.add_task(process_and_upload, folder_path, batch_id, max_workers)
        logger.info(f"[{batch_id}] Background processing started")

        return JSONResponse({
            "message":  "Upload started in background",
            "batch_id": batch_id,
            "status":   "pending"
        })
    except Exception as e:
        logger.error(f"Failed to start bulk upload: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post('/single_upload')
async def single_upload(image: UploadFile = File(...)):
    """
    Process and upload a single image immediately.

    The returned request_id is a 16-digit timestamp integer (same format as
    bulk_upload batch_ids) so that all IDs in the system are consistent and
    the job appears in GET /jobs.
    """
    request_id = "unknown"
    start_time = time.time()
    image_path = None

    try:
        request_id = create_single_upload_job(image.filename)
        logger.info(f"[{request_id}] === SINGLE UPLOAD REQUEST START ===")
        logger.info(f"[{request_id}] Image: '{image.filename}'")

        image_path = TEMP_DIR / f"single_{request_id}_{image.filename}"


        content = await image.read()
        with open(image_path, "wb") as f:
            f.write(content)
        file_size = image_path.stat().st_size
        logger.info(f"[{request_id}] File saved: {file_size:,} bytes")

        UPLOAD_DIR = Path("uploads") / request_id
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        process_start = time.time()
        point = await asyncio.get_event_loop().run_in_executor(
            None, process_single_image, str(image_path), UPLOAD_DIR, request_id
        )
        process_time = time.time() - process_start

        if point is None:
            logger.error(f"[{request_id}] Failed to process image — no face detected")
            complete_single_upload_job(request_id, success=False, error_msg="No face detected in image")
            return JSONResponse(
                status_code=400,
                content={"error": "No face detected in image", "request_id": request_id}
            )

        upload_start = time.time()
        await asyncio.get_event_loop().run_in_executor(
            None, client.upsert, COLLECTION_NAME, [point]
        )
        upload_time = time.time() - upload_start

        complete_single_upload_job(request_id, success=True)

        total_time = time.time() - start_time
        logger.info(f"[{request_id}] SUCCESS - Process: {process_time:.2f}s | Upload: {upload_time:.2f}s | Total: {total_time:.2f}s")

        return JSONResponse({
            "message":    "Image processed and uploaded successfully",
            "request_id": request_id,
            "status":     "completed"
        })
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] SINGLE UPLOAD ERROR after {total_time:.2f}s: {str(e)}", exc_info=True)
        if request_id != "unknown":
            complete_single_upload_job(request_id, success=False, error_msg=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "status": "failed"})
    finally:
        await asyncio.to_thread(cleanup_file, image_path)
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] === SINGLE UPLOAD COMPLETED in {total_time:.2f}s ===")


# --- Status --- #
@app.get('/batch/{batch_id}/status')
def get_batch_status(batch_id: str):
    """Get the status of a batch processing job."""
    logger.debug(f"[{batch_id}] Batch status requested")

    job = get_batch_job(batch_id)

    if not job:
        logger.warning(f"[{batch_id}] Batch not found")
        return JSONResponse(
            status_code=404,
            content={"error": "Batch not found", "batch_id": batch_id}
        )

    response = {
        "batch_id":          job.batch_id,
        "status":            job.status,
        "folder_path":       job.folder_path,
        "created_at":        job.created_at.isoformat() if job.created_at else None,
        "total_images":      job.total_images,
        "processed_images":  job.processed_images,
    }

    if job.started_at:
        response["started_at"] = job.started_at.isoformat()
    if job.completed_at:
        response["completed_at"] = job.completed_at.isoformat()
    if job.error_message:
        response["error_message"] = job.error_message

    if job.total_images > 0:
        response["progress_percentage"] = round(
            (job.processed_images / job.total_images) * 100, 2
        )
    else:
        response["progress_percentage"] = 0

    return JSONResponse(response)


@app.get('/batches')
def get_all_batches():
    """Get all batch processing jobs."""
    logger.debug("All batches status requested")
    jobs = get_all_batch_jobs()
    logger.debug(f"Retrieved {len(jobs)} batch jobs")

    batches = []
    for job in jobs:
        batch_data = {
            "batch_id":         job.batch_id,
            "status":           job.status,
            "folder_path":      job.folder_path,
            "max_workers":      job.max_workers,
            "created_at":       job.created_at.isoformat() if job.created_at else None,
            "total_images":     job.total_images,
            "processed_images": job.processed_images,
        }

        if job.started_at:
            batch_data["started_at"] = job.started_at.isoformat()
        if job.completed_at:
            batch_data["completed_at"] = job.completed_at.isoformat()
        if job.error_message:
            batch_data["error_message"] = job.error_message

        if job.total_images > 0:
            batch_data["progress_percentage"] = round(
                (job.processed_images / job.total_images) * 100, 2
            )
        else:
            batch_data["progress_percentage"] = 0

        batches.append(batch_data)

    logger.debug(f"Returning {len(batches)} batch jobs")
    return JSONResponse({"batches": batches, "total_count": len(batches)})


# --- Cancel --- #
@app.post('/batch/{batch_id}/cancel')
def cancel_batch(batch_id: str):
    """Cancel a pending or in-progress batch job."""
    logger.info(f"[{batch_id}] Cancel requested")

    result = cancel_batch_job(batch_id)

    if "error" in result:
        error_map = {
            "not_found":         (404, "Batch not found."),
            "already_completed": (400, "Cannot cancel — job already completed."),
            "already_failed":    (400, "Cannot cancel — job already failed."),
            "already_cancelled": (400, "Job is already cancelled."),
        }
        code, msg = error_map.get(result["error"], (400, "Cannot cancel job."))
        return JSONResponse(status_code=code, content={"error": msg, "batch_id": batch_id})

    return JSONResponse({
        "batch_id":        batch_id,
        "status":          "cancelled",
        "message":         "Job cancelled successfully.",
    })