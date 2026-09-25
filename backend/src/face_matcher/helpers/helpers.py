import os
import time
import uuid
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import torch
import asyncio
import logging
from pathlib import Path
from skimage import transform as trans
from qdrant_client import models
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from face_matcher.helpers.pre_process import (
    add_padding_to_image, denoise_image_if_needed,
    preprocess_image_for_model, adjust_coordinates_after_padding,
    scale_landmark_template, read_image_any_format,
    normalize_image_orientation
)

from face_matcher.components.adaface_config import load_adaface_model
from face_matcher.components.yunet_config import detect_faces, get_thread_detector
from face_matcher.helpers.constants import (TEMP_DIR, FACE_SIZE, PADDING_RATIO, 
                                    MODEL_PATH, MATCH_THRESHOLD, COLLECTION_NAME)



logger = logging.getLogger(__name__)

def _get_client():
    """Get the Qdrant client singleton (initialized during lifespan)."""
    from face_matcher.components.qdrant_config import get_qdrant_client
    return get_qdrant_client()

adaface_model = load_adaface_model(MODEL_PATH)


# --- 1:1 --- #

# Step 1: Extract and align faces (compare)
def extract_faces(image_path: str = None, image: np.ndarray = None, target_size: tuple = FACE_SIZE, request_id: str = None) -> list:
    """Extract and align all faces detected in an image.
    
    Accepts either a file path or a pre-loaded NumPy array (BGR format).
    If both are provided, the NumPy array takes priority.
    
    Args:
        image_path: Path to image file (used if `image` is None).
        image: Pre-loaded image as NumPy array in BGR format (skips file loading).
        target_size: Output face size for alignment.
        request_id: Optional request ID for logging.
    
    Returns:
        List of dicts with keys: face_id, aligned_face, bbox, landmarks, confidence.
    """
    req_prefix = f"[{request_id}] " if request_id else ""
    img_name = Path(image_path).name if image_path else "in-memory"
    logger.debug(f"{req_prefix}Processing {img_name}: loading and detecting faces...")
    
    try:
        # Load image from disk or use provided array
        if image is not None:
            loaded_image = image
        elif image_path:
            loaded_image = read_image_any_format(image_path)
        else:
            logger.error(f"{req_prefix}No image_path or image array provided")
            return []
            
        if loaded_image is None:
            logger.error(f"{req_prefix}Failed to load image: {img_name}")
            return []

        logger.debug(f"{req_prefix}{img_name}: image loaded {loaded_image.shape}, converting to RGB and detecting...")
        
        # If image was loaded from disk, it's BGR → convert to RGB
        # If image was passed as array, caller is responsible for format,
        # but the in-memory path from normalize_and_extract_faces passes RGB already
        if image is not None:
            # Caller provides RGB (from normalize_image_orientation)
            image_rgb = loaded_image
        else:
            image_rgb = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
        
        detected_faces = detect_faces(get_thread_detector(), image_rgb)
        pad_height = pad_width = 0

        # Try with padding if no faces detected initially
        if len(detected_faces) == 0:
            logger.info(f"{req_prefix}{img_name}: No faces detected initially, trying with {PADDING_RATIO*100}% padding...")
            padded_image, pad_height, pad_width = add_padding_to_image(image_rgb, PADDING_RATIO)
            detected_faces = detect_faces(get_thread_detector(), padded_image)
            if len(detected_faces) == 0:
                logger.warning(f"{req_prefix}{img_name}: No faces detected even with padding")
                return []
            logger.info(f"{req_prefix}{img_name}: Found {len(detected_faces)} face(s) after padding")
        else:
            logger.info(f"{req_prefix}{img_name}: Found {len(detected_faces)} face(s) directly")

        aligned_faces = []
        landmark_template = scale_landmark_template(target_size)
        logger.debug(f"{req_prefix}{img_name}: Aligning {len(detected_faces)} faces to {target_size}...")

        for face_idx, face in enumerate(detected_faces):
            landmarks = face.kps.copy()
            bbox = face.bbox.copy()
            confidence = face.det_score
            
            logger.debug(f"{req_prefix}{img_name}: Face {face_idx} - confidence={confidence:.3f}, bbox={bbox[:4].astype(int)}")
            
            # Adjust coordinates if padding was applied
            if pad_height > 0 or pad_width > 0:
                landmarks, bbox = adjust_coordinates_after_padding(landmarks, bbox, pad_height, pad_width)
                logger.debug(f"{req_prefix}{img_name}: Face {face_idx} coordinates adjusted for padding")
            
            # Compute alignment transformation
            transform = trans.SimilarityTransform()
            transform.estimate(landmarks, landmark_template)
            transform_matrix = transform.params[0:2, :]

            # Apply alignment to original image
            aligned_face = cv2.warpAffine(image_rgb, transform_matrix, target_size, borderValue=0.0)
            
            aligned_faces.append({
                'face_id': face_idx,
                'aligned_face': aligned_face,
                'bbox': bbox,
                'landmarks': landmarks,
                'confidence': confidence
            })

        logger.info(f"{req_prefix}{img_name}: Successfully aligned {len(aligned_faces)} faces")
        return aligned_faces

    except Exception as e:
        logger.error(f"{req_prefix}{img_name}: Error extracting faces - {str(e)}")
        return []


def normalize_and_extract_faces(image_path: str, target_size: tuple = FACE_SIZE, request_id: str = None) -> list:
    """Normalize image orientation then extract faces — the 1:1 compare entry point.
    
    Pipeline:
        1. Load image from disk
        2. Normalize orientation (coarse-to-fine rotation search)
        3. Extract and align faces from the upright image (in memory, no temp files)
    
    Args:
        image_path: Path to the image file.
        target_size: Output face size for alignment.
        request_id: Optional request ID for logging.
    
    Returns:
        List of aligned face dicts (same format as extract_faces).
    """
    req_prefix = f"[{request_id}] " if request_id else ""
    img_name = Path(image_path).name
    logger.info(f"{req_prefix}normalize_and_extract_faces: starting for {img_name}")
    
    try:
        # 1. Load image
        loaded_image = read_image_any_format(image_path)
        if loaded_image is None:
            logger.error(f"{req_prefix}Failed to load image: {img_name}")
            return []
        
        image_rgb = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
        
        # 2. Normalize orientation (returns RGB image)
        upright_rgb, best_angle, confidence = normalize_image_orientation(image_rgb, request_id)
        logger.info(f"{req_prefix}{img_name}: orientation normalized to {best_angle}° (confidence={confidence:.3f})")
        
        # 3. Extract faces from the upright image (pass as RGB array, skip file loading)
        faces = extract_faces(image=upright_rgb, target_size=target_size, request_id=request_id)
        logger.info(f"{req_prefix}{img_name}: extracted {len(faces)} faces after orientation normalization")
        return faces
        
    except Exception as e:
        logger.error(f"{req_prefix}{img_name}: Error in normalize_and_extract_faces - {str(e)}")
        return []


# Step 2: Compute embeddings (compare)
def get_embeddings(aligned_faces: list, request_id: str = None) -> list:
    """Extract feature embeddings for aligned faces.
    
    Uses torch.no_grad() to disable gradient tracking (saves ~15-30% CPU time)
    and batches multiple faces into a single forward pass when possible.
    """
    req_prefix = f"[{request_id}] " if request_id else ""
    logger.debug(f"{req_prefix}Computing embeddings for {len(aligned_faces)} faces")
    embeddings = []
    with torch.no_grad():
        if len(aligned_faces) > 1:
            # Batch all faces into a single tensor for one forward pass
            batch_tensors = [preprocess_image_for_model(fd['aligned_face']) for fd in aligned_faces]
            batch = torch.cat(batch_tensors, dim=0)  # (N, 3, 112, 112)
            batch_embeddings, _ = adaface_model(batch)
            for i in range(len(aligned_faces)):
                embedding = batch_embeddings[i:i+1]  # Keep 2D shape (1, 512)
                embeddings.append(embedding)
                logger.debug(f"{req_prefix}Embedding {i}: shape={embedding.shape}, confidence={aligned_faces[i]['confidence']:.3f}")
        else:
            for i, face_data in enumerate(aligned_faces):
                aligned_face = face_data['aligned_face']
                embedding, _ = adaface_model(preprocess_image_for_model(aligned_face))
                embeddings.append(embedding)
                logger.debug(f"{req_prefix}Embedding {i}: shape={embedding.shape}, confidence={face_data['confidence']:.3f}")
    return embeddings

# Step 3: Compare faces and find matches (compare)
def find_matches(faces1: list, faces2: list, embeddings1: list, embeddings2: list, threshold: float = MATCH_THRESHOLD, request_id: str = None) -> tuple:
    """Find matching face pairs based on similarity threshold."""
    req_prefix = f"[{request_id}] " if request_id else ""
    if not faces1 or not faces2 or not embeddings1 or not embeddings2:
        logger.warning(f"{req_prefix}Empty inputs for face matching")
        return [], None
    
    logger.debug(f"{req_prefix}Computing similarity matrix: {len(embeddings1)}x{len(embeddings2)}")
    # Compute similarity matrix
    embeddings1_tensor = torch.cat(embeddings1, dim=0)
    embeddings2_tensor = torch.cat(embeddings2, dim=0)
    similarity_matrix = torch.mm(embeddings1_tensor, embeddings2_tensor.t())
    similarity_array = similarity_matrix.detach().numpy()
    
    logger.debug(f"{req_prefix}Similarity matrix shape: {similarity_array.shape}, max={similarity_array.max():.4f}, min={similarity_array.min():.4f}")
    
    
    # Find matches above threshold
    matches = []
    for i in range(len(faces1)):
        for j in range(len(faces2)):
            similarity_score = similarity_array[i, j]
            match_percentage = round(((similarity_score + 1) / 2) * 100, 2) 
            if similarity_score > threshold:
                matches.append({
                    'face1_id': i,
                    'face2_id': j,
                    'similarity_score': float(similarity_score),
                    'match_percentage': match_percentage,
                    'face1_info': faces1[i],
                    'face2_info': faces2[j]
                })
    
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    logger.debug(f"{req_prefix}Found {len(matches)} matches above threshold {threshold}")
    return matches, similarity_array

# 1:1
async def compare_two_images(image1_path: str, image2_path: str, request_id: str = None, threshold: float = MATCH_THRESHOLD) -> tuple:
    """Compare two images with full 360° rotation support.
    
    Strategy (Normalize-then-Match):
        1. Normalize each image's orientation independently (coarse-to-fine search)
        2. Extract faces from both upright images
        3. Generate embeddings once
        4. Match once
    
    No temp files are created. Rotation search uses only lightweight face detection.
    Embeddings are generated exactly once per image.
    """
    req_prefix = f"[{request_id}] " if request_id else ""
    logger.info(f"{req_prefix}Starting face comparison: {Path(image1_path).name} vs {Path(image2_path).name}, threshold={threshold}")
    start_time = time.time()

    # Step 1: Normalize orientations + extract faces (both images in parallel)
    logger.info(f"{req_prefix}Step 1: Normalizing orientations and extracting faces...")
    faces1, faces2 = await asyncio.gather(
        asyncio.to_thread(normalize_and_extract_faces, image1_path, FACE_SIZE, request_id),
        asyncio.to_thread(normalize_and_extract_faces, image2_path, FACE_SIZE, request_id)
    )

    if not faces1 or not faces2:
        total_time = time.time() - start_time
        logger.warning(f"{req_prefix}No faces detected: img1={len(faces1) if faces1 else 0}, img2={len(faces2) if faces2 else 0} (total: {total_time:.1f}s)")
        return None, "No faces detected", None, None, None

    # Step 2: Generate embeddings (once per image, in parallel)
    logger.info(f"{req_prefix}Step 2: Generating embeddings ({len(faces1)} + {len(faces2)} faces)...")
    embeddings1, embeddings2 = await asyncio.gather(
        asyncio.to_thread(get_embeddings, faces1, request_id),
        asyncio.to_thread(get_embeddings, faces2, request_id)
    )

    # Step 3: Match (once)
    logger.info(f"{req_prefix}Step 3: Computing similarity...")
    matches, similarity_matrix = find_matches(faces1, faces2, embeddings1, embeddings2, threshold, request_id)
    max_sim = max([m['similarity_score'] for m in matches]) if matches else 0.0

    total_time = time.time() - start_time
    logger.info(f"{req_prefix}RESULT: max_similarity={max_sim:.4f}, {len(matches)} matches (total: {total_time:.1f}s)")
    if matches:
        for match in matches[:3]:
            logger.info(f"{req_prefix}Match: face1_id={match['face1_id']}, face2_id={match['face2_id']}, similarity={match['similarity_score']:.4f}, percentage={match['match_percentage']:.1f}%")

    status_msg = f"{len(matches)} match(es), max similarity: {max_sim:.3f}"
    return matches, status_msg, faces1, faces2, similarity_matrix


# --- 1:N --- #

# Step 1: Extract and align faces (search)
def extract_face_v2(image_path: str, target_size: tuple = FACE_SIZE, request_id: str = None) -> dict:
    """Extract and align the face with highest confidence detected in an image.
    
    Tries detection without padding first; falls back to padding if no faces found.
    """
    req_prefix = f"[{request_id}] " if request_id else ""
    img_name = Path(image_path).name
    
    try:
        image = read_image_any_format(image_path)

        if image is None:
            logger.error(f"{req_prefix}Failed to load image: {img_name}")
            return {}

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Try detection without padding first (saves creating a 69% larger image)
        detected_faces = detect_faces(get_thread_detector(), image_rgb)
        pad_height = pad_width = 0
        
        # Fall back to padding if no faces detected
        if len(detected_faces) == 0:
            logger.info(f"{req_prefix}{img_name}: No faces without padding, trying with {PADDING_RATIO*100}% padding...")
            padded_image, pad_height, pad_width = add_padding_to_image(image_rgb, PADDING_RATIO)
            detected_faces = detect_faces(get_thread_detector(), padded_image)
        
        if len(detected_faces) == 0:
            logger.warning(f"{req_prefix}{img_name}: No faces detected")
            return {}

        # Select face with highest confidence score
        best_face = max(detected_faces, key=lambda face: face.det_score)
        confidence = best_face.det_score
        
        landmark_template = scale_landmark_template(target_size)
        landmarks = best_face.kps.copy()
        bbox = best_face.bbox.copy()
        
        # Adjust coordinates for padding
        landmarks, bbox = adjust_coordinates_after_padding(landmarks, bbox, pad_height, pad_width)
        
        # Compute alignment transformation
        transform = trans.SimilarityTransform()
        transform.estimate(landmarks, landmark_template)
        transform_matrix = transform.params[0:2, :]

        # Apply alignment to original image
        aligned_face = cv2.warpAffine(image_rgb, transform_matrix, target_size, borderValue=0.0)
        
        return {
            'aligned_face': aligned_face,
            'bbox': bbox,
            'landmarks': landmarks,
            'confidence': confidence
        }

    except Exception as e:
        logger.error(f"{req_prefix}{img_name}: Error extracting faces - {str(e)}")
        return {}

# Step 2: Compute embeddings (search)
def compute_face_embeddings_v2(face_data: dict, request_id: str = None) -> np.ndarray:
    """Extract feature embedding for an aligned face.
    
    Uses torch.no_grad() to disable gradient tracking during inference.
    
    Args:
        face_data: Dictionary containing 'aligned_face', 'bbox', 'landmarks', and 'confidence'
        request_id: Optional request ID for logging
        
    Returns:
        Face embedding array, or None if face_data is empty
    """
    req_prefix = f"[{request_id}] " if request_id else ""
    
    if not face_data:
        logger.warning(f"{req_prefix}No face data provided for embedding computation")
        return None
    
    try:
        aligned_face = face_data['aligned_face']
        confidence = face_data['confidence']
        
        logger.info(f"{req_prefix}Computing embedding for face with confidence={confidence:.3f}")
        
        with torch.no_grad():
            embedding, _ = adaface_model(preprocess_image_for_model(aligned_face))
        
        logger.info(f"{req_prefix}Embedding computed: shape={embedding.shape}, confidence={confidence:.3f}")
        
        return embedding
        
    except Exception as e:
        logger.error(f"{req_prefix}Error computing embedding: {str(e)}")
        return None

# 1:N
# def search_vector(image, request_id, top_k):
#     """Search for similar images in the Qdrant vector database."""
#     #read image and save it
#     file_path = os.path.join(TEMP_DIR, image.filename)
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(image.file, buffer)
        
#     face = extract_face_v2(file_path, request_id=request_id)
#     img_embeddings = compute_face_embeddings_v2(face, request_id=request_id).tolist()[0]
    
#     hits = client.query_points(
#         collection_name=f"{COLLECTION_NAME}",
#         query=img_embeddings,
#         limit=top_k,
#         score_threshold=0.5
#     )

#     images = []
#     for hit in hits:
#         path = hit.payload['image_path']
#         images.append({
#             "id": hit.id,
#             "image_path": path,
#             "score": hit.score
#         })
#     if os.path.exists(file_path):
#         os.remove(file_path)
#     return images

def search_vector(image, request_id, top_k):
    """Search for similar images in the Qdrant vector database."""
    #read image and save it
    file_path = os.path.join(TEMP_DIR, image.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
        
    # denoised_img = denoise_image_if_needed(str(file_path), request_id)
    
    # if denoised_img is not None:
    #     cv2.imwrite(str(file_path), denoised_img)
    #     logger.info(f"[{request_id}] Image1 denoised - queued for saving")
    # else:
    #     logger.debug(f"[{request_id}] Image1 - no noise detected")
        
    face = extract_face_v2(file_path, request_id=request_id)
    img_embeddings_tensor = compute_face_embeddings_v2(face, request_id=request_id)
    if img_embeddings_tensor is None:
        logger.error(f"[{request_id}] Could not compute embeddings for query image")
        return []
    
    img_embeddings = img_embeddings_tensor.tolist()[0]
    
    hits = _get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=img_embeddings,
        limit=top_k,
        score_threshold=0.5
    ).points

    images = []
    for hit in hits:
        path = hit.payload['image_path']
        images.append({
            "id": hit.id,
            "image_path": path,
            "score": hit.score
        })
    if os.path.exists(file_path):
        os.remove(file_path)
    return images


# --- Upload --- #

def process_single_image(file_path: str, upload_dir: Path, request_id: str) -> models.PointStruct:
    """Process a single image and return a PointStruct for batch upload."""
    try:
        # Copy file to upload directory
        dest_path = shutil.copy(file_path, upload_dir / os.path.basename(file_path))
        
        # Extract face and compute embeddings
        face = extract_face_v2(file_path, request_id=request_id)
        if not face:
            logger.warning(f"No face detected in {file_path}")
            return None
            
        embeddings = compute_face_embeddings_v2(face, request_id)
        if embeddings is None:
            logger.warning(f"Failed to compute embeddings for {file_path}")
            return None
        
        return models.PointStruct(
            id=str(uuid.uuid4()),
            payload={"image_path": dest_path},
            vector=embeddings.tolist()[0],
        )
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

def process_and_upload(folder_path: str, batch_id: str, max_workers: int = 4):
    """Process images concurrently and save them to the Qdrant vector database."""
    from ..database.database import update_batch_status, get_batch_job
    
    from face_matcher import settings
    start = time.time()
    UPLOAD_DIR = settings.UPLOAD_DIR / batch_id
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic'}
    
    BATCH_SIZE = settings.BATCH_SIZE
    UPDATE_INTERVAL = settings.UPDATE_INTERVAL
    
    try:
        update_batch_status(batch_id, 'processing')
        
        image_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        update_batch_status(batch_id, 'processing', total_images=len(image_files))
        logger.info(f"Found {len(image_files)} images to process with {max_workers} workers")
        
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            process_func = partial(process_single_image, upload_dir=UPLOAD_DIR, request_id=batch_id)
            
            for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Processing batches"):
                # ✅ Check once per batch before submitting any work
                job = get_batch_job(batch_id)
                if job and job.status == "cancelled":
                    logger.info(f"[{batch_id}] Cancelled — stopping before batch {i}")
                    update_batch_status(batch_id, 'cancelled', processed_images=processed_count)
                    return
                batch_files = image_files[i:i + BATCH_SIZE]
                
                futures = [executor.submit(process_func, file_path) for file_path in batch_files]
                
                batch_points = []
                for future in futures:
                    try:
                        # ✅ Check DB before each image
                        job = get_batch_job(batch_id)
                        if job and job.status == "cancelled":
                            logger.info(f"[{batch_id}] Cancelled — stopping before next image")
                            update_batch_status(batch_id, 'cancelled', processed_images=processed_count)
                            executor.shutdown(wait=False, cancel_futures=True)
                            return

                        result = future.result(timeout=30)
                        if result:
                            batch_points.append(result)
                            processed_count += 1
                            
                            if processed_count % UPDATE_INTERVAL == 0:
                                update_batch_status(batch_id, 'processing', processed_images=processed_count)
                    except Exception as e:
                        logger.error(f"Error processing image: {str(e)}")
                
                if batch_points:
                    _get_client().upsert(collection_name=COLLECTION_NAME, points=batch_points)
                    logger.info(f"Uploaded batch of {len(batch_points)} images")
        
        total = time.time() - start
        logger.info(f"Processed {processed_count}/{len(image_files)} images in {total:.2f} seconds")
        
        update_batch_status(batch_id, 'completed', processed_images=processed_count)
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        update_batch_status(batch_id, 'failed', error_message=str(e))
        raise
 
# --- evaluation --- #

def compare_two_images_eval(image1_path: str, image2_path: str, request_id: str = None, threshold: float = MATCH_THRESHOLD) -> tuple:
    """Compare two images directly and return the result."""
    req_prefix = f"[{request_id}] " if request_id else ""
    logger.info(f"{req_prefix}Starting face comparison: {Path(image1_path).name} vs {Path(image2_path).name}, threshold={threshold}")
    
    import time
    start_time = time.time()
    
    # Extract faces from images
    faces1 = extract_faces(image1_path, request_id=request_id)
    faces2 = extract_faces(image2_path, request_id=request_id)
    
    if not faces1 or not faces2:
        logger.warning(f"{req_prefix}No faces detected: faces1={len(faces1) if faces1 else 0}, faces2={len(faces2) if faces2 else 0}")
        return None, "No faces detected", None, None, None
    
    logger.info(f"{req_prefix}Computing embeddings: {len(faces1)} vs {len(faces2)} faces")
    embeddings1 = get_embeddings(faces1, request_id)
    embeddings2 = get_embeddings(faces2, request_id)
    
    matches, similarity_matrix = find_matches(faces1, faces2, embeddings1, embeddings2, threshold, request_id)
    max_sim = max([m['similarity_score'] for m in matches]) if matches else 0.0
    
    total_time = time.time() - start_time
    logger.info(f"{req_prefix}Comparison complete: {len(matches)} matches, max_similarity={max_sim:.4f} (total: {total_time:.1f}s)")
    
    if matches:
        for match in matches[:3]:  # Log top 3 matches
            logger.info(f"{req_prefix}Match: face1_id={match['face1_id']}, face2_id={match['face2_id']}, similarity={match['similarity_score']:.4f}, percentage={match['match_percentage']:.1f}%")
    
    return matches, f"{len(matches)} match(es), max similarity: {max_sim:.3f}", faces1, faces2, similarity_matrix


# --- Cleaning --- #

def cleanup_file(file_path: Path) -> None:
    """Safely delete a file."""
    try:
        if file_path and file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to delete {file_path}: {str(e)}")