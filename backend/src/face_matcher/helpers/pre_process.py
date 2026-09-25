import cv2
import torch
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from face_matcher.helpers.constants import TEMP_DIR, FACE_SIZE, FACE_LANDMARK_TEMPLATE
from face_matcher.components.yunet_config import get_thread_detector

logger = logging.getLogger(__name__)

def read_image_any_format(image_path: str) -> np.ndarray:
    ext = Path(image_path).suffix.lower()
    if ext == '.heic':
        from pillow_heif import register_heif_opener
        register_heif_opener()
    if ext in ('.heic', '.gif'):
        with Image.open(image_path) as pil_img:
            if ext == '.gif':
                pil_img.seek(0)
            pil_img = pil_img.convert('RGB')
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv2.imread(image_path)


# ── Rotation Helpers ──────────────────────────────────────────────

# Fast lookup for exact 90° multiples → use cv2.rotate (no interpolation, faster)
_CV2_ROTATE_MAP = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by any angle (degrees, clockwise) in memory.

    For exact multiples of 90° uses the faster cv2.rotate().
    For arbitrary angles uses cv2.warpAffine() with border replication.
    Returns the rotated image as a NumPy array. No disk I/O.

    Args:
        image: Input image as NumPy array (BGR or RGB).
        angle: Rotation angle in degrees (0–360, clockwise).

    Returns:
        Rotated image (same dtype as input).
    """
    # Normalize angle to [0, 360)
    angle = angle % 360
    if angle == 0:
        return image

    # Fast path for exact 90° multiples
    int_angle = int(round(angle))
    if int_angle in _CV2_ROTATE_MAP and abs(angle - int_angle) < 0.01:
        return cv2.rotate(image, _CV2_ROTATE_MAP[int_angle])

    # Arbitrary angle: use affine warp
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    # Negative angle because cv2 rotates counter-clockwise by default
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

    # Compute new bounding box size to avoid cropping
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_w = int(h * sin_val + w * cos_val)
    new_h = int(h * cos_val + w * sin_val)

    # Adjust the rotation matrix for the new center
    rotation_matrix[0, 2] += (new_w / 2.0) - center[0]
    rotation_matrix[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return rotated


# ── Coarse-to-Fine Orientation Normalization (Legacy constants, kept for compat) ─
_CARDINAL_ANGLES = [0, 90, 180, 270]
_DIAGONAL_ANGLES = [45, 135, 225, 315]
_FINE_OFFSETS = [-15, -10, -5, 5, 10, 15]
_EARLY_EXIT_CONFIDENCE = 0.85


# ── Hierarchical Beam Search — 360° Orientation Normalization ─────

# Stage definitions: (step_degrees, offsets_per_side, beam_width_out)
_COARSE_ANGLES = list(range(0, 360, 30))        # 12 angles at 30° spacing
_STAGE2_HALF_RANGE = 12                           # ±12° around each candidate
_STAGE2_STEP = 6                                  # 6° steps in Stage 2
_STAGE2_BEAM = 2                                  # keep top-2 after Stage 2
_STAGE3_HALF_RANGE = 4                            # ±4° around each candidate
_STAGE3_STEP = 2                                  # 2° steps in Stage 3
_STAGE4_OFFSETS = [-1.5, -0.75, 0.75, 1.5]       # sub-degree polish
_COARSE_BEAM = 3                                  # beam width after coarse scan
_COMPOSITE_EARLY_EXIT = 0.92                      # early exit on composite score

# Composite score weights — sum to 1.0
_W_DET_SCORE = 0.30          # YuNet detection confidence
_W_EYE_HORIZ = 0.30          # eye-line horizontality (strongest uprightness signal)
_W_SYMMETRY = 0.15           # left-right landmark symmetry
_W_NOSE_MOUTH = 0.15         # nose-mouth vertical alignment
_W_ASPECT = 0.10             # bounding box aspect ratio

# Ideal upright face aspect ratio (height / width) — empirically ~1.2–1.4
_IDEAL_ASPECT_RATIO = 1.30
_ASPECT_RATIO_TOLERANCE = 0.35  # scores drop outside [0.95, 1.65]


def _detect_at_angle(image_rgb: np.ndarray, angle: float, detector):
    """Rotate image and run face detection. Returns (angle, faces_list, confidence).

    'confidence' is the highest detection score among found faces, or 0.0 if none.
    """
    from face_matcher.components.yunet_config import detect_faces
    rotated = rotate_image(image_rgb, angle)
    faces = detect_faces(detector, rotated)
    if faces:
        best_conf = max(f.det_score for f in faces)
        return angle, faces, best_conf, rotated
    return angle, [], 0.0, rotated


def _compute_landmark_quality(face) -> dict:
    """Compute geometric quality sub-scores from a detected face's 5-point landmarks.

    Landmark order (YuNet / InsightFace convention):
        [0] right_eye, [1] left_eye, [2] nose, [3] right_mouth, [4] left_mouth

    Returns a dict with individual sub-scores, each in [0.0, 1.0]:
        - eye_horizontality: how close the eye-line is to perfectly horizontal
        - symmetry:          left-right balance of landmarks around vertical nose axis
        - nose_mouth_vert:   vertical alignment of nose above mouth center
        - aspect_ratio:      how close the bbox aspect ratio is to ideal upright face
    """
    kps = face.kps   # shape (5, 2), float32
    bbox = face.bbox  # [x1, y1, x2, y2], float32

    # ── Eye-line horizontality ──
    # A perfectly upright face has eyes on a horizontal line (angle = 0°).
    right_eye, left_eye = kps[0], kps[1]
    dx_eye = left_eye[0] - right_eye[0]
    dy_eye = left_eye[1] - right_eye[1]
    eye_angle_deg = abs(np.degrees(np.arctan2(dy_eye, dx_eye + 1e-9)))
    # Map: 0° → 1.0, 15° → 0.5, 30° → 0.0 (clamped)
    eye_horizontality = float(np.clip(1.0 - (eye_angle_deg / 30.0), 0.0, 1.0))

    # ── Landmark symmetry ──
    # Measure how well left-side and right-side landmarks mirror around the nose x-axis.
    nose = kps[2]
    right_mouth, left_mouth = kps[3], kps[4]

    # Horizontal distances from nose center
    d_right_eye = abs(right_eye[0] - nose[0])
    d_left_eye = abs(left_eye[0] - nose[0])
    d_right_mouth = abs(right_mouth[0] - nose[0])
    d_left_mouth = abs(left_mouth[0] - nose[0])

    # Symmetry ratio: perfect symmetry → 1.0
    eye_sym = min(d_right_eye, d_left_eye) / (max(d_right_eye, d_left_eye) + 1e-6)
    mouth_sym = min(d_right_mouth, d_left_mouth) / (max(d_right_mouth, d_left_mouth) + 1e-6)
    symmetry = float((eye_sym + mouth_sym) / 2.0)

    # ── Nose-mouth vertical alignment ──
    # In an upright face: nose.y < mouth_center.y (nose is ABOVE mouth),
    # and nose.x ≈ mouth_center.x (horizontally centered).
    mouth_center_x = (right_mouth[0] + left_mouth[0]) / 2.0
    mouth_center_y = (right_mouth[1] + left_mouth[1]) / 2.0
    eye_center_y = (right_eye[1] + left_eye[1]) / 2.0

    # Vertical ordering: nose should be between eyes and mouth
    face_height = abs(mouth_center_y - eye_center_y) + 1e-6
    nose_vert_frac = (nose[1] - eye_center_y) / face_height  # ideal ≈ 0.4–0.6
    vert_order_ok = 1.0 if (0.2 < nose_vert_frac < 0.8) else 0.3

    # Horizontal centering of nose relative to mouth
    horiz_offset = abs(nose[0] - mouth_center_x) / (face_height + 1e-6)
    horiz_center_score = float(np.clip(1.0 - horiz_offset * 3.0, 0.0, 1.0))

    nose_mouth_vert = float(vert_order_ok * horiz_center_score)

    # ── Bounding box aspect ratio ──
    bw = abs(bbox[2] - bbox[0])
    bh = abs(bbox[3] - bbox[1])
    if bw < 1 or bh < 1:
        aspect_score = 0.0
    else:
        aspect = bh / bw
        aspect_dev = abs(aspect - _IDEAL_ASPECT_RATIO) / _ASPECT_RATIO_TOLERANCE
        aspect_score = float(np.clip(1.0 - aspect_dev, 0.0, 1.0))

    return {
        'eye_horizontality': eye_horizontality,
        'symmetry': symmetry,
        'nose_mouth_vert': nose_mouth_vert,
        'aspect_ratio': aspect_score,
    }


def _compute_composite_score(det_score: float, landmark_quality: dict) -> float:
    """Weighted combination of detection confidence and geometric landmark quality.

    Returns a single float in [0.0, 1.0].  Higher = more likely to be upright.
    """
    return (
        _W_DET_SCORE  * min(det_score, 1.0)
        + _W_EYE_HORIZ  * landmark_quality['eye_horizontality']
        + _W_SYMMETRY   * landmark_quality['symmetry']
        + _W_NOSE_MOUTH * landmark_quality['nose_mouth_vert']
        + _W_ASPECT     * landmark_quality['aspect_ratio']
    )


def _evaluate_angle(image_rgb: np.ndarray, angle: float, detector, req_prefix: str = ""):
    """Detect faces at a given rotation angle and compute composite score.

    Returns:
        dict with keys: angle, faces, det_score, composite, rotated_image, landmark_quality
        or None if no face detected.
    """
    _, faces, det_score, rotated = _detect_at_angle(image_rgb, angle, detector)
    if not faces:
        return None

    # Score the best-confidence face's geometry
    best_face = max(faces, key=lambda f: f.det_score)
    lq = _compute_landmark_quality(best_face)
    composite = _compute_composite_score(det_score, lq)

    logger.debug(
        f"{req_prefix}Orientation: {angle:>6.1f}° → det={det_score:.3f} "
        f"eye_h={lq['eye_horizontality']:.2f} sym={lq['symmetry']:.2f} "
        f"nm={lq['nose_mouth_vert']:.2f} ar={lq['aspect_ratio']:.2f} "
        f"→ composite={composite:.3f}"
    )
    return {
        'angle': angle,
        'faces': faces,
        'det_score': det_score,
        'composite': composite,
        'rotated_image': rotated,
        'landmark_quality': lq,
    }


def _run_beam_stage(
    image_rgb: np.ndarray,
    detector,
    candidate_angles: list,
    half_range: float,
    step: float,
    beam_width: int,
    seen: set,
    all_results: list,
    req_prefix: str = "",
) -> list:
    """Run one refinement stage of the beam search.

    For each candidate angle, probe [angle - half_range, angle + half_range]
    at the given step size, skipping already-seen angles.  Re-rank all
    accumulated results and return the top beam_width candidates.
    """
    new_angles = []
    for center in candidate_angles:
        offset = -half_range
        while offset <= half_range + 1e-9:
            a = round((center + offset) % 360, 2)
            if a not in seen:
                new_angles.append(a)
                seen.add(a)
            offset += step

    for angle in new_angles:
        result = _evaluate_angle(image_rgb, angle, detector, req_prefix)
        if result is not None:
            all_results.append(result)

    # Sort all accumulated results by composite score descending
    all_results.sort(key=lambda r: r['composite'], reverse=True)
    # Return top beam_width angle values
    return [r['angle'] for r in all_results[:beam_width]]


def _apply_eyeline_correction(
    image_rgb: np.ndarray,
    beam_result: dict,
    req_prefix: str = "",
) -> tuple:
    """Apply analytical eye-line correction on top of a beam search result.

    Uses the eye landmarks detected at the beam search's best angle to
    mathematically calculate the exact residual tilt via arctan2, then
    rotates the ORIGINAL image by the corrected angle in a single step
    (avoiding double-interpolation quality loss).

    Args:
        image_rgb:   The original (un-rotated) input image in RGB format.
        beam_result: The winning result dict from the beam search, containing
                     'angle', 'faces', 'composite', 'det_score', etc.
        req_prefix:  Logging prefix string.

    Returns:
        Tuple of (corrected_image_rgb, final_angle, composite_score).
    """
    beam_angle = beam_result['angle']
    composite = beam_result['composite']

    # Select the highest-confidence face from the beam result
    best_face = max(beam_result['faces'], key=lambda f: f.det_score)

    # Extract eye landmarks (YuNet convention: [0]=right_eye, [1]=left_eye)
    right_eye = best_face.kps[0]
    left_eye = best_face.kps[1]

    # Calculate the residual tilt of the eye-line in the rotated image.
    # A perfectly upright face has eyes on a horizontal line (angle = 0°).
    dx = left_eye[0] - right_eye[0]
    dy = left_eye[1] - right_eye[1]
    residual_tilt = np.degrees(np.arctan2(dy, dx))

    # Subtract the residual tilt from the beam angle to get the true correction.
    # This gives the single rotation angle that makes the face perfectly upright.
    final_angle = (beam_angle - residual_tilt) % 360

    logger.info(
        f"{req_prefix}Eye-line correction: beam_angle={beam_angle:.2f}°, "
        f"residual_tilt={residual_tilt:.2f}°, final_angle={final_angle:.2f}°"
    )

    # Rotate the ORIGINAL image by the corrected angle (single rotation,
    # no double-interpolation since we never use the beam's rotated image).
    corrected_image = rotate_image(image_rgb, final_angle)

    return corrected_image, final_angle, composite


def normalize_image_orientation(image_rgb: np.ndarray, request_id: str = None) -> tuple:
    """Find the best rotation to make faces detectable and perfectly upright.

    Strategy (Hierarchical Beam Search + Analytical Eye-line Correction):
        Stage 1 (Coarse):   Test 12 angles at 30° spacing (full 360° coverage).
                             Early exit if composite score >= 0.92.
                             Keep top-3 candidates (beam).
        Stage 2 (Medium):   Refine ±12° around each beam candidate at 6° steps.
                             Keep top-2 candidates.
        Stage 3 (Fine):     Refine ±4° around each beam candidate at 2° steps.
                             Pick the single best.
        Stage 4 (Polish):   Test ±1.5° and ±0.75° around the best for sub-degree
                             precision.
        Stage 5 (Exact):    Use arctan2 on detected eye landmarks to calculate
                             the exact residual tilt, then apply one final precise
                             rotation to the original image.

    Composite scoring combines YuNet detection confidence with geometric
    landmark analysis (eye-line horizontality, left-right symmetry,
    nose-mouth vertical alignment, bounding box aspect ratio) to
    distinguish truly upright faces from merely detectable tilted ones.

    Args:
        image_rgb: Input image in RGB format (NumPy array).
        request_id: Optional request ID for logging.

    Returns:
        Tuple of (upright_image_rgb, final_angle, best_composite_score).
    """
    
    req_prefix = f"[{request_id}] " if request_id else ""
    detector = get_thread_detector()

    all_results: list[dict] = []   # accumulated across all stages
    seen: set[float] = set()       # angles already probed

    # ── Stage 1: Coarse Scan (30° steps, 12 angles) ──
    logger.info(f"{req_prefix}Orientation: Stage 1 — coarse scan (12 angles at 30° steps)...")
    for angle in _COARSE_ANGLES:
        a = float(angle)
        if a in seen:
            continue
        seen.add(a)
        result = _evaluate_angle(image_rgb, a, detector, req_prefix)
        if result is not None:
            all_results.append(result)
            # Early exit: very high composite means face is both detected AND geometrically perfect
            if result['composite'] >= _COMPOSITE_EARLY_EXIT:
                logger.info(
                    f"{req_prefix}Orientation: early exit at {a}° "
                    f"(composite={result['composite']:.3f} >= {_COMPOSITE_EARLY_EXIT})"
                )
                # Stage 5: Apply analytical eye-line correction for sub-degree precision
                return _apply_eyeline_correction(image_rgb, result, req_prefix)

    if not all_results:
        logger.warning(f"{req_prefix}Orientation: no face found at any coarse angle. Returning original 0° image.")
        return image_rgb, 0.0, 0.0

    # Select beam candidates from Stage 1
    all_results.sort(key=lambda r: r['composite'], reverse=True)
    beam_angles = [r['angle'] for r in all_results[:_COARSE_BEAM]]
    beam_summary = ", ".join(
        f"{a:.1f}°({all_results[i]['composite']:.3f})" for i, a in enumerate(beam_angles)
    )
    logger.info(
        f"{req_prefix}Orientation: Stage 1 done — "
        f"top-{_COARSE_BEAM} beams: [{beam_summary}]"
    )

    # ── Stage 2: Medium Refinement (±12° at 6° steps, beam=2) ──
    logger.info(f"{req_prefix}Orientation: Stage 2 — medium refinement (±{_STAGE2_HALF_RANGE}° at {_STAGE2_STEP}° steps)...")
    beam_angles = _run_beam_stage(
        image_rgb, detector, beam_angles,
        _STAGE2_HALF_RANGE, _STAGE2_STEP, _STAGE2_BEAM,
        seen, all_results, req_prefix,
    )
    logger.info(
        f"{req_prefix}Orientation: Stage 2 done — "
        f"top-{_STAGE2_BEAM}: {[f'{a:.1f}°' for a in beam_angles]}"
    )

    # ── Stage 3: Fine Refinement (±4° at 2° steps, pick best) ──
    logger.info(f"{req_prefix}Orientation: Stage 3 — fine refinement (±{_STAGE3_HALF_RANGE}° at {_STAGE3_STEP}° steps)...")
    beam_angles = _run_beam_stage(
        image_rgb, detector, beam_angles,
        _STAGE3_HALF_RANGE, _STAGE3_STEP, 1,  # beam_width=1 → pick single best
        seen, all_results, req_prefix,
    )

    # ── Stage 4: Sub-degree Polish ──
    logger.info(f"{req_prefix}Orientation: Stage 4 — sub-degree polish around {beam_angles[0]:.1f}°...")
    best_center = beam_angles[0]
    for offset in _STAGE4_OFFSETS:
        a = round((best_center + offset) % 360, 2)
        if a not in seen:
            seen.add(a)
            result = _evaluate_angle(image_rgb, a, detector, req_prefix)
            if result is not None:
                all_results.append(result)

    # Final selection: best composite across all stages
    all_results.sort(key=lambda r: r['composite'], reverse=True)
    winner = all_results[0]

    logger.info(
        f"{req_prefix}Orientation: beam search best = {winner['angle']:.2f}° "
        f"(composite={winner['composite']:.3f}, det_score={winner['det_score']:.3f}, "
        f"eye_h={winner['landmark_quality']['eye_horizontality']:.2f}, "
        f"sym={winner['landmark_quality']['symmetry']:.2f}) "
        f"[{len(seen)} angles probed]"
    )

    # ── Stage 5: Analytical Eye-line Correction ──
    # The beam search found the approximately correct angle. Now use the eye
    # landmarks to mathematically calculate the exact remaining tilt and apply
    # a single precise rotation to the original image.
    return _apply_eyeline_correction(image_rgb, winner, req_prefix)


def add_padding_to_image(image: np.ndarray, padding_ratio: float) -> tuple:
    """Add white padding around image and return padded image with offset values."""
    height, width = image.shape[:2]
    pad_height = int(height * padding_ratio)
    pad_width = int(width * padding_ratio)
    
    padded_image = np.full((height + 2*pad_height, width + 2*pad_width, 3), 255, dtype=np.uint8)
    padded_image[pad_height:pad_height+height, pad_width:pad_width+width] = image
    
    return padded_image, pad_height, pad_width

def detect_noise_in_image(image_path, threshold=0.04):
    """
    Detect noise in an image using multiple robust methods.

    Args:
        image_path: Path to input image or numpy array
        threshold: Noise threshold (default 0.04)

    Returns:
        bool: True if noisy, False otherwise
    """
    # Read image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}
    else:
        img = image_path

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Method 1: Median filter difference (impulse noise)
    median_filtered = cv2.medianBlur(gray, 5)
    noise_map = cv2.absdiff(gray, median_filtered)
    median_score = np.mean(noise_map) / 255.0

    # Method 2: Gaussian vs Bilateral filter difference (Gaussian noise)
    gaussian = cv2.GaussianBlur(gray, (5, 5), 1.0)
    bilateral = cv2.bilateralFilter(gray, 5, 50, 50)
    noise_diff = cv2.absdiff(gaussian, bilateral)
    bilateral_score = np.mean(noise_diff) / 255.0

    # Method 3: Local standard deviation
    gray_norm = gray.astype(np.float32) / 255.0
    kernel = np.ones((7, 7), np.float32) / 49
    local_mean = cv2.filter2D(gray_norm, -1, kernel)
    local_sq_mean = cv2.filter2D(gray_norm**2, -1, kernel)
    local_var = local_sq_mean - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 0))
    std_score = np.mean(local_std)

    # Combined noise score
    combined_score = (median_score * 0.4 + bilateral_score * 0.3 + std_score * 0.3)
    is_noisy = combined_score > threshold

    return is_noisy
 
def classify_image_hsv(image_path):
    """
    Classifies using HSV color space (more accurate)
    """
    img = read_image_any_format(image_path)

    
    if img is None:
        return "Error: Could not load image"
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Extract saturation channel
    saturation = hsv[:, :, 1]

    # Calculate mean saturation
    mean_saturation = np.mean(saturation)
    print(f"Mean Saturation: {mean_saturation}")
    # Threshold for classification 
    threshold = 10  # Low saturation = B&W
    
    if mean_saturation < threshold:
        return "baw"
    else:
        return "rgb"

def denoise_image_if_needed(image_path, request_id: str = None):
    """
    Detect noise and apply appropriate denoising based on image type.
    
    Args:
        image_path: Path to input image
        
    Returns:
        numpy.ndarray: Denoised image or original if no noise detected
    """
    req_prefix = f"[{request_id}] " if request_id else ""
    img = read_image_any_format(image_path)

    if img is None:
        logger.error(f"{req_prefix}Failed to read image: {Path(image_path).name}")
        return None
        
    # Check if image has noise
    has_noise = detect_noise_in_image(image_path)
    logger.debug(f"{req_prefix}Noise detection for {Path(image_path).name}: {'DETECTED' if has_noise else 'NOT DETECTED'}")
    
    if has_noise:
        # Classify image type
        result = classify_image_hsv(image_path)
        logger.debug(f"{req_prefix}Image classification for {Path(image_path).name}: {result}")
        
        # Apply appropriate denoising
        if result == "rgb":
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            logger.info(f"{req_prefix}Applied RGB denoising to {Path(image_path).name}")
        else:
            dst = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
            logger.info(f"{req_prefix}Applied grayscale denoising to {Path(image_path).name}")
        
        return dst
    
    logger.debug(f"{req_prefix}No denoising needed for {Path(image_path).name}")
    return None

from functools import lru_cache

@lru_cache(maxsize=8)
def _compute_scaled_template(target_w: int, target_h: int) -> tuple:
    """Compute and cache the scaled landmark template (immutable tuple form)."""
    if (target_w, target_h) == FACE_SIZE:
        return tuple(map(tuple, FACE_LANDMARK_TEMPLATE.tolist()))
    
    scale_x = target_w / FACE_SIZE[0]
    scale_y = target_h / FACE_SIZE[1]
    scaled = FACE_LANDMARK_TEMPLATE.copy()
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return tuple(map(tuple, scaled.tolist()))

def scale_landmark_template(target_size: tuple) -> np.ndarray:
    """Scale the landmark template to match target face size (cached).
    
    Returns a fresh copy each time (safe for mutation by caller),
    but avoids recomputing the scaling on every call.
    """
    cached = _compute_scaled_template(target_size[0], target_size[1])
    return np.array(cached, dtype=np.float32)

def adjust_coordinates_after_padding(landmarks: np.ndarray, bbox: np.ndarray, pad_height: int, pad_width: int):
    """Adjust face coordinates from padded image back to original image space."""
    adjusted_landmarks = landmarks.copy()
    adjusted_landmarks[:, 0] -= pad_width
    adjusted_landmarks[:, 1] -= pad_height
    
    adjusted_bbox = bbox.copy()
    adjusted_bbox[0] -= pad_width
    adjusted_bbox[1] -= pad_height
    adjusted_bbox[2] -= pad_width
    adjusted_bbox[3] -= pad_height
    
    return adjusted_landmarks, adjusted_bbox

def preprocess_image_for_model(rgb_image: np.ndarray) -> torch.Tensor:
    """Convert RGB image to model input tensor.
    
    Pipeline: RGB→BGR channel swap → normalize to [-1, 1] → CHW layout → torch tensor.
    Uses np.ascontiguousarray + torch.from_numpy to minimize memory copies.
    """
    # Channel swap (RGB→BGR) + normalize to [-1, 1] range
    bgr_normalized = ((rgb_image[:, :, ::-1] / 255.) - 0.5) / 0.5
    # Transpose to CHW and ensure contiguous memory layout (single copy)
    chw = np.ascontiguousarray(bgr_normalized.transpose(2, 0, 1), dtype=np.float32)
    # from_numpy creates a zero-copy tensor; unsqueeze adds batch dim without copying
    tensor = torch.from_numpy(chw).unsqueeze(0)
    return tensor