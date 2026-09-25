# Block 1 – Import Required Libraries
import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from face_matcher import settings

# Block 2 – Logger Initialization
logger = logging.getLogger(__name__)


# Block 3 – DetectedFace Class, This block defines a custom DetectedFace class that stores the detected 
# face's bounding box, facial landmarks, and confidence score while maintaining compatibility with the
#  previous InsightFace face object format used throughout the project.

class DetectedFace:
    """Adapter that mimics the InsightFace face object interface.

    Attributes:
        kps:       (5, 2) float32 ndarray — facial landmark coordinates
                   [right_eye, left_eye, nose, right_mouth, left_mouth]
        bbox:      (4,) float32 ndarray — bounding box as [x1, y1, x2, y2]
        det_score: float — detection confidence (0‒1)
    """

    def __init__(self, kps: np.ndarray, bbox: np.ndarray, det_score: float):
        self.kps = kps
        self.bbox = bbox
        self.det_score = det_score


# Block 4 – Download YuNet Model,This block checks whether the YuNet ONNX model exists locally and 
# automatically downloads it from the configured URL if it is missing, ensuring the application always 
# has the required face detection model.

def _download_yunet_model(model_path: Path, model_url: str) -> None:
    """Download the YuNet ONNX model if it does not already exist."""
    if model_path.exists():
        logger.info(f"YuNet model already exists: {model_path}")
        return

    logger.info(f"Downloading YuNet model from {model_url} ...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(model_url, str(model_path))
        logger.info(f"YuNet model downloaded successfully: {model_path}")
    except Exception as e:
        logger.error(f"Failed to download YuNet model: {e}")
        raise


# This block initializes the OpenCV YuNet face detector by loading the ONNX model, applying detector configurations such as input size, confidence threshold, NMS threshold, and maximum detections, making it ready for face detection.
def load_face_detector() -> cv2.FaceDetectorYN:
    """Initialize and return an OpenCV YuNet face detector.

    The model file is downloaded automatically on first run.
    """
    model_path = settings.DETECTOR_MODEL_PATH
    model_url = settings.DETECTOR_MODEL_URL

    # Download model if needed
    _download_yunet_model(model_path, model_url)

# Gets detector input size.
    det_w, det_h = settings.DETECTOR_SIZE

# This loads the ONNX model.
    detector = cv2.FaceDetectorYN.create(
        model=str(model_path),
        config="",
        input_size=(det_w, det_h),
        score_threshold=settings.DETECTOR_SCORE_THRESHOLD,
        nms_threshold=settings.DETECTOR_NMS_THRESHOLD,
        top_k=settings.DETECTOR_TOP_K,
    )
    logger.info(
        f"YuNet face detector loaded (input_size={det_w}x{det_h}, "
        f"score_threshold={settings.DETECTOR_SCORE_THRESHOLD}, "
        f"nms_threshold={settings.DETECTOR_NMS_THRESHOLD})"
    )
    return detector

import threading

# Thread-Local Storage: each thread gets its own YuNet detector instance.
# This eliminates the global lock that serialized all detection across threads.
_thread_local = threading.local()


#Give me the YuNet face detector that belongs to the current thread. If this thread doesn't have one yet, create one

def get_thread_detector() -> cv2.FaceDetectorYN:
    """Return a per-thread YuNet detector instance (created lazily on first call).
    
    OpenCV DNN models are safe to instantiate independently per thread.
    Using thread-local storage avoids the need for a global lock, enabling
    true parallel detection across ThreadPoolExecutor workers.
    """
    #Does the current thread's thread-local storage already contain a detector?
    if not hasattr(_thread_local, 'detector'):
        _thread_local.detector = load_face_detector()
        logger.debug(f"Created thread-local YuNet detector for thread {threading.current_thread().name}")
    return _thread_local.detector


def detect_faces(detector: cv2.FaceDetectorYN, image: np.ndarray) -> list[DetectedFace]:
    """Run YuNet face detection on an image and return results.

    Args:
        detector: An initialised cv2.FaceDetectorYN instance.
                  For thread-safe usage, pass the result of get_thread_detector().
        image:    BGR or RGB uint8 image (OpenCV YuNet works with either).

    Returns:
        List of DetectedFace objects compatible with the InsightFace interface
        (each has .kps, .bbox, .det_score attributes).
    """
    h, w = image.shape[:2]
    
    # No lock needed — each thread uses its own detector instance
    detector.setInputSize((w, h))
    retval, raw_detections = detector.detect(image)

    if raw_detections is None or retval < 1:
        return []

    faces: list[DetectedFace] = []
    for det in raw_detections:
        # Bounding box: YuNet gives [x, y, w, h] → convert to [x1, y1, x2, y2]
        x, y, bw, bh = det[0:4]
        bbox = np.array([x, y, x + bw, y + bh], dtype=np.float32)

        # 5 landmarks (right_eye, left_eye, nose, right_mouth, left_mouth)
        landmarks = det[4:14].reshape(5, 2).astype(np.float32)

        # Confidence
        confidence = float(det[14])

        faces.append(DetectedFace(kps=landmarks, bbox=bbox, det_score=confidence))

    return faces


# This module implements the complete face detection stage of the Face Matching pipeline by automatically 
# downloading and loading the YuNet ONNX model, performing thread-safe face detection on input images, 
# converting the detection results into a standardized DetectedFace format compatible with the existing 
# InsightFace workflow, and providing the detected face information to the subsequent face alignment and 
# recognition modules.