import numpy as np
from face_matcher import settings

TEMP_DIR = settings.TEMP_DIR
PRETRAINED_DIR = settings.PRETRAINED_DIR
LOG_DIR = settings.LOG_DIR

FACE_SIZE = settings.FACE_SIZE
MATCH_THRESHOLD = settings.MATCH_THRESHOLD
PADDING_RATIO = settings.PADDING_RATIO
MAX_ROTATION_DEPTH = settings.MAX_ROTATION_DEPTH

MODEL_FILE_ID = settings.MODEL_FILE_ID
MODEL_PATH = settings.MODEL_PATH

# Face alignment landmark reference points
FACE_LANDMARK_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

COLLECTION_NAME = settings.COLLECTION_NAME

