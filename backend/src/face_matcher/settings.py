import os
import configparser
from pathlib import Path
from dotenv import load_dotenv

# Base Directory paths
FACE_MATCHER_DIR = Path(__file__).parent.resolve()
SRC_DIR = FACE_MATCHER_DIR.parent
CONFIG_DIR = SRC_DIR / "config"

# Paths to config files
ENV_PATH = CONFIG_DIR / ".env"
SETTINGS_INI_PATH = CONFIG_DIR / "settings.ini"

# Load environment variables
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Read settings.ini
config = configparser.ConfigParser()
if SETTINGS_INI_PATH.exists():
    config.read(SETTINGS_INI_PATH)
else:
    raise FileNotFoundError(f"Configuration settings.ini not found at {SETTINGS_INI_PATH}")

# [app] settings
ROOT_PATH = config.get("app", "root_path", fallback="/face_matching")
ALLOW_ORIGINS = [orig.strip() for orig in config.get("app", "allow_origins", fallback="*").split(",")]

# Directories
TEMP_DIR = SRC_DIR / config.get("app", "temp_dir", fallback="temp_uploads")
PRETRAINED_DIR = SRC_DIR / config.get("app", "pretrained_dir", fallback="pretrained")
LOG_DIR = SRC_DIR / config.get("app", "log_dir", fallback="logs")
SEARCH_RESULTS_DIR = SRC_DIR / config.get("app", "search_results_dir", fallback="temp_uploads/search_results")
UPLOAD_DIR = SRC_DIR / config.get("app", "upload_dir", fallback="uploads")

# Ensure critical directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)
PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_ID_TTL_SECONDS = config.getint("app", "search_id_ttl_seconds", fallback=3600)

# [database] settings
DB_HOST = config.get("database", "host", fallback="127.0.0.1")
DB_PORT = config.getint("database", "port", fallback=3306)
DB_NAME = config.get("database", "database", fallback="ai_db4")
DB_CHARSET = config.get("database", "charset", fallback="utf8mb4")

# [model] settings
FACE_SIZE = (
    config.getint("model", "face_size_width", fallback=112),
    config.getint("model", "face_size_height", fallback=112)
)
MATCH_THRESHOLD = config.getfloat("model", "match_threshold", fallback=0.34)
PADDING_RATIO = config.getfloat("model", "padding_ratio", fallback=0.3)
MAX_ROTATION_DEPTH = config.getint("model", "max_rotation_depth", fallback=2)
MODEL_FILE_ID = config.get("model", "model_file_id", fallback="1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI")
MODEL_PATH = SRC_DIR / config.get("model", "model_path", fallback="pretrained/adaface_ir50_ms1mv2.ckpt")
COLLECTION_NAME = config.get("model", "collection_name", fallback="my_collection")

# [detector] settings — OpenCV YuNet face detector
DETECTOR_MODEL_PATH = PRETRAINED_DIR / config.get("detector", "model_path", fallback="face_detection_yunet_2023mar.onnx")
DETECTOR_MODEL_URL = config.get("detector", "model_url", fallback="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
DETECTOR_SIZE = (
    config.getint("detector", "det_size_width", fallback=640),
    config.getint("detector", "det_size_height", fallback=640)
)
DETECTOR_SCORE_THRESHOLD = config.getfloat("detector", "score_threshold", fallback=0.6)
DETECTOR_NMS_THRESHOLD = config.getfloat("detector", "nms_threshold", fallback=0.3)
DETECTOR_TOP_K = config.getint("detector", "top_k", fallback=5000)

# [ingestion] settings
BATCH_SIZE = config.getint("ingestion", "batch_size", fallback=20)
UPDATE_INTERVAL = config.getint("ingestion", "update_interval", fallback=10)
# Secrets loaded from environment (fallback logic included)
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OTEL_ENDPOINT = os.getenv("OTEL_ENDPOINT", "http://otel-collector:4317")
