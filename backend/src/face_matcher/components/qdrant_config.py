import logging
from qdrant_client import QdrantClient, models
from face_matcher import settings

logger = logging.getLogger(__name__)

# Module-level singleton — created once, shared across the application.
_client: QdrantClient = None


def get_qdrant_client() -> QdrantClient:
    """Initialize and return a Qdrant client (singleton).

    Strategy:
    1. Try connecting to the Qdrant server at QDRANT_URL (Docker / remote).
    2. If the server is unreachable, fall back to local file-based storage
       at  <SRC_DIR>/qdrant_data  which gives full persistence without Docker.
    """
    global _client
    if _client is not None:
        return _client

    # --- Attempt 1: remote / Docker Qdrant server ---
    try:
        trial = QdrantClient(settings.QDRANT_URL, timeout=5)
        trial.get_collections()          # lightweight connectivity check
        _client = trial
        logger.info(f"Qdrant client connected to server: {settings.QDRANT_URL}")
        return _client
    except Exception as e:
        logger.warning(
            f"Qdrant server at {settings.QDRANT_URL} unreachable ({e}). "
            "Falling back to local file-based Qdrant storage."
        )

    # --- Attempt 2: local file-based storage (no Docker needed) ---
    qdrant_data_dir = settings.SRC_DIR / "qdrant_data"
    qdrant_data_dir.mkdir(parents=True, exist_ok=True)
    _client = QdrantClient(path=str(qdrant_data_dir))
    logger.info(f"Qdrant client using local file storage: {qdrant_data_dir}")
    return _client


def init_collection(client: QdrantClient) -> None:
    """Create Qdrant collection if it doesn't already exist."""
    if not client.collection_exists(collection_name=settings.COLLECTION_NAME):
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE)
        )
        logger.info(f"Collection '{settings.COLLECTION_NAME}' created.")
    else:
        logger.info(f"Collection '{settings.COLLECTION_NAME}' already exists; skipping creation.")