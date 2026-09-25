import logging
from pathlib import Path
from face_matcher import settings
from .config import LOGS_DIR

def get_evaluation_logger(name: str) -> logging.Logger:
    """Creates and returns a structured logger for the evaluation module."""
    logger = logging.getLogger(f"eval_{name}")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate logs if already added
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s'
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        log_file = LOGS_DIR / "evaluation.log"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

