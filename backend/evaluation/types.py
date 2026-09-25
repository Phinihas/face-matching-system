from typing import TypedDict, Optional, List, Tuple
from pathlib import Path

class ImagePair(TypedDict):
    image1_path: Path
    image2_path: Path
    is_match: bool
    pair_id: str

class PairResult(TypedDict):
    pair_id: str
    image1_path: str
    image2_path: str
    is_match: bool
    predicted_match: bool
    similarity_score: float
    threshold: float
    success: bool
    error_message: Optional[str]
    detection_time: float
    alignment_time: float
    embedding_time: float
    similarity_time: float
    total_time: float
    
class ThresholdResult(TypedDict):
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    far: float
    frr: float
    eer: float
