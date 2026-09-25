from pathlib import Path
from typing import Optional

def validate_image_path(path_str: str) -> Optional[Path]:
    """Validates if an image exists and returns a Path object."""
    path = Path(path_str)
    if path.exists() and path.is_file():
        return path
    return None

def resolve_lfw_path(base_dir: Path, name: str, index: str) -> Path:
    """Resolves LFW image path format: base_dir/name/name_000index.jpg"""
    # LFW format zero-pads to 4 digits
    padded_index = str(index).zfill(4)
    filename = f"{name}_{padded_index}.jpg"
    return base_dir / name / filename
