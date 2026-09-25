import csv
import os
from pathlib import Path
from typing import Iterator
from .types import ImagePair
from .utils import resolve_lfw_path, validate_image_path
from .logger import get_evaluation_logger

logger = get_evaluation_logger("dataset_loader")

class LFWDatasetLoader:
    def __init__(self, data_dir: Path, pairs_file: Path):
        self.data_dir = data_dir
        self.pairs_file = pairs_file
        
    def _read_pairs(self) -> Iterator[ImagePair]:
        """Reads the pairs.csv file and yields ImagePair objects."""
        if not self.pairs_file.exists():
            logger.error(f"Pairs file not found: {self.pairs_file}")
            return
            
        logger.info(f"Loading dataset pairs from {self.pairs_file}")
        
        with open(self.pairs_file, 'r') as f:
            # Skip the first row (10 300) in pairs.txt or header in pairs.csv if present
            # We will use csv.reader
            # Wait, LFW pairs.txt is tab separated, pairs.csv is comma separated
            
            # Detect delimiter
            sample = f.read(1024)
            f.seek(0)
            delimiter = ',' if ',' in sample else '\t'
            
            reader = csv.reader(f, delimiter=delimiter)
            
            for row_idx, row in enumerate(reader):
                # Remove trailing empty strings if any
                row = [x.strip() for x in row if x.strip()]
                
                if len(row) == 2 and row_idx == 0:
                    continue # header format: sets, matches per set
                if len(row) == 0:
                    continue
                
                if len(row) == 3:
                    # Matched pair: name, idx1, idx2
                    name, idx1, idx2 = row
                    img1_path = resolve_lfw_path(self.data_dir, name, idx1)
                    img2_path = resolve_lfw_path(self.data_dir, name, idx2)
                    is_match = True
                elif len(row) == 4:
                    # Mismatched pair: name1, idx1, name2, idx2
                    name1, idx1, name2, idx2 = row
                    img1_path = resolve_lfw_path(self.data_dir, name1, idx1)
                    img2_path = resolve_lfw_path(self.data_dir, name2, idx2)
                    is_match = False
                else:
                    logger.warning(f"Skipping malformed row {row_idx}: {row}")
                    continue
                
                # Validate
                valid_img1 = validate_image_path(str(img1_path))
                valid_img2 = validate_image_path(str(img2_path))
                
                if valid_img1 and valid_img2:
                    yield ImagePair(
                        image1_path=valid_img1,
                        image2_path=valid_img2,
                        is_match=is_match,
                        pair_id=f"pair_{row_idx}"
                    )
                else:
                    if not valid_img1:
                        logger.debug(f"Skipping pair {row_idx}: Image not found {img1_path}")
                    if not valid_img2:
                        logger.debug(f"Skipping pair {row_idx}: Image not found {img2_path}")

    def load(self) -> list[ImagePair]:
        pairs = list(self._read_pairs())
        logger.info(f"Successfully loaded {len(pairs)} valid image pairs.")
        return pairs
