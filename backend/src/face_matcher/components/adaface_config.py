import gdown
import logging
import torch
from pathlib import Path
import face_matcher.helpers.net as net
logger = logging.getLogger(__name__)


# Download pretrained model
def download_pretrained_model(file_id: str, output_path: Path) -> None:
    """Download model from Google Drive if not already present."""
    if output_path.exists():
        logger.info(f"Model already exists: {output_path}")
        return

    try:
        gdown.download(f'https://drive.google.com/uc?id={file_id}', str(output_path), quiet=False)
        logger.info(f"Model downloaded: {output_path}")
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

# Load AdaFace model
def load_adaface_model(model_path: Path):
    """Load the AdaFace face recognition model."""
    logger.info("Loading AdaFace model...")
    model = net.build_model('ir_50')
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)['state_dict']
    model_state_dict = {key[6:]: val for key, val in state_dict.items() if key.startswith('model.')}
    model.load_state_dict(model_state_dict)
    model.eval()
    logger.info("AdaFace model loaded successfully!")
    return model