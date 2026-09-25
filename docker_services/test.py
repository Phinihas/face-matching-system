import os
import time
import shutil
from pathlib import Path

def process_and_upload(folder_path: str, request_id: str):
    """Process images in a folder and save them to the Qdrant vector database."""
    start = time.time()
    UPLOAD_DIR = Path("uploads") / request_id
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic'}

    for root, _, files in os.walk(folder_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                file = os.path.join(root, file)
                path = shutil.copy(file, UPLOAD_DIR / os.path.basename(file))
                print("path: ",path)
                break
                # faces = extract_faces_from_image(file, request_id=request_id)
                # embeddings = compute_face_embeddings(faces, request_id) 
    total = time.time() - start
    print(f"Processed and uploaded images in {total:.2f} seconds.")
process_and_upload("./images", "test_request_id")