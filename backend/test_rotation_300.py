"""
Test script: Validate the existing 360 degree orientation-normalization feature
against 300 pre-rotated face images.

This script does NOT modify any existing backend code.
It imports and calls normalize_image_orientation() directly --
the same function used by compare_two_images().

Usage:
    cd /d D:\\test\\face-matching_yunet\\backend
    .venv\\Scripts\\python.exe test_rotation_300.py
"""

import os
import sys
import io
import csv
import re
import time
import logging
import traceback

# Force UTF-8 output on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

# ── Setup sys.path so we can import from the project ──────────────────────
BACKEND_DIR = Path(__file__).parent.resolve()
SRC_DIR = BACKEND_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# ── Configure logging (show INFO from orientation stages) ─────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("test_rotation_300")

# ── Import project functions (read-only, no modifications) ────────────────
from face_matcher.helpers.pre_process import (
    normalize_image_orientation,
    read_image_any_format,
)

# ── Directories ───────────────────────────────────────────────────────────
INPUT_DIR = BACKEND_DIR / "test_images"
OUTPUT_DIR = BACKEND_DIR / "output_images"
CSV_REPORT_PATH = BACKEND_DIR / "rotation_test_results.csv"

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif", ".heic"}

# Regex to extract rotation angle from filename like image_001_rot_117.jpeg
ANGLE_PATTERN = re.compile(r"_rot_(\d+)\.")


def extract_angle_from_filename(filename: str):
    """Extract the rotation angle embedded in the filename, or None."""
    match = ANGLE_PATTERN.search(filename)
    if match:
        return int(match.group(1))
    return None


def process_single_image(image_path: Path, output_dir: Path) -> dict:
    """Process one image through normalize_image_orientation and save output.

    Returns a dict with all captured metadata for the CSV report.
    """
    filename = image_path.name
    result = {
        "input_filename": filename,
        "output_filename": "N/A",
        "status": "FAILED",
        "filename_angle": extract_angle_from_filename(filename),
        "detected_angle": "N/A",
        "composite_score": "N/A",
        "det_score": "N/A",
        "eye_horizontality": "N/A",
        "symmetry": "N/A",
        "nose_mouth_vert": "N/A",
        "aspect_ratio_score": "N/A",
        "processing_time_ms": "N/A",
        "face_detected": "NO",
        "output_saved": "NO",
        "error_message": "",
    }

    start = time.perf_counter()

    try:
        # 1. Load image using the project's own loader
        loaded_image = read_image_any_format(str(image_path))
        if loaded_image is None:
            result["error_message"] = "Failed to load image (read_image_any_format returned None)"
            return result

        # 2. Convert BGR → RGB (normalize_image_orientation expects RGB)
        image_rgb = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)

        # 3. Call the EXISTING orientation-normalization function
        upright_rgb, best_angle, composite_score = normalize_image_orientation(
            image_rgb, request_id=f"test_{filename}"
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result["processing_time_ms"] = round(elapsed_ms, 2)
        result["detected_angle"] = round(best_angle, 2)
        result["composite_score"] = round(composite_score, 4)

        # Determine if a face was actually detected
        if composite_score > 0.0:
            result["face_detected"] = "YES"
        else:
            result["face_detected"] = "NO"
            result["status"] = "NO_FACE"
            result["error_message"] = "No face detected at any angle (composite=0.0)"

        # 4. Save the normalized output image (RGB → BGR for cv2.imwrite)
        output_filename = f"normalized_{filename}"
        # Ensure output extension is always writable
        out_stem = Path(output_filename).stem
        out_ext = Path(output_filename).suffix.lower()
        if out_ext not in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
            output_filename = out_stem + ".jpg"

        output_path = output_dir / output_filename
        upright_bgr = cv2.cvtColor(upright_rgb, cv2.COLOR_RGB2BGR)
        write_success = cv2.imwrite(str(output_path), upright_bgr)

        if write_success:
            result["output_filename"] = output_filename
            result["output_saved"] = "YES"
            if result["face_detected"] == "YES":
                result["status"] = "SUCCESS"
        else:
            result["error_message"] = f"cv2.imwrite failed for {output_path}"

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result["processing_time_ms"] = round(elapsed_ms, 2)
        result["error_message"] = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Error processing {filename}: {traceback.format_exc()}")

    return result


def main():
    print("=" * 70)
    print("  ROTATION NORMALIZATION TEST — 300 Pre-Rotated Images")
    print("=" * 70)
    print()

    # ── Validate input directory ──────────────────────────────────────────
    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    # Gather image files
    image_files = sorted(
        f for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    total_images = len(image_files)
    print(f"  Input directory  : {INPUT_DIR}")
    print(f"  Images found     : {total_images}")
    print(f"  Output directory : {OUTPUT_DIR}")
    print(f"  CSV report       : {CSV_REPORT_PATH}")
    print()

    if total_images == 0:
        print("ERROR: No images found in test_images/")
        sys.exit(1)

    # ── Create output directory ───────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Process all images ────────────────────────────────────────────────
    results = []
    overall_start = time.perf_counter()

    for idx, image_path in enumerate(image_files, 1):
        print(f"  [{idx:>3}/{total_images}] Processing: {image_path.name} ... ", end="", flush=True)
        result = process_single_image(image_path, OUTPUT_DIR)
        results.append(result)

        status_icon = "✓" if result["status"] == "SUCCESS" else ("⚠" if result["status"] == "NO_FACE" else "✗")
        time_str = f"{result['processing_time_ms']}ms" if result["processing_time_ms"] != "N/A" else "N/A"
        angle_str = f"angle={result['detected_angle']}°" if result["detected_angle"] != "N/A" else "no angle"
        score_str = f"score={result['composite_score']}" if result["composite_score"] != "N/A" else ""
        print(f"{status_icon}  {time_str}  {angle_str}  {score_str}")

    overall_elapsed = time.perf_counter() - overall_start

    # ── Write CSV report ──────────────────────────────────────────────────
    csv_columns = [
        "input_filename", "output_filename", "status", "filename_angle",
        "detected_angle", "composite_score", "det_score",
        "eye_horizontality", "symmetry", "nose_mouth_vert", "aspect_ratio_score",
        "processing_time_ms", "face_detected", "output_saved", "error_message",
    ]

    with open(CSV_REPORT_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # ── Compute statistics ────────────────────────────────────────────────
    successful = [r for r in results if r["status"] == "SUCCESS"]
    no_face = [r for r in results if r["status"] == "NO_FACE"]
    failed = [r for r in results if r["status"] == "FAILED"]

    times = [r["processing_time_ms"] for r in results if isinstance(r["processing_time_ms"], (int, float))]
    scores = [r["composite_score"] for r in results if isinstance(r["composite_score"], (int, float))]
    detected_angles = [r["detected_angle"] for r in results if isinstance(r["detected_angle"], (int, float))]

    success_rate = (len(successful) / total_images * 100) if total_images > 0 else 0.0
    face_detected_rate = ((len(successful) + len(no_face)) / total_images * 100) if total_images > 0 else 0.0

    # ── Print summary ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  ROTATION NORMALIZATION TEST — RESULTS")
    print("=" * 70)
    print()
    print(f"  Total images          : {total_images}")
    print(f"  Successful (face+save): {len(successful)}")
    print(f"  No face detected      : {len(no_face)}")
    print(f"  Failed (errors)       : {len(failed)}")
    print(f"  Success rate          : {success_rate:.2f}%")
    print(f"  Face detection rate   : {face_detected_rate:.2f}%")
    print()

    if times:
        print(f"  Total wall time       : {overall_elapsed:.2f}s")
        print(f"  Average time/image    : {np.mean(times):.2f} ms")
        print(f"  Median time/image     : {np.median(times):.2f} ms")
        print(f"  Minimum time          : {np.min(times):.2f} ms")
        print(f"  Maximum time          : {np.max(times):.2f} ms")
        print(f"  Std deviation         : {np.std(times):.2f} ms")
        print()

    if scores:
        print(f"  Composite score stats:")
        print(f"    Average             : {np.mean(scores):.4f}")
        print(f"    Median              : {np.median(scores):.4f}")
        print(f"    Minimum             : {np.min(scores):.4f}")
        print(f"    Maximum             : {np.max(scores):.4f}")
        print(f"    Std deviation       : {np.std(scores):.4f}")
        low_confidence = [r for r in results if isinstance(r["composite_score"], (int, float)) and r["composite_score"] < 0.5]
        print(f"    Low confidence (<0.5): {len(low_confidence)}")
        print()

    if detected_angles:
        print(f"  Detected angle stats:")
        print(f"    Average             : {np.mean(detected_angles):.2f}°")
        print(f"    Std deviation       : {np.std(detected_angles):.2f}°")
        # Angle distribution in 45° buckets
        buckets = defaultdict(int)
        for a in detected_angles:
            bucket = int(a // 45) * 45
            buckets[bucket] += 1
        print(f"    Angle distribution (45° buckets):")
        for bucket_start in sorted(buckets.keys()):
            bucket_end = bucket_start + 45
            count = buckets[bucket_start]
            bar = "█" * (count // 2) + ("▌" if count % 2 else "")
            print(f"      {bucket_start:>3}°–{bucket_end:<3}° : {count:>3}  {bar}")
        print()

    # ── List problematic images ───────────────────────────────────────────
    if no_face:
        print(f"  ⚠ Images with NO face detected ({len(no_face)}):")
        for r in no_face:
            print(f"    - {r['input_filename']}  (filename_angle={r['filename_angle']}°)")
        print()

    if failed:
        print(f"  ✗ FAILED images ({len(failed)}):")
        for r in failed:
            print(f"    - {r['input_filename']}: {r['error_message']}")
        print()

    if low_confidence := [r for r in results if isinstance(r["composite_score"], (int, float)) and 0 < r["composite_score"] < 0.5]:
        print(f"  ⚠ Low confidence images (0 < score < 0.5) ({len(low_confidence)}):")
        for r in low_confidence:
            print(f"    - {r['input_filename']}  angle={r['detected_angle']}°  score={r['composite_score']}")
        print()

    # ── Correlation: filename angle vs detected angle ─────────────────────
    paired = [
        (r["filename_angle"], r["detected_angle"])
        for r in results
        if r["filename_angle"] is not None and isinstance(r["detected_angle"], (int, float))
    ]
    if paired:
        fn_angles, det_angles = zip(*paired)
        print(f"  Filename angle vs detected angle ({len(paired)} images):")
        # Check how many have detected_angle near 0 (meaning successful upright normalization)
        near_zero = sum(1 for da in det_angles if da <= 5 or da >= 355)
        print(f"    Detected angle near 0° (±5°)  : {near_zero} / {len(paired)}  ({near_zero/len(paired)*100:.1f}%)")
        near_low = sum(1 for da in det_angles if da <= 15 or da >= 345)
        print(f"    Detected angle near 0° (±15°) : {near_low} / {len(paired)}  ({near_low/len(paired)*100:.1f}%)")
        print()

    print(f"  Output directory : {OUTPUT_DIR}")
    print(f"  CSV report       : {CSV_REPORT_PATH}")
    print()

    # ── Limitations notice ────────────────────────────────────────────────
    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │  IMPORTANT LIMITATION                                          │")
    print("  │                                                                │")
    print("  │  This test validates that normalize_image_orientation()        │")
    print("  │  executes without errors, detects faces, and returns a         │")
    print("  │  rotation angle + confidence score.                            │")
    print("  │                                                                │")
    print("  │  It CANNOT automatically verify that the output image is       │")
    print("  │  visually upright. The composite score is a proxy for          │")
    print("  │  uprightness, not a guarantee.                                 │")
    print("  │                                                                │")
    print("  │  RECOMMENDATION: Manually inspect output images, especially    │")
    print("  │  those with low composite scores or unusual detected angles.   │")
    print("  └─────────────────────────────────────────────────────────────────┘")
    print()
    print("=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
