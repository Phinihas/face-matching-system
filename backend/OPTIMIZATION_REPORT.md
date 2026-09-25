# Optimization Report

## 1. Global Thread Lock in YuNet Detector

**Problem**: 
In `face_matcher/components/yunet_config.py`, there is a global `_YUNET_LOCK = threading.Lock()` that wraps the `detector.setInputSize()` and `detector.detect()` calls.

**Impact**: 
High. This forces all face detections to run sequentially, completely negating the benefits of the `ThreadPoolExecutor` in `bulk_upload` and the new `--workers` flag in the evaluation framework. The CPU will be vastly underutilized because all worker threads will bottleneck waiting for this lock.

**Recommendation**: 
Instead of a single global detector instance wrapped in a lock, use a Thread-Local Storage (TLS) instance of the detector, or create a pool of detectors (one per worker thread). OpenCV DNN models can be instantiated per thread safely.

**Estimated Improvement**: 
N-times speedup in detection/alignment throughput, where N is the number of CPU cores or worker threads used.

## 2. Redundant Preprocessing in Comparison

**Problem**:
In `face_matcher/helpers/helpers.py`, the `compare_two_images` tests 4 rotation combinations using `asyncio.gather`. Each of these tests triggers `extract_faces`.

**Impact**:
Medium. This means the same images are read from disk, converted to RGB, padded, and run through the face detector multiple times unnecessarily for rotations that might not even contain faces or be useful if the primary orientation is correct.

**Recommendation**:
Only compute rotations if the primary orientation fails to yield high-confidence matches. Alternatively, run detection on the base image once, and if faces are found, don't bother running detection on the 180-degree flipped version unless confidence is extremely low.

**Estimated Improvement**: 
Up to 75% reduction in CPU time for 1:1 matching endpoints in successful (well-oriented) cases.

## 3. Synchronous File Operations in Async Functions

**Problem**:
Functions like `cleanup_file` and reading/writing images (`cv2.imread`, `cv2.imwrite`) are often called directly within `async def` endpoints (e.g., in `/compare` `finally` block) without being dispatched to a thread pool.

**Impact**:
Low to Medium. This blocks the main FastAPI event loop, potentially reducing concurrent throughput of the API under load.

**Recommendation**:
Wrap all blocking I/O calls in `asyncio.to_thread()` when called from within an asynchronous route handler.

**Estimated Improvement**:
Improved API responsiveness and higher concurrency limits under heavy load.
