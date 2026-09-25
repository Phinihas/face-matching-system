import numpy as np
from typing import List
from .types import PairResult
from .models import BenchmarkMetrics

def calculate_benchmarks(results: List[PairResult], total_duration: float) -> BenchmarkMetrics:
    """Calculates latency benchmarks from evaluation results."""
    if not results:
        return BenchmarkMetrics()

    detection_times = [r['detection_time'] for r in results if r['success']]
    alignment_times = [r['alignment_time'] for r in results if r['success']]
    embedding_times = [r['embedding_time'] for r in results if r['success']]
    similarity_times = [r['similarity_time'] for r in results if r['success']]
    total_times = [r['total_time'] for r in results if r['success']]

    if not total_times:
        return BenchmarkMetrics()

    num_images = len(total_times) * 2
    num_pairs = len(total_times)

    return BenchmarkMetrics(
        avg_detection_time=np.mean(detection_times),
        avg_alignment_time=np.mean(alignment_times),
        avg_embedding_time=np.mean(embedding_times),
        avg_similarity_time=np.mean(similarity_times),
        avg_pipeline_time=np.mean(total_times),
        median_time=np.median(total_times),
        min_time=np.min(total_times),
        max_time=np.max(total_times),
        std_dev_time=np.std(total_times),
        percentile_95_time=np.percentile(total_times, 95),
        percentile_99_time=np.percentile(total_times, 99),
        images_per_second=num_images / total_duration if total_duration > 0 else 0,
        pairs_per_second=num_pairs / total_duration if total_duration > 0 else 0
    )
