import time
import asyncio
from typing import List, Tuple
from .types import ImagePair, PairResult
from .models import EvaluationSummary
from .logger import get_evaluation_logger
from .metrics import calculate_metrics
from .benchmark import calculate_benchmarks
from .threshold import evaluate_thresholds, find_best_thresholds
from .profiler import SystemProfiler
from face_matcher.helpers.helpers import extract_faces, get_embeddings, find_matches

logger = get_evaluation_logger("evaluator")

class FaceMatchingEvaluator:
    def __init__(self, threshold: float = 0.42):
        self.threshold = threshold

    def _process_image(self, image_path: str, request_id: str, pbar=None) -> Tuple[List, List, float, float]:
        """Runs detection, alignment, and embedding for a single image."""
        if pbar:
            pbar.set_description("Face detection & Preprocessing")
        det_start = time.time()
        
        # 1. Detection and Alignment
        faces = extract_faces(image_path, request_id=request_id)
        det_align_time = time.time() - det_start
        
        if not faces:
            return [], [], det_align_time, 0.0
            
        # 2. Embedding
        if pbar:
            pbar.set_description("Embedding generation")
        emb_start = time.time()
        embeddings = get_embeddings(faces, request_id=request_id)
        emb_time = time.time() - emb_start
        
        return faces, embeddings, det_align_time, emb_time

# Evaluates a single image pair using the exact production pipeline.
    def evaluate_pair(self, pair: ImagePair, pbar=None) -> PairResult:
        """Evaluates a single image pair using the production pipeline."""
        request_id = pair['pair_id']
        logger.debug(f"Evaluating pair {request_id}")
        
        start_time = time.time()
        
        img1_path = str(pair['image1_path'])
        img2_path = str(pair['image2_path'])
        
        try:
            # Process Image 1
            faces1, emb1, t_det1, t_emb1 = self._process_image(img1_path, request_id, pbar)
            if not faces1:
                raise ValueError("No faces detected in image 1")
                
            # Process Image 2
            faces2, emb2, t_det2, t_emb2 = self._process_image(img2_path, request_id, pbar)
            if not faces2:
                raise ValueError("No faces detected in image 2")
                
            # 3. Similarity Matching
            if pbar:
                pbar.set_description("Similarity calculation")
            sim_start = time.time()
            matches, _ = find_matches(faces1, faces2, emb1, emb2, threshold=0.0, request_id=request_id) # Get raw score using 0 threshold
            sim_time = time.time() - sim_start
            
            max_sim = max([m['similarity_score'] for m in matches]) if matches else 0.0
            predicted_match = max_sim > self.threshold
            
            total_time = time.time() - start_time
            
            result = PairResult(
                pair_id=request_id,
                image1_path=img1_path,
                image2_path=img2_path,
                is_match=pair['is_match'],
                predicted_match=predicted_match,
                similarity_score=max_sim,
                threshold=self.threshold,
                success=True,
                error_message=None,
                detection_time=(t_det1 + t_det2) / 2, # Approximation, detection + alignment
                alignment_time=0.0, # Combined in extract_faces in production
                embedding_time=(t_emb1 + t_emb2) / 2,
                similarity_time=sim_time,
                total_time=total_time
            )
            
        except Exception as e:
            logger.error(f"Failed pair {request_id}: {e}")
            total_time = time.time() - start_time
            result = PairResult(
                pair_id=request_id,
                image1_path=img1_path,
                image2_path=img2_path,
                is_match=pair['is_match'],
                predicted_match=False,
                similarity_score=0.0,
                threshold=self.threshold,
                success=False,
                error_message=str(e),
                detection_time=0.0,
                alignment_time=0.0,
                embedding_time=0.0,
                similarity_time=0.0,
                total_time=total_time
            )
            
        if pbar:
            pbar.update(1)
            
        return result

    def run_evaluation(self, pairs: List[ImagePair], workers: int = 1, pbar=None) -> Tuple[EvaluationSummary, List[PairResult], List]:
        """Runs evaluation over the dataset."""
        logger.info(f"Starting evaluation on {len(pairs)} pairs using {workers} workers")
        
        profiler = SystemProfiler()
        profiler.start()
        
        start_time = time.time()
        
        results = []
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            from functools import partial
            with ThreadPoolExecutor(max_workers=workers) as executor:
                eval_func = partial(self.evaluate_pair, pbar=pbar)
                results = list(executor.map(eval_func, pairs))
        else:
            for pair in pairs:
                results.append(self.evaluate_pair(pair, pbar=pbar))
                
        total_duration = time.time() - start_time
        system_profile = profiler.stop()
        
        if pbar:
            pbar.set_description("Metric computation")
            
        # Calculate Metrics
        metrics = calculate_metrics(results)
        benchmarks = calculate_benchmarks(results, total_duration)
        
        # Threshold analysis
        threshold_results = evaluate_thresholds(results)
        best_thresholds = find_best_thresholds(threshold_results)
        
        successful_pairs = sum(1 for r in results if r['success'])
        
        summary = EvaluationSummary(
            total_pairs=len(pairs),
            successful_pairs=successful_pairs,
            failed_pairs=len(pairs) - successful_pairs,
            metrics=metrics,
            benchmark=benchmarks,
            system=system_profile,
            optimal_threshold_accuracy=best_thresholds.get("best_accuracy", 0.0),
            optimal_threshold_f1=best_thresholds.get("best_f1", 0.0),
            optimal_threshold_eer=best_thresholds.get("lowest_eer", 0.0)
        )
        
        return summary, results, threshold_results
