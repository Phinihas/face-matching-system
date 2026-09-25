from typing import List
import numpy as np
from .types import PairResult, ThresholdResult
from .config import THRESHOLD_RANGE
from .logger import get_evaluation_logger

logger = get_evaluation_logger("threshold")

def evaluate_thresholds(results: List[PairResult]) -> List[ThresholdResult]:
    """Evaluates metrics across a range of thresholds using vectorized computation.
    
    Uses NumPy broadcasting to compute TP/FP/TN/FN for all thresholds at once,
    avoiding repeated iteration over results per threshold.
    """
    if not results:
        return []
    
    # Extract ground truth and scores into arrays (once)
    y_true = np.array([1 if r['is_match'] else 0 for r in results if r['success']])
    y_scores = np.array([r['similarity_score'] for r in results if r['success']])
    
    if len(y_true) == 0:
        return []

    # Build threshold array including fine sweep around optimal region
    coarse = set(THRESHOLD_RANGE)
    fine = {round(x * 0.005, 3) for x in range(20, 71)}  # 0.10 to 0.35 in 0.005 steps
    all_thresholds = sorted(coarse | fine)
    
    logger.info(f"Evaluating {len(all_thresholds)} thresholds from {min(all_thresholds)} to {max(all_thresholds)}")
    
    # Vectorized: predict for ALL thresholds at once
    # y_scores is (N,), thresholds is (T,) → predictions is (T, N)
    thresholds_arr = np.array(all_thresholds)
    predictions = (y_scores[np.newaxis, :] > thresholds_arr[:, np.newaxis]).astype(int)  # (T, N)
    
    # Compute confusion matrix entries for all thresholds simultaneously
    positives = (y_true == 1)
    negatives = (y_true == 0)
    
    tp = (predictions[:, positives] == 1).sum(axis=1)  # (T,)
    fn = (predictions[:, positives] == 0).sum(axis=1)  # (T,)
    fp = (predictions[:, negatives] == 1).sum(axis=1)  # (T,)
    tn = (predictions[:, negatives] == 0).sum(axis=1)  # (T,)
    
    total = len(y_true)
    n_pos = positives.sum()
    n_neg = negatives.sum()
    
    # Compute metrics from confusion matrix entries
    accuracy = (tp + tn) / total
    precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    recall = np.where(n_pos > 0, tp / n_pos, 0.0)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    far = np.where(n_neg > 0, fp / n_neg, 0.0)
    frr = np.where(n_pos > 0, fn / n_pos, 0.0)
    
    # EER: compute from ROC curve (once, shared across thresholds)
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        eer_index = np.nanargmin(np.absolute(fnr - fpr))
        eer_value = float(fpr[eer_index])
    except Exception:
        eer_value = 0.0
    
    threshold_results = []
    for i, t in enumerate(all_thresholds):
        threshold_results.append(ThresholdResult(
            threshold=t,
            accuracy=float(accuracy[i]),
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            far=float(far[i]),
            frr=float(frr[i]),
            eer=eer_value  # EER is a global property, same for all thresholds
        ))
        
    return threshold_results

def find_best_thresholds(threshold_results: List[ThresholdResult]) -> dict:
    """Finds best thresholds for Accuracy, F1, and EER."""
    if not threshold_results:
        return {}
        
    best_acc = max(threshold_results, key=lambda x: x['accuracy'])
    best_f1 = max(threshold_results, key=lambda x: x['f1'])
    best_eer = min(threshold_results, key=lambda x: abs(x['far'] - x['frr']))  # Point where FAR ≈ FRR
    
    return {
        "best_accuracy": best_acc['threshold'],
        "best_f1": best_f1['threshold'],
        "lowest_eer": best_eer['threshold']
    }
