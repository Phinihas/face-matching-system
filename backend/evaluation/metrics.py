import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
from typing import List
from .models import EvaluationMetrics
from .types import PairResult

# Computes all evaluation metrics from pair results.
def calculate_metrics(results: List[PairResult], threshold: float = None) -> EvaluationMetrics:
    y_true = []
    y_pred = []
    y_scores = []
    
    for r in results:
        if r['success']:
            y_true.append(1 if r['is_match'] else 0)
            
            # Use specific threshold if provided, else use the one from result
            if threshold is not None:
                predicted = 1 if r['similarity_score'] > threshold else 0
            else:
                predicted = 1 if r['predicted_match'] else 0
                
            y_pred.append(predicted)
            y_scores.append(r['similarity_score'])
            
    if not y_true:
        return EvaluationMetrics()
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Rates
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # EER and ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        fnr = 1 - tpr
        eer_index = np.nanargmin(np.absolute((fnr - fpr)))
        eer = fpr[eer_index]
        ap = average_precision_score(y_true, y_scores)
    except Exception:
        roc_auc = 0.0
        eer = 0.0
        ap = 0.0
    
    return EvaluationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        specificity=specificity,
        sensitivity=sensitivity,
        f1_score=f1_score(y_true, y_pred, zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        roc_auc=roc_auc,
        average_precision=ap,
        equal_error_rate=eer,
        false_acceptance_rate=far,
        false_rejection_rate=frr,
        true_positive=int(tp),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        matthews_correlation=matthews_corrcoef(y_true, y_pred),
        cohen_kappa=cohen_kappa_score(y_true, y_pred)
    )
