import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from typing import List
from .types import PairResult, ThresholdResult
from .config import PLOTS_DIR

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 300

def generate_roc_curve(results: List[PairResult], save_path: Path = PLOTS_DIR / "roc_curve.png"):
    set_style()
    
    y_true = [1 if r['is_match'] else 0 for r in results if r['success']]
    y_scores = [r['similarity_score'] for r in results if r['success']]
    
    if not y_true:
        return
        
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def generate_pr_curve(results: List[PairResult], save_path: Path = PLOTS_DIR / "pr_curve.png"):
    set_style()
    
    y_true = [1 if r['is_match'] else 0 for r in results if r['success']]
    y_scores = [r['similarity_score'] for r in results if r['success']]
    
    if not y_true:
        return
        
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def generate_confusion_matrix(results: List[PairResult], save_path: Path = PLOTS_DIR / "confusion_matrix.png"):
    set_style()
    
    y_true = [1 if r['is_match'] else 0 for r in results if r['success']]
    y_pred = [1 if r['predicted_match'] else 0 for r in results if r['success']]
    
    if not y_true:
        return
        
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Impostor', 'Genuine'], 
                yticklabels=['Impostor', 'Genuine'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def generate_score_distributions(results: List[PairResult], save_path: Path = PLOTS_DIR / "score_distribution.png"):
    set_style()
    
    genuine_scores = [r['similarity_score'] for r in results if r['success'] and r['is_match']]
    impostor_scores = [r['similarity_score'] for r in results if r['success'] and not r['is_match']]
    
    if not genuine_scores and not impostor_scores:
        return
        
    plt.figure()
    if genuine_scores:
        sns.histplot(genuine_scores, color='blue', label='Genuine', kde=True, stat='density', alpha=0.5, bins=50)
    if impostor_scores:
        sns.histplot(impostor_scores, color='red', label='Impostor', kde=True, stat='density', alpha=0.5, bins=50)
        
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.title('Distribution of Genuine and Impostor Scores')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def generate_threshold_curves(threshold_results: List[ThresholdResult], save_path: Path = PLOTS_DIR / "threshold_curves.png"):
    set_style()
    
    if not threshold_results:
        return
        
    thresholds = [r['threshold'] for r in threshold_results]
    accuracy = [r['accuracy'] for r in threshold_results]
    precision = [r['precision'] for r in threshold_results]
    recall = [r['recall'] for r in threshold_results]
    f1 = [r['f1'] for r in threshold_results]
    
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, accuracy, label='Accuracy', lw=2)
    plt.plot(thresholds, precision, label='Precision', lw=2)
    plt.plot(thresholds, recall, label='Recall', lw=2)
    plt.plot(thresholds, f1, label='F1 Score', lw=2)
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def generate_far_frr_curve(threshold_results: List[ThresholdResult], save_path: Path = PLOTS_DIR / "far_frr_curve.png"):
    set_style()
    
    if not threshold_results:
        return
        
    thresholds = [r['threshold'] for r in threshold_results]
    far = [r['far'] for r in threshold_results]
    frr = [r['frr'] for r in threshold_results]
    
    plt.figure()
    plt.plot(thresholds, far, label='False Acceptance Rate (FAR)', color='red', lw=2)
    plt.plot(thresholds, frr, label='False Rejection Rate (FRR)', color='blue', lw=2)
    
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.title('FAR and FRR vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def generate_latency_histogram(results: List[PairResult], save_path: Path = PLOTS_DIR / "latency_histogram.png"):
    set_style()
    
    total_times = [r['total_time'] * 1000 for r in results if r['success']] # in ms
    
    if not total_times:
        return
        
    plt.figure()
    sns.histplot(total_times, color='green', kde=True, bins=50)
    plt.xlabel('Pipeline Time (ms)')
    plt.ylabel('Count')
    plt.title('Latency Histogram')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def generate_all_plots(results: List[PairResult], threshold_results: List[ThresholdResult]):
    """Generates and saves all requested visualizations."""
    generate_roc_curve(results)
    generate_pr_curve(results)
    generate_confusion_matrix(results)
    generate_score_distributions(results)
    generate_threshold_curves(threshold_results)
    generate_far_frr_curve(threshold_results)
    generate_latency_histogram(results)
