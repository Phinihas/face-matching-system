from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class EvaluationMetrics:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0
    sensitivity: float = 0.0
    f1_score: float = 0.0
    balanced_accuracy: float = 0.0
    roc_auc: float = 0.0
    average_precision: float = 0.0
    equal_error_rate: float = 0.0
    false_acceptance_rate: float = 0.0
    false_rejection_rate: float = 0.0
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0
    matthews_correlation: float = 0.0
    cohen_kappa: float = 0.0

@dataclass
class BenchmarkMetrics:
    avg_detection_time: float = 0.0
    avg_alignment_time: float = 0.0
    avg_embedding_time: float = 0.0
    avg_similarity_time: float = 0.0
    avg_pipeline_time: float = 0.0
    median_time: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    std_dev_time: float = 0.0
    percentile_95_time: float = 0.0
    percentile_99_time: float = 0.0
    images_per_second: float = 0.0
    pairs_per_second: float = 0.0

@dataclass
class SystemProfile:
    avg_cpu_usage: float = 0.0
    peak_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0

@dataclass
class EvaluationSummary:
    total_pairs: int = 0
    successful_pairs: int = 0
    failed_pairs: int = 0
    metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    benchmark: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)
    system: SystemProfile = field(default_factory=SystemProfile)
    optimal_threshold_accuracy: float = 0.0
    optimal_threshold_f1: float = 0.0
    optimal_threshold_eer: float = 0.0
