import json
import csv
from dataclasses import asdict
from typing import List, Dict, Any
from pathlib import Path
from .models import EvaluationSummary
from .types import PairResult
from .config import CSV_DIR, JSON_DIR, FAILURES_DIR
from .logger import get_evaluation_logger

logger = get_evaluation_logger("output")

def save_json(data: Any, path: Path):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.debug(f"Saved JSON to {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")

def save_csv(data: List[Dict], path: Path):
    if not data:
        return
    try:
        keys = data[0].keys()
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        logger.debug(f"Saved CSV to {path}")
    except Exception as e:
        logger.error(f"Failed to save CSV to {path}: {e}")

def export_results(summary: EvaluationSummary, all_results: List[PairResult]):
    """Exports all results to JSON and CSV formats."""
    
    # 1. Summary JSON & CSV
    summary_dict = asdict(summary)
    save_json(summary_dict, JSON_DIR / "summary.json")
    
    # Flatten summary for CSV
    flat_summary = {
        "total_pairs": summary.total_pairs,
        "successful_pairs": summary.successful_pairs,
        "failed_pairs": summary.failed_pairs,
        "optimal_threshold_accuracy": summary.optimal_threshold_accuracy,
        "optimal_threshold_f1": summary.optimal_threshold_f1,
        "optimal_threshold_eer": summary.optimal_threshold_eer,
        **asdict(summary.metrics),
        **asdict(summary.benchmark),
        **asdict(summary.system)
    }
    save_csv([flat_summary], CSV_DIR / "summary.csv")
    
    # 2. Metrics JSON & CSV
    save_json(asdict(summary.metrics), JSON_DIR / "metrics.json")
    save_csv([asdict(summary.metrics)], CSV_DIR / "metrics.csv")
    
    # 3. Benchmark JSON & CSV
    save_json(asdict(summary.benchmark), JSON_DIR / "benchmark.json")
    save_csv([asdict(summary.benchmark)], CSV_DIR / "benchmark.csv")
    
    # 4. Failures
    failures = [r for r in all_results if not r['success'] or (r['success'] and r['predicted_match'] != r['is_match'])]
    if failures:
        # Save detailed failures
        save_json(failures, JSON_DIR / "failures.json")
        save_csv([{k: v for k, v in f.items() if k != 'error_message' or v} for f in failures], FAILURES_DIR / "failures.csv")

