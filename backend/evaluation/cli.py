import argparse
from pathlib import Path
import sys
import torch
import cv2

# --- CPU Optimization ---
# Restrict PyTorch and OpenCV internal threads to 1.
# Since we use ThreadPoolExecutor to process multiple faces concurrently,
# allowing PyTorch/OpenCV to spawn their own threads per-worker causes
# severe CPU thrashing (100% CPU lockup and 3x slower execution).
torch.set_num_threads(1)
cv2.setNumThreads(1)
from .config import DEFAULT_DATASET_DIR, DEFAULT_PAIRS_FILE, DEFAULT_THRESHOLD, DEFAULT_WORKERS, OUTPUT_DIR
from .logger import get_evaluation_logger
from .dataset_loader import LFWDatasetLoader
from .evaluator import FaceMatchingEvaluator
from .output import export_results
from .report import generate_reports
from .plots import generate_all_plots

logger = get_evaluation_logger("cli")

def print_final_summary(summary, output_folder):
    """Prints the final summary to the console as requested."""
    
    print("\n" + "="*50)
    print(f"Total Pairs: {summary.total_pairs}")
    print(f"Successful Pairs: {summary.successful_pairs}")
    print(f"Failed Pairs: {summary.failed_pairs}")
    print(f"Accuracy: {summary.metrics.accuracy:.4f}")
    print(f"Precision: {summary.metrics.precision:.4f}")
    print(f"Recall: {summary.metrics.recall:.4f}")
    print(f"Specificity: {summary.metrics.specificity:.4f}")
    print(f"F1: {summary.metrics.f1_score:.4f}")
    print(f"ROC AUC: {summary.metrics.roc_auc:.4f}")
    print(f"Average Precision: {summary.metrics.average_precision:.4f}")
    print(f"EER: {summary.metrics.equal_error_rate:.4f}")
    print(f"Optimal Threshold: {summary.optimal_threshold_accuracy:.2f}")
    print(f"Average Detection Time: {summary.benchmark.avg_detection_time*1000:.2f} ms")
    print(f"Average Embedding Time: {summary.benchmark.avg_embedding_time*1000:.2f} ms")
    print(f"Average Similarity Time: {summary.benchmark.avg_similarity_time*1000:.2f} ms")
    print(f"Average Pipeline Time: {summary.benchmark.avg_pipeline_time*1000:.2f} ms")
    print(f"Pairs Per Second: {summary.benchmark.pairs_per_second:.2f}")
    print(f"Average CPU: {summary.system.avg_cpu_usage}%")
    print(f"Peak CPU: {summary.system.peak_cpu_usage}%")
    print(f"Average Memory: {summary.system.avg_memory_usage} MB")
    print(f"Peak Memory: {summary.system.peak_memory_usage} MB")
    print(f"Output Folder: {output_folder}")
    print("="*50 + "\n")

def run_cli():
    parser = argparse.ArgumentParser(description="Face Matching Evaluation Framework")
    
    parser.add_argument('--dataset', type=str, default=str(DEFAULT_DATASET_DIR),
                        help='Path to the image dataset directory')
    parser.add_argument('--pairs', type=str, default=str(DEFAULT_PAIRS_FILE),
                        help='Path to the pairs file (csv or txt)')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='Similarity threshold for predictions')
    parser.add_argument('--save-plots', action='store_true',
                        help='Generate and save evaluation plots')
    parser.add_argument('--benchmark', action='store_true',
                        help='Include system and latency benchmarks')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help='Number of concurrent workers')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of pairs to evaluate (for testing)')
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset)
    pairs_file = Path(args.pairs)
    
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)
        
    from tqdm import tqdm
    
    logger.info(f"Starting evaluation with threshold {args.threshold} and {args.workers} workers.")
    
    # Initialize tqdm progress bar
    pbar = tqdm(desc="Stage: Loading dataset", unit="pair", 
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    
    # 1. Load Dataset
    loader = LFWDatasetLoader(dataset_dir, pairs_file)
    pairs = loader.load()
    
    if args.limit:
        pairs = pairs[:args.limit]
        logger.info(f"Limited evaluation to {args.limit} pairs.")
        
    if not pairs:
        logger.error("No valid pairs found to evaluate.")
        sys.exit(1)
        
    # Update progress bar total now that pairs are loaded
    pbar.total = len(pairs)
    pbar.refresh()
        
    # 2. Run Evaluation
    evaluator = FaceMatchingEvaluator(threshold=args.threshold)
    summary, results, threshold_results = evaluator.run_evaluation(pairs, workers=args.workers, pbar=pbar)
    
    # 3. Export Data
    pbar.set_description("Stage: Exporting Data")
    logger.info("Exporting results to JSON and CSV...")
    export_results(summary, results)
    
    # 4. Generate Reports
    pbar.set_description("Stage: Generating Reports")
    logger.info("Generating Markdown and HTML reports...")
    generate_reports(summary)
    
    # 5. Generate Plots
    if args.save_plots:
        pbar.set_description("Stage: Generating Plots")
        logger.info("Generating visualizations...")
        generate_all_plots(results, threshold_results)
        
    pbar.set_description("Stage: Evaluation Completed")
    pbar.close()
    
    # 6. Final Console Summary
    print_final_summary(summary, str(OUTPUT_DIR))
    
    logger.info("Evaluation completed successfully.")

if __name__ == "__main__":
    run_cli()
