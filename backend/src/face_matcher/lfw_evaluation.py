import os
import csv
import time
import logging
import json
import asyncio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
from concurrent.futures import ThreadPoolExecutor, as_completed

# Updated import
from face_matcher.main import compare_faces_eval
from face_matcher.helpers.constants import MATCH_THRESHOLD

logger = logging.getLogger(__name__)


def parse_lfw_pairs(pairs_csv: str, lfw_dir: str):
    """Parse LFW pairs.csv (4-column CSV, empty 4th column = match)"""
    pairs, labels = [], []
    missing_count = 0

    with open(pairs_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header if present

        for row in reader:
            row = [r.strip() for r in row]
            if not row or len(row) < 3:
                continue

            # Matched pair
            if len(row) >= 4 and row[3] == '':
                name = row[0]
                img1_num = row[1].zfill(4)
                img2_num = row[2].zfill(4)
                img1_path = os.path.join(lfw_dir, name, f"{name}_{img1_num}.jpg")
                img2_path = os.path.join(lfw_dir, name, f"{name}_{img2_num}.jpg")
                label = 1

            # Mismatched pair
            elif len(row) >= 4 and row[3] != '':
                name1 = row[0]
                img1_num = row[1].zfill(4)
                name2 = row[2]
                img2_num = row[3].zfill(4)
                img1_path = os.path.join(lfw_dir, name1, f"{name1}_{img1_num}.jpg")
                img2_path = os.path.join(lfw_dir, name2, f"{name2}_{img2_num}.jpg")
                label = 0
            else:
                continue

            # Check existence
            if os.path.exists(img1_path) and os.path.exists(img2_path):
                pairs.append((img1_path, img2_path))
                labels.append(label)
            else:
                missing_count += 1
                logger.debug(f"Missing file(s): {img1_path if not os.path.exists(img1_path) else ''} "
                             f"{img2_path if not os.path.exists(img2_path) else ''}")

    print(f"✅ Loaded {len(pairs)} valid pairs")
    print(f"⚠️ Skipped {missing_count} pairs due to missing images")
    print(f"Matches: {labels.count(1)}, Mismatches: {labels.count(0)}")

    return pairs, labels


async def process_pair_async(idx, img1, img2, threshold):
    """Async version of process_pair"""
    try:
        result = await compare_faces_eval(img1, img2)  # async call
        if hasattr(result, "body"):
            result_data = json.loads(result.body.decode()) if isinstance(result.body, bytes) else json.loads(result.body)
        else:
            result_data = result

        match_percentage = result_data.get("match_percentage", 0)
        prediction = 1 if result_data.get("result") == "match" else 0

        return idx, prediction, match_percentage, False

    except Exception as e:
        logger.error(f"Error processing pair {idx}: {e}")
        return idx, 0, -1.0, True


def calculate_comprehensive_metrics(labels, predictions):
    """Calculate metrics including confusion matrix safely"""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)

    # Ensure 2x2 confusion matrix
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'fpr': fpr,
        'fnr': fnr,
        'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
    }


def save_results_to_file(results, filename="lfw_evaluation_results.json"):
    results['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")


async def evaluate_lfw_async(lfw_dir: str, pairs_csv: str, threshold: float = MATCH_THRESHOLD,
                             max_pairs: int = None, max_workers: int = 4, save_file: str = None):
    print(f"=== LFW ASYNC EVALUATION START ===")
    pairs, labels = parse_lfw_pairs(pairs_csv, lfw_dir)
    if max_pairs:
        pairs, labels = pairs[:max_pairs], labels[:max_pairs]

    pair_data = [(i, img1, img2, threshold) for i, (img1, img2) in enumerate(pairs)]
    predictions, similarities = [0] * len(pairs), [-1.0] * len(pairs)
    failed_pairs = 0
    start_time = time.time()

    semaphore = asyncio.Semaphore(max_workers)  # limit concurrency

    async def sem_task(idx, img1, img2, threshold):
        async with semaphore:
            return await process_pair_async(idx, img1, img2, threshold)

    tasks = [sem_task(idx, img1, img2, threshold) for idx, img1, img2, threshold in pair_data]
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing pairs"):
        idx, prediction, similarity, failed = await future
        predictions[idx], similarities[idx] = prediction, similarity
        if failed:
            failed_pairs += 1

    metrics = calculate_comprehensive_metrics(labels, predictions)
    valid_indices = [i for i, sim in enumerate(similarities) if sim > -1.0]

    auc = 0.0
    if valid_indices:
        valid_labels = [labels[i] for i in valid_indices]
        valid_similarities = [similarities[i] for i in valid_indices]
        auc = roc_auc_score(valid_labels, valid_similarities)

    total_time = time.time() - start_time
    num_matches = labels.count(1)
    num_mismatches = labels.count(0)

    results = {
        **metrics,
        'auc': auc,
        'threshold': threshold,
        'total_pairs': len(pairs),
        'failed_pairs': failed_pairs,
        'num_matches': num_matches,
        'num_mismatches': num_mismatches,
        'processing_time': total_time,
        'avg_time_per_pair': total_time / len(pairs) if pairs else 0
    }

    print(f"=== LFW ASYNC EVALUATION RESULTS ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}, AUC: {auc:.4f}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")
    print(f"Matches: {num_matches}, Mismatches: {num_mismatches}")
    print(f"Failed pairs: {failed_pairs}/{len(pairs)} | Time: {total_time:.2f}s")

    if save_file:
        save_results_to_file(results, save_file)

    return results

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()  # only needed in Jupyter/notebooks

    lfw_dataset_path = "/mnt/data/aidata/vamshi/face maface_matcher/lfw-deepfunneled"
    pairs_file = "/mnt/data/aidata/vamshi/face maface_matcher/pairs.csv"

    print(">>> Evaluating on ALL valid pairs in CSV (ASYNC)...")
    results = asyncio.run(evaluate_lfw_async(
        lfw_dataset_path,
        pairs_file,
        max_pairs=None,
        max_workers=32,
        save_file="lfw_all_pairs_results_async.json"
    ))

    print("\n✅ Async evaluation completed!")
    print(json.dumps(results, indent=2))

