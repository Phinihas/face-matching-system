# Face Matching Evaluation Report

## 1. Overall Summary
- **Total Pairs**: 6000
- **Successful Pairs**: 6000
- **Failed Pairs**: 0

## 2. Key Metrics
- **Accuracy**: 0.9983
- **Precision**: 1.0000
- **Recall**: 0.9967
- **Specificity**: 1.0000
- **F1 Score**: 0.9983
- **ROC AUC**: 0.9986
- **Equal Error Rate (EER)**: 0.0040
- **Optimal Threshold (Accuracy)**: 0.20

## 3. Performance Benchmark (Latency)
- **Average Detection Time**: 37.43 ms
- **Average Alignment Time**: 0.00 ms
- **Average Embedding Time**: 747.94 ms
- **Average Similarity Time**: 0.50 ms
- **Average Total Pipeline Time**: 1575.79 ms
- **Throughput**: 2.54 pairs/sec (5.07 images/sec)

## 4. System Profile
- **Average CPU Usage**: 71.96%
- **Peak CPU Usage**: 100.0%
- **Average Memory Usage**: 13002.6 MB
- **Peak Memory Usage**: 13558.25 MB

## 5. Visualizations
You can find the generated plots in the `plots/` directory:
- `roc_curve.png`
- `pr_curve.png`
- `confusion_matrix.png`
- `score_distribution.png`
- `threshold_curves.png`
- `far_frr_curve.png`
- `latency_histogram.png`
