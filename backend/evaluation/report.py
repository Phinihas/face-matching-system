from .models import EvaluationSummary
from .config import REPORTS_DIR

def generate_markdown_report(summary: EvaluationSummary):
    report_path = REPORTS_DIR / "evaluation_report.md"
    
    md = f"""# Face Matching Evaluation Report

## 1. Overall Summary
- **Total Pairs**: {summary.total_pairs}
- **Successful Pairs**: {summary.successful_pairs}
- **Failed Pairs**: {summary.failed_pairs}

## 2. Key Metrics
- **Accuracy**: {summary.metrics.accuracy:.4f}
- **Precision**: {summary.metrics.precision:.4f}
- **Recall**: {summary.metrics.recall:.4f}
- **Specificity**: {summary.metrics.specificity:.4f}
- **F1 Score**: {summary.metrics.f1_score:.4f}
- **ROC AUC**: {summary.metrics.roc_auc:.4f}
- **Equal Error Rate (EER)**: {summary.metrics.equal_error_rate:.4f}
- **Optimal Threshold (Accuracy)**: {summary.optimal_threshold_accuracy:.2f}

## 3. Performance Benchmark (Latency)
- **Average Detection Time**: {summary.benchmark.avg_detection_time*1000:.2f} ms
- **Average Alignment Time**: {summary.benchmark.avg_alignment_time*1000:.2f} ms
- **Average Embedding Time**: {summary.benchmark.avg_embedding_time*1000:.2f} ms
- **Average Similarity Time**: {summary.benchmark.avg_similarity_time*1000:.2f} ms
- **Average Total Pipeline Time**: {summary.benchmark.avg_pipeline_time*1000:.2f} ms
- **Throughput**: {summary.benchmark.pairs_per_second:.2f} pairs/sec ({summary.benchmark.images_per_second:.2f} images/sec)

## 4. System Profile
- **Average CPU Usage**: {summary.system.avg_cpu_usage}%
- **Peak CPU Usage**: {summary.system.peak_cpu_usage}%
- **Average Memory Usage**: {summary.system.avg_memory_usage} MB
- **Peak Memory Usage**: {summary.system.peak_memory_usage} MB

## 5. Visualizations
You can find the generated plots in the `plots/` directory:
- `roc_curve.png`
- `pr_curve.png`
- `confusion_matrix.png`
- `score_distribution.png`
- `threshold_curves.png`
- `far_frr_curve.png`
- `latency_histogram.png`
"""
    
    with open(report_path, "w") as f:
        f.write(md)

def generate_html_report(summary: EvaluationSummary):
    report_path = REPORTS_DIR / "evaluation_report.html"
    
    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Face Matching Evaluation Report</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
    h1, h2 {{ color: #333; }}
    table {{ border-collapse: collapse; width: 50%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
</style>
</head>
<body>

<h1>Face Matching Evaluation Report</h1>

<h2>1. Overall Summary</h2>
<ul>
    <li><b>Total Pairs:</b> {summary.total_pairs}</li>
    <li><b>Successful Pairs:</b> {summary.successful_pairs}</li>
    <li><b>Failed Pairs:</b> {summary.failed_pairs}</li>
</ul>

<h2>2. Key Metrics</h2>
<table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Accuracy</td><td>{summary.metrics.accuracy:.4f}</td></tr>
    <tr><td>Precision</td><td>{summary.metrics.precision:.4f}</td></tr>
    <tr><td>Recall</td><td>{summary.metrics.recall:.4f}</td></tr>
    <tr><td>Specificity</td><td>{summary.metrics.specificity:.4f}</td></tr>
    <tr><td>F1 Score</td><td>{summary.metrics.f1_score:.4f}</td></tr>
    <tr><td>ROC AUC</td><td>{summary.metrics.roc_auc:.4f}</td></tr>
    <tr><td>Equal Error Rate (EER)</td><td>{summary.metrics.equal_error_rate:.4f}</td></tr>
    <tr><td>Optimal Threshold (Accuracy)</td><td>{summary.optimal_threshold_accuracy:.2f}</td></tr>
</table>

<h2>3. Performance Benchmark (Latency)</h2>
<table>
    <tr><th>Component</th><th>Average Time (ms)</th></tr>
    <tr><td>Detection</td><td>{summary.benchmark.avg_detection_time*1000:.2f}</td></tr>
    <tr><td>Alignment</td><td>{summary.benchmark.avg_alignment_time*1000:.2f}</td></tr>
    <tr><td>Embedding</td><td>{summary.benchmark.avg_embedding_time*1000:.2f}</td></tr>
    <tr><td>Similarity</td><td>{summary.benchmark.avg_similarity_time*1000:.2f}</td></tr>
    <tr><td>Total Pipeline</td><td>{summary.benchmark.avg_pipeline_time*1000:.2f}</td></tr>
</table>
<p><b>Throughput:</b> {summary.benchmark.pairs_per_second:.2f} pairs/sec ({summary.benchmark.images_per_second:.2f} images/sec)</p>

<h2>4. System Profile</h2>
<ul>
    <li><b>Average CPU Usage:</b> {summary.system.avg_cpu_usage}%</li>
    <li><b>Peak CPU Usage:</b> {summary.system.peak_cpu_usage}%</li>
    <li><b>Average Memory Usage:</b> {summary.system.avg_memory_usage} MB</li>
    <li><b>Peak Memory Usage:</b> {summary.system.peak_memory_usage} MB</li>
</ul>

</body>
</html>
"""
    
    with open(report_path, "w") as f:
        f.write(html)

def generate_reports(summary: EvaluationSummary):
    generate_markdown_report(summary)
    generate_html_report(summary)
