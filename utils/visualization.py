import os
import matplotlib.pyplot as plt

def generate_comparison_charts(metrics_results, video_path):
    """Generate visual comparison of tracker performance."""
    if not metrics_results:
        return

    # Create output directory for charts
    output_dir = "tracking_metrics"
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.basename(video_path).split('.')[0]
    output_base = os.path.join(output_dir, f"{video_name}_comparison")

    # Create bar charts for key metrics
    metrics_to_plot = {
        'effective_tracking_percentage': {'title': 'Effective Tracking (higher is better)', 'percentage': True},
        'drift_percentage': {'title': 'Background Drift (lower is better)', 'percentage': True},
        'avg_appearance_similarity': {'title': 'Appearance Similarity (higher is better)', 'percentage': True},
        'composite_score': {'title': 'Overall Score (higher is better)', 'percentage': False},
        'failure_rate': {'title': 'Failure Rate (%) — Lower is Better', 'percentage': True}
    }

    num_metrics = len(metrics_to_plot)
    cols = 2
    rows = (num_metrics + 1) // 2
    plt.figure(figsize=(6 * cols, 4.5 * rows))

    for i, (metric_name, config) in enumerate(metrics_to_plot.items(), 1):
        plt.subplot(rows, cols, i)

        trackers = list(metrics_results.keys())
        values = [metrics_results[t].get(metric_name, 0) for t in trackers]

        # Define color map for visual clarity
        if metric_name in ['avg_appearance_similarity']:
            colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in values]
        elif metric_name in ['drift_percentage', 'failure_rate']:
            colors = ['green' if v < 10 else 'orange' if v < 30 else 'red' for v in values]
        elif metric_name == 'effective_tracking_percentage':
            colors = ['green' if v > 80 else 'orange' if v > 50 else 'red' for v in values]
        elif metric_name == 'composite_score':
            colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in values]
        else:
            colors = 'skyblue'

        bars = plt.bar(trackers, values, color=colors)

        plt.title(config['title'])
        plt.xticks(rotation=45, ha='right')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            label = f"{height:.1f}%" if config['percentage'] else f"{height:.3f}"
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     label, ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_base}_metrics.png")
    print(f"Comparison charts saved to {output_base}_metrics.png")

    # Save numerical results as CSV
    csv_path = f"{output_base}_metrics.csv"
    with open(csv_path, 'w') as f:
        # Write header
        metrics_header = list(next(iter(metrics_results.values())).keys())
        f.write("tracker," + ",".join(metrics_header) + "\n")

        # Write values
        for tracker, metrics in metrics_results.items():
            f.write(f"{tracker}")
            for metric in metrics_header:
                value = metrics.get(metric, "N/A")
                if isinstance(value, float):
                    f.write(f",{value:.4f}")
                else:
                    f.write(f",{value}")
            f.write("\n")

    print(f"Metrics data saved to {csv_path}")

    # Return to command line that metrics are ready
    print("\nTracking comparison completed. See results in the tracking_metrics directory.")