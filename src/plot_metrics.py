"""
Accuracy Metric Plots
---------------------
Side‑by‑side comparison of SVM vs VQC classification metrics
across different numbers of input features.
"""

import matplotlib.pyplot as plt

# ── SVM experiment results ───────────────────────────────────────────────────
SVM_NUM_FEATURES = [16, 36, 64, 100, 144, 196, 256, 324, 400, 484, 576, 676, 784]
SVM_ACCURACY     = [0.89, 0.95, 0.96, 0.97, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]
SVM_PRECISION    = [0.89, 0.95, 0.96, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]
SVM_RECALL       = [0.89, 0.95, 0.96, 0.97, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]
SVM_F1_SCORE     = [0.89, 0.95, 0.96, 0.97, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]

# ── VQC experiment results ───────────────────────────────────────────────────
VQC_NUM_FEATURES = [16, 36, 64, 100, 144, 196, 256, 324, 400, 484, 576, 676, 784]
VQC_ACCURACY     = [0.49, 0.94, 0.93, 0.94, 0.92, 0.93, 0.93, 0.93, 0.96, 0.97, 0.97, 0.97, 0.95]
VQC_PRECISION    = [0.49, 0.94, 0.84, 1.00, 0.98, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
VQC_RECALL       = [0.76, 0.94, 0.92, 0.88, 0.86, 0.90, 0.92, 0.86, 0.92, 0.94, 0.88, 0.94, 0.90]
VQC_F1_SCORE     = [0.60, 0.94, 0.93, 0.94, 0.91, 0.95, 0.96, 0.92, 0.96, 0.97, 0.94, 0.97, 0.95]


# ── Plotting helper ──────────────────────────────────────────────────────────
def _plot_metrics(ax, num_features, accuracy, precision, recall, f1, title):
    """Draw four metric lines on the given Axes."""
    ax.plot(num_features, accuracy,  marker="o", label="Accuracy")
    ax.plot(num_features, precision, marker="o", label="Precision")
    ax.plot(num_features, recall,    marker="o", label="Recall")
    ax.plot(num_features, f1,        marker="o", label="F1 Score")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Metrics Value")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    ax.legend()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    fig, (ax_svm, ax_vqc) = plt.subplots(1, 2, figsize=(12, 6))

    _plot_metrics(
        ax_svm, SVM_NUM_FEATURES,
        SVM_ACCURACY, SVM_PRECISION, SVM_RECALL, SVM_F1_SCORE,
        title='Accuracy Metrics of SVM for Classes "0" and "8"',
    )
    _plot_metrics(
        ax_vqc, VQC_NUM_FEATURES,
        VQC_ACCURACY, VQC_PRECISION, VQC_RECALL, VQC_F1_SCORE,
        title='Accuracy Metrics of VQC for Classes "0" and "8"',
    )

    plt.tight_layout(w_pad=3)
    plt.show()


if __name__ == "__main__":
    main()
