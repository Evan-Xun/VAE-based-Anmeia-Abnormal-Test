"""Create the required plot-generation task deliverables."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from dataset import DEFAULT_DATA_PATH


def load_history(history_path: Path) -> dict:
    """Load the saved training-history dictionary."""
    history = np.load(history_path, allow_pickle=True)
    if getattr(history, "shape", ()) == ():
        history = history.item()
    if not isinstance(history, dict):
        raise TypeError(f"Expected a dict-like history in {history_path}, got {type(history)!r}")
    return history


def save_training_loss(history: dict, output_dir: Path, warmup_end_epoch: int) -> None:
    """Plot total, reconstruction, and KL losses across epochs."""
    epochs = np.arange(1, len(history["total"]) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["total"], color="tab:blue", label="Total Loss")
    plt.plot(epochs, history["recon"], color="tab:orange", label="Reconstruction Loss")
    plt.plot(epochs, history["kl"], color="tab:green", label="KL Loss")
    plt.axvline(
        warmup_end_epoch,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="KL Warmup End",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss Curve")
    plt.xlim(0, max(len(epochs), warmup_end_epoch))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss_curve.png", dpi=150)
    plt.close()


def save_score_distribution(
    y_true: np.ndarray,
    scores: np.ndarray,
    output_dir: Path,
    threshold: float,
) -> None:
    """Plot anomaly score histograms for healthy and anemia samples."""
    healthy_scores = scores[y_true == 0]
    abnormal_scores = scores[y_true == 1]
    plt.figure(figsize=(10, 5))
    plt.hist(
        healthy_scores,
        bins=40,
        alpha=0.6,
        label="Healthy",
        color="tab:blue",
    )
    plt.hist(
        abnormal_scores,
        bins=40,
        alpha=0.6,
        label="Anemia",
        color="tab:red",
    )
    plt.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold = {threshold:.4f}",
    )
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.title("Anomaly Score Distribution — Test Set")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "anomaly_score_distribution.png", dpi=150)
    plt.close()


def save_roc_curve_comparison(
    y_true: np.ndarray,
    vae_scores: np.ndarray,
    supervised_scores: np.ndarray,
    hybrid_scores: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot ROC curves for the baseline, supervised, and hybrid models."""
    plt.figure(figsize=(8, 7))
    for name, scores, color in [
        ("Baseline VAE", vae_scores, "tab:blue"),
        ("Supervised LR", supervised_scores, "tab:orange"),
        ("Hybrid LR", hybrid_scores, "tab:green"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auroc = roc_auc_score(y_true, scores)
        plt.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUROC = {auroc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (AUROC = 0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve_comparison.png", dpi=150)
    plt.close()


def create_plots(args) -> None:
    """Create the three required plot-generation task deliverables."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = load_history(args.history)
    baseline_outputs = np.load(args.outputs)
    hybrid_outputs = np.load(args.hybrid_outputs)

    y_true = baseline_outputs["y_true"]
    vae_scores = baseline_outputs["scores"]
    hybrid_y_true = hybrid_outputs["test_y"]
    if not np.array_equal(y_true, hybrid_y_true):
        raise ValueError("Baseline and hybrid prediction files do not use the same test split.")

    save_training_loss(history, args.output_dir, warmup_end_epoch=args.kl_warmup_epochs)
    save_score_distribution(y_true, vae_scores, args.output_dir, threshold=args.threshold)
    save_roc_curve_comparison(
        y_true,
        vae_scores,
        hybrid_outputs["supervised_only_logistic_regression_test_probability"],
        hybrid_outputs["hybrid_logistic_regression_test_probability"],
        args.output_dir,
    )

    print(f"Saved plots to {args.output_dir}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/final_reports_only_keep_duplicates/evaluation_outputs.npz"),
    )
    parser.add_argument(
        "--hybrid-outputs",
        type=Path,
        default=Path("results/final_reports_only_keep_duplicates/hybrid_predictions.npz"),
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("results/final_reports_only_keep_duplicates/training_history.npy"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--threshold", type=float, default=0.4767)
    parser.add_argument("--kl-warmup-epochs", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    create_plots(parse_args())
