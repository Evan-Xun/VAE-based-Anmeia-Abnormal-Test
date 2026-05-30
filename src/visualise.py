"""Create required plots for the anemia VAE baseline."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from dataset import DEFAULT_DATA_PATH, load_anemia_data
from evaluate import load_model
from model import anomaly_scores


def save_training_loss(history: dict, output_dir: Path) -> None:
    """Plot total, reconstruction, and KL losses across epochs."""
    epochs = np.arange(1, len(history["total"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["total"], label="Total loss")
    plt.plot(epochs, history["recon"], label="Reconstruction loss")
    plt.plot(epochs, history["kl"], label="KL loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss_curve.png", dpi=200)
    plt.close()


def save_score_distribution(y_true: np.ndarray, scores: np.ndarray, output_dir: Path, positive_label: str) -> None:
    """Plot anomaly score histograms for normal and anemia samples."""
    healthy_scores = scores[y_true == 0]
    abnormal_scores = scores[y_true == 1]
    plt.figure(figsize=(8, 5))
    plot_max = float(np.percentile(scores, 99))
    bins = np.linspace(float(scores.min()), plot_max, 31)
    plt.hist(
        abnormal_scores[abnormal_scores <= plot_max],
        bins=bins,
        histtype="step",
        linewidth=2.2,
        label=f"{positive_label} (n={len(abnormal_scores)})",
        color="tab:orange",
    )
    plt.hist(
        healthy_scores[healthy_scores <= plot_max],
        bins=bins,
        histtype="step",
        linewidth=2.2,
        label=f"Healthy (n={len(healthy_scores)})",
        color="tab:blue",
    )
    clipped_count = int((scores > plot_max).sum())
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.title(f"Anomaly Score Distribution (99th percentile view, {clipped_count} hidden)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "anomaly_score_distribution.png", dpi=200)
    plt.close()


def save_roc_curve(y_true: np.ndarray, scores: np.ndarray, results: np.lib.npyio.NpzFile, output_dir: Path) -> None:
    """Plot ROC curve with AUROC annotation."""
    auroc = roc_auc_score(y_true, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(results["fpr"], results["tpr"], label=f"VAE AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()


def save_latent_scatter(
    model,
    test_x: torch.Tensor,
    y_true: np.ndarray,
    output_dir: Path,
    device: torch.device,
    positive_label: str,
) -> None:
    """Plot the 2D latent mean space colored by label."""
    with torch.no_grad():
        mu, _ = model.encode(test_x.to(device))
    mu_np = mu.cpu().numpy()

    plt.figure(figsize=(7, 6))
    healthy_count = int((y_true == 0).sum())
    abnormal_count = int((y_true == 1).sum())
    plt.scatter(mu_np[y_true == 0, 0], mu_np[y_true == 0, 1], s=18, alpha=0.75, label=f"Healthy (n={healthy_count})")
    plt.scatter(mu_np[y_true == 1, 0], mu_np[y_true == 1, 1], s=18, alpha=0.75, label=f"{positive_label} (n={abnormal_count})")
    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.title("Latent Space Scatter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "latent_space_scatter.png", dpi=200)
    plt.close()


def save_reconstruction_examples(model, data, y_true: np.ndarray, output_dir: Path, device: torch.device) -> None:
    """Show 3 normal and 3 anemia records with reconstructed values."""
    normal_idx = np.where(y_true == 0)[0][:3]
    anemia_idx = np.where(y_true == 1)[0][:3]
    selected = np.concatenate([normal_idx, anemia_idx])

    with torch.no_grad():
        mu, _ = model.encode(data.test_x[selected].to(device))
        recon_scaled = model.decode(mu).cpu().numpy()

    original = data.scaler.inverse_transform(data.test_x[selected].numpy())
    reconstructed = data.scaler.inverse_transform(recon_scaled)

    rows = []
    positive_label = getattr(data, "positive_label", "Abnormal")
    labels = ["Healthy"] * len(normal_idx) + [positive_label] * len(anemia_idx)
    for i, label in enumerate(labels):
        for j, feature in enumerate(data.feature_columns):
            rows.append(
                {
                    "Sample": f"{label} {i % 3 + 1}",
                    "Feature": feature,
                    "Original": original[i, j],
                    "Reconstructed": reconstructed[i, j],
                }
            )

    table_df = pd.DataFrame(rows)
    pivot_text = [
        [
            row["Sample"],
            row["Feature"],
            f"{row['Original']:.2f}",
            f"{row['Reconstructed']:.2f}",
        ]
        for _, row in table_df.iterrows()
    ]

    figure_height = max(7, 0.28 * len(pivot_text) + 1.5)
    plt.figure(figsize=(9, figure_height))
    plt.axis("off")
    table = plt.table(
        cellText=pivot_text,
        colLabels=["Sample", "Feature", "Original", "Reconstructed"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    plt.title("Reconstruction Examples")
    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_examples.png", dpi=200)
    plt.close()


def create_plots(args) -> None:
    """Create all five required Session 6 plots."""
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_anemia_data(args.data, seed=args.seed, clean_data=not args.no_cleaning, target=args.target)
    model, checkpoint = load_model(args.model, device)
    if checkpoint.get("feature_columns") != data.feature_columns:
        raise ValueError(
            "The model was trained with different feature columns than the current dataset. "
            "Retrain the model or pass the matching --data file."
        )
    if checkpoint.get("target", data.target) != data.target:
        raise ValueError("The model target does not match the current --target setting.")
    outputs = np.load(args.outputs)
    y_true = outputs["y_true"]
    scores = outputs["scores"]
    if len(data.test_x) != len(y_true):
        raise ValueError(
            "The evaluation outputs do not match the current data cleaning setting. "
            "Run evaluation again with the same --data and cleaning options."
        )

    save_training_loss(checkpoint["history"], args.output_dir)
    save_score_distribution(y_true, scores, args.output_dir, data.positive_label)
    save_roc_curve(y_true, scores, outputs, args.output_dir)
    save_latent_scatter(model, data.test_x, y_true, args.output_dir, device, data.positive_label)
    save_reconstruction_examples(model, data, y_true, args.output_dir, device)

    print(f"Saved plots to {args.output_dir}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=Path("results/vae_anemia.pt"))
    parser.add_argument("--outputs", type=Path, default=Path("results/evaluation_outputs.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--no-cleaning", action="store_true", help="Disable CBC value-range data cleaning.")
    parser.add_argument("--target", choices=["anemia", "abnormal"], default="anemia")
    return parser.parse_args()


if __name__ == "__main__":
    create_plots(parse_args())
