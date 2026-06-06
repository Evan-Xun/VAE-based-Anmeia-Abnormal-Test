from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "report_assets"
ASSET_DIR.mkdir(exist_ok=True)


def read_single_row(csv_path: Path, match: dict[str, str]) -> dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if all(row.get(k) == v for k, v in match.items()):
                return row
    raise ValueError(f"No row matched {match} in {csv_path}")


def draw_box(ax, x, y, w, h, text, fc="#EAF2FF", ec="#4A6FA5", fontsize=12):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.4,
            color="#4A4A4A",
        )
    )


def make_workflow_figure():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.03, 0.35, 0.16, 0.28, "Data Cleaning\nand Filtering"),
        (0.22, 0.35, 0.16, 0.28, "Train / Val / Test\nSplit"),
        (0.41, 0.35, 0.16, 0.28, "Baseline VAE\nTraining"),
        (0.60, 0.35, 0.16, 0.28, "Model Comparison\n(Supervised / Hybrid /\nEnd-to-End)"),
        (0.79, 0.35, 0.16, 0.28, "Result Analysis\nand Reporting"),
    ]
    for x, y, w, h, t in boxes:
        draw_box(ax, x, y, w, h, t)
    for i in range(len(boxes) - 1):
        x, y, w, h, _ = boxes[i]
        nx, ny, nw, nh, _ = boxes[i + 1]
        arrow(ax, x + w, y + h / 2, nx, ny + nh / 2)

    ax.set_title("Figure 1. Overall Experimental Workflow", fontsize=16, pad=16)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "figure1_workflow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_model_routes_figure():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # baseline lane
    ax.text(0.17, 0.92, "Baseline VAE", fontsize=14, fontweight="bold", ha="center")
    draw_box(ax, 0.05, 0.74, 0.18, 0.11, "CBC Features")
    draw_box(ax, 0.05, 0.56, 0.18, 0.11, "Encoder / Decoder")
    draw_box(ax, 0.05, 0.38, 0.18, 0.11, "Anomaly Score")
    draw_box(ax, 0.05, 0.20, 0.18, 0.11, "Threshold-based\nPrediction")
    arrow(ax, 0.14, 0.74, 0.14, 0.67)
    arrow(ax, 0.14, 0.56, 0.14, 0.49)
    arrow(ax, 0.14, 0.38, 0.14, 0.31)

    # hybrid lane
    ax.text(0.50, 0.92, "Loose-Coupled Hybrid", fontsize=14, fontweight="bold", ha="center")
    draw_box(ax, 0.38, 0.74, 0.24, 0.11, "CBC Features")
    draw_box(ax, 0.38, 0.56, 0.24, 0.11, "Train VAE and Extract\nLatent / Reconstruction Features")
    draw_box(ax, 0.38, 0.38, 0.24, 0.11, "Concatenate Raw Features\nand VAE-derived Features")
    draw_box(ax, 0.38, 0.20, 0.24, 0.11, "External Classifier\n(LR / RF)")
    arrow(ax, 0.50, 0.74, 0.50, 0.67)
    arrow(ax, 0.50, 0.56, 0.50, 0.49)
    arrow(ax, 0.50, 0.38, 0.50, 0.31)

    # e2e lane
    ax.text(0.83, 0.92, "End-to-End Semi-supervised VAE", fontsize=14, fontweight="bold", ha="center")
    draw_box(ax, 0.71, 0.74, 0.24, 0.11, "CBC Features")
    draw_box(ax, 0.71, 0.56, 0.24, 0.11, "Encoder -> Latent z")
    draw_box(ax, 0.71, 0.38, 0.11, 0.11, "Decoder")
    draw_box(ax, 0.84, 0.38, 0.11, 0.11, "Classifier")
    draw_box(ax, 0.71, 0.20, 0.24, 0.11, "Joint Loss:\nRecon + KL + Classification")
    arrow(ax, 0.83, 0.74, 0.83, 0.67)
    arrow(ax, 0.83, 0.56, 0.76, 0.49)
    arrow(ax, 0.83, 0.56, 0.89, 0.49)
    arrow(ax, 0.76, 0.38, 0.76, 0.31)
    arrow(ax, 0.89, 0.38, 0.89, 0.31)

    ax.set_title("Figure 2. Model Routes Compared in This Project", fontsize=16, pad=16)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "figure2_model_routes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_baseline_cross_dataset_figure():
    datasets = ["v4", "reports-only", "merged\n(latent=4)"]
    auroc = [0.9223, 0.5146, 0.6509]
    f1 = [0.9155, 0.6619, 0.7500]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].bar(datasets, auroc, color=["#5B8FF9", "#61DDAA", "#65789B"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Baseline VAE AUROC Across Datasets")
    axes[0].set_ylabel("AUROC")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(datasets, f1, color=["#5B8FF9", "#61DDAA", "#65789B"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Baseline VAE F1 Across Datasets")
    axes[1].set_ylabel("F1")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Figure 3. Sensitivity of Baseline VAE to Dataset Conditions", fontsize=15)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "figure3_baseline_cross_dataset.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_grouped_bar_figure(filename: str, title: str, labels: list[str], auroc: list[float], f1: list[float]):
    x = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar([i - width / 2 for i in x], auroc, width=width, label="AUROC", color="#5B8FF9")
    ax.bar([i + width / 2 for i in x], f1, width=width, label="F1", color="#F6BD16")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    make_workflow_figure()
    make_model_routes_figure()
    make_baseline_cross_dataset_figure()

    make_grouped_bar_figure(
        "figure4_v4_comparison.png",
        "Figure 4. v4 Dataset Model Comparison",
        ["Baseline VAE", "Supervised RF", "Hybrid RF", "End-to-End"],
        [0.9223, 1.0000, 0.9999, 0.9868],
        [0.9155, 1.0000, 0.9905, 0.9712],
    )

    make_grouped_bar_figure(
        "figure5_report_comparison.png",
        "Figure 5. Reports-only Dataset Model Comparison",
        ["Baseline VAE", "Supervised LR", "Hybrid LR", "End-to-End"],
        [0.5146, 0.6202, 0.6332, 0.6339],
        [0.6619, 0.7013, 0.6862, 0.6507],
    )

    make_grouped_bar_figure(
        "figure6_merged_comparison.png",
        "Figure 6. Merged Dataset Comparison (latent_dim = 4)",
        ["Baseline VAE", "Supervised RF", "Hybrid RF", "End-to-End"],
        [0.6509, 0.9062, 0.8947, 0.8992],
        [0.7500, 0.8552, 0.8632, 0.8537],
    )

    shutil.copy2(ROOT / "plots" / "training_loss_curve.png", ASSET_DIR / "figure7_training_loss.png")
    shutil.copy2(ROOT / "plots" / "anomaly_score_distribution.png", ASSET_DIR / "figure8_anomaly_distribution.png")


if __name__ == "__main__":
    main()
