"""Evaluate the trained anemia VAE and report anomaly detection metrics."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from dataset import DEFAULT_DATA_PATH, load_anemia_baseline_split
from model import VAE, anomaly_scores


def load_model(model_path: Path, device: torch.device) -> Tuple[VAE, dict]:
    """Load a trained VAE checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    input_dim = checkpoint.get("input_dim", len(checkpoint["feature_columns"]))
    model = VAE(input_dim=input_dim, latent_dim=checkpoint["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def metric_row(
    split_name: str,
    threshold_method: str,
    threshold: float,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    """Evaluate anomaly scores at a given threshold."""
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "split": split_name,
        "threshold_method": threshold_method,
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "specificity": float(specificity),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_abnormal": int(y_pred.sum()),
    }


def validation_threshold(
    method: str,
    checkpoint_threshold: float,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    val_y: np.ndarray,
    min_recall: float,
    train_quantile: float,
) -> Dict[str, float]:
    """Choose a threshold without touching the test set."""
    if method == "checkpoint_fixed":
        threshold = checkpoint_threshold
        return metric_row("validation", method, threshold, val_y, val_scores)

    if method == "train_quantile":
        threshold = float(np.quantile(train_scores, train_quantile / 100.0))
        return metric_row("validation", f"{method}_p{train_quantile:g}", threshold, val_y, val_scores)

    candidate_thresholds = np.unique(val_scores)
    rows = []
    for threshold in candidate_thresholds:
        row = metric_row("validation", method, float(threshold), val_y, val_scores)
        if method == "validation_recall_floor" and row["recall"] < min_recall:
            continue
        rows.append(row)

    if method == "validation_youden":
        fpr, tpr, thresholds = roc_curve(val_y, val_scores)
        best_index = int(np.argmax(tpr - fpr))
        return metric_row("validation", method, float(thresholds[best_index]), val_y, val_scores)

    if not rows:
        rows = [metric_row("validation", method, float(threshold), val_y, val_scores) for threshold in candidate_thresholds]

    return max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))


def evaluate(args) -> Dict[str, float]:
    """Compute validation-selected threshold metrics on the held-out test set."""
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    data = load_anemia_baseline_split(
        args.data,
        seed=args.seed,
        clean_data=not args.no_cleaning,
        target=args.target,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    model, checkpoint = load_model(args.model, device)
    if checkpoint.get("feature_columns") != data.feature_columns:
        raise ValueError(
            "The model was trained with different feature columns than the current dataset. "
            "Retrain the model or pass the matching --data file."
        )
    if checkpoint.get("target", data.target) != data.target:
        raise ValueError("The model target does not match the current --target setting.")

    train_scores, _, _ = anomaly_scores(model, data.train_x.to(device))
    val_scores, _, _ = anomaly_scores(model, data.val_x.to(device))
    test_scores, test_recon_error, test_latent_dist = anomaly_scores(model, data.test_x.to(device))
    train_scores_np = train_scores.cpu().numpy()
    val_scores_np = val_scores.cpu().numpy()
    test_scores_np = test_scores.cpu().numpy()

    selected = validation_threshold(
        method=args.threshold_method,
        checkpoint_threshold=float(checkpoint["threshold"]),
        train_scores=train_scores_np,
        val_scores=val_scores_np,
        val_y=data.val_y.astype(int),
        min_recall=args.min_recall,
        train_quantile=args.train_quantile,
    )
    test_metrics = metric_row(
        "test",
        selected["threshold_method"],
        float(selected["threshold"]),
        data.test_y.astype(int),
        test_scores_np,
    )

    fpr, tpr, roc_thresholds = roc_curve(data.test_y.astype(int), test_scores_np)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "evaluation_outputs.npz",
        y_true=data.test_y.astype(int),
        scores=test_scores_np,
        y_pred=(test_scores_np >= selected["threshold"]).astype(int),
        fpr=fpr,
        tpr=tpr,
        roc_thresholds=roc_thresholds,
        recon_error=test_recon_error.cpu().numpy(),
        latent_dist=test_latent_dist.cpu().numpy(),
        test_x_scaled=data.test_x.numpy(),
        validation_scores=val_scores_np,
        validation_y=data.val_y.astype(int),
    )

    baseline_metrics = pd.DataFrame(
        [
            {**selected},
            {**test_metrics},
        ]
    )
    baseline_metrics.to_csv(args.output_dir / "baseline_metrics.csv", index=False)

    print(f"Validation threshold method: {selected['threshold_method']}")
    print(f"Validation threshold: {selected['threshold']:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Test AUPRC: {test_metrics['auprc']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1-score: {test_metrics['f1']:.4f}")
    print(f"Test Specificity: {test_metrics['specificity']:.4f}")
    return test_metrics


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=Path("results/vae_anemia.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--no-cleaning", action="store_true", help="Disable CBC value-range data cleaning.")
    parser.add_argument("--target", choices=["anemia", "abnormal"], default="anemia")
    parser.add_argument(
        "--threshold-method",
        choices=[
            "checkpoint_fixed",
            "train_quantile",
            "validation_f1",
            "validation_recall_floor",
            "validation_youden",
        ],
        default="validation_f1",
    )
    parser.add_argument("--train-quantile", type=float, default=95.0)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
