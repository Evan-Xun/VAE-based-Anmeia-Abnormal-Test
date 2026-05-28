"""Evaluate the trained anemia VAE and report anomaly detection metrics."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from dataset import DEFAULT_DATA_PATH, load_anemia_data
from model import VAE, anomaly_scores


def load_model(model_path: Path, device: torch.device) -> Tuple[VAE, dict]:
    """Load a trained VAE checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    input_dim = checkpoint.get("input_dim", len(checkpoint["feature_columns"]))
    model = VAE(input_dim=input_dim, latent_dim=checkpoint["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate(args) -> Dict[str, float]:
    """Compute AUROC and threshold metrics on the full held-out test set."""
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    data = load_anemia_data(args.data, seed=args.seed, clean_data=not args.no_cleaning, target=args.target)
    model, checkpoint = load_model(args.model, device)
    if checkpoint.get("feature_columns") != data.feature_columns:
        raise ValueError(
            "The model was trained with different feature columns than the current dataset. "
            "Retrain the model or pass the matching --data file."
        )
    if checkpoint.get("target", data.target) != data.target:
        raise ValueError("The model target does not match the current --target setting.")

    scores, recon_error, latent_dist = anomaly_scores(model, data.test_x.to(device))
    scores_np = scores.cpu().numpy()
    y_true = data.test_y.astype(int)
    y_pred = (scores_np >= checkpoint["threshold"]).astype(int)

    fpr, tpr, roc_thresholds = roc_curve(y_true, scores_np)
    metrics = {
        "auroc": float(roc_auc_score(y_true, scores_np)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(checkpoint["threshold"]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "evaluation_outputs.npz",
        y_true=y_true,
        scores=scores_np,
        y_pred=y_pred,
        fpr=fpr,
        tpr=tpr,
        roc_thresholds=roc_thresholds,
        recon_error=recon_error.cpu().numpy(),
        latent_dist=latent_dist.cpu().numpy(),
        test_x_scaled=data.test_x.numpy(),
    )

    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"Threshold: {metrics['threshold']:.4f}")
    return metrics


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=Path("results/vae_anemia.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--no-cleaning", action="store_true", help="Disable CBC value-range data cleaning.")
    parser.add_argument("--target", choices=["anemia", "abnormal"], default="anemia")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
