"""Compare anomaly detection metrics across several threshold choices."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve

from dataset import DEFAULT_DATA_PATH, load_anemia_data
from evaluate import load_model
from model import anomaly_scores


def metric_row(name: str, threshold: float, y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    return {
        "method": name,
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_abnormal": int(y_pred.sum()),
    }


def threshold_sweep(args) -> pd.DataFrame:
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    data = load_anemia_data(args.data, seed=args.seed, clean_data=not args.no_cleaning, target=args.target)
    model, checkpoint = load_model(args.model, device)
    if checkpoint.get("feature_columns") != data.feature_columns:
        raise ValueError("The model feature columns do not match the current dataset.")
    if checkpoint.get("target", data.target) != data.target:
        raise ValueError("The model target does not match the current --target setting.")

    train_scores, _, _ = anomaly_scores(model, data.train_x.to(device))
    test_scores, _, _ = anomaly_scores(model, data.test_x.to(device))
    train_scores_np = train_scores.cpu().numpy()
    test_scores_np = test_scores.cpu().numpy()
    y_true = data.test_y.astype(int)

    rows: List[Dict[str, float]] = []
    rows.append(metric_row("checkpoint_fixed", checkpoint["threshold"], y_true, test_scores_np))
    for q in args.quantiles:
        threshold = np.quantile(train_scores_np, q / 100.0)
        rows.append(metric_row(f"train_p{q:g}", threshold, y_true, test_scores_np))

    candidate_thresholds = np.unique(test_scores_np)
    f1_rows = [metric_row("f1_best", threshold, y_true, test_scores_np) for threshold in candidate_thresholds]
    rows.append(max(f1_rows, key=lambda row: row["f1"]))

    fpr, tpr, roc_thresholds = roc_curve(y_true, test_scores_np)
    youden_index = int(np.argmax(tpr - fpr))
    rows.append(metric_row("youden", roc_thresholds[youden_index], y_true, test_scores_np))

    table = pd.DataFrame(rows).sort_values(["f1", "recall"], ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "threshold_sweep.csv", index=False)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved threshold comparison to {args.output_dir / 'threshold_sweep.csv'}")
    return table


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=Path("results/vae_anemia.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--quantiles", type=float, nargs="+", default=[80, 85, 90, 95, 97.5, 99])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--no-cleaning", action="store_true", help="Disable CBC value-range data cleaning.")
    parser.add_argument("--target", choices=["anemia", "abnormal"], default="anemia")
    return parser.parse_args()


if __name__ == "__main__":
    threshold_sweep(parse_args())
