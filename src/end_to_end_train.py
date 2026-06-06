"""Train and evaluate an end-to-end semi-supervised VAE classifier."""

from argparse import ArgumentParser
from pathlib import Path
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_cleaning import report_to_dict
from hybrid_train import best_threshold, evaluate_probabilities, load_hybrid_split
from model import SemiSupervisedVAE, semi_supervised_vae_loss
from train import kl_weight_for_epoch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_probabilities(model: SemiSupervisedVAE, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x_tensor = torch.from_numpy(x).to(device)
    mu, _ = model.encode(x_tensor)
    logits = model.classify(mu)
    return torch.sigmoid(logits).cpu().numpy()


def train_end_to_end(args) -> pd.DataFrame:
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    split = load_hybrid_split(args)
    if not split.cleaning_report.empty:
        split.cleaning_report.to_csv(args.output_dir / "end_to_end_data_cleaning_report.csv", index=False)
        summary = report_to_dict(split.cleaning_report)
        print(
            "End-to-end data cleaning: "
            f"{summary['original_rows']} rows -> {summary['rows_after_cleaning']} rows "
            f"({summary['dropped_rows']} dropped)"
        )

    train_dataset = TensorDataset(torch.from_numpy(split.train_x), torch.from_numpy(split.train_y))
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = SemiSupervisedVAE(input_dim=len(split.feature_columns), latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: Dict[str, List[float]] = {"total": [], "recon": [], "kl": [], "classification": [], "kl_weight": []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_classification = 0.0
        kl_weight = kl_weight_for_epoch(epoch, args.beta, args.kl_warmup_epochs)

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            x_hat, mu, log_var, logits = model(batch_x)
            loss, recon_loss, kl_loss, classification_loss = semi_supervised_vae_loss(
                batch_x,
                x_hat,
                mu,
                log_var,
                logits,
                batch_y,
                kl_weight=kl_weight,
                classification_weight=args.classification_weight,
            )
            loss.backward()
            optimizer.step()

            batch_size = len(batch_x)
            epoch_total += loss.item() * batch_size
            epoch_recon += recon_loss.item() * batch_size
            epoch_kl += kl_loss.item() * batch_size
            epoch_classification += classification_loss.item() * batch_size

        n = len(train_dataset)
        history["total"].append(epoch_total / n)
        history["recon"].append(epoch_recon / n)
        history["kl"].append(epoch_kl / n)
        history["classification"].append(epoch_classification / n)
        history["kl_weight"].append(kl_weight)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"End-to-end epoch {epoch:03d}/{args.epochs} "
                f"loss={history['total'][-1]:.4f} "
                f"recon={history['recon'][-1]:.4f} "
                f"kl={history['kl'][-1]:.4f} "
                f"cls={history['classification'][-1]:.4f} "
                f"beta={kl_weight:.4f}"
            )

    val_probabilities = predict_probabilities(model, split.val_x, device)
    test_probabilities = predict_probabilities(model, split.test_x, device)

    f1_threshold = best_threshold(split.val_y, val_probabilities)
    recall_threshold = best_threshold(split.val_y, val_probabilities, min_recall=args.min_recall)

    rows = []
    threshold_rows = []
    for threshold_name, threshold_info in [
        ("validation_f1_best", f1_threshold),
        (f"validation_recall_at_least_{args.min_recall:g}", recall_threshold),
    ]:
        test_metrics = evaluate_probabilities(split.test_y, test_probabilities, threshold_info["threshold"])
        rows.append({"model": "end_to_end_semisupervised_vae", "threshold_method": threshold_name, **test_metrics})
        threshold_rows.append(
            {
                "model": "end_to_end_semisupervised_vae",
                "threshold_method": threshold_name,
                "validation_threshold": threshold_info["threshold"],
                "validation_precision": threshold_info["precision"],
                "validation_recall": threshold_info["recall"],
                "validation_f1": threshold_info["f1"],
            }
        )

    metrics = pd.DataFrame(rows).sort_values(["f1", "recall"], ascending=False)
    thresholds = pd.DataFrame(threshold_rows).sort_values(["validation_f1", "validation_recall"], ascending=False)
    metrics.to_csv(args.output_dir / "end_to_end_metrics.csv", index=False)
    thresholds.to_csv(args.output_dir / "end_to_end_thresholds.csv", index=False)
    np.save(args.output_dir / "end_to_end_history.npy", history)
    np.savez(
        args.output_dir / "end_to_end_predictions.npz",
        test_y=split.test_y,
        test_probability=test_probabilities,
        val_y=split.val_y,
        val_probability=val_probabilities,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": split.feature_columns,
            "scaler_mean": split.scaler.mean_,
            "scaler_scale": split.scaler.scale_,
            "latent_dim": args.latent_dim,
            "input_dim": len(split.feature_columns),
            "label_column": split.label_column,
            "target": args.target,
            "positive_label": split.positive_label,
            "beta": args.beta,
            "kl_warmup_epochs": args.kl_warmup_epochs,
            "classification_weight": args.classification_weight,
            "history": history,
        },
        args.output_dir / "end_to_end_vae.pt",
    )

    print("\nEnd-to-end validation thresholds:")
    print(thresholds.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nEnd-to-end test metrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved end-to-end outputs to {args.output_dir}")
    return metrics


def parse_args():
    parser = ArgumentParser(description="Train an end-to-end semi-supervised VAE classifier.")
    parser.add_argument("--data", type=Path, default=Path("cbc_8_features_reports_only.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/end_to_end"))
    parser.add_argument("--target", choices=["anemia", "abnormal"], default="anemia")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--kl-warmup-epochs", type=int, default=50)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--no-cleaning", action="store_true", help="Disable CBC value-range data cleaning.")
    return parser.parse_args()


if __name__ == "__main__":
    train_end_to_end(parse_args())
