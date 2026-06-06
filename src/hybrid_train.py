"""Train a semi-supervised hybrid VAE classifier for CBC anemia detection."""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import random
import warnings
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data_cleaning import report_to_dict
from dataset import DEFAULT_DATA_PATH, TARGET_LABELS, _prepare_features_and_labels, load_report_dataset
from model import VAE, anomaly_scores, latent_kl_divergence, vae_loss
from train import kl_weight_for_epoch


@dataclass
class HybridSplit:
    """Container for the hybrid train/validation/test split."""

    train_x: np.ndarray
    val_x: np.ndarray
    test_x: np.ndarray
    train_y: np.ndarray
    val_y: np.ndarray
    test_y: np.ndarray
    scaler: StandardScaler
    feature_columns: List[str]
    label_column: str
    cleaning_report: pd.DataFrame
    positive_label: str


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible hybrid training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_hybrid_split(args) -> HybridSplit:
    """Load data and create a stratified 60/20/20 split for semi-supervised learning."""
    data_path = Path(args.data)
    if data_path.is_dir():
        features, y, feature_columns, label_column, cleaning_report = load_report_dataset(
            data_path,
            clean_data=not args.no_cleaning,
            target=args.target,
        )
    else:
        df = pd.read_csv(data_path)
        features, y, feature_columns, label_column, cleaning_report = _prepare_features_and_labels(
            df,
            data_path,
            clean_data=not args.no_cleaning,
            target=args.target,
        )
    x = features.astype("float32").to_numpy()

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    validation_fraction = args.val_size / (1.0 - args.test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=validation_fraction,
        random_state=args.seed,
        stratify=y_train_val,
    )

    normal_train_x = x_train[y_train == 0]
    if len(normal_train_x) == 0:
        raise ValueError("No normal samples were found in the hybrid training split.")

    scaler = StandardScaler()
    scaler.fit(normal_train_x)

    return HybridSplit(
        train_x=scaler.transform(x_train).astype("float32"),
        val_x=scaler.transform(x_val).astype("float32"),
        test_x=scaler.transform(x_test).astype("float32"),
        train_y=y_train.astype(int),
        val_y=y_val.astype(int),
        test_y=y_test.astype(int),
        scaler=scaler,
        feature_columns=feature_columns,
        label_column=label_column,
        cleaning_report=cleaning_report,
        positive_label=TARGET_LABELS[args.target],
    )


def train_vae(split: HybridSplit, args, device: torch.device) -> Tuple[VAE, Dict[str, List[float]]]:
    """Train the optimized VAE on healthy training samples only."""
    normal_x = split.train_x[split.train_y == 0]
    train_tensor = torch.from_numpy(normal_x)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=args.batch_size, shuffle=True)

    model = VAE(input_dim=len(split.feature_columns), latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {"total": [], "recon": [], "kl": [], "kl_weight": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        kl_weight = kl_weight_for_epoch(epoch, args.beta, args.kl_warmup_epochs)

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            x_hat, mu, log_var = model(batch_x)
            loss, recon_loss, kl_loss = vae_loss(batch_x, x_hat, mu, log_var, kl_weight=kl_weight)
            loss.backward()
            optimizer.step()

            batch_size = len(batch_x)
            epoch_total += loss.item() * batch_size
            epoch_recon += recon_loss.item() * batch_size
            epoch_kl += kl_loss.item() * batch_size

        n = len(train_tensor)
        history["total"].append(epoch_total / n)
        history["recon"].append(epoch_recon / n)
        history["kl"].append(epoch_kl / n)
        history["kl_weight"].append(kl_weight)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Hybrid VAE epoch {epoch:03d}/{args.epochs} "
                f"loss={history['total'][-1]:.4f} "
                f"recon={history['recon'][-1]:.4f} "
                f"kl={history['kl'][-1]:.4f} "
                f"beta={kl_weight:.4f}"
            )

    return model, history


@torch.no_grad()
def build_hybrid_features(
    model: VAE,
    x: np.ndarray,
    feature_columns: List[str],
    device: torch.device,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """Create original CBC + VAE-derived features for a split."""
    x_tensor = torch.from_numpy(x).to(device)
    model.eval()
    mu, log_var = model.encode(x_tensor)
    x_hat = model.decode(mu)
    score, recon_error, kl_div = anomaly_scores(model, x_tensor)
    abs_recon_error = torch.abs(x_tensor - x_hat)
    squared_recon_error = (x_tensor - x_hat).pow(2)

    arrays = [
        x,
        recon_error.cpu().numpy()[:, None],
        kl_div.cpu().numpy()[:, None],
        score.cpu().numpy()[:, None],
        mu.cpu().numpy(),
        log_var.cpu().numpy(),
        abs_recon_error.cpu().numpy(),
        squared_recon_error.cpu().numpy(),
    ]
    names = (
        [f"scaled_{name}" for name in feature_columns]
        + ["vae_reconstruction_error", "vae_kl_divergence", "vae_anomaly_score"]
        + [f"latent_mu_{index + 1}" for index in range(mu.shape[1])]
        + [f"latent_logvar_{index + 1}" for index in range(log_var.shape[1])]
        + [f"abs_recon_error_{name}" for name in feature_columns]
        + [f"squared_recon_error_{name}" for name in feature_columns]
    )
    diagnostics = {
        "vae_score": score.cpu().numpy(),
        "recon_error": recon_error.cpu().numpy(),
        "kl_divergence": latent_kl_divergence(mu, log_var, reduction="mean").cpu().numpy(),
    }
    features = np.concatenate(arrays, axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=20.0, neginf=-20.0)
    features = np.clip(features, -20.0, 20.0)
    return features, names, diagnostics


def select_feature_set(feature_set: str, all_features: np.ndarray, feature_names: List[str], original_feature_count: int):
    """Return the requested feature subset for comparison experiments."""
    if feature_set == "hybrid":
        return all_features, feature_names
    if feature_set == "supervised_only":
        return all_features[:, :original_feature_count], feature_names[:original_feature_count]
    if feature_set == "vae_only":
        return all_features[:, original_feature_count:], feature_names[original_feature_count:]
    raise ValueError(f"Unsupported feature set: {feature_set}")


def best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    min_recall: Optional[float] = None,
) -> Dict[str, float]:
    """Select the validation threshold that maximizes F1, optionally under a recall floor."""
    thresholds = np.unique(probabilities)
    rows = []
    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        if min_recall is not None and recall < min_recall:
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )

    if not rows:
        return best_threshold(y_true, probabilities, min_recall=None)
    return max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Evaluate probabilistic classifier predictions at a selected threshold."""
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "specificity": float(specificity),
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def make_classifiers(seed: int) -> Dict[str, object]:
    """Return the supervised decision heads used by the hybrid pipeline."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=seed,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def predict_positive_probability(classifier, features: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities while keeping separable data warnings out of logs."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return classifier.predict_proba(features)[:, 1]


def save_feature_importance(
    feature_set: str,
    classifier_name: str,
    classifier,
    feature_names: List[str],
    output_dir: Path,
) -> None:
    """Save feature importance or coefficients when the classifier exposes them."""
    if classifier_name == "random_forest":
        values = classifier.feature_importances_
    elif classifier_name == "logistic_regression":
        values = classifier.named_steps["classifier"].coef_[0]
    else:
        return

    importance = pd.DataFrame({"feature": feature_names, "importance": values})
    importance["abs_importance"] = importance["importance"].abs()
    importance = importance.sort_values("abs_importance", ascending=False)
    importance.to_csv(output_dir / f"{feature_set}_{classifier_name}_feature_importance.csv", index=False)


def train_hybrid(args) -> pd.DataFrame:
    """Train and evaluate the full semi-supervised hybrid VAE classifier."""
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    split = load_hybrid_split(args)
    if not split.cleaning_report.empty:
        split.cleaning_report.to_csv(args.output_dir / "hybrid_data_cleaning_report.csv", index=False)
        summary = report_to_dict(split.cleaning_report)
        print(
            "Hybrid data cleaning: "
            f"{summary['original_rows']} rows -> {summary['rows_after_cleaning']} rows "
            f"({summary['dropped_rows']} dropped)"
        )

    model, history = train_vae(split, args, device)
    np.save(args.output_dir / "hybrid_vae_training_history.npy", history)

    train_features, feature_names, train_diag = build_hybrid_features(
        model, split.train_x, split.feature_columns, device
    )
    val_features, _, val_diag = build_hybrid_features(model, split.val_x, split.feature_columns, device)
    test_features, _, test_diag = build_hybrid_features(model, split.test_x, split.feature_columns, device)

    rows = []
    threshold_rows = []
    predictions: Dict[str, np.ndarray] = {
        "test_y": split.test_y,
        "test_vae_score": test_diag["vae_score"],
        "test_recon_error": test_diag["recon_error"],
        "test_kl_divergence": test_diag["kl_divergence"],
    }
    original_feature_count = len(split.feature_columns)
    for feature_set in args.feature_sets:
        train_subset, subset_feature_names = select_feature_set(
            feature_set,
            train_features,
            feature_names,
            original_feature_count,
        )
        val_subset, _ = select_feature_set(feature_set, val_features, feature_names, original_feature_count)
        test_subset, _ = select_feature_set(feature_set, test_features, feature_names, original_feature_count)

        classifiers = make_classifiers(args.seed)
        for classifier_name, classifier in classifiers.items():
            classifier.fit(train_subset, split.train_y)
            val_probabilities = predict_positive_probability(classifier, val_subset)
            test_probabilities = predict_positive_probability(classifier, test_subset)

            f1_threshold = best_threshold(split.val_y, val_probabilities)
            recall_threshold = best_threshold(
                split.val_y,
                val_probabilities,
                min_recall=args.min_recall,
            )

            for threshold_name, threshold_info in [
                ("validation_f1_best", f1_threshold),
                (f"validation_recall_at_least_{args.min_recall:g}", recall_threshold),
            ]:
                test_metrics = evaluate_probabilities(
                    split.test_y,
                    test_probabilities,
                    threshold_info["threshold"],
                )
                rows.append(
                    {
                        "feature_set": feature_set,
                        "model": classifier_name,
                        "threshold_method": threshold_name,
                        **test_metrics,
                    }
                )
                threshold_rows.append(
                    {
                        "feature_set": feature_set,
                        "model": classifier_name,
                        "threshold_method": threshold_name,
                        "validation_threshold": threshold_info["threshold"],
                        "validation_precision": threshold_info["precision"],
                        "validation_recall": threshold_info["recall"],
                        "validation_f1": threshold_info["f1"],
                    }
                )

            predictions[f"{feature_set}_{classifier_name}_test_probability"] = test_probabilities
            joblib.dump(classifier, args.output_dir / f"{feature_set}_{classifier_name}.joblib")
            save_feature_importance(feature_set, classifier_name, classifier, subset_feature_names, args.output_dir)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": split.feature_columns,
            "hybrid_feature_names": feature_names,
            "scaler_mean": split.scaler.mean_,
            "scaler_scale": split.scaler.scale_,
            "latent_dim": args.latent_dim,
            "input_dim": len(split.feature_columns),
            "label_column": split.label_column,
            "target": args.target,
            "positive_label": split.positive_label,
            "beta": args.beta,
            "kl_warmup_epochs": args.kl_warmup_epochs,
            "history": history,
        },
        args.output_dir / "hybrid_vae.pt",
    )
    np.savez(args.output_dir / "hybrid_predictions.npz", **predictions)

    metrics = pd.DataFrame(rows).sort_values(["f1", "recall"], ascending=False)
    thresholds = pd.DataFrame(threshold_rows).sort_values(["validation_f1", "validation_recall"], ascending=False)
    metrics.to_csv(args.output_dir / "hybrid_metrics.csv", index=False)
    thresholds.to_csv(args.output_dir / "hybrid_thresholds.csv", index=False)

    print("\nHybrid validation thresholds:")
    print(thresholds.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nHybrid test metrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved hybrid outputs to {args.output_dir}")
    return metrics


def parse_args():
    parser = ArgumentParser(description="Train a semi-supervised hybrid VAE classifier.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--target", choices=["anemia", "abnormal"], default="anemia")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--kl-warmup-epochs", type=int, default=50)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--no-cleaning", action="store_true", help="Disable CBC value-range data cleaning.")
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["supervised_only", "vae_only", "hybrid"],
        default=["supervised_only", "vae_only", "hybrid"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_hybrid(parse_args())
