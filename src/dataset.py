"""Data loading and preprocessing for the CBC VAE baseline."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from data_cleaning import (
    REPORT_CORE_FEATURE_COLUMNS,
    canonical_report_field,
    clean_cbc_dataframe,
    parse_report_value,
)


DEFAULT_DATA_PATH = Path("cbc_8_features_reports_only.csv")
OLD_FEATURE_COLUMNS = ["Hemoglobin", "MCH", "MCHC", "MCV"]
OLD_LABEL_COLUMN = "Result"
DIAGNOSIS_LABEL_COLUMN = "Diagnosis"
NORMAL_DIAGNOSIS = "Healthy"
ANEMIA_DIAGNOSES = {
    "Anemia",
    "Normocytic hypochromic anemia",
    "Normocytic normochromic anemia",
    "Iron deficiency anemia",
    "Other microcytic anemia",
    "Macrocytic anemia",
}
TARGET_LABELS = {
    "anemia": "Anemia",
    "abnormal": "Abnormal",
}


@dataclass
class AnemiaData:
    """Container for preprocessed train/test tensors and metadata."""

    train_dataset: TensorDataset
    train_x: torch.Tensor
    test_x: torch.Tensor
    test_y: np.ndarray
    train_y: np.ndarray
    scaler: StandardScaler
    feature_columns: List[str]
    label_column: str
    cleaning_report: pd.DataFrame
    target: str
    positive_label: str


@dataclass
class BaselineSplit:
    """Container for baseline train/validation/test tensors and metadata."""

    train_dataset: TensorDataset
    train_x: torch.Tensor
    val_x: torch.Tensor
    test_x: torch.Tensor
    val_y: np.ndarray
    test_y: np.ndarray
    train_y: np.ndarray
    scaler: StandardScaler
    feature_columns: List[str]
    label_column: str
    cleaning_report: pd.DataFrame
    target: str
    positive_label: str


def _prepare_features_and_labels(
    df: pd.DataFrame,
    csv_path: Path,
    clean_data: bool = True,
    target: str = "anemia",
) -> Tuple[pd.DataFrame, np.ndarray, List[str], str, pd.DataFrame]:
    """Return numeric features and binary labels for supported CBC datasets."""
    if DIAGNOSIS_LABEL_COLUMN in df.columns:
        label_column = DIAGNOSIS_LABEL_COLUMN
        feature_columns = [
            col for col in df.select_dtypes(include=[np.number]).columns if col != label_column
        ]
        if not feature_columns:
            raise ValueError(f"No numeric feature columns found in {csv_path}")

        if target not in TARGET_LABELS:
            raise ValueError(f"Unsupported target '{target}'. Choose one of: {sorted(TARGET_LABELS)}")

        if target == "anemia":
            allowed_diagnoses = ANEMIA_DIAGNOSES | {NORMAL_DIAGNOSIS}
            df = df[df[label_column].astype(str).str.strip().isin(allowed_diagnoses)].copy()

        if clean_data:
            data, cleaning_report = clean_cbc_dataframe(
                df,
                feature_columns,
                label_column,
                drop_duplicates=False,
            )
        else:
            data = df[feature_columns + [label_column]].dropna().copy()
            cleaning_report = pd.DataFrame()

        if target == "anemia":
            y = data[label_column].astype(str).str.strip().isin(ANEMIA_DIAGNOSES).astype(int).to_numpy()
        else:
            y = (data[label_column].astype(str).str.strip() != NORMAL_DIAGNOSIS).astype(int).to_numpy()
        return data[feature_columns], y, feature_columns, label_column, cleaning_report

    if OLD_LABEL_COLUMN in df.columns:
        missing = [col for col in OLD_FEATURE_COLUMNS + [OLD_LABEL_COLUMN] if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        if clean_data:
            data, cleaning_report = clean_cbc_dataframe(
                df,
                OLD_FEATURE_COLUMNS,
                OLD_LABEL_COLUMN,
                drop_duplicates=False,
            )
        else:
            data = df[OLD_FEATURE_COLUMNS + [OLD_LABEL_COLUMN]].dropna().copy()
            cleaning_report = pd.DataFrame()
        y = data[OLD_LABEL_COLUMN].astype(int).to_numpy()
        return data[OLD_FEATURE_COLUMNS], y, OLD_FEATURE_COLUMNS, OLD_LABEL_COLUMN, cleaning_report

    raise ValueError(
        f"Unsupported dataset format in {csv_path}. Expected a '{DIAGNOSIS_LABEL_COLUMN}' "
        f"column for the new CBC dataset or a '{OLD_LABEL_COLUMN}' column for the old dataset."
    )


def _resolve_report_directories(path: Path) -> Tuple[Path, Path]:
    """Resolve the paired anemia and normal report directories from a path."""
    if path.is_dir() and path.name == "Anemia_CBC_reports":
        anemia_dir = path
        normal_dir = path.parent / "Normal_CBC_reports"
    elif path.is_dir() and path.name == "Normal_CBC_reports":
        normal_dir = path
        anemia_dir = path.parent / "Anemia_CBC_reports"
    elif path.is_dir() and (path / "Anemia_CBC_reports").is_dir() and (path / "Normal_CBC_reports").is_dir():
        anemia_dir = path / "Anemia_CBC_reports"
        normal_dir = path / "Normal_CBC_reports"
    else:
        raise ValueError(
            f"Could not resolve report dataset from {path}. "
            "Pass the Anemia_CBC_reports directory, the Normal_CBC_reports directory, "
            "or a parent directory containing both."
        )

    if not anemia_dir.is_dir() or not normal_dir.is_dir():
        raise FileNotFoundError("Both Anemia_CBC_reports and Normal_CBC_reports directories are required.")
    return anemia_dir, normal_dir


def _load_report_rows(report_dir: Path, diagnosis: str) -> List[dict]:
    """Parse one directory of text CBC reports into row dictionaries."""
    rows = []
    for path in sorted(report_dir.glob("*.txt")):
        row = {
            DIAGNOSIS_LABEL_COLUMN: diagnosis,
            "__source_file__": path.name,
        }
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            canonical_name = canonical_report_field(parts[0])
            if canonical_name is None or canonical_name in row:
                continue
            row[canonical_name] = parse_report_value(parts[1])
        rows.append(row)
    return rows


def load_report_dataset(
    path: Union[str, Path],
    clean_data: bool = True,
    target: str = "anemia",
) -> Tuple[pd.DataFrame, np.ndarray, List[str], str, pd.DataFrame]:
    """Load the paired CBC text-report dataset."""
    if target not in TARGET_LABELS:
        raise ValueError(f"Unsupported target '{target}'. Choose one of: {sorted(TARGET_LABELS)}")

    anemia_dir, normal_dir = _resolve_report_directories(Path(path))
    rows = _load_report_rows(anemia_dir, "Anemia") + _load_report_rows(normal_dir, NORMAL_DIAGNOSIS)
    df = pd.DataFrame(rows)

    feature_columns = REPORT_CORE_FEATURE_COLUMNS
    label_column = DIAGNOSIS_LABEL_COLUMN
    if clean_data:
        data, cleaning_report = clean_cbc_dataframe(
            df,
            feature_columns,
            label_column,
            drop_duplicates=False,
        )
    else:
        data = df[feature_columns + [label_column]].dropna().copy()
        cleaning_report = pd.DataFrame()

    y = (data[label_column].astype(str).str.strip() != NORMAL_DIAGNOSIS).astype(int).to_numpy()
    return data[feature_columns], y, feature_columns, label_column, cleaning_report


def _train_val_test_split(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float,
    val_size: float,
):
    """Create a stratified train/validation/test split."""
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    validation_fraction = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y_train_val,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def load_anemia_data(
    csv_path: Union[str, Path],
    seed: int = 42,
    clean_data: bool = True,
    target: str = "anemia",
) -> AnemiaData:
    """Load anemia CSV data and prepare the unsupervised VAE split.

    The VAE is trained only on normal samples. For the new CBC diagnosis
    dataset, Diagnosis == "Healthy" is treated as normal and every other
    diagnosis is treated as abnormal for evaluation.
    """
    csv_path = Path(csv_path)
    if csv_path.is_dir():
        features, y, feature_columns, label_column, cleaning_report = load_report_dataset(
            csv_path,
            clean_data=clean_data,
            target=target,
        )
    else:
        df = pd.read_csv(csv_path)
        features, y, feature_columns, label_column, cleaning_report = _prepare_features_and_labels(
            df,
            csv_path,
            clean_data=clean_data,
            target=target,
        )
    x = features.astype("float32").to_numpy()

    x_train_all, x_test, y_train_all, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    normal_mask = y_train_all == 0
    x_train_normal = x_train_all[normal_mask]
    y_train_normal = y_train_all[normal_mask]
    if len(x_train_normal) == 0:
        raise ValueError("No normal samples were found for VAE training.")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_normal).astype("float32")
    x_test_scaled = scaler.transform(x_test).astype("float32")

    train_x = torch.from_numpy(x_train_scaled)
    test_x = torch.from_numpy(x_test_scaled)

    return AnemiaData(
        train_dataset=TensorDataset(train_x),
        train_x=train_x,
        test_x=test_x,
        test_y=y_test,
        train_y=y_train_normal,
        scaler=scaler,
        feature_columns=feature_columns,
        label_column=label_column,
        cleaning_report=cleaning_report,
        target=target,
        positive_label=TARGET_LABELS[target],
    )


def load_anemia_baseline_split(
    csv_path: Union[str, Path],
    seed: int = 42,
    clean_data: bool = True,
    target: str = "anemia",
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> BaselineSplit:
    """Load anemia CSV data with a stratified train/validation/test split.

    The VAE is trained on healthy samples from the training split only, while
    validation and test splits retain both classes for threshold selection and
    final reporting.
    """
    csv_path = Path(csv_path)
    if csv_path.is_dir():
        features, y, feature_columns, label_column, cleaning_report = load_report_dataset(
            csv_path,
            clean_data=clean_data,
            target=target,
        )
    else:
        df = pd.read_csv(csv_path)
        features, y, feature_columns, label_column, cleaning_report = _prepare_features_and_labels(
            df,
            csv_path,
            clean_data=clean_data,
            target=target,
        )
    x = features.astype("float32").to_numpy()

    x_train, x_val, x_test, y_train, y_val, y_test = _train_val_test_split(
        x,
        y,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
    )

    normal_mask = y_train == 0
    x_train_normal = x_train[normal_mask]
    y_train_normal = y_train[normal_mask]
    if len(x_train_normal) == 0:
        raise ValueError("No normal samples were found for VAE training.")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_normal).astype("float32")
    x_val_scaled = scaler.transform(x_val).astype("float32")
    x_test_scaled = scaler.transform(x_test).astype("float32")

    return BaselineSplit(
        train_dataset=TensorDataset(torch.from_numpy(x_train_scaled)),
        train_x=torch.from_numpy(x_train_scaled),
        val_x=torch.from_numpy(x_val_scaled),
        test_x=torch.from_numpy(x_test_scaled),
        val_y=y_val.astype(int),
        test_y=y_test.astype(int),
        train_y=y_train_normal.astype(int),
        scaler=scaler,
        feature_columns=feature_columns,
        label_column=label_column,
        cleaning_report=cleaning_report,
        target=target,
        positive_label=TARGET_LABELS[target],
    )
