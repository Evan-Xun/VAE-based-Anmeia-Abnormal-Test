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

from data_cleaning import clean_cbc_dataframe


DEFAULT_DATA_PATH = Path("diagnosed_cbc_data_v4.csv")
OLD_FEATURE_COLUMNS = ["Hemoglobin", "MCH", "MCHC", "MCV"]
OLD_LABEL_COLUMN = "Result"
DIAGNOSIS_LABEL_COLUMN = "Diagnosis"
NORMAL_DIAGNOSIS = "Healthy"
ANEMIA_DIAGNOSES = {
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
            data, cleaning_report = clean_cbc_dataframe(df, feature_columns, label_column)
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
            data, cleaning_report = clean_cbc_dataframe(df, OLD_FEATURE_COLUMNS, OLD_LABEL_COLUMN)
        else:
            data = df[OLD_FEATURE_COLUMNS + [OLD_LABEL_COLUMN]].dropna().copy()
            cleaning_report = pd.DataFrame()
        y = data[OLD_LABEL_COLUMN].astype(int).to_numpy()
        return data[OLD_FEATURE_COLUMNS], y, OLD_FEATURE_COLUMNS, OLD_LABEL_COLUMN, cleaning_report

    raise ValueError(
        f"Unsupported dataset format in {csv_path}. Expected a '{DIAGNOSIS_LABEL_COLUMN}' "
        f"column for the new CBC dataset or a '{OLD_LABEL_COLUMN}' column for the old dataset."
    )


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
