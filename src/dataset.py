"""Data loading and preprocessing for the anemia VAE baseline."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


FEATURE_COLUMNS = ["Hemoglobin", "MCH", "MCHC", "MCV"]
LABEL_COLUMN = "Result"


@dataclass
class AnemiaData:
    """Container for preprocessed train/test tensors and metadata."""

    train_dataset: TensorDataset
    train_x: torch.Tensor
    test_x: torch.Tensor
    test_y: np.ndarray
    train_y: np.ndarray
    scaler: StandardScaler
    feature_columns: list[str]


def load_anemia_data(csv_path: str | Path, seed: int = 42) -> AnemiaData:
    """Load anemia CSV data and prepare the unsupervised VAE split.

    The VAE is trained only on normal samples where Result == 0. Labels are
    retained only for evaluation on the test split.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    missing = [col for col in FEATURE_COLUMNS + [LABEL_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    df = df[FEATURE_COLUMNS + [LABEL_COLUMN]].dropna().copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    x = df[FEATURE_COLUMNS].astype("float32").to_numpy()
    y = df[LABEL_COLUMN].to_numpy()

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
        feature_columns=FEATURE_COLUMNS,
    )
