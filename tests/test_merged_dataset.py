import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import load_anemia_baseline_split  # noqa: E402


def test_merged_dataset_has_expected_schema():
    df = pd.read_csv("cbc_8_features_reports_only.csv")

    assert list(df.columns) == [
        "WBC",
        "RBC",
        "HGB",
        "HCT",
        "MCV",
        "MCH",
        "MCHC",
        "PLT",
        "Diagnosis",
        "Source",
    ]
    assert set(df["Diagnosis"].unique()) == {"Anemia", "Healthy"}


def test_merged_dataset_loads_into_baseline_split():
    split = load_anemia_baseline_split(Path("cbc_8_features_reports_only.csv"), clean_data=True, target="anemia")

    assert split.feature_columns == ["WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT"]
    assert len(split.train_x) > 0
    assert len(split.val_x) > 0
    assert len(split.test_x) > 0
