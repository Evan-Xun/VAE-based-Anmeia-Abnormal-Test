"""Data quality checks and parsing helpers for CBC tabular data."""

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


CBC_VALUE_RANGES = {
    "WBC": (0.1, 200.0),
    "LYMp": (0.0, 100.0),
    "NEUTp": (0.0, 100.0),
    "LYMn": (0.0, 100.0),
    "NEUTn": (0.0, 100.0),
    "RBC": (0.5, 10.0),
    "HGB": (1.0, 25.0),
    "Hemoglobin": (1.0, 25.0),
    "HCT": (5.0, 75.0),
    "MCV": (40.0, 150.0),
    "MCH": (10.0, 60.0),
    "MCHC": (15.0, 45.0),
    "PLT": (1.0, 1500.0),
    "PDW": (5.0, 30.0),
    "PCT": (0.0, 2.0),
}

REPORT_FIELD_ALIASES = {
    "WBC": "WBC",
    "RBC": "RBC",
    "HGB": "HGB",
    "Hemoglobin": "HGB",
    "HCT": "HCT",
    "MCV": "MCV",
    "MCH": "MCH",
    "MCHC": "MCHC",
    "PLT": "PLT",
    "MPV": "MPV",
    "RDW---CV": "RDW_CV",
    "%RDW---CV": "RDW_CV",
    "Neut": "NEUT_PCT",
    "%Neut": "NEUT_PCT",
    "Neutrophil": "NEUT_PCT",
    "LYMP": "LYMP_PCT",
    "%LYMP": "LYMP_PCT",
    "Lymphocyte": "LYMP_PCT",
    "MONO": "MONO_PCT",
    "%MONO": "MONO_PCT",
    "Monocyte": "MONO_PCT",
    "EOS": "EOS_PCT",
    "%EOS": "EOS_PCT",
    "Eosinophil": "EOS_PCT",
    "BASO": "BASO_PCT",
    "%BASO": "BASO_PCT",
    "Basophil": "BASO_PCT",
    "#NEUT": "NEUT_ABS",
    "#LYMP": "LYMP_ABS",
    "#MONO": "MONO_ABS",
    "#EOS": "EOS_ABS",
    "#BASO": "BASO_ABS",
}

REPORT_CORE_FEATURE_COLUMNS = ["WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT"]
NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def canonical_report_field(raw_name: str) -> Optional[str]:
    """Return the canonical field name for a raw CBC report line."""
    return REPORT_FIELD_ALIASES.get(raw_name.strip())


def parse_report_value(raw_value: str) -> Optional[float]:
    """Extract the leading numeric value from a CBC report cell."""
    text = raw_value.replace("*", " ").replace("−", "-").strip()
    match = NUMERIC_PATTERN.search(text)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def clean_cbc_dataframe(
    df: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop duplicate rows plus missing, non-numeric, or implausible CBC values."""
    cleaned = df[feature_columns + [label_column]].copy()
    report_rows = []
    out_of_range_any = pd.Series(False, index=cleaned.index)

    for column in feature_columns:
        raw_values = cleaned[column]
        numeric_values = pd.to_numeric(raw_values, errors="coerce")
        non_numeric_count = int(numeric_values.isna().sum() - raw_values.isna().sum())
        cleaned[column] = numeric_values

        if column in CBC_VALUE_RANGES:
            lower, upper = CBC_VALUE_RANGES[column]
            out_of_range = numeric_values.notna() & ((numeric_values < lower) | (numeric_values > upper))
            out_of_range_any = out_of_range_any | out_of_range
            out_of_range_count = int(out_of_range.sum())
        else:
            lower, upper = None, None
            out_of_range_count = 0

        report_rows.append(
            {
                "column": column,
                "min_allowed": lower,
                "max_allowed": upper,
                "missing_count": int(raw_values.isna().sum()),
                "non_numeric_count": non_numeric_count,
                "out_of_range_count": out_of_range_count,
            }
        )

    required_columns = feature_columns + [label_column]
    missing_or_non_numeric = cleaned[required_columns].isna().any(axis=1)
    drop_mask = missing_or_non_numeric | out_of_range_any
    cleaned = cleaned.loc[~drop_mask].copy()
    duplicate_rows = 0
    if drop_duplicates:
        duplicate_mask = cleaned.duplicated(subset=required_columns, keep="first")
        duplicate_rows = int(duplicate_mask.sum())
        cleaned = cleaned.loc[~duplicate_mask].copy()

    summary_rows = [
        {"column": "__summary__", "metric": "original_rows", "value": len(df)},
        {"column": "__summary__", "metric": "rows_after_cleaning", "value": len(cleaned)},
        {"column": "__summary__", "metric": "dropped_rows", "value": int(drop_mask.sum()) + duplicate_rows},
        {
            "column": "__summary__",
            "metric": "missing_or_non_numeric_rows",
            "value": int(missing_or_non_numeric.sum()),
        },
        {"column": "__summary__", "metric": "out_of_range_rows", "value": int(out_of_range_any.sum())},
        {"column": "__summary__", "metric": "duplicate_rows", "value": duplicate_rows},
    ]
    report = pd.concat([pd.DataFrame(summary_rows), pd.DataFrame(report_rows)], ignore_index=True)
    return cleaned, report


def report_to_dict(report: pd.DataFrame) -> Dict[str, int]:
    """Return compact numeric summary values from a cleaning report."""
    summary = report[report["column"] == "__summary__"]
    return {str(row["metric"]): int(row["value"]) for _, row in summary.iterrows()}
