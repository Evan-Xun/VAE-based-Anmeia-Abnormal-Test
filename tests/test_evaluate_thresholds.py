import sys
from pathlib import Path

import pytest
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import validation_threshold  # noqa: E402


def test_validation_threshold_uses_train_quantile():
    result = validation_threshold(
        method="train_quantile",
        checkpoint_threshold=0.97,
        train_scores=np.array([0.1, 0.2, 0.3, 0.4]),
        val_scores=np.array([0.15, 0.35, 0.45, 0.5]),
        val_y=np.array([0, 0, 1, 1]),
        min_recall=0.95,
        train_quantile=75,
    )

    assert result["threshold"] == pytest.approx(0.325)
    assert result["threshold_method"] == "train_quantile_p75"


def test_validation_threshold_recall_floor_falls_back_when_needed():
    result = validation_threshold(
        method="validation_recall_floor",
        checkpoint_threshold=0.97,
        train_scores=np.array([0.1, 0.2, 0.3, 0.4]),
        val_scores=np.array([0.1, 0.4, 0.6, 0.9]),
        val_y=np.array([0, 0, 1, 1]),
        min_recall=1.1,
        train_quantile=95,
    )

    assert result["recall"] == pytest.approx(1.0)
    assert result["threshold"] == pytest.approx(0.6)
