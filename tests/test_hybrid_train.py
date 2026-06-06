import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_train import best_threshold, evaluate_probabilities, select_feature_set  # noqa: E402


def test_best_threshold_can_enforce_recall_floor():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = best_threshold(y_true, probabilities, min_recall=1.0)

    assert threshold["threshold"] == pytest.approx(0.6)
    assert threshold["recall"] == pytest.approx(1.0)
    assert threshold["precision"] == pytest.approx(1.0)


def test_evaluate_probabilities_reports_confusion_matrix_counts():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.2, 0.8, 0.7, 0.9])

    metrics = evaluate_probabilities(y_true, probabilities, threshold=0.75)

    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_positive"] == 1


def test_select_feature_set_slices_original_and_vae_features():
    all_features = np.arange(24, dtype=float).reshape(3, 8)
    feature_names = [f"f{i}" for i in range(8)]

    supervised, supervised_names = select_feature_set("supervised_only", all_features, feature_names, 3)
    vae_only, vae_names = select_feature_set("vae_only", all_features, feature_names, 3)
    hybrid, hybrid_names = select_feature_set("hybrid", all_features, feature_names, 3)

    assert supervised.shape == (3, 3)
    assert vae_only.shape == (3, 5)
    assert hybrid.shape == (3, 8)
    assert supervised_names == ["f0", "f1", "f2"]
    assert vae_names == ["f3", "f4", "f5", "f6", "f7"]
    assert hybrid_names == feature_names
