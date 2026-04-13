"""
Unit tests for wine_classification.py
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wine_classification import (
    load_data,
    preprocess,
    train_model,
    evaluate_model,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "wine.csv")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wine_df():
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def split_data(wine_df):
    return preprocess(wine_df)


@pytest.fixture(scope="module")
def trained_model(split_data):
    X_train, _, y_train, _, _ = split_data
    return train_model(X_train, y_train)


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------


def test_load_data_shape(wine_df):
    """Dataset should have 178 rows and 14 columns (target + 13 features)."""
    assert wine_df.shape == (178, 14)


def test_load_data_columns(wine_df):
    """Dataset must contain the target column and 13 feature columns."""
    assert "target" in wine_df.columns
    assert wine_df.shape[1] == 14


def test_load_data_target_classes(wine_df):
    """Target should only contain classes 1, 2, and 3."""
    assert set(wine_df["target"].unique()) == {1, 2, 3}


def test_load_data_no_missing_values(wine_df):
    """There should be no missing values in the dataset."""
    assert wine_df.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


def test_preprocess_split_sizes(split_data):
    """20 % test split of 178 samples → ~36 test, ~142 train."""
    X_train, X_test, y_train, y_test, _ = split_data
    assert len(y_test) == 36
    assert len(y_train) == 142


def test_preprocess_feature_standardisation(split_data):
    """Training features should be approximately zero-mean, unit-variance."""
    X_train, _, _, _, _ = split_data
    assert abs(X_train.mean()) < 1e-10
    assert abs(X_train.std() - 1.0) < 0.01


def test_preprocess_no_data_leakage(split_data):
    """Test set mean should NOT be exactly zero (scaler fitted on train only)."""
    _, X_test, _, _, _ = split_data
    assert abs(X_test.mean()) > 1e-10


# ---------------------------------------------------------------------------
# Model training tests
# ---------------------------------------------------------------------------


def test_model_is_fitted(trained_model):
    """Model should expose coef_ after training."""
    assert hasattr(trained_model, "coef_")
    assert trained_model.coef_.shape[0] == 3  # three classes


def test_model_predict_shape(trained_model, split_data):
    """Predictions array should match the test set length."""
    _, X_test, _, y_test, _ = split_data
    preds = trained_model.predict(X_test)
    assert len(preds) == len(y_test)


def test_model_predict_valid_classes(trained_model, split_data):
    """All predicted labels should be in {1, 2, 3}."""
    _, X_test, _, _, _ = split_data
    preds = trained_model.predict(X_test)
    assert set(preds).issubset({1, 2, 3})


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


def test_evaluate_accuracy_range(trained_model, split_data):
    """Logistic regression on this dataset should achieve > 90 % accuracy."""
    _, X_test, _, y_test, _ = split_data
    results = evaluate_model(trained_model, X_test, y_test)
    assert results["accuracy"] > 0.90


def test_evaluate_confusion_matrix_shape(trained_model, split_data):
    """Confusion matrix should be 3×3."""
    _, X_test, _, y_test, _ = split_data
    results = evaluate_model(trained_model, X_test, y_test)
    assert results["confusion_matrix"].shape == (3, 3)


def test_evaluate_report_contains_classes(trained_model, split_data):
    """Classification report should mention all three cultivar names."""
    _, X_test, _, y_test, _ = split_data
    results = evaluate_model(trained_model, X_test, y_test)
    for cls in ["Cultivar 1", "Cultivar 2", "Cultivar 3"]:
        assert cls in results["report"]
