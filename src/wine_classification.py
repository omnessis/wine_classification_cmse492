"""
Wine Classification using Logistic Regression

Trains and evaluates a logistic regression model on the UCI Wine dataset.
The dataset contains 178 samples of wine from three cultivars, described
by 13 chemical measurements.

Usage:
    python src/wine_classification.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = "data/wine.csv"
TARGET_COLUMN = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CLASS_NAMES = ["Cultivar 1", "Cultivar 2", "Cultivar 3"]


def load_data(path: str) -> pd.DataFrame:
    """Load the wine dataset from a CSV file."""
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame):
    """Split features/target and standardise features."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train) -> LogisticRegression:
    """Train a logistic regression classifier."""
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, X_test, y_test) -> dict:
    """Return accuracy and a classification report dict."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    cm = confusion_matrix(y_test, y_pred)
    return {"accuracy": accuracy, "report": report, "confusion_matrix": cm, "y_pred": y_pred}


def plot_confusion_matrix(cm: np.ndarray, output_path: str = "confusion_matrix.png") -> None:
    """Save a heatmap of the confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Confusion Matrix — Wine Classification")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def main():
    # 1. Load data
    df = load_data(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1] - 1} features")

    # 2. Preprocess
    X_train, X_test, y_train, y_test, _ = preprocess(df)
    print(f"Train samples: {len(y_train)} | Test samples: {len(y_test)}")

    # 3. Train
    model = train_model(X_train, y_train)
    print("Model training complete.")

    # 4. Evaluate
    results = evaluate_model(model, X_test, y_test)
    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results["report"])

    # 5. Plot confusion matrix
    plot_confusion_matrix(results["confusion_matrix"])


if __name__ == "__main__":
    main()
