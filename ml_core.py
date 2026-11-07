"""Shared schema, saved classifiers, and held-out evaluation for the Dry Bean study."""

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
TARGET = "Class"
FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]
CLASSES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


@dataclass
class Evaluation:
    """Metrics and diagnostics for one classifier on one labeled test dataset."""

    metrics: dict[str, float | None]
    predictions: pd.DataFrame
    confusion: pd.DataFrame
    report: pd.DataFrame


def validate_data(frame: pd.DataFrame, *, max_rows: int = 20_000) -> pd.DataFrame:
    """Validate a labeled test CSV and return features in their training order."""
    if frame.empty:
        raise ValueError("The CSV is empty. Upload labeled test rows.")
    if len(frame) > max_rows:
        raise ValueError(f"Upload test data only: at most {max_rows:,} rows are allowed.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    required = [*FEATURES, TARGET]
    missing = sorted(set(required) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(required))
    if missing or extra:
        details = []
        if missing:
            details.append("Missing columns: " + ", ".join(missing))
        if extra:
            details.append("Unexpected columns: " + ", ".join(extra))
        raise ValueError(". ".join(details) + ". Use the downloadable test CSV schema.")
    clean = frame.loc[:, required].copy()
    try:
        clean[FEATURES] = clean[FEATURES].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError("All 16 input features must be numeric.") from error
    if not np.isfinite(clean[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Input features contain missing or infinite values. Correct these rows.")
    if clean[TARGET].isna().any():
        raise ValueError("The Class column contains missing labels.")
    invalid = sorted(set(clean[TARGET].astype(str)) - set(CLASSES))
    if invalid:
        raise ValueError(
            "Unknown Class labels: " + ", ".join(invalid) + ". Labels are case-sensitive."
        )
    return clean


def read_test_csv(source: bytes | Path) -> pd.DataFrame:
    """Parse CSV without allowing pandas to silently rename duplicate headers."""
    try:
        text = (
            source.decode("utf-8-sig")
            if isinstance(source, bytes)
            else source.read_text(encoding="utf-8-sig")
        )
    except UnicodeDecodeError as error:
        raise ValueError("Save the CSV with UTF-8 encoding and upload it again.") from error
    reader = csv.reader(io.StringIO(text), strict=True)
    try:
        header = next(reader, [])
        for line, row in enumerate(reader, start=2):
            if row and len(row) != len(header):
                raise ValueError(f"CSV row {line} has {len(row)} fields; expected {len(header)}.")
    except csv.Error as error:
        raise ValueError("Malformed CSV quoting. Save a valid comma-separated CSV.") from error
    if len(header) != len(set(header)):
        raise ValueError("Duplicate column names are not allowed.")
    try:
        frame = pd.read_csv(io.StringIO(text), on_bad_lines="error")
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError(
            "Cannot parse this CSV. Check the header, commas, and row lengths."
        ) from error
    return validate_data(frame)


def load_models(model_dir: Path = ROOT / "model") -> dict[str, Pipeline]:
    """Load only repository-controlled model files, never user-uploaded artifacts."""
    missing = [
        filename for filename in MODEL_FILES.values() if not (model_dir / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Saved models are missing. Run uv run train.py first: " + ", ".join(missing)
        )
    return {name: joblib.load(model_dir / filename) for name, filename in MODEL_FILES.items()}


def evaluate(model: Pipeline | ClassifierMixin, frame: pd.DataFrame) -> Evaluation:
    """Compute macro metrics and OVR AUC using the classifier's probability column order."""
    clean = validate_data(frame)
    features, truth = clean[FEATURES], clean[TARGET]
    predicted = model.predict(features)
    probabilities = model.predict_proba(features)
    labels = model.classes_.tolist()
    auc = None
    if set(truth) == set(labels):
        auc = float(
            roc_auc_score(truth, probabilities, labels=labels, multi_class="ovr", average="macro")
        )
    metrics = {
        "Accuracy": float(accuracy_score(truth, predicted)),
        "AUC": auc,
        "Precision": float(
            precision_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "Recall": float(
            recall_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "F1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "MCC": (
            0.0
            if len(set(truth) | set(predicted)) == 1
            else float(matthews_corrcoef(truth, predicted))
        ),
    }
    predictions = pd.DataFrame(
        {
            "Actual": truth.to_numpy(),
            "Predicted": predicted,
            "Confidence": probabilities.max(axis=1),
        }
    )
    confusion = pd.DataFrame(
        confusion_matrix(truth, predicted, labels=labels),
        index=pd.Index(labels, name="Actual"),
        columns=pd.Index(labels, name="Predicted"),
    )
    report = pd.DataFrame(
        classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0)
    ).T
    return Evaluation(metrics, predictions, confusion, report)
