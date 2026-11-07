"""Reproduce the dataset split, fitted pipelines, and evaluation artifacts.

Usage: uv run train.py
"""

import hashlib
import json
import logging
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_core import CLASSES, FEATURES, MODEL_FILES, ROOT, TARGET, evaluate, validate_data

LOGGER = logging.getLogger(__name__)
SOURCE_SHA256 = "0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f"
SEED = 42


def read_source(path: Path) -> pd.DataFrame:
    """Read the verified UCI workbook from its original download archive."""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("The source archive checksum does not match the verified UCI download.")
    with ZipFile(BytesIO(content)) as archive:
        frame = pd.read_excel(
            BytesIO(archive.read("DryBeanDataset/Dry_Bean_Dataset.xlsx")), sheet_name=0
        )
    frame = validate_data(frame)
    if frame.shape != (13611, 17) or set(frame[TARGET]) != set(CLASSES):
        raise ValueError("Expected 13,611 rows, 16 features and the seven UCI Dry Bean classes.")
    return frame


def create_models() -> dict[str, Pipeline]:
    """Create five fixed-baseline classifiers without looking at the held-out data."""
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=1.0, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=3, random_state=SEED
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(C=3.0, kernel="rbf", probability=True, random_state=SEED),
    }
    return {
        name: Pipeline([("scale", StandardScaler()), ("classifier", estimator)])
        for name, estimator in estimators.items()
    }


def train() -> None:
    """Train only on the training partition and persist reproducible evaluation outputs."""
    raw = read_source(ROOT / "data" / "dry-bean-source.zip")
    exact_duplicates = int(raw.duplicated().sum())
    frame = raw.drop_duplicates().reset_index(drop=True)
    training, test = train_test_split(
        frame, test_size=0.20, stratify=frame[TARGET], random_state=SEED
    )
    training.to_csv(ROOT / "data" / "train_data.csv", index=False, float_format="%.17g")
    test.to_csv(ROOT / "test_data.csv", index=False, float_format="%.17g")
    # Evaluate the serialized CSV to make CLI and uploaded-file results identical.
    test = validate_data(pd.read_csv(ROOT / "test_data.csv"))
    model_dir = ROOT / "model"
    results_dir = ROOT / "results"
    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    rows = []
    for name, pipeline in create_models().items():
        LOGGER.info("Training %s on %s rows", name, len(training))
        pipeline.fit(training[FEATURES], training[TARGET])
        joblib.dump(pipeline, model_dir / MODEL_FILES[name], compress=3)
        result = evaluate(pipeline, test)
        rows.append({"Model": name, **result.metrics})
        stem = MODEL_FILES[name].removesuffix(".joblib")
        result.confusion.to_csv(results_dir / f"{stem}_confusion.csv")
        result.report.to_csv(results_dir / f"{stem}_report.csv")
    comparison = pd.DataFrame(rows)
    comparison.to_csv(results_dir / "comparison.csv", index=False)
    metadata = {
        "source": "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset",
        "doi": "https://doi.org/10.24432/C50S4B",
        "license": "CC BY 4.0",
        "source_sha256": SOURCE_SHA256,
        "raw_rows": len(raw),
        "exact_duplicates_removed": exact_duplicates,
        "usable_rows": len(frame),
        "training_rows": len(training),
        "test_rows": len(test),
        "features": FEATURES,
        "classes": CLASSES,
        "training_class_counts": training[TARGET].value_counts().sort_index().to_dict(),
        "test_class_counts": test[TARGET].value_counts().sort_index().to_dict(),
        "random_state": SEED,
        "test_fraction": 0.20,
        "sklearn_version": sklearn.__version__,
        "averaging": "macro over all seven labels; AUC is macro one-vs-rest",
        "winner_rule": "Highest held-out macro F1; descriptive comparison, not model tuning",
        "test_csv_sha256": hashlib.sha256((ROOT / "test_data.csv").read_bytes()).hexdigest(),
    }
    (results_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Comparison results:\n%s", comparison.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train()
