"""Verify validation, leakage controls, saved pipelines, and all required metrics."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from streamlit.testing.v1 import AppTest

from ml_core import CLASSES, FEATURES, METRICS, ROOT, evaluate, load_models, read_test_csv
from train import create_models, read_source


@pytest.fixture(scope="module")
def test_frame() -> pd.DataFrame:
    """Load the same held-out CSV used by the app and reported comparison."""
    return read_test_csv(ROOT / "test_data.csv")


@pytest.fixture(scope="module")
def models() -> dict:
    """Load trained artifacts once for integration tests."""
    return load_models()


def test_given_source_when_loaded_then_meets_assignment_minimums() -> None:
    frame = read_source(ROOT / "data" / "dry-bean-source.zip")

    assert frame.shape == (13611, 17)
    assert len(FEATURES) >= 12 and len(frame) >= 500
    assert set(frame["Class"]) == set(CLASSES)


def test_given_split_when_compared_then_no_shared_rows(test_frame: pd.DataFrame) -> None:
    training = pd.read_csv(ROOT / "data" / "train_data.csv")

    overlap = training.merge(test_frame, on=[*FEATURES, "Class"], how="inner")

    assert overlap.empty
    assert set(training["Class"]) == set(test_frame["Class"]) == set(CLASSES)


def test_given_models_when_scalers_checked_then_fit_only_training_rows(models: dict) -> None:
    training = pd.read_csv(ROOT / "data" / "train_data.csv")

    for model in models.values():
        scaler = model.named_steps["scale"]

        assert scaler.n_samples_seen_ == len(training)
        np.testing.assert_allclose(scaler.mean_, training[FEATURES].mean(), rtol=1e-10)


def test_given_factories_when_created_then_all_five_models_are_unfitted() -> None:
    models = create_models()

    assert len(models) == 5
    assert all(not hasattr(model.named_steps["scale"], "mean_") for model in models.values())


@pytest.mark.parametrize("model_name", list(create_models()))
def test_given_saved_model_when_evaluated_then_matches_metrics(
    model_name: str, models: dict, test_frame: pd.DataFrame
) -> None:
    expected = pd.read_csv(ROOT / "results" / "comparison.csv").set_index("Model")
    model = models[model_name]

    result = evaluate(model, test_frame)
    truth = test_frame["Class"]
    predicted = model.predict(test_frame[FEATURES])
    probabilities = model.predict_proba(test_frame[FEATURES])

    assert list(result.metrics) == METRICS
    np.testing.assert_allclose(
        list(result.metrics.values()), expected.loc[model_name, METRICS].to_numpy(dtype=float)
    )
    assert result.metrics["Accuracy"] == pytest.approx(accuracy_score(truth, predicted))
    assert result.metrics["F1"] == pytest.approx(f1_score(truth, predicted, average="macro"))
    assert result.metrics["MCC"] == pytest.approx(matthews_corrcoef(truth, predicted))
    assert result.metrics["AUC"] == pytest.approx(
        roc_auc_score(
            truth, probabilities, labels=model.classes_, multi_class="ovr", average="macro"
        )
    )
    assert result.confusion.shape == (7, 7)
    assert result.confusion.to_numpy().sum() == len(test_frame)
    assert len(result.predictions) == len(test_frame)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "Missing columns"),
        ("extra", "Unexpected columns"),
        ("nonnumeric", "numeric"),
        ("nan", "missing or infinite"),
        ("infinite", "missing or infinite"),
        ("label", "Unknown Class"),
        ("missing_label", "missing labels"),
        ("empty", "empty"),
    ],
)
def test_given_invalid_csv_when_read_then_explains_problem(
    case: str, message: str, test_frame: pd.DataFrame
) -> None:
    frame = test_frame.head(7).copy()
    if case == "missing":
        frame = frame.drop(columns=["Area"])
    elif case == "extra":
        frame["unexpected"] = 1
    elif case == "nonnumeric":
        frame["Area"] = "not-a-number"
    elif case == "nan":
        frame.loc[frame.index[0], "Area"] = np.nan
    elif case == "infinite":
        frame["Area"] = frame["Area"].astype(float)
        frame.loc[frame.index[0], "Area"] = np.inf
    elif case == "label":
        frame.loc[frame.index[0], "Class"] = "UNKNOWN"
    elif case == "missing_label":
        frame.loc[frame.index[0], "Class"] = None
    else:
        frame = frame.iloc[:0]

    with pytest.raises(ValueError, match=message):
        read_test_csv(frame.to_csv(index=False).encode())


@pytest.mark.parametrize("content", [b"", b"\xff\xfe", b"a,a\n1,2", b"a,b\n1,2,3"])
def test_given_malformed_csv_when_read_then_rejects(content: bytes) -> None:
    with pytest.raises(ValueError):
        read_test_csv(content)


def test_given_reordered_columns_when_read_then_uses_training_order(
    test_frame: pd.DataFrame,
) -> None:
    reordered = test_frame.iloc[:, ::-1].to_csv(index=False).encode()

    clean = read_test_csv(reordered)

    assert clean.columns.tolist() == [*FEATURES, "Class"]


def test_given_too_many_rows_when_read_then_rejects(test_frame: pd.DataFrame) -> None:
    content = pd.concat([test_frame.head(1)] * 20001).to_csv(index=False).encode()

    with pytest.raises(ValueError, match="20,000"):
        read_test_csv(content)


def test_given_single_class_when_evaluated_then_auc_is_explicitly_unavailable(
    test_frame: pd.DataFrame, models: dict
) -> None:
    subset = test_frame[test_frame["Class"] == "SEKER"].head(5)

    result = evaluate(models["Logistic Regression"], subset)

    assert result.metrics["AUC"] is None
    assert result.confusion.to_numpy().sum() == 5


def test_given_saved_metadata_when_loaded_then_counts_are_consistent() -> None:
    metadata = json.loads((ROOT / "results" / "metadata.json").read_text())

    assert metadata["training_rows"] + metadata["test_rows"] == metadata["usable_rows"]
    assert metadata["usable_rows"] + metadata["exact_duplicates_removed"] == 13611


def test_given_missing_artifacts_when_loading_then_explains_training(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="train.py"):
        load_models(tmp_path)


def test_given_app_when_switching_models_then_no_retraining(monkeypatch) -> None:
    def forbid_fit(*args, **kwargs):
        raise AssertionError("The evaluation app must never fit a pipeline.")

    monkeypatch.setattr("sklearn.pipeline.Pipeline.fit", forbid_fit)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()

    assert not app.exception
    for name in create_models():
        app.selectbox[0].select(name).run()

        assert not app.exception
        assert any(name in heading.value for heading in app.subheader)
    app.radio[0].set_value("Upload test CSV").run()
    assert not app.exception
    assert len(app.info) == 1
