"""Interactive held-out evaluation of five saved Dry Bean classifier pipelines."""

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from ml_core import (
    CLASSES,
    FEATURES,
    METRICS,
    ROOT,
    Evaluation,
    evaluate,
    load_models,
    read_test_csv,
)

LOGGER = logging.getLogger(__name__)
st.set_page_config(page_title="BeanScope | Classifier Lab", page_icon="🌱", layout="wide")


@st.cache_resource
def saved_models() -> dict[str, Pipeline]:
    """Cache repository-owned pipelines without fitting any estimators."""
    return load_models()


@st.cache_data(show_spinner=False)
def score_dataset(csv_bytes: bytes) -> dict[str, Evaluation]:
    """Cache five model evaluations for the exact uploaded bytes."""
    frame = read_test_csv(csv_bytes)
    return {name: evaluate(model, frame) for name, model in saved_models().items()}


def main() -> None:
    """Render the classifier comparison, selected-model diagnostics, and CSV controls."""
    st.caption("MACHINE LEARNING ASSIGNMENT 2  /  MULTICLASS CLASSIFICATION")
    st.title("BeanScope")
    st.markdown("### Seven bean varieties. Five classifiers. One fair comparison.")
    st.write(
        "Explore how shape and size measurements distinguish dry bean varieties. "
        "Every result below comes from a saved model, never a model fitted to your upload."
    )
    with st.sidebar:
        st.header("Evaluation workspace")
        selected = st.selectbox("Select classifier", list(saved_models()))
        source = st.radio("Test data source", ["Bundled held-out test data", "Upload test CSV"])
        upload = None
        if source == "Upload test CSV":
            upload = st.file_uploader("Upload labeled test CSV", type=["csv"])
        st.download_button(
            "Download test CSV",
            (ROOT / "test_data.csv").read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )
        st.divider()
        st.caption("16 numeric features + Class. Maximum 20,000 rows and 10 MB.")
        st.caption("Precision, recall and F1 use macro averaging across all seven classes.")
        st.link_button(
            "View original UCI dataset", "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset"
        )
    if source == "Upload test CSV" and upload is None:
        st.info("Upload a test CSV with all 16 features and the Class column to start evaluation.")
        with st.expander("Required CSV columns", expanded=True):
            st.code(",".join([*FEATURES, "Class"]), language="text")
            st.write("Allowed labels: " + ", ".join(CLASSES))
            st.caption("The original UCI spellings AspectRation and roundness are preserved.")
        return
    content = upload.getvalue() if upload is not None else (ROOT / "test_data.csv").read_bytes()
    try:
        frame = read_test_csv(content)
    except (ValueError, TypeError) as error:
        st.error(str(error))
        return
    with st.spinner("Evaluating saved classifiers..."):
        evaluations = score_dataset(content)
    comparison = pd.DataFrame(
        [{"Model": name, **result.metrics} for name, result in evaluations.items()]
    ).set_index("Model")
    result = evaluations[selected]
    top = comparison["F1"].idxmax()
    overview = st.columns(4)
    overview[0].metric("Test samples", f"{len(frame):,}")
    overview[1].metric("Input features", len(FEATURES))
    overview[2].metric("Classes present", f"{frame['Class'].nunique()} / 7")
    overview[3].metric("Highest macro F1", "SVM" if top == "Support Vector Machine" else top)
    st.caption(
        f"Evaluating: {upload.name if upload is not None else 'bundled test_data.csv'} "
        "| Models trained offline on a disjoint, stratified 80% training split."
    )
    st.subheader(f"{selected} | Evaluation metrics")
    for column, metric in zip(st.columns(6), METRICS, strict=True):
        value = result.metrics[metric]
        column.metric(metric, "N/A" if value is None else f"{value:.4f}")
    if result.metrics["AUC"] is None:
        st.warning(
            "AUC is undefined because this upload does not contain all seven classes. "
            "Other metrics use the fixed seven-class label set; absent classes contribute zero."
        )
    st.caption(
        "AUC: macro one-vs-rest, using predict_proba. MCC: multiclass correlation coefficient."
    )
    compare_tab, diagnostics_tab, data_tab = st.tabs(
        [
            "Compare all models",
            "Model diagnostics",
            "Dataset & predictions",
        ]
    )
    with compare_tab:
        st.subheader("Same test rows, side-by-side")
        st.dataframe(comparison.style.format("{:.4f}", na_rep="N/A"), width="stretch")
        st.bar_chart(comparison[["Accuracy", "F1"]], color=["#16745B", "#A1BC72"], stack=False)
        st.info(
            f"{top} has the highest macro F1 on these rows "
            f"({comparison.loc[top, 'F1']:.4f}). This is a descriptive result, not a guarantee "
            "on unseen data. Hyperparameters were not tuned on this test set."
        )
        st.download_button(
            "Download comparison",
            comparison.to_csv().encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
        )
    with diagnostics_tab:
        left, right = st.columns([1.1, 1])
        with left:
            st.subheader("Confusion matrix")
            figure, axis = plt.subplots(figsize=(7, 5))
            image = axis.imshow(result.confusion, cmap="Greens")
            axis.set_xticks(range(7), CLASSES, rotation=45, ha="right")
            axis.set_yticks(range(7), CLASSES)
            axis.set_xlabel("Predicted class")
            axis.set_ylabel("Actual class")
            threshold = result.confusion.to_numpy().max() / 2
            for row in range(7):
                for col in range(7):
                    count = result.confusion.iloc[row, col]
                    axis.text(
                        col,
                        row,
                        str(count),
                        ha="center",
                        va="center",
                        color="white" if count > threshold else "#172A23",
                        fontsize=8,
                    )
            figure.colorbar(image, ax=axis, shrink=0.7)
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)
        with right:
            st.subheader("Classification report")
            st.dataframe(result.report.style.format("{:.3f}"), width="stretch")
            st.caption("Support is the number of actual examples per class.")
    with data_tab:
        st.subheader("Uploaded test data" if upload is not None else "Bundled held-out test data")
        st.dataframe(frame.head(100), width="stretch")
        st.subheader("Class distribution")
        st.bar_chart(frame["Class"].value_counts().sort_index())
        st.subheader("Predictions")
        st.dataframe(result.predictions.head(100), width="stretch")
        st.caption(
            "Confidence is the maximum predicted class probability, not a calibrated guarantee."
        )
        st.download_button(
            "Download predictions",
            result.predictions.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
    with st.expander("Dataset choice & experimental protocol"):
        st.write(
            "UCI Dry Bean (2020), DOI 10.24432/C50S4B, CC BY 4.0. "
            "13,611 source samples; 16 numerical shape and size features; seven classes. "
            "Exact duplicate labeled rows are removed before the 80/20 stratified split "
            "(random_state=42). StandardScaler is fitted only on training data and is saved "
            "inside each pipeline. No target column is used as an input feature."
        )
        st.caption(
            "This learning reference must be adapted and executed on BITS Virtual Lab "
            "before submission."
        )


if __name__ == "__main__":
    main()
