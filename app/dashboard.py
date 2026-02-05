from __future__ import annotations

import io
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

from tabular_ml_lab.lab.datasets import BUILTIN_DATASETS, load_builtin
from tabular_ml_lab.lab.modeling import TrainResult, train_sklearn
from tabular_ml_lab.lab.schema import infer_feature_types
from tabular_ml_lab.lab.torch_trainer import TorchResult, train_and_eval_binary
from tabular_ml_lab.lab.utils import parse_list, run_pandas_script


st.set_page_config(page_title="Tabular ML Lab", layout="wide")

st.title("Tabular ML Lab")
st.write(
    "Build, evaluate, and compare tabular models. Start with a dataset, explore statistics, "
    "apply transformations, then train a model with optional grid search."
)


if "df" not in st.session_state:
    st.session_state.df = None
if "target" not in st.session_state:
    st.session_state.target = None
if "results" not in st.session_state:
    st.session_state.results = None
if "task" not in st.session_state:
    st.session_state.task = None


def _auto_task(target_series: pd.Series) -> str:
    if target_series.dtype == "object" or target_series.nunique() <= 20:
        return "classification"
    return "regression"


def _render_metrics(metrics: Dict[str, Any]):
    cols = st.columns(4)
    for idx, (k, v) in enumerate(metrics.items()):
        cols[idx % 4].metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))


def _plot_classification(y_true, y_pred, y_proba):
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax_cm, cmap="Blues")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    if y_proba is not None and y_proba.ndim == 1:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr)
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_title("ROC Curve")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        st.pyplot(fig_roc)

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(recall, precision)
        ax_pr.set_title("Precision-Recall Curve")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        st.pyplot(fig_pr)


def _plot_regression(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    st.pyplot(fig)


def _dataset_loaded() -> bool:
    return st.session_state.df is not None and st.session_state.target is not None


tab_data, tab_transform, tab_model, tab_results = st.tabs(
    ["Dataset", "Transform", "Model", "Results"]
)

with tab_data:
    st.subheader("Select Data")
    source = st.selectbox("Source", ["Built-in", "Upload CSV"])

    df: pd.DataFrame | None = None
    target: str | None = None

    if source == "Built-in":
        dataset_name = st.selectbox("Dataset", list(BUILTIN_DATASETS.keys()))
        if st.button("Load dataset"):
            df, target = load_builtin(dataset_name)
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)

    if df is not None:
        st.session_state.df = df
        st.session_state.target = target
        st.success("Dataset loaded.")

    if st.session_state.df is not None:
        st.write("Preview")
        st.dataframe(st.session_state.df.head(20))

        st.write("Shape", st.session_state.df.shape)

        target = st.selectbox(
            "Target column",
            st.session_state.df.columns,
            index=st.session_state.df.columns.get_loc(st.session_state.target)
            if st.session_state.target in st.session_state.df.columns
            else 0,
        )
        st.session_state.target = target

with tab_transform:
    st.subheader("Transform Data")

    if not _dataset_loaded():
        st.info("Load a dataset first.")
    else:
        df = st.session_state.df.copy()
        target = st.session_state.target

        drop_cols = st.multiselect(
            "Drop columns",
            [col for col in df.columns if col != target],
            default=[],
        )

        num_strategy = st.selectbox(
            "Numeric missing values",
            ["median", "mean", "zero", "drop rows"],
            index=0,
        )
        cat_strategy = st.selectbox(
            "Categorical missing values",
            ["most_frequent", "missing", "drop rows"],
            index=0,
        )

        st.write("Custom pandas script (optional)")
        script = st.text_area(
            "Script",
            value="# df is a pandas DataFrame\n# Example: df['ratio'] = df['a'] / (df['b'] + 1)\n",
            height=150,
        )

        if st.button("Apply transformations"):
            if drop_cols:
                df = df.drop(columns=drop_cols)

            numeric_features, categorical_features = infer_feature_types(df, target)

            if num_strategy == "drop rows" and numeric_features:
                df = df.dropna(subset=numeric_features)
            elif numeric_features:
                if num_strategy == "median":
                    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
                elif num_strategy == "mean":
                    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
                elif num_strategy == "zero":
                    df[numeric_features] = df[numeric_features].fillna(0)

            if cat_strategy == "drop rows" and categorical_features:
                df = df.dropna(subset=categorical_features)
            elif categorical_features:
                if cat_strategy == "most_frequent":
                    for col in categorical_features:
                        df[col] = df[col].fillna(df[col].mode(dropna=True)[0])
                elif cat_strategy == "missing":
                    df[categorical_features] = df[categorical_features].fillna("missing")

            if script.strip():
                df = run_pandas_script(df, script)

            st.session_state.df = df
            st.success("Transformations applied.")

        st.write("Preview")
        st.dataframe(st.session_state.df.head(20))

with tab_model:
    st.subheader("Model Training")

    if not _dataset_loaded():
        st.info("Load a dataset first.")
    else:
        df = st.session_state.df
        target = st.session_state.target
        numeric_features, categorical_features = infer_feature_types(df, target)

        st.write("Features")
        numeric_features = st.multiselect(
            "Numeric features",
            [c for c in df.columns if c != target],
            default=numeric_features,
        )
        categorical_features = st.multiselect(
            "Categorical features",
            [c for c in df.columns if c != target and c not in numeric_features],
            default=categorical_features,
        )

        test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        scale_numeric = st.checkbox("Scale numeric features", value=True)

        inferred_task = _auto_task(df[target])
        task = st.selectbox(
            "Task",
            ["classification", "regression"],
            index=0 if inferred_task == "classification" else 1,
        )

        model_choices = (
            ["Logistic Regression", "Random Forest", "Gradient Boosting", "PyTorch MLP"]
            if task == "classification"
            else ["Linear Regression", "Random Forest", "Gradient Boosting"]
        )
        model_name = st.selectbox("Model", model_choices)

        use_grid = st.checkbox("Use grid search (sklearn models only)", value=False)

        params: Dict[str, Any] = {}
        grid: Dict[str, Any] = {}

        if model_name == "Logistic Regression":
            params["C"] = st.number_input("C", min_value=0.01, max_value=100.0, value=1.0)
            params["max_iter"] = st.number_input("Max Iter", min_value=100, max_value=2000, value=500)
            params["solver"] = st.selectbox("Solver", ["liblinear", "lbfgs", "saga"])
            if use_grid:
                grid["C"] = parse_list(st.text_input("Grid C (comma-separated)", "0.1,1,10"), float)
        elif model_name == "Random Forest" and task == "classification":
            params["n_estimators"] = st.number_input("Trees", min_value=50, max_value=500, value=200)
            params["max_depth"] = st.number_input("Max depth", min_value=2, max_value=40, value=10)
            params["min_samples_split"] = st.number_input("Min samples split", min_value=2, max_value=20, value=2)
            if use_grid:
                grid["n_estimators"] = parse_list(st.text_input("Grid trees", "100,200,300"), int)
                grid["max_depth"] = parse_list(st.text_input("Grid depth", "5,10,15"), int)
        elif model_name == "Gradient Boosting" and task == "classification":
            params["n_estimators"] = st.number_input("Estimators", min_value=50, max_value=500, value=200)
            params["learning_rate"] = st.number_input("Learning rate", min_value=0.01, max_value=1.0, value=0.1)
            params["max_depth"] = st.number_input("Max depth", min_value=1, max_value=10, value=3)
            if use_grid:
                grid["n_estimators"] = parse_list(st.text_input("Grid estimators", "100,200,300"), int)
                grid["learning_rate"] = parse_list(st.text_input("Grid lr", "0.05,0.1,0.2"), float)
        elif model_name == "Linear Regression":
            params["fit_intercept"] = st.checkbox("Fit intercept", value=True)
        elif model_name == "Random Forest" and task == "regression":
            params["n_estimators"] = st.number_input("Trees", min_value=50, max_value=500, value=200)
            params["max_depth"] = st.number_input("Max depth", min_value=2, max_value=40, value=10)
            params["min_samples_split"] = st.number_input("Min samples split", min_value=2, max_value=20, value=2)
            if use_grid:
                grid["n_estimators"] = parse_list(st.text_input("Grid trees", "100,200,300"), int)
                grid["max_depth"] = parse_list(st.text_input("Grid depth", "5,10,15"), int)
        elif model_name == "Gradient Boosting" and task == "regression":
            params["n_estimators"] = st.number_input("Estimators", min_value=50, max_value=500, value=200)
            params["learning_rate"] = st.number_input("Learning rate", min_value=0.01, max_value=1.0, value=0.1)
            params["max_depth"] = st.number_input("Max depth", min_value=1, max_value=10, value=3)
            if use_grid:
                grid["n_estimators"] = parse_list(st.text_input("Grid estimators", "100,200,300"), int)
                grid["learning_rate"] = parse_list(st.text_input("Grid lr", "0.05,0.1,0.2"), float)
        elif model_name == "PyTorch MLP":
            hidden = st.text_input("Hidden layers (comma-separated)", "256,128")
            params["hidden_layers"] = [int(x) for x in hidden.split(",") if x.strip()]
            params["dropout"] = st.slider("Dropout", 0.0, 0.7, 0.2, 0.05)
            params["lr"] = st.number_input("Learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
            params["weight_decay"] = st.number_input("Weight decay", min_value=0.0, max_value=1e-2, value=1e-4, format="%.5f")
            params["epochs"] = st.number_input("Epochs", min_value=5, max_value=200, value=30)
            params["batch_size"] = st.number_input("Batch size", min_value=32, max_value=1024, value=256)
            params["early_stopping_patience"] = st.number_input("Early stopping patience", min_value=1, max_value=10, value=4)
            params["val_split"] = st.slider("Validation split", 0.1, 0.4, 0.2, 0.05)

        if st.button("Train"):
            with st.spinner("Training..."):
                if model_name == "PyTorch MLP":
                    result = train_and_eval_binary(
                        df,
                        target,
                        numeric_features,
                        categorical_features,
                        params,
                        test_size=test_size,
                        scale_numeric=scale_numeric,
                    )
                    st.session_state.results = result
                    st.session_state.task = task
                else:
                    result = train_sklearn(
                        df,
                        target,
                        numeric_features,
                        categorical_features,
                        task,
                        model_name,
                        params,
                        grid=grid if use_grid else None,
                        test_size=test_size,
                        scale_numeric=scale_numeric,
                    )
                    st.session_state.results = result
                    st.session_state.task = task

                st.success("Training complete.")

with tab_results:
    st.subheader("Results")

    if st.session_state.results is None:
        st.info("Train a model to see results.")
    else:
        results = st.session_state.results
        if isinstance(results, TrainResult):
            _render_metrics(results.metrics)
            if st.session_state.task == "classification":
                _plot_classification(results.y_true, results.y_pred, results.y_proba)
            else:
                _plot_regression(results.y_true, results.y_pred)
        elif isinstance(results, TorchResult):
            _render_metrics(results.metrics)
            _plot_classification(results.y_true, results.y_pred, results.y_proba)

        st.write("Download metrics")
        metrics_payload = results.metrics
        buffer = io.StringIO()
        pd.Series(metrics_payload).to_json(buffer)
        st.download_button("Download JSON", buffer.getvalue(), file_name="metrics.json")
