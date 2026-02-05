from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

from tabular_ml_lab.lab.datasets import BUILTIN_DATASETS, load_builtin, load_openml
from tabular_ml_lab.lab.experiments import export_active, save_run
from tabular_ml_lab.lab.modeling import TrainResult, train_sklearn
from tabular_ml_lab.lab.schema import infer_feature_types
from tabular_ml_lab.lab.torch_trainer import TorchResult, train_and_eval_binary
from tabular_ml_lab.lab.utils import parse_list, run_pandas_script


BASE_DIR = Path(__file__).resolve().parents[1]

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
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "dataset_description" not in st.session_state:
    st.session_state.dataset_description = None
if "results" not in st.session_state:
    st.session_state.results = None
if "task" not in st.session_state:
    st.session_state.task = None
if "last_run" not in st.session_state:
    st.session_state.last_run = None


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


def _dataset_overview(df: pd.DataFrame):
    missing = df.isna().mean().sort_values(ascending=False)
    duplicates = df.duplicated().sum()
    st.write("Rows", df.shape[0])
    st.write("Columns", df.shape[1])
    st.write("Duplicates", int(duplicates))
    st.write("Missing (top 10)")
    st.dataframe(missing.head(10))


def _short_description(text: str | None) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text[:400] + ("..." if len(text) > 400 else "")


(tab_data, tab_eda, tab_transform, tab_model, tab_results) = st.tabs(
    ["Dataset", "EDA", "Transform", "Model", "Results"]
)

with tab_data:
    st.subheader("Select Data")
    source = st.selectbox("Source", ["Built-in", "OpenML", "Upload CSV"])

    df: pd.DataFrame | None = None
    target: str | None = None
    dataset_name: str | None = None
    dataset_description: str | None = None

    if source == "Built-in":
        dataset_name = st.selectbox("Dataset", list(BUILTIN_DATASETS.keys()))
        if st.button("Load dataset"):
            df, target, info = load_builtin(dataset_name)
            dataset_name = info.get("name", dataset_name)
            dataset_description = info.get("description")
    elif source == "OpenML":
        identifier = st.text_input("OpenML dataset id or name", "")
        if st.button("Load OpenML"):
            if identifier.strip():
                with st.spinner("Downloading OpenML dataset..."):
                    df, target, info = load_openml(identifier.strip())
                    dataset_name = info.get("name", identifier)
                    dataset_description = info.get("description")
            else:
                st.warning("Enter a dataset id or name.")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            dataset_name = "Uploaded CSV"
            dataset_description = "User uploaded CSV file."

    if df is not None:
        st.session_state.df = df
        st.session_state.target = target
        st.session_state.dataset_name = dataset_name
        st.session_state.dataset_description = dataset_description
        st.success("Dataset loaded.")

    if st.session_state.df is not None:
        if st.session_state.dataset_name:
            st.write("Dataset", st.session_state.dataset_name)
        if st.session_state.dataset_description:
            st.write(_short_description(st.session_state.dataset_description))

        st.write("Preview")
        st.dataframe(st.session_state.df.head(20))
        _dataset_overview(st.session_state.df)

        target = st.selectbox(
            "Target column",
            st.session_state.df.columns,
            index=st.session_state.df.columns.get_loc(st.session_state.target)
            if st.session_state.target in st.session_state.df.columns
            else 0,
        )
        st.session_state.target = target

with tab_eda:
    st.subheader("EDA")

    if not _dataset_loaded():
        st.info("Load a dataset first.")
    else:
        df = st.session_state.df
        target = st.session_state.target

        st.write("Overview")
        _dataset_overview(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

        st.write("Summary statistics")
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().transpose())

        st.write("Categorical frequency (top 10)")
        if categorical_cols:
            cat_col = st.selectbox("Categorical column", categorical_cols)
            st.dataframe(df[cat_col].value_counts().head(10))

        st.write("Distributions")
        if numeric_cols:
            num_col = st.selectbox("Numeric column", numeric_cols)
            bins = st.slider("Bins", min_value=5, max_value=80, value=30)
            fig, ax = plt.subplots()
            ax.hist(df[num_col].dropna(), bins=bins, color="#4C78A8", alpha=0.8)
            ax.set_title(f"Distribution: {num_col}")
            st.pyplot(fig)

        if categorical_cols:
            cat_dist_col = st.selectbox("Categorical distribution", categorical_cols, key="cat_dist")
            top_n = st.slider("Top N categories", min_value=5, max_value=30, value=10)
            counts = df[cat_dist_col].value_counts().head(top_n)
            fig, ax = plt.subplots()
            counts.plot(kind="bar", ax=ax, color="#F58518")
            ax.set_title(f"Top {top_n}: {cat_dist_col}")
            st.pyplot(fig)

        st.write("Relationships")
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X", numeric_cols, key="x_col")
            y_col = st.selectbox("Y", numeric_cols, key="y_col")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=target if target in df.columns else None, ax=ax)
            ax.set_title("Scatter")
            st.pyplot(fig)

            corr_cols = st.multiselect("Correlation columns", numeric_cols, default=numeric_cols[:6])
            if len(corr_cols) >= 2:
                fig, ax = plt.subplots()
                sns.heatmap(df[corr_cols].corr(), cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)

        st.write("Target distribution")
        if target in df.columns:
            fig, ax = plt.subplots()
            df[target].value_counts().plot(kind="bar", ax=ax, color="#54A24B")
            ax.set_title("Target distribution")
            st.pyplot(fig)

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
                metadata = {
                    "dataset": st.session_state.dataset_name or "dataset",
                    "description": st.session_state.dataset_description or "",
                    "target": target,
                    "task": task,
                    "model": model_name,
                    "params": params,
                    "grid": grid if use_grid else None,
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "numeric_features": numeric_features,
                    "categorical_features": categorical_features,
                }

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
                    metadata["model_type"] = "torch"
                    run_dir, artifacts = save_run(
                        BASE_DIR,
                        metadata=metadata,
                        metrics=result.metrics,
                        model=result.checkpoint,
                        model_type="torch",
                        preprocessor=result.preprocessor,
                    )
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
                    metadata["model_type"] = "sklearn"
                    if result.best_params:
                        metadata["best_params"] = result.best_params
                    run_dir, artifacts = save_run(
                        BASE_DIR,
                        metadata=metadata,
                        metrics=result.metrics,
                        model=result.model,
                        model_type="sklearn",
                    )

                export_active(
                    BASE_DIR,
                    metadata=metadata,
                    artifacts=artifacts,
                )

                st.session_state.results = result
                st.session_state.task = task
                st.session_state.last_run = str(run_dir)

                st.success("Training complete.")

with tab_results:
    st.subheader("Results")

    if st.session_state.results is None:
        st.info("Train a model to see results.")
    else:
        results = st.session_state.results
        if isinstance(results, TrainResult):
            _render_metrics(results.metrics)
            if results.best_params:
                st.write("Best params", results.best_params)
            if st.session_state.task == "classification":
                _plot_classification(results.y_true, results.y_pred, results.y_proba)
            else:
                _plot_regression(results.y_true, results.y_pred)
        elif isinstance(results, TorchResult):
            _render_metrics(results.metrics)
            _plot_classification(results.y_true, results.y_pred, results.y_proba)

        if st.session_state.last_run:
            st.write("Run saved to", st.session_state.last_run)
            st.write("Active model exported to models/active")

        st.write("Download metrics")
        metrics_payload = results.metrics
        buffer = io.StringIO()
        pd.Series(metrics_payload).to_json(buffer)
        st.download_button("Download JSON", buffer.getvalue(), file_name="metrics.json")

        st.write("API example")
        sample = st.session_state.df.drop(columns=[st.session_state.target]).iloc[0].to_dict()
        st.code(
            "curl -X POST http://localhost:8000/predict \\\n  -H \"Content-Type: application/json\" \\\n  -d '" + str({"features": sample}).replace("'", "\"") + "'",
            language="bash",
        )
