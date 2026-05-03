import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np

from core.data_loader import load_data
from core.validator import validate_dataset, generate_warnings
from core.preprocessor import (
    remove_high_missing,
    impute_data,
    remove_non_numeric,
    transpose_if_needed,
    normalize_data,
    match_samples
)
from core.feature_selector import select_features
from pipelines.clustering import clustering_pipeline
from core.visualizer import plot_with_labels, plot_tsne_clusters, plot_survival
from core.interpreter import interpret_clusters

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Bioinformatics Dashboard",
    page_icon="🧬",
    layout="wide"
)

# =========================
# CENTERED LAYOUT
# =========================
left, center, right = st.columns([1, 3, 1])

with center:

    st.title("🧬 Multi-Omics Breast Cancer Analyzer")
    st.markdown("Analyze gene expression data and discover patient subtypes")

    # =========================
    # FILE UPLOAD
    # =========================
    st.header("1️⃣ Upload Dataset")

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        uploaded_file = st.file_uploader(
            "📂 Upload Gene Expression Dataset (CSV / TSV / GZ)",
            type=["csv", "tsv", "gz"]
        )

        uploaded_file_2 = st.file_uploader(
            "📂 Optional: Upload Second Dataset (Multi-Omics) (CSV / TSV / GZ)",
            type=["csv", "tsv", "gz"],
            key="second_dataset"
        )

        survival_file = st.file_uploader(
            "📂 Optional: Upload Survival Data (CSV with Sample, time, event)",
            type=["csv"],
            key="survival_data"
        )

    # =========================
    # MAIN PIPELINE
    # =========================
    if uploaded_file:

        # LOAD DATA
        status = st.status("Loading dataset...", expanded=True)
        with status:
            st.write("Reading file...")
            df = load_data(uploaded_file)
        status.update(label="✅ Dataset loaded successfully!", state="complete")

        # MULTI-OMICS
        if uploaded_file_2:

            status = st.status("Processing multi-omics datasets...", expanded=True)

            with status:
                st.write("Loading second dataset...")
                df2 = load_data(uploaded_file_2)

                st.write("Matching samples...")
                df, df2 = match_samples(df, df2)

                st.write("Merging datasets...")
                df = pd.concat([df, df2], axis=1)

            status.update(label="✅ Multi-omics integration complete!", state="complete")

            # 🔥 NEW: show matched samples
            st.info(f"Number of matched patients: {df.shape[1]}")

        # PREVIEW
        st.subheader("Preview (sampled)")
        df_preview = df.copy()
        df_preview = df_preview.apply(lambda col: pd.to_numeric(col, errors='coerce'))
        df_preview = df_preview.dropna(axis=1, how='all')
        df_preview = df_preview.iloc[:100, :50]
        st.dataframe(df_preview)

        # VALIDATION
        st.header("2️⃣ Data Validation")
        report = validate_dataset(df)
        warnings = generate_warnings(report)

        st.subheader("Dataset Summary")
        st.json(report)

        if warnings:
            for w in warnings:
                st.warning(w)

        # PREPROCESSING
        st.header("3️⃣ Preprocessing")

        status = st.status("Preprocessing data...", expanded=True)

        with status:
            st.write("Transposing if needed...")
            df_clean = transpose_if_needed(df, report["orientation"])
            st.write("Removing missing features...")
            df_clean = remove_high_missing(df_clean)
            st.write("Imputing missing values...")
            df_clean = impute_data(df_clean)
            st.write("Filtering numeric data...")
            df_clean = remove_non_numeric(df_clean)

        status.update(label="✅ Preprocessing complete!", state="complete")

        if df_clean.shape[0] == 0 or df_clean.shape[1] == 0:
            st.error("Dataset became empty after preprocessing.")
            st.stop()

        col1, col2 = st.columns(2)
        col1.metric("Patients", f"{df_clean.shape[0]:,}")
        col2.metric("Features", f"{df_clean.shape[1]:,}")

        # NORMALIZATION
        X = normalize_data(df_clean)

        # FEATURE SELECTION
        st.header("4️⃣ Feature Selection")
        k_features = st.slider("Number of features", 500, 5000, 2000)

        status = st.status("Selecting features...", expanded=True)
        with status:
            X_selected = select_features(X, k_features)
        status.update(label="✅ Feature selection complete!", state="complete")

        # MODEL SETTINGS
        st.header("5️⃣ Model Selection")
        k_clusters = st.slider("Number of clusters (K)", 2, 10, 5)
        st.write(f"Selected number of clusters: {k_clusters}")

        # RUN ANALYSIS
        if st.button("🧬 Run Analysis"):

            # 🔥 CRITICAL SAFETY CHECKS
            if X_selected is None or len(X_selected) == 0:
                st.error("No data available after preprocessing.")
                st.stop()

            if np.isnan(X_selected).any():
                st.error("NaN values detected after preprocessing.")
                st.stop()

            if k_clusters >= len(X_selected):
                st.error("K must be less than number of samples.")
                st.stop()

            st.header("6️⃣ Results")

            status = st.status("Running clustering models...", expanded=True)

            with status:
                st.write("Running KMeans...")
                res_kmeans = clustering_pipeline(X_selected, "kmeans", k_clusters)

                st.write("Running Hierarchical...")
                res_hier = clustering_pipeline(X_selected, "hierarchical", k_clusters)

            status.update(label="✅ Clustering complete!", state="complete")

            # MODEL COMPARISON
            st.subheader("Model Comparison")

            col1, col2 = st.columns(2)
            col1.metric("KMeans Score", round(res_kmeans["score"], 3))
            col2.metric("Hierarchical Score", round(res_hier["score"], 3))

            if res_kmeans["score"] >= res_hier["score"]:
                results = res_kmeans
                best_model = "KMeans"
            else:
                results = res_hier
                best_model = "Hierarchical"

            st.info(f"Best performing model: {best_model}")
            if len(set(results["labels"])) < 2:
                st.warning("Only one cluster detected. Results may not be meaningful.")

            st.metric("Silhouette Score", round(results["score"], 3))

            # SURVIVAL ANALYSIS
            if survival_file:

                st.subheader("Survival Analysis (Kaplan-Meier)")

                status = st.status("Running survival analysis...", expanded=True)

                with status:
                    try:
                        surv_df = pd.read_csv(survival_file)

                        if not {"Sample", "time", "event"}.issubset(surv_df.columns):
                            st.error("Survival file must contain: Sample, time, event")
                        else:

                            merged = pd.DataFrame({
                                "Sample": df_clean.index,
                                "Cluster": results["labels"]
                            }).merge(surv_df, on="Sample")

                            merged = merged.dropna()

                            if merged.shape[0] == 0:
                                st.warning("No matching samples for survival analysis.")
                            else:
                                fig_surv = plot_survival(
                                    merged["time"].values,
                                    merged["event"].values,
                                    merged["Cluster"].values
                                )
                                st.pyplot(fig_surv)
                                status.update(label="✅ Survival analysis complete!", state="complete")
                    except Exception as e:
                        st.error("Error occurred while running survival analysis!")

            # VISUALIZATION
            st.subheader("Cluster Visualization (PCA)")
            st.pyplot(plot_with_labels(results["X_reduced"], results["labels"]))

            st.subheader("t-SNE Visualization (Non-linear Projection)")
            st.pyplot(plot_tsne_clusters(X_selected, results["labels"]))

            # INTERPRETATION
            st.subheader("Interpretation")
            st.write(interpret_clusters(results["labels"])["message"])

            # =========================
            # BIOLOGICAL INSIGHT
            # =========================
            st.subheader("Biological Insight")

            st.write("""
            The identified clusters may represent distinct molecular subtypes of breast cancer.
            Strong separation suggests meaningful biological grouping.
            """)
            # DOWNLOAD
            df_results = pd.DataFrame({
                "Sample": df_clean.index,
                "Cluster": results["labels"]
            })

            st.download_button(
                "📥 Download Results",
                df_results.to_csv(index=False).encode("utf-8"),
                "clustering_results.csv",
                "text/csv"
            )

    else:
        st.info("Please upload a dataset to begin.")