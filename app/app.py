import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np

# =========================
# SESSION STATE
# =========================
if "last_uploaded_file" not in st.session_state:
    st.session_state["last_uploaded_file"] = None

if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False

if "results" not in st.session_state:
    st.session_state["results"] = None

if "df_results" not in st.session_state:
    st.session_state["df_results"] = None

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
from core.visualizer import (
    plot_with_labels,
    plot_tsne_clusters,
    plot_survival
)
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
# CACHE DATA LOADING
# =========================
@st.cache_data
def cached_load(file):
    return load_data(file)

# =========================
# CACHE CLUSTERING
# =========================
@st.cache_data
def cached_clustering(X, model_type, k):
    return clustering_pipeline(X, model_type, k)

# =========================
# CACHE FEATURE SELECTION
# =========================
@st.cache_data
def cached_feature_selection(X, k):
    return select_features(X, k)

# =========================
# CENTERED LAYOUT
# =========================
left, center, right = st.columns([1, 3, 1])

with center:

    st.title("🧬 Multi-Omics Breast Cancer Analyzer")
    st.markdown(
        "Analyze gene expression data and discover patient subtypes"
    )

    # =========================
    # FILE UPLOAD
    # =========================
    st.header("1️⃣ Upload Dataset")

    st.info(
        """
        - Upload a gene expression dataset (CSV/TSV/GZ) with TCGA sample IDs as columns.
        - Optionally upload a second multi-omics dataset for integration.
        - Optionally upload a survival dataset for Kaplan-Meier analysis.
        """
    )

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:

        gene_expression_file = st.file_uploader(
            "📂 Upload Gene Expression Dataset (CSV / TSV / GZ)",
            type=["csv", "tsv", "gz"]
        )

        multi_omics_file = st.file_uploader(
            "📂 Optional: Upload Second Dataset (Multi-Omics)",
            type=["csv", "tsv", "gz"],
            key="second_dataset"
        )

        survival_file = st.file_uploader(
            "📂 Optional: Upload Survival Data",
            type=["csv", "tsv", "gz"],
            key="survival_data"
        )


    # =========================
    # MAIN PIPELINE
    # =========================
    if gene_expression_file:

        # Reset previous analysis if new file uploaded
        if "last_uploaded_file" not in st.session_state:

            st.session_state["last_uploaded_file"] = gene_expression_file.name

        elif (
            st.session_state["last_uploaded_file"]
            != gene_expression_file.name
        ):

            st.session_state["analysis_done"] = False
            st.session_state["results"] = None
            st.session_state["df_results"] = None

            st.session_state["last_uploaded_file"] = gene_expression_file.name

        # =========================
        # LOAD DATA
        # =========================
        status = st.status(
            "Loading dataset...",
            expanded=True
        )

        with status:

            st.write(
                "Reading gene expression dataset..."
            )

            df = cached_load(gene_expression_file)

        status.update(
            label="✅ Dataset loaded successfully!",
            state="complete"
        )

        # =========================
        # MULTI-OMICS INTEGRATION
        # =========================
        if multi_omics_file:

            status = st.status(
                "Processing multi-omics datasets...",
                expanded=True
            )

            with status:

                st.write("Loading second dataset...")
                df2 = cached_load(multi_omics_file)

                st.write("Matching TCGA patient barcodes...")
                df, df2 = match_samples(df, df2)

                st.write("Merging matched datasets...")
                df = pd.concat([df, df2], axis=1)

            status.update(
                label="✅ Multi-omics integration complete!",
                state="complete"
            )

            st.info(
                f"Matched patients across datasets: {df.shape[1]}"
            )

            st.caption("""
            Different omics datasets may contain different patient subsets.
            Samples were aligned using TCGA barcodes and only common
            patients were retained for integrated analysis.
            """)

        # =========================
        # DATA PREVIEW
        # =========================
        st.header("2️⃣ Dataset Preview")

        st.info(
        """
        - Only the first 10 rows and 50 columns are shown for performance.
        - Non-numeric columns are hidden in the preview but retained for analysis.
        - Ensure that TCGA sample IDs are present as column headers for proper analysis.
        """
        )

        df_preview = df.copy()

        df_preview = df_preview.apply(
            lambda col: pd.to_numeric(col, errors='coerce')
        )

        df_preview = df_preview.dropna(axis=1, how='all')

        df_preview = df_preview.iloc[:10, :50]

        st.dataframe(
            df_preview,
            width="stretch"
        )

        # =========================
        # VALIDATION
        # =========================
        st.header("3️⃣ Data Validation")

        st.info(
        """
        - Checks dataset orientation (samples vs features).
        - Identifies missing values and high-missing features.
        - Detects non-numeric features that may require special handling.
        - Provides warnings for potential issues that may affect analysis.
        """
        )

        report = validate_dataset(df)
        warnings = generate_warnings(report)

        st.subheader("Dataset Summary")
        st.json(report)

        if warnings:
            st.subheader("⚠ Warnings")

            for w in warnings:
                st.warning(w)

        # =========================
        # PREPROCESSING
        # =========================
        st.header("4️⃣ Preprocessing")

        st.info(
        """
        - Transposes dataset if samples are in rows instead of columns.
        - Removes features with >50% missing values.
        - Imputes remaining missing values using mean imputation.
        - Filters out non-numeric features that cannot be used for clustering.
        """
        )

        status = st.status(
            "Preprocessing genomic data...",
            expanded=True
        )

        with status:

            st.write("Checking dataset orientation...")
            df_clean = transpose_if_needed(
                df,
                report["orientation"]
            )

            st.write("Removing high-missing features...")
            df_clean = remove_high_missing(df_clean)

            st.write("Imputing missing values...")
            df_clean = impute_data(df_clean)

            st.write("Filtering numeric features...")
            df_clean = remove_non_numeric(df_clean)

        status.update(
            label="✅ Preprocessing complete!",
            state="complete"
        )

        # =========================
        # SAFETY CHECKS
        # =========================
        if df_clean.shape[0] == 0 or df_clean.shape[1] == 0:
            st.error(
                "Dataset became empty after preprocessing."
            )
            st.stop()

        # =========================
        # DATASET METRICS
        # =========================
        col1, col2 = st.columns(2)

        col1.metric(
            "Patients",
            f"{df_clean.shape[0]:,}"
        )

        col2.metric(
            "Features",
            f"{df_clean.shape[1]:,}"
        )

        # =========================
        # NORMALIZATION
        # =========================
        X = normalize_data(df_clean)

        # =========================
        # FEATURE SELECTION
        # =========================
        st.header("5️⃣ Feature Selection")

        st.info(
        """
        - Reduces dimensionality to improve clustering performance.
        - Selects top K most informative features based on variance.
        """
        )

        k_features = st.slider(
            "Number of features",
            500,
            5000,
            2000
        )

        status = st.status(
            "Selecting top informative features...",
            expanded=True
        )

        with status:

            X_selected = cached_feature_selection(
                X,
                k_features
            )

        status.update(
            label="✅ Feature selection complete!",
            state="complete"
        )

        st.write(
            "Selected feature shape:",
            X_selected.shape
        )

        top_features = df_clean.var().sort_values(
            ascending=False
        ).head(10)

        # =========================
        # MODEL SETTINGS
        # =========================
        st.header("6️⃣ Model Selection")

        st.info(
        """
        - Choose the number of clusters (K) for analysis.
        - K should be smaller than the number of samples.
        """
        )

        k_clusters = st.slider(
            "Number of clusters (K)",
            2,
            10,
            5
        )

        st.write(
            f"Selected number of clusters: {k_clusters}"
        )

        # =========================
        # RUN ANALYSIS
        # =========================
        run_analysis = st.button("🧬 Run Analysis")

        if run_analysis or st.session_state["analysis_complete"]:

            try:

                # =========================
                # SAFETY CHECKS
                # =========================
                if X_selected is None or len(X_selected) == 0:
                    st.error(
                        "No data available after preprocessing."
                    )
                    st.stop()

                if np.isnan(X_selected).any():
                    st.error(
                        "NaN values detected after preprocessing."
                    )
                    st.stop()

                if k_clusters >= len(X_selected):
                    st.error(
                        "K must be smaller than number of samples."
                    )
                    st.stop()

                # =========================
                # CLUSTERING
                # =========================
                st.header("7️⃣ Results")

                st.info(
                """
                - Compares KMeans and Hierarchical clustering models.
                - Evaluates clustering quality using silhouette score.
                - Visualizes clusters using PCA and t-SNE.
                - Provides interpretation of identified clusters.
                - If survival data is provided, performs Kaplan-Meier analysis to compare survival between clusters.
                """
                )

                status = st.status(
                    "Running clustering models...",
                    expanded=True
                )

                with status:

                    st.write("Running KMeans clustering...")
                    res_kmeans = cached_clustering(
                        X_selected,
                        "kmeans",
                        k_clusters
                    )

                    st.write("Running Hierarchical clustering...")
                    res_hier = cached_clustering(
                        X_selected,
                        "hierarchical",
                        k_clusters
                    )

                status.update(
                    label="✅ Clustering complete!",
                    state="complete"
                )

                # =========================
                # MODEL COMPARISON
                # =========================
                st.subheader("Model Comparison")

                col1, col2 = st.columns(2)

                col1.metric(
                    "KMeans Score",
                    round(res_kmeans["score"], 3)
                )

                col2.metric(
                    "Hierarchical Score",
                    round(res_hier["score"], 3)
                )

                if res_kmeans["score"] >= res_hier["score"]:
                    results = res_kmeans
                    best_model = "KMeans"
                else:
                    results = res_hier
                    best_model = "Hierarchical"

                st.success(
                    f"Best Performing Model: {best_model}"
                )

                # =========================
                # SCORE
                # =========================
                st.subheader("Clustering Quality")

                st.metric(
                    "Silhouette Score",
                    round(results["score"], 3)
                )

                if len(set(results["labels"])) < 2:
                    st.warning(
                        "Only one cluster detected."
                    )

                # =========================
                # SURVIVAL ANALYSIS
                # =========================
                if survival_file:

                    st.subheader("Survival Analysis (Kaplan-Meier)")

                    status = st.status(
                        "Running survival analysis...",
                        expanded=True
                    )

                    with status:

                        if survival_file.name.endswith(".gz"):

                            surv_df = pd.read_csv(
                                survival_file,
                                sep="\t",
                                compression="gzip"
                            )

                        elif survival_file.name.endswith(".tsv"):

                            surv_df = pd.read_csv(
                                survival_file,
                                sep="\t"
                            )

                        else:

                            surv_df = pd.read_csv(survival_file)

                        # =========================
                        # TCGA SURVIVAL EXTRACTION
                        # =========================

                        if "sample" not in surv_df.columns:
                            st.error(
                                "Survival file missing 'sample' column."
                            )
                            st.stop()

                        st.write(
                                    "Standardizing TCGA patient IDs..."
                                )

                        surv_df["Sample"] = (
                            surv_df["sample"]
                            .astype(str)
                            .str[:12]
                        )

                        st.write(
                                    "Preparing survival events..."
                                )

                        # =========================
                        # DETECT VITAL STATUS COLUMN
                        # =========================
                        vital_col = None

                        possible_vital_cols = [

                            "vital_status.demographic",

                            "vital_status",

                            "patient.vital_status",

                            "OS"

                        ]

                        for col in possible_vital_cols:

                            if col in surv_df.columns:

                                vital_col = col

                                break

                        if vital_col is None:

                            st.error(
                                "Could not find a valid "
                                "vital status column."
                            )

                            st.write(
                                "Available columns:"
                            )

                            st.write(
                                surv_df.columns.tolist()
                            )

                            st.stop()

                        # =========================
                        # CREATE EVENT COLUMN
                        # =========================
                        surv_df["event"] = (

                            surv_df[vital_col]

                            .astype(str)

                            .str.lower()

                            .map({

                                "dead": 1,

                                "deceased": 1,

                                "alive": 0

                            })

                        )

                        st.write(
                                    "Preparing survival times..."
                                )

                        # =========================
                        # SURVIVAL TIME
                        # =========================

                        surv_df["time"] = np.nan

                        if (
                            "days_to_death.demographic"
                            in surv_df.columns
                        ):

                            surv_df["time"] = pd.to_numeric(

                                surv_df[
                                    "days_to_death.demographic"
                                ],

                                errors="coerce"
                            )

                        if (
                            "days_to_last_follow_up.diagnoses"
                            in surv_df.columns
                        ):

                            follow_up = pd.to_numeric(

                                surv_df[
                                    "days_to_last_follow_up.diagnoses"
                                ],

                                errors="coerce"

                            )

                            surv_df["time"] = surv_df[
                                "time"
                            ].fillna(
                                follow_up
                            )

                        surv_df = surv_df[
                            ["Sample", "time", "event"]
                        ]

                        surv_df = surv_df.dropna()

                        # =========================
                        # MERGE WITH CLUSTERS
                        # =========================
                        merged = pd.DataFrame({
                            "Sample": df_clean.index,
                            "Cluster": results["labels"]
                        }).merge(
                            surv_df,
                            on="Sample"
                        )

                        merged = merged.dropna()

                        st.write(
                                    "Matched survival samples:",
                                    merged.shape[0]
                                )
                        
                        st.write(
                                    "Detected clusters:",
                                    merged["Cluster"].unique()
                                )

                        if merged.shape[0] == 0:

                            st.warning(
                                "No overlapping samples found "
                                "between clustering results "
                                "and survival dataset."
                            )

                        else:

                            st.write(
                                "Generating Kaplan-Meier curves..."
                            )

                            # =========================
                            # PLOT SURVIVAL
                            # =========================

                    st.pyplot(
                        plot_survival(

                            merged["time"].values,

                            merged["event"].values,

                            merged["Cluster"].values

                        )
                    )

                    st.caption(
                        """
                        Kaplan-Meier survival curves
                        comparing patient clusters.
                        """
                    )

                    status.update(

                        label="✅ Survival analysis complete!",

                        state="complete"

                    )

                # =========================
                # VISUALIZATION
                # =========================
                st.subheader(
                    "PCA Cluster Visualization"
                )

                st.pyplot(
                    plot_with_labels(
                        results["X_reduced"],
                        results["labels"]
                    )
                )

                st.subheader(
                    "t-SNE Visualization"
                )

                st.pyplot(
                    plot_tsne_clusters(
                        X_selected,
                        results["labels"]
                    )
                )

                # =========================
                # INTERPRETATION
                # =========================
                st.subheader("Interpretation")

                st.write(
                    interpret_clusters(
                        results["labels"]
                    )["message"]
                )

                # =========================
                # BIOLOGICAL INSIGHT
                # =========================
                st.subheader(
                    "Biological Insight"
                )

                st.write("""
                The identified clusters may represent
                biologically distinct molecular subtypes
                of breast cancer.

                Clinical phenotype labels can be used
                later for validation and downstream
                biological interpretation.
                """)

                df_results = pd.DataFrame({
                    "Sample": df_clean.index,
                    "Cluster": results["labels"]
                })

                st.session_state["results"] = results
                st.session_state["df_results"] = df_results
                st.session_state["analysis_done"] = True
                st.session_state["analysis_complete"] = True

            except Exception as e:

                st.error(
                     f"An error occurred during analysis: {e}"
                )

        else:
            st.info(
                "Please click the 'Run Analysis' button to begin."
            )

    else:
        st.info(
            "Please upload a dataset to begin."
        )

    # =========================
    # DOWNLOAD RESULTS
    # =========================
    if st.session_state.get("analysis_done", False):

        st.subheader("Download Results")

        try:

            st.download_button(
                "📥 Download Results",
                st.session_state["df_results"]
                    .to_csv(index=False)
                    .encode("utf-8"),
                "clustering_results.csv",
                "text/csv"
            )

            st.info(
                """
                The CSV file contains:
                - TCGA sample IDs
                - Assigned cluster labels

                These results can be used for:
                - downstream biological interpretation
                - survival validation
                - subtype comparison
                """
            )

        except Exception as e:

            st.error(
                "An error occurred while preparing the download."
            )