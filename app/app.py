import sys
import os

# Fix module import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd

from core.data_loader import load_data
from core.validator import validate_dataset, generate_warnings
from core.preprocessor import (
    remove_high_missing,
    impute_data,
    remove_non_numeric,
    transpose_if_needed,
    normalize_data
)
from core.feature_selector import select_features
from pipelines.clustering import clustering_pipeline
from core.visualizer import (
    plot_with_labels,
    plot_tsne_clusters
)
from core.interpreter import interpret_clusters

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Bioinformatics Dashboard", page_icon="🧬", layout="wide")

left, center, right = st.columns([1, 3, 1])

with center:

    st.title("🧬 Multi-Omics Breast Cancer Analyzer")
    st.markdown("Analyze gene expression data and discover patient subtypes")

    # =========================
    # FILE UPLOAD
    # =========================
    st.header("1️⃣ Upload Dataset")

    with center:
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            uploaded_file = st.file_uploader(
                "📂 Upload Gene Expression Dataset (CSV / TSV / GZ)",
                type=["csv", "tsv", "gz"]
            )

    # =========================
    # MAIN PIPELINE
    # =========================
    if uploaded_file:

        # =========================
        # LOAD DATA
        # =========================
        df = load_data(uploaded_file)

        st.success(f"✅ {uploaded_file.name} - Dataset has been loaded successfully!")

        # =========================
        # SAFE PREVIEW
        # =========================
        st.subheader("Preview (sampled for performance)")

        df_preview = df.copy()
        df_preview = df_preview.apply(lambda col: pd.to_numeric(col, errors='coerce'))
        df_preview = df_preview.dropna(axis=1, how='all')
        df_preview = df_preview.iloc[:100, :50]

        st.dataframe(df_preview)

        # =========================
        # VALIDATION
        # =========================
        st.header("2️⃣ Data Validation")

        report = validate_dataset(df)
        warnings = generate_warnings(report)

        st.subheader("Dataset Summary")
        st.json(report)

        st.write("Unique data types BEFORE cleaning:")
        st.write([str(x) for x in df.dtypes.unique()])

        if warnings:
            st.subheader("⚠ Warnings")
            for w in warnings:
                st.warning(w)

        # =========================
        # PREPROCESSING
        # =========================
        st.header("3️⃣ Preprocessing")
        st.info("Cleaning and preparing genomic data... 🧬")

        df_clean = transpose_if_needed(df, report["orientation"])
        df_clean = remove_high_missing(df_clean)
        df_clean = impute_data(df_clean)
        df_clean = remove_non_numeric(df_clean)

        # 🔥 Safety check
        if df_clean.shape[0] == 0 or df_clean.shape[1] == 0:
            st.error("Dataset became empty after preprocessing. Please check input data.")
            st.stop()

        st.write("Unique data types AFTER cleaning:")
        st.write([str(x) for x in df_clean.dtypes.unique()])

        st.write(f"Final dataset: {df_clean.shape[0]} patients × {df_clean.shape[1]} features")

        col1, col2 = st.columns(2)
        col1.metric("Patients", df_clean.shape[0])
        col2.metric("Features", df_clean.shape[1])

        # Normalize
        X = normalize_data(df_clean)

        # =========================
        # FEATURE SELECTION
        # =========================
        st.header("4️⃣ Feature Selection")

        k_features = st.slider("Number of features", 500, 5000, 2000)
        X_selected = select_features(X, k_features)

        st.write("Selected feature shape:", X_selected.shape)

        # =========================
        # MODEL SELECTION
        # =========================
        st.header("5️⃣ Model Selection")

        k_clusters = st.slider("Number of clusters (K)", 2, 10, 5)
        st.write(f"Selected number of clusters: {k_clusters}")

        # =========================
        # RUN ANALYSIS
        # =========================
        if st.button("🧬 Run Analysis"):

            st.header("6️⃣ Results")

            with st.spinner("Running clustering models... 🧬"):

                res_kmeans = clustering_pipeline(X_selected, model_type="kmeans", k=k_clusters)
                res_hier = clustering_pipeline(X_selected, model_type="hierarchical", k=k_clusters)

            st.success("Analysis Complete!")

            # =========================
            # MODEL COMPARISON
            # =========================
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

            # =========================
            # RESULTS
            # =========================
            st.subheader("Clustering Score")
            st.metric("Silhouette Score", round(results["score"], 3))

            # =========================
            # VISUALIZATION
            # =========================
            st.subheader("Cluster Visualization (PCA)")

            fig = plot_with_labels(results["X_reduced"], results["labels"])
            st.pyplot(fig)

            st.caption("Each color represents a different patient cluster")

            st.subheader("t-SNE Visualization (Non-linear Projection)")

            fig_tsne = plot_tsne_clusters(X_selected, results["labels"])
            st.pyplot(fig_tsne)

            # =========================
            # INTERPRETATION
            # =========================
            st.subheader("Interpretation")

            interpretation = interpret_clusters(results["labels"])
            st.write(interpretation["message"])

            # =========================
            # BIOLOGICAL INSIGHT
            # =========================
            st.subheader("Biological Insight")

            st.write("""
            The identified clusters may represent distinct molecular subtypes of breast cancer.

            Strong separation indicates biologically meaningful grouping of patients,
            while weaker separation may reflect overlapping molecular characteristics.

            Integration with clinical labels (e.g., PAM50) would enable robust biological validation
            and subtype confirmation.
            """)

            # =========================
            # FEEDBACK
            # =========================
            st.subheader("Model Feedback")

            if results["score"] > 0.5:
                st.success("Clusters are well separated (strong structure detected)")
            else:
                st.warning("""
                Cluster separation is weak.

                Consider:
                • tuning cluster number (K)  
                • improving feature selection  
                • trying alternative models  
                """)

            # =========================
            # SUMMARY REPORT
            # =========================
            st.subheader("Summary Report")

            st.markdown(f"""
            ### 🧬 Analysis Summary

            - **Dataset Size:** {df_clean.shape[0]} patients × {df_clean.shape[1]} features  
            - **Best Model:** {best_model}  
            - **Silhouette Score:** {round(results["score"], 3)}  

            ### 📊 Interpretation

            The clustering analysis identified distinct patient subgroups based on gene expression patterns.

            - High silhouette score → strong separation  
            - Low score → overlapping biological signals  

            ### 🧪 Biological Insight

            These clusters may represent molecular subtypes of breast cancer.  
            Further validation with clinical labels (e.g., PAM50) is recommended.

            ---
            """)

            # =========================
            # DOWNLOAD RESULTS
            # =========================
            st.subheader("Download Results")

            df_results = pd.DataFrame({
                "Sample": df_clean.index,
                "Cluster": results["labels"]
            })

            csv = df_results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Clustering Results (CSV)",
                data=csv,
                file_name="clustering_results.csv",
                mime="text/csv"
            )

    else:
        st.info("Please upload a dataset to begin.")