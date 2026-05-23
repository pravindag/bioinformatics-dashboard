# 🧬 Multi-Omics Breast Cancer Analyzer

An interactive bioinformatics dashboard for analyzing high-dimensional TCGA breast cancer datasets using machine learning, multi-omics integration, survival analysis, and advanced visualization techniques.

This system enables users to upload genomic datasets, preprocess high-dimensional biological data, compare clustering models, visualize molecular patterns, and extract biologically meaningful insights through an interactive Streamlit interface.

---

# 📌 Project Overview

Modern bioinformatics combines multiple biological data types (multi-omics) to better understand diseases such as cancer.

This project focuses on:

* Gene Expression Analysis
* Multi-Omics Integration
* Dimensionality Reduction (PCA & t-SNE)
* Clustering (K-Means & Hierarchical)
* Survival Analysis
* Biological Interpretation

🎯 The primary goal is to identify hidden patient subtypes and explore biologically meaningful molecular patterns in breast cancer datasets.

---

# 🚀 Features

## 📥 Data Handling

* Upload `.csv`, `.tsv`, or `.gz` datasets
* Automatic format detection
* Safe preview for large datasets
* TCGA-compatible dataset handling
* Dataset validation and reporting

---

## 🧬 Multi-Omics Integration

* Integration of multiple TCGA omics datasets
* TCGA barcode harmonization
* Shared-patient matching across datasets
* Automatic duplicate sample removal
* Multi-omics feature merging

---

## 🔍 Validation

* Missing-value detection
* Duplicate patient detection
* Automatic orientation detection
* Data integrity warnings
* Arrow-safe dataframe handling

---

## 🧹 Preprocessing

* Removal of features with >20% missing values
* Mean-based imputation
* Strict numeric conversion
* Removal of non-numeric metadata columns
* Automatic dataset transposition
* Standard normalization (`StandardScaler`)
* Large-dataset optimization

---

## ✂️ Feature Selection

* Variance-based feature selection
* Adjustable feature count (500–5000)
* Default selection: 2000 features

---

## 🤖 Modeling

* K-Means clustering
* Hierarchical clustering
* Automatic model comparison
* Best-model selection using silhouette score
* PCA dimensionality reduction

---

## ❤️ Survival Analysis

* Kaplan-Meier survival analysis
* Cluster-wise survival comparison
* Clinical phenotype integration
* TCGA clinical data support

---

## 📊 Evaluation

* Silhouette score evaluation
* Side-by-side clustering comparison
* Cluster quality assessment

---

## 🎨 Visualization

* PCA cluster visualization
* t-SNE non-linear visualization
* Kaplan-Meier survival curves
* Color-coded cluster plots
* Cluster comparison visualizations

---

## 🧬 Biological Interpretation

* Cluster summaries
* Molecular subtype interpretation
* Clinical significance discussion
* Model feedback based on clustering quality

---

# 🏗️ Project Structure

```text
bioinformatics_dashboard/
│
├── app/
│   └── app.py
│
├── core/
│   ├── data_loader.py
│   ├── validator.py
│   ├── preprocessor.py
│   ├── feature_selector.py
│   ├── models.py
│   ├── evaluator.py
│   ├── visualizer.py
│   └── interpreter.py
│
├── pipelines/
│   ├── clustering.py  
│
├── config/
    └── config.yaml
```
---

# 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Lifelines
- TCGA / UCSC Xena

---

# ⚙️ Installation

## 1️⃣ Clone Repository

git clone https://github.com/pravindag/bioinformatics-dashboard.git
cd bioinformatics-dashboard

## 2️⃣ Install Dependencies

pip install -r requirements.txt

## ▶️ Running the Application

streamlit run app/app.py

---

# 📂 Supported Input Datasets

## Primary Dataset

Gene expression dataset:

* TCGA RNA-Seq
* STAR-FPKM
* .csv, .tsv, .gz

## Optional Multi-Omics Dataset

Examples:

* DNA Methylation
* miRNA expression
* Copy-number variation

## Optional Clinical Dataset

TCGA clinical phenotype data:

* survival information
* patient metadata
* vital status

---

# 🔬 Workflow

1. Upload TCGA gene expression dataset
2. Optionally upload second omics dataset
3. Match TCGA patient barcodes
4. Validate and preprocess data
5. Normalize and select features
6. Run clustering models
7. Compare clustering performance
8. Perform survival analysis
9. Visualize PCA / t-SNE projections
10. Interpret biological significance
11. Download clustering results

---

# 📊 Example Outputs

* Dataset summary
* Validation warnings
* PCA visualization
* t-SNE visualization
* Kaplan-Meier survival curves
* Clustering comparison scores
* Biological interpretation
* Downloadable cluster assignments

---

# 🧠 Methodology

## Preprocessing

* Missing-value filtering
* Numeric enforcement
* Mean imputation
* Standard normalization

## Multi-Omics Integration

* TCGA barcode standardization
* Shared patient matching
* Cross-omics feature integration

## Dimensionality Reduction

* Principal Component Analysis (PCA)
* t-distributed Stochastic Neighbor Embedding (t-SNE)

## Clustering

* K-Means clustering
* Hierarchical Agglomerative Clustering

## Evaluation

* Silhouette score

## Survival Analysis

* Kaplan-Meier survival estimation
* Cluster-wise clinical comparison

---

# 🧬 Biological Interpretation

The discovered clusters may represent biologically distinct molecular subtypes of breast cancer, such as:

* Luminal A
* Luminal B
* HER2-enriched
* Basal-like

Interpretation guidelines:

* High silhouette score → strong cluster separation
* Low silhouette score → overlapping molecular patterns

Further biological validation can be performed using:

* PAM50 subtype labels
* Clinical survival data
* External validation cohorts

---

# ⚡ Performance Optimizations

To support large TCGA datasets, the system includes:

* Cached preprocessing
* Cached clustering pipelines
* Session-state optimization
* Memory-efficient float32 conversion
* Large-dataset preview sampling
* Adaptive t-SNE configuration

---

# 🎯 Learning Objectives

This project demonstrates:

* Handling high-dimensional biological datasets
* Building modular bioinformatics workflows
* Applying machine learning to genomics
* Multi-omics integration techniques
* Survival analysis in cancer research
* Interactive dashboard development
* Biological interpretation of clustering results

---

# 🏆 Key Highlights

* TCGA multi-omics integration
* TCGA barcode harmonization
* KMeans vs Hierarchical clustering comparison
* PCA and t-SNE visualization
* Kaplan-Meier survival analysis
* Robust preprocessing pipeline
* Large dataset optimization
* Cached Streamlit architecture
* Downloadable clustering outputs
* Biological interpretation support

---

# 🚀 Future Improvements

* Deep learning models (Autoencoders / VAEs)
* SHAP-based feature interpretation
* UMAP visualization
* Pathway enrichment analysis
* Cloud deployment for large-scale datasets
* External validation datasets

---

# 👨‍💻 Author

M D P U Gunathilake
MSc in Data Science & Artificial Intelligence

DSA 611 – Bioinformatics
Multi-Omics Breast Cancer Analysis Project



