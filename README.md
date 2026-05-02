# 🧬 Multi-Omics Breast Cancer Analyzer

An interactive bioinformatics dashboard for analyzing high-dimensional cancer datasets using machine learning and advanced visualization techniques.

This system enables users to upload gene expression data, perform preprocessing, compare multiple clustering models, visualize results, and extract biological insights through a user-friendly interface.

---

## 📌 Project Overview

Modern bioinformatics integrates multiple biological data types (multi-omics) to better understand diseases like cancer.

This project focuses on:

* Gene Expression Analysis
* Dimensionality Reduction (PCA & t-SNE)
* Clustering (K-Means & Hierarchical)
* Model Comparison
* Biological Interpretation

🎯 The goal is to identify hidden patient subtypes and explore meaningful biological patterns.

---

## 🚀 Features

### 📥 Data Handling

* Upload `.csv`, `.tsv`, or `.gz` datasets
* Automatic format detection
* Safe dataset preview (large-data friendly)
* Dataset summary and validation

### 🔍 Validation

* Detect missing values
* Identify incorrect data orientation
* Generate warnings for potential issues
* Arrow-safe reporting (no serialization errors)

### 🧹 Preprocessing

* Remove features with >20% missing values
* Mean imputation for missing data
* Automatic data transposition
* Strict numeric conversion
* Standard normalization

### ✂️ Feature Selection

* Variance-based feature selection
* Adjustable number of features (500–5000)
* Default: 2000

### 🤖 Modeling

* K-Means clustering
* Hierarchical clustering
* PCA for dimensionality reduction
* Automatic model comparison and best model selection

### 📊 Evaluation

* Silhouette score
* Side-by-side model comparison

### 🎨 Visualization

* PCA cluster visualization
* t-SNE visualization (non-linear projection)
* Color-coded clusters

### 🧬 Interpretation

* Cluster summaries
* Biological insight explanation
* Model feedback based on score

---

## 🏗️ Project Structure

```
bioinformatics_dashboard/
│
├── app/
│   └── app.py                 # Streamlit dashboard
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
│   ├── supervised.py
│   └── hybrid.py
│
├── config/
│   └── config.yaml
│
└── notebook/
    └── experiments.ipynb
```

---

## ⚙️ Installation

### 1. Clone or Download Project

```
git clone [Bioinformatics Dashboard](https://github.com/pravindag/bioinformatics-dashboard.git) 
cd bioinformatics-dashboard
```

---

### 2. Install Dependencies

```
pip install streamlit pandas numpy scikit-learn matplotlib
```

---

## ▶️ Running the Application

From the project root directory:

```
streamlit run app/app.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 📂 Input Data Requirements

* Format: `.csv`, `.tsv`, or `.gz`
* Structure:

  * Rows = genes
  * Columns = samples (patients)

⚠️ The system will automatically transpose data if needed.

---

## 🔬 Workflow

1. Upload dataset
2. Validate dataset
3. Preprocess data
4. Select features
5. Run clustering models
6. Compare models
7. Visualize (PCA + t-SNE)
8. Interpret results
9. Download results
---

## 📊 Example Output

* Dataset summary
* Data warnings
* PCA plot
* t-SNE visualization
* Model comparison scores
* Cluster interpretation
* Biological insights
* Downloadable results

---

## 🧠 Methodology

### Preprocessing

* Missing value handling
* Feature filtering
* Numeric enforcement
* Standard scaling

### Dimensionality Reduction

* Principal Component Analysis (PCA)
* t-distributed Stochastic Neighbor Embedding (t-SNE)

### Clustering

* K-Means
* Hierarchical clustering

### Evaluation

* Silhouette score

---

## 🧬 Biological Interpretation

Clusters may represent distinct molecular subtypes of breast cancer.

* High silhouette score → strong separation
* Low score → overlapping patterns

Further validation can be performed using:

* PAM50 subtype labels
* Clinical survival data

---

## 🎯 Learning Objectives

This project demonstrates:

* Handling high-dimensional biological data
* Applying machine learning techniques
* Building modular pipelines
* Developing interactive dashboards
* Interpreting results biologically

---

## 🏆 Key Highlights
* Model comparison (KMeans vs Hierarchical)
* Advanced visualization (t-SNE + PCA)
* Robust error handling
* Downloadable outputs
* Clean UI/UX
* Scalable architecture

---

## 🚀 Future Improvements

* Integrate clinical labels (PAM50)
* Add supervised models
* Feature importance (SHAP)
* UMAP visualization
* Multi-omics integration

---

## 👨‍💻 Author

**M D P U Gunathilake**  
MSc in Data Science & Artificial Intelligence  

Bioinformatics Project – Multi-Omics Breast Cancer Analysis  
DSA 611 – Bioinformatics

---

