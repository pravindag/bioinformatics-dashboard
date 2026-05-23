from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# =========================
# REMOVE HIGH MISSING
# =========================
def remove_high_missing(df, threshold=0.1):

    return df.loc[
        df.isnull().mean(axis=1) < threshold
    ]

# =========================
# KEEP ONLY NUMERIC
# =========================
def remove_non_numeric(df):

    return df.select_dtypes(
        include=[np.number]
    )

# =========================
# IMPUTATION
# =========================
def impute_data(df):
    """
    Clean and impute genomic data safely.
    Optimized for large TCGA datasets.
    """

    df = df.replace(
        ["NA", "NaN", "null", "--", "?", ""],
        np.nan
    )

    df = df.apply(
        lambda col: pd.to_numeric(
            col,
            errors='coerce'
        )
    )

    df = df.dropna(axis=1, how='all')

    df = df.dropna(axis=0, how='all')

    df = df.select_dtypes(
        include=[np.number]
    )

    if df.shape[1] > 5000:

        variances = df.var()

        top_features = variances.sort_values(
            ascending=False
        ).head(5000).index

        df = df[top_features]

    if df.shape[0] > 2000:

        df = df.sample(
            2000,
            random_state=42
        )

    # =========================
    # MEAN IMPUTATION
    # =========================
    df = df.fillna(
        df.mean(numeric_only=True)
    )

    # =========================
    # MEMORY OPTIMIZATION
    # =========================
    df = df.astype(np.float32)

    return df

# =========================
# TRANSPOSE
# =========================
def transpose_if_needed(df, orientation):

    if orientation == "genes_as_rows":
        return df.T

    return df

# =========================
# NORMALIZATION
# =========================
def normalize_data(df):

    scaler = StandardScaler()

    return scaler.fit_transform(df)

# =========================
# TCGA SAMPLE MATCHING
# =========================
def match_samples(df1, df2):
    """
    Match patients across multi-omics datasets
    using TCGA barcodes.
    """

    df1.columns = [
        str(c)[:12]
        for c in df1.columns
    ]

    df2.columns = [
        str(c)[:12]
        for c in df2.columns
    ]

    common_samples = (
        df1.columns
        .intersection(df2.columns)
    )

    if len(common_samples) == 0:

        raise ValueError(
            "No matching TCGA samples found "
            "between datasets."
        )

    df1_matched = df1[common_samples]
    df2_matched = df2[common_samples]

    df1_matched = df1_matched.loc[
        :,
        ~df1_matched.columns.duplicated()
    ]

    df2_matched = df2_matched.loc[
        :,
        ~df2_matched.columns.duplicated()
    ]

    return df1_matched, df2_matched