from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def remove_high_missing(df, threshold=0.2):
    return df.loc[df.isnull().mean(axis=1) < threshold]

def remove_non_numeric(df):
    return df.select_dtypes(include=['number'])

def impute_data(df):
    """
    STRICT cleaning:
    - Force everything to numeric
    - Drop anything that cannot be converted
    - Then safely impute
    """

    # Step 1: Replace common non-numeric placeholders
    df = df.replace(["NA", "NaN", "null", "--", "?"], np.nan)

    # Step 2: Convert EVERYTHING to numeric (force)
    df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

    # Step 3: Drop columns that are NOT usable (too many NaNs)
    df = df.dropna(axis=1, how='all')

    # Step 4: CRITICAL — ensure ONLY numeric columns remain
    df = df.select_dtypes(include=[np.number])

    # Step 5: Now safe to compute mean
    df = df.fillna(df.mean())

    return df

def transpose_if_needed(df, orientation):
    if orientation == "genes_as_rows":
        return df.T
    return df

def normalize_data(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)

def match_samples(df1, df2):
    """
    Match samples using TCGA barcodes (intersection)
    """
    common_samples = df1.columns.intersection(df2.columns)

    df1_matched = df1[common_samples]
    df2_matched = df2[common_samples]

    return df1_matched, df2_matched