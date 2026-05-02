import pandas as pd
import numpy as np

def load_data(file):
    """
    Load and clean dataset safely for bioinformatics pipeline.
    Supports: .csv, .tsv, .tsv.gz
    """

    try:
        # =========================
        # LOAD FILE
        # =========================
        if file.name.endswith(".gz"):
            df = pd.read_csv(file, sep="\t", compression="gzip")
        elif file.name.endswith(".tsv"):
            df = pd.read_csv(file, sep="\t")
        else:
            df = pd.read_csv(file)

        # =========================
        # HANDLE GENE IDENTIFIER
        # =========================
        if "Gene Symbol" in df.columns:
            df.set_index("Gene Symbol", inplace=True)

        # =========================
        # CLEAN VALUES
        # =========================

        # Replace common placeholders
        df.replace(["NA", "NaN", "null", "--", "?"], np.nan, inplace=True)

        # Strip whitespace safely (FIX for applymap warning)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.strip()

        # =========================
        # FORCE NUMERIC CONVERSION (CRITICAL FIX)
        # =========================
        df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

        # =========================
        # REMOVE BAD COLUMNS
        # =========================
        df = df.dropna(axis=1, how='all')

        # =========================
        # FINAL TYPE SAFETY (FIX Arrow error)
        # =========================
        df = df.astype(float)

        return df

    except Exception as e:
        raise ValueError(f"Error loading file: {e}")