import pandas as pd
import numpy as np

def load_data(file):
    """
    Load and clean TCGA-style omics datasets safely.

    Supports:
    - .csv
    - .tsv
    - .tsv.gz
    """

    try:

        # =========================
        # LOAD FILE
        # =========================
        if file.name.endswith(".gz"):

            df = pd.read_csv(
                file,
                sep="\t",
                compression="gzip"
            )

        elif file.name.endswith(".tsv"):

            df = pd.read_csv(
                file,
                sep="\t"
            )

        else:

            df = pd.read_csv(file)

        # =========================
        # HANDLE GENE IDENTIFIERS
        # =========================
        possible_gene_cols = [
            "Gene Symbol",
            "gene",
            "symbol",
            "Hybridization REF",
            "Composite Element REF"
        ]

        for col in possible_gene_cols:

            if col in df.columns:
                df.set_index(col, inplace=True)
                break

        # =========================
        # REMOVE DUPLICATES
        # =========================
        df = df.loc[~df.index.duplicated(keep="first")]

        # =========================
        # CLEAN VALUES
        # =========================
        df.replace(
            ["NA", "NaN", "null", "--", "?", ""],
            np.nan,
            inplace=True
        )

        for col in df.columns:

            if df[col].dtype == "object":

                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                )

        # =========================
        # KEEP ONLY TCGA COLUMNS
        # =========================
        tcga_cols = [
            c for c in df.columns
            if "TCGA" in str(c)
        ]

        if len(tcga_cols) > 0:
            df = df[tcga_cols]

        # =========================
        # STANDARDIZE TCGA BARCODES
        # =========================
        df.columns = [
            str(c)[:12]
            for c in df.columns
        ]

        # =========================
        # REMOVE DUPLICATE PATIENTS
        # =========================
        df = df.loc[:, ~df.columns.duplicated()]

        # =========================
        # FORCE NUMERIC CONVERSION
        # =========================
        df = df.apply(
            lambda col: pd.to_numeric(
                col,
                errors='coerce'
            )
        )

        # =========================
        # REMOVE EMPTY COLUMNS
        # =========================
        df = df.dropna(axis=1, how='all')

        # =========================
        # REMOVE EMPTY ROWS
        # =========================
        df = df.dropna(axis=0, how='all')

        # =========================
        # FINAL MEMORY OPTIMIZATION
        # =========================
        df = df.astype(np.float32)

        return df

    except Exception as e:

        raise ValueError(
            f"Error loading file: {e}"
        )