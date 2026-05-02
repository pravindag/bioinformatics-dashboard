def validate_dataset(df):
    report = {}

    report["rows"] = df.shape[0]
    report["columns"] = df.shape[1]

    # Missing values
    report["missing_ratio"] = df.isnull().mean().mean()

    # Duplicate columns (sample IDs)
    report["duplicate_columns"] = df.columns.duplicated().sum()

    # Heuristic: genes usually > samples
    if df.shape[0] > df.shape[1]:
        report["orientation"] = "genes_as_rows"
    else:
        report["orientation"] = "samples_as_rows"

    return report

def generate_warnings(report):
    warnings = []

    if report["missing_ratio"] > 0.2:
        warnings.append("⚠ High missing values (>20%)")

    if report["orientation"] == "genes_as_rows":
        warnings.append("⚠ Data likely needs transposition")

    if report["duplicate_columns"] > 0:
        warnings.append("⚠ Duplicate sample IDs detected")

    return warnings