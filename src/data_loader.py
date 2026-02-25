from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    sample_id: pd.Series


def load_dataset(data_path: str, labels_path: str) -> Dataset:
    data = pd.read_csv(data_path)
    labels = pd.read_csv(labels_path)

    id_col = "Unnamed: 0"
    target_col = "Class"

    if id_col not in data.columns or id_col not in labels.columns:
        raise ValueError("Missing ID column 'Unnamed: 0'.")
    if target_col not in labels.columns:
        raise ValueError("Missing target column 'Class' in labels file.")

    if data[id_col].duplicated().any():
        raise ValueError("Duplicate sample IDs found in data file.")
    if labels[id_col].duplicated().any():
        raise ValueError("Duplicate sample IDs found in labels file.")

    data[id_col] = data[id_col].astype(str)
    labels[id_col] = labels[id_col].astype(str)
    labels[target_col] = labels[target_col].astype(str)

    df = data.merge(labels[[id_col, target_col]], on=id_col, how="inner", validate="one_to_one")
    if df.shape[0] != data.shape[0] or df.shape[0] != labels.shape[0]:
        raise ValueError("Merge mismatch: some samples are missing after merge.")

    sample_id = df[id_col]
    y = df[target_col]
    X = df.drop(columns=[id_col, target_col])

    # Gene-expression features should all be numeric for downstream modeling.
    non_numeric_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric_cols[:5]}")

    if X.isna().any().any():
        raise ValueError("NaNs found in features. Handle explicitly.")
    if y.isna().any():
        raise ValueError("NaNs found in labels. Handle explicitly.")

    return Dataset(X=X, y=y, sample_id=sample_id)
