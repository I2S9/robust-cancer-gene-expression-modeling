import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create reproducible, stratified train/test splits."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encode string labels into integer class IDs."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


def get_label_mapping(le: LabelEncoder) -> dict[int, str]:
    """Return class-id to class-name mapping from a fitted LabelEncoder."""
    return {int(i): str(label) for i, label in enumerate(le.classes_)}


def save_label_mapping(le: LabelEncoder, output_path: str) -> None:
    """Persist label mapping as JSON for reproducible decoding."""
    mapping = get_label_mapping(le)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=True)


def scale_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit scaler on train features only, then transform train/test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
