from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
    target_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute a standard evaluation bundle for any classifier."""
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    aggregate_keys = {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}
    per_class_report = {k: v for k, v in report_dict.items() if k not in aggregate_keys}

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": conf_mat,
        "classification_report": report_dict,
        "per_class_report": per_class_report,
        "classification_report_text": report_text,
    }
