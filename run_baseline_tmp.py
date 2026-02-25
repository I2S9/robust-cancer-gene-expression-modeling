from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

project_root = Path.cwd().resolve()
if not (project_root / "src").exists():
    project_root = project_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import load_dataset
from src.evaluation import compute_metrics
from src.models import make_logreg_l2
from src.preprocessing import encode_labels, make_splits, scale_train_test
from src.visualization import save_figure


def main() -> None:
    sns.set_theme(style="whitegrid")
    fig_dir = project_root / "results" / "figures"
    tables_dir = project_root / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        data_path=str(project_root / "data" / "raw" / "data.csv"),
        labels_path=str(project_root / "data" / "raw" / "labels.csv"),
    )

    X = ds.X
    y_enc, label_encoder = encode_labels(ds.y)

    X_train, X_test, y_train, y_test = make_splits(X, y_enc, test_size=0.2, seed=42)
    X_train_scaled, X_test_scaled, _ = scale_train_test(X_train, X_test)

    model = make_logreg_l2(seed=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    metrics = compute_metrics(
        y_true=y_test,
        y_pred=y_pred,
        labels=list(range(len(label_encoder.classes_))),
        target_names=list(label_encoder.classes_),
    )

    summary_df = pd.DataFrame(
        [
            {
                "model": "logreg_l2",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "n_train": X_train_scaled.shape[0],
                "n_test": X_test_scaled.shape[0],
                "n_features": X_train_scaled.shape[1],
                "seed": 42,
                "test_size": 0.2,
            }
        ]
    )
    summary_df.to_csv(tables_dir / "03_logreg_l2_metrics.csv", index=False)

    per_class_df = pd.DataFrame(metrics["per_class_report"]).T
    per_class_df.to_csv(tables_dir / "03_logreg_l2_per_class_report.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Logistic Regression (L2) - Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    save_figure(fig, str(fig_dir / "03_logreg_l2_confusion_matrix.png"))
    plt.close()

    print(
        "OK",
        round(metrics["accuracy"], 4),
        round(metrics["macro_f1"], 4),
        str(tables_dir / "03_logreg_l2_metrics.csv"),
    )


if __name__ == "__main__":
    main()
