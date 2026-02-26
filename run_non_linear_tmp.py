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
from src.models import make_gradient_boosting, make_random_forest
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

    non_linear_factories = {
        "random_forest": lambda: make_random_forest(seed=42, n_estimators=60),
        "gradient_boosting": lambda: make_gradient_boosting(seed=42),
    }
    non_linear_experiments = [
        ("raw_features", X_train, X_test),
        ("standardized_features", X_train_scaled, X_test_scaled),
    ]

    non_linear_rows = []
    for setting, Xtr, Xte in non_linear_experiments:
        for model_name, factory in non_linear_factories.items():
            model_nl = factory()
            model_nl.fit(Xtr, y_train)
            y_pred_nl = model_nl.predict(Xte)
            m_nl = compute_metrics(
                y_true=y_test,
                y_pred=y_pred_nl,
                labels=list(range(len(label_encoder.classes_))),
                target_names=list(label_encoder.classes_),
            )
            non_linear_rows.append(
                {
                    "model": model_name,
                    "feature_setting": setting,
                    "accuracy": m_nl["accuracy"],
                    "macro_f1": m_nl["macro_f1"],
                    "n_features": Xtr.shape[1],
                    "seed": 42,
                }
            )

    non_linear_df = pd.DataFrame(non_linear_rows).sort_values(
        ["model", "feature_setting"]
    ).reset_index(drop=True)
    out_csv = tables_dir / "non_linear_baseline_comparison.csv"
    non_linear_df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=non_linear_df, x="model", y="macro_f1", hue="feature_setting", ax=ax)
    ax.set_title("Non-linear baseline comparison (macro F1)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro F1")
    out_fig = fig_dir / "03_non_linear_baseline_comparison.png"
    save_figure(fig, str(out_fig))
    plt.close()

    print(non_linear_df)
    print(out_csv)
    print(out_fig)


if __name__ == "__main__":
    main()
