import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.preprocess import preprocess_data


def _positive_class_shap_values(shap_values):
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])

    values = np.asarray(shap_values)
    if values.ndim == 3:
        if values.shape[-1] == 2:
            return values[:, :, 1]
        if values.shape[0] == 2:
            return values[1]
    return values


def generate_shap_plots(model, X_test, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_values = _positive_class_shap_values(shap_values)

    summary_path = output_dir / "shap_summary.png"
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(summary_path, bbox_inches="tight", dpi=150)
    plt.close()

    predictions = model.predict(X_test)
    fail_positions = np.where(predictions == 1)[0]
    sample_position = int(fail_positions[0]) if len(fail_positions) else 0
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if np.asarray(expected_value).ndim > 0 else float(expected_value)

    explanation = shap.Explanation(
        values=shap_values[sample_position],
        base_values=expected_value,
        data=X_test.iloc[sample_position].values,
        feature_names=list(X_test.columns),
    )
    waterfall_path = output_dir / "shap_waterfall.png"
    shap.waterfall_plot(explanation, show=False)
    plt.savefig(waterfall_path, bbox_inches="tight", dpi=150)
    plt.close()

    mean_abs_values = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(X_test.columns, mean_abs_values))
    top_features = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:5]

    print("Top 5 most important features by mean(|SHAP value|):")
    for feature, value in top_features:
        print(f"{feature}: {value:.6f}")

    return importance


def main():
    models_dir = BASE_DIR / "models"
    data_dir = BASE_DIR / "data"
    reports_dir = BASE_DIR / "reports"

    model = joblib.load(models_dir / "best_model.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    feature_names = joblib.load(models_dir / "feature_names.pkl")

    df = pd.read_csv(data_dir / "builds.csv")
    X, y, _ = preprocess_data(df, fit_scaler=False, scaler=scaler)
    X = X.reindex(columns=feature_names, fill_value=0)

    _, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    generate_shap_plots(model, X_test, reports_dir)
    print("SHAP plots saved to reports/")


if __name__ == "__main__":
    main()
