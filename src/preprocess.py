from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def engineer_features(df):
    data = df.copy()
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_late_night"] = ((data["hour_of_commit"] > 22) | (data["hour_of_commit"] < 6)).astype(int)
    data["lines_change_ratio"] = data["lines_added"] / (data["lines_deleted"] + 1)
    data["large_diff"] = (data["files_changed"] > 10).astype(int)
    data["is_short_message"] = (data["commit_msg_length"] < 15).astype(int)
    data["branch_risk"] = data["branch"].map({"main": 0, "feature": 1, "hotfix": 2, "release": 1}).fillna(1).astype(int)
    data = pd.get_dummies(data, columns=["language"], prefix="lang")
    return data.drop(columns=["build_id", "branch", "language"], errors="ignore")


def preprocess_data(df, fit_scaler=False, scaler=None):
    engineered = engineer_features(df)

    y = None
    if "result" in engineered.columns:
        y = engineered["result"].astype(int)
        X = engineered.drop(columns=["result"])
    else:
        X = engineered

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled_array = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("A fitted scaler is required when fit_scaler=False.")
        if hasattr(scaler, "feature_names_in_"):
            X = X.reindex(columns=list(scaler.feature_names_in_), fill_value=0)
        X_scaled_array = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)
    return X_scaled, y, scaler


def load_and_preprocess(csv_path, fit_scaler=True, scaler=None):
    df = pd.read_csv(csv_path)
    return preprocess_data(df, fit_scaler=fit_scaler, scaler=scaler)


def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y, scaler = load_and_preprocess(data_dir / "builds.csv", fit_scaler=True)

    X.to_csv(data_dir / "features.csv", index=False)
    if y is not None:
        y.to_csv(data_dir / "target.csv", index=False)
    joblib.dump(scaler, models_dir / "scaler.pkl")

    print("Feature names:")
    for feature in X.columns:
        print(f"- {feature}")
    print(f"Saved features to {data_dir / 'features.csv'}")
    print(f"Saved target to {data_dir / 'target.csv'}")
    print(f"Saved scaler to {models_dir / 'scaler.pkl'}")


if __name__ == "__main__":
    main()
