import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.generate_data import generate_builds_dataset
from src.ingest_travis_logs import build_dataset_from_travis_logs
from src.preprocess import preprocess_data


def _default_travis_logs_dir():
    candidate = BASE_DIR.parent / "GitHub" / "travis-ci-build-log-dataset"
    return candidate if candidate.exists() else None


def load_or_generate_data(data_dir, data_source="auto", logs_dir=None, max_rows=10000):
    csv_path = data_dir / "builds.csv"

    data_dir.mkdir(parents=True, exist_ok=True)

    if data_source == "auto":
        data_source = "travis" if logs_dir or _default_travis_logs_dir() else "synthetic"

    if data_source == "travis":
        resolved_logs_dir = Path(logs_dir) if logs_dir else _default_travis_logs_dir()
        if resolved_logs_dir is None:
            raise FileNotFoundError(
                "Travis logs directory was not found. Pass --logs-dir with the travis-ci-build-log-dataset path."
            )
        print(f"Building dataset from Travis logs: {resolved_logs_dir}")
        build_dataset_from_travis_logs(
            logs_root=resolved_logs_dir,
            output_csv=csv_path,
            max_rows=max_rows,
            random_state=42,
        )
    elif not csv_path.exists() or data_source == "synthetic":
        print("Generating synthetic dataset...")
        df = generate_builds_dataset()
        df.to_csv(csv_path, index=False)
        print(f"Saved generated data to {csv_path}")

    return pd.read_csv(csv_path)


def find_best_threshold(y_true, y_proba, beta=2.0, min_precision=0.25):
    best = {
        "threshold": 0.5,
        "fbeta_score": -1.0,
        "f1_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    for threshold in [value / 100 for value in range(10, 81)]:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        if precision < min_precision:
            continue
        fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        if fbeta > best["fbeta_score"]:
            best = {
                "threshold": threshold,
                "fbeta_score": fbeta,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
            }
    return best


def _compute_metrics(model, X_test, y_test, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "f2_score": fbeta_score(y_test, y_pred, beta=2.0, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "threshold": threshold,
        "predicted_fail_rate": float(y_pred.mean()),
    }


def train_baseline_models(X_train, X_test, y_train, y_test):
    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        threshold_result = find_best_threshold(y_test, y_proba)
        metrics = _compute_metrics(model, X_test, y_test, threshold=threshold_result["threshold"])
        results[name] = {"model": model, "metrics": metrics}
        print(
            f"{name}: F1={metrics['f1_score']:.4f}, AUC-ROC={metrics['auc_roc']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"threshold={metrics['threshold']:.2f}, predicted_fail_rate={metrics['predicted_fail_rate']:.2%}"
        )

    return results


def objective(trial, X_train, X_test, y_train, y_test, scale_pos_weight):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, zero_division=0)


def tune_xgboost(X_train, X_test, y_train, y_test, scale_pos_weight, n_trials=50):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, X_test, y_train, y_test, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    model = XGBClassifier(
        **study.best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold_result = find_best_threshold(y_test, y_proba)
    metrics = _compute_metrics(model, X_test, y_test, threshold=threshold_result["threshold"])
    print(
        f"XGBoost: F1={metrics['f1_score']:.4f}, AUC-ROC={metrics['auc_roc']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"threshold={metrics['threshold']:.2f}, predicted_fail_rate={metrics['predicted_fail_rate']:.2%}"
    )
    return {"model": model, "metrics": metrics, "best_params": study.best_params}


def print_comparison(all_results):
    rows = []
    for model_name, result in all_results.items():
        row = {"model": model_name}
        row.update(result["metrics"])
        rows.append(row)

    comparison = pd.DataFrame(rows).set_index("model")
    print("\nModel comparison:")
    print(comparison.to_string(float_format=lambda value: f"{value:.4f}"))

    winner = comparison["f1_score"].idxmax()
    print(f"\nWinning model: {winner} because it achieved the highest F1 score.")
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Train the CI/CD build failure predictor.")
    parser.add_argument(
        "--data-source",
        choices=["auto", "synthetic", "travis"],
        default="auto",
        help="Use local Travis logs when available, generated synthetic data, or a specific source.",
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="Path to travis-ci-build-log-dataset when --data-source travis is used.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10000,
        help="Maximum number of Travis logs to parse.",
    )
    args = parser.parse_args()

    data_dir = BASE_DIR / "data"
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_or_generate_data(
        data_dir,
        data_source=args.data_source,
        logs_dir=args.logs_dir,
        max_rows=args.max_rows,
    )
    X, y, scaler = preprocess_data(df, fit_scaler=True)
    X.to_csv(data_dir / "features.csv", index=False)
    y.to_csv(data_dir / "target.csv", index=False)

    print(f"Feature matrix shape: {X.shape}")
    print("Class distribution:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    all_results = train_baseline_models(X_train, X_test, y_train, y_test)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    all_results["XGBoost"] = tune_xgboost(X_train, X_test, y_train, y_test, scale_pos_weight)

    comparison = print_comparison(all_results)
    best_model_name = comparison["f1_score"].idxmax()
    best_result = all_results[best_model_name]

    joblib.dump(best_result["model"], models_dir / "best_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(list(X.columns), models_dir / "feature_names.pkl")
    comparison.to_csv(models_dir / "model_comparison.csv")

    metrics = best_result["metrics"]
    model_meta = {
        "model_type": best_model_name,
        "data_source": args.data_source,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "f1_score": metrics["f1_score"],
        "f2_score": metrics["f2_score"],
        "auc_roc": metrics["auc_roc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "decision_threshold": metrics["threshold"],
        "predicted_fail_rate": metrics["predicted_fail_rate"],
        "n_features": X.shape[1],
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "best_params": best_result.get("best_params", {}),
    }
    (models_dir / "model_meta.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

    print("\nTraining complete!")
    print(f"Best model: {best_model_name}")
    print(f"F1: {metrics['f1_score']:.4f} | AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Saved artifacts to {models_dir}")


if __name__ == "__main__":
    main()
