import json
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schema import BuildCommit, LogPredictionRequest, PredictionResult
from src.preprocess import preprocess_data

BASE_DIR = Path(__file__).parent.parent
app_state = {}


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = BASE_DIR / "models"
    model_path = models_dir / "best_model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    feature_names_path = models_dir / "feature_names.pkl"
    meta_path = models_dir / "model_meta.json"

    if not model_path.exists() or not scaler_path.exists() or not feature_names_path.exists():
        raise RuntimeError("Models not found. Run: python src/train.py")

    app_state["model"] = joblib.load(model_path)
    app_state["scaler"] = joblib.load(scaler_path)
    app_state["feature_names"] = joblib.load(feature_names_path)
    app_state["model_meta"] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    app_state["explainer"] = shap.TreeExplainer(app_state["model"])
    yield
    app_state.clear()


app = FastAPI(title="CI/CD Build Failure Predictor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in app_state}


@app.get("/model-info")
def model_info():
    return app_state.get("model_meta", {})


def _predict_commit(commit):
    record = commit.model_dump() if isinstance(commit, BuildCommit) else commit
    df = pd.DataFrame([record])
    X_input, _, _ = preprocess_data(df, fit_scaler=False, scaler=app_state["scaler"])
    X_input = X_input.reindex(columns=app_state["feature_names"], fill_value=0)

    model = app_state["model"]
    probabilities = model.predict_proba(X_input)[0]
    fail_probability = float(probabilities[1])
    threshold = float(app_state.get("model_meta", {}).get("decision_threshold", 0.5))
    prediction_value = int(fail_probability >= threshold)
    prediction = "FAIL" if prediction_value == 1 else "PASS"
    confidence = fail_probability if prediction == "FAIL" else float(probabilities[0])

    if prediction == "FAIL":
        if fail_probability < threshold:
            risk_level = "low"
        elif fail_probability < min(0.7, threshold + 0.18):
            risk_level = "medium"
        else:
            risk_level = "high"
    else:
        risk_level = "low"

    shap_values = app_state["explainer"].shap_values(X_input)
    shap_row = _positive_class_shap_values(shap_values)[0]
    top_indices = np.argsort(np.abs(shap_row))[::-1][:3]
    top_risk_factors = [
        {"feature": app_state["feature_names"][idx], "impact": round(float(shap_row[idx]), 4)}
        for idx in top_indices
    ]

    return PredictionResult(
        prediction=prediction,
        confidence=round(confidence, 4),
        risk_level=risk_level,
        top_risk_factors=top_risk_factors,
    )


def _extract_int(patterns, text, default, minimum=None, maximum=None):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = int(match.group(1).replace(",", ""))
            if minimum is not None:
                value = max(minimum, value)
            if maximum is not None:
                value = min(maximum, value)
            return value
    return default


def _extract_branch(text):
    match = re.search(r"(?:branch|TRAVIS_BRANCH|GITHUB_REF_NAME)[:=]\s*([A-Za-z0-9_./-]+)", text, re.IGNORECASE)
    branch = match.group(1).lower() if match else ""
    if "hotfix" in branch:
        return "hotfix"
    if "release" in branch:
        return "release"
    if branch in {"main", "master"}:
        return "main"
    return "feature"


def _extract_language(text):
    languages = ["TypeScript", "JavaScript", "Python", "Java", "Ruby", "C++", "Go"]
    lowered = text.lower()
    for language in languages:
        if re.search(rf"\b{re.escape(language.lower())}\b", lowered):
            return language
    if "node" in lowered or "npm" in lowered or "yarn" in lowered:
        return "JavaScript"
    if "pytest" in lowered or "pip install" in lowered:
        return "Python"
    if "maven" in lowered or "gradle" in lowered:
        return "Java"
    return "Python"


def _extract_timestamp_parts(text):
    patterns = [
        r"commit_time[:=]\s*([0-9T:Z+\-\s]+)",
        r"timestamp[:=]\s*([0-9T:Z+\-\s]+)",
        r"started_at[:=]\s*([0-9T:Z+\-\s]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
            return parsed.hour, parsed.weekday()
        except ValueError:
            continue
    return 14, 2


def parse_log_to_commit(log_text):
    text = log_text[:50000]
    line_count = max(1, len(text.splitlines()))
    error_hits = len(re.findall(r"\b(error|failed|failure|exception|traceback)\b", text, re.IGNORECASE))
    warning_hits = len(re.findall(r"\b(warn|warning)\b", text, re.IGNORECASE))
    exit_match = re.search(r"exited with\s+(\d+)", text, flags=re.IGNORECASE)
    exit_code = int(exit_match.group(1)) if exit_match else int(error_hits > 0)

    files_changed = _extract_int(
        [r"files changed[:=]\s*(\d+)", r"changed files[:=]\s*(\d+)", r"(\d+)\s+files?\s+changed"],
        text,
        default=max(1, min(50, round(line_count / 18))),
        minimum=1,
    )
    lines_added = _extract_int(
        [r"lines added[:=]\s*(\d+)", r"insertions?\D+(\d+)", r"\+(\d+)\s+lines?"],
        text,
        default=min(5000, line_count * 4 + warning_hits * 12),
        minimum=0,
    )
    lines_deleted = _extract_int(
        [r"lines deleted[:=]\s*(\d+)", r"deletions?\D+(\d+)", r"-(\d+)\s+lines?"],
        text,
        default=min(5000, error_hits * 18 + warning_hits * 4),
        minimum=0,
    )
    commit_msg_length = _extract_int(
        [r"commit message length[:=]\s*(\d+)"],
        text,
        default=len(re.search(r"commit message[:=]\s*(.+)", text, re.IGNORECASE).group(1).strip())
        if re.search(r"commit message[:=]\s*(.+)", text, re.IGNORECASE)
        else 50,
        minimum=1,
    )
    author_id = _extract_int([r"author[_ -]?id[:=]\s*(\d+)"], text, default=42, minimum=1, maximum=500)
    prior_failure_rate = min(1.0, max(0.0, 0.18 + error_hits * 0.035 + warning_hits * 0.01 + (0.25 if exit_code else 0)))
    hour_of_commit, day_of_week = _extract_timestamp_parts(text)

    return {
        "author_id": author_id,
        "files_changed": files_changed,
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
        "commit_msg_length": commit_msg_length,
        "hour_of_commit": hour_of_commit,
        "day_of_week": day_of_week,
        "branch": _extract_branch(text),
        "language": _extract_language(text),
        "prior_failure_rate": round(prior_failure_rate, 2),
        "error_count": error_hits,
        "warning_count": warning_hits,
        "log_line_count": line_count,
        "exit_code": exit_code,
    }


@app.post("/predict", response_model=PredictionResult)
def predict(commit: BuildCommit):
    return _predict_commit(commit)


@app.post("/predict-log")
def predict_log(payload: LogPredictionRequest):
    commit = parse_log_to_commit(payload.log_text)
    prediction = _predict_commit(commit)
    return {"features": commit, "prediction": prediction.model_dump()}


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
