# CI/CD Build Failure Predictor

A complete machine learning pipeline that predicts whether a CI/CD build will fail from commit metadata, serves predictions through FastAPI, and explains model decisions with SHAP.

## What Is Included

- Travis CI build-log ingestion from `travis-ci-build-log-dataset`.
- Synthetic Travis CI-style dataset generation as a fallback/demo option.
- Feature engineering for commit, branch, timing, diff size, language, and author history signals.
- Model comparison across Logistic Regression, Random Forest, and Optuna-tuned XGBoost.
- Saved production model artifacts in `models/`.
- SHAP summary and waterfall plots in `reports/`.
- FastAPI REST API with a dark themed web UI in `static/index.html`.

## Project Structure

```text
PipelineIQ/
  api/
    main.py
    schema.py
  data/
    builds.csv
    features.csv
    target.csv
  models/
    best_model.pkl
    scaler.pkl
    feature_names.pkl
    model_meta.json
    model_comparison.csv
  notebooks/
    EDA.ipynb
  reports/
    shap_summary.png
    shap_waterfall.png
  src/
    generate_data.py
    preprocess.py
    train.py
    evaluate.py
    ingest_travis_logs.py
  static/
    index.html
  requirements.txt
  README.md
```

## Setup

Tested with Python 3.12.

```bash
pip install -r requirements.txt
```

## Data Sources

This project supports two data sources:

- Real Travis logs from `travis-ci-build-log-dataset`
- Synthetic CI-style data from `src/generate_data.py`

The committed model artifacts are trained on 20,000 parsed Travis CI logs from:

```text
C:\Users\BK\Documents\GitHub\travis-ci-build-log-dataset
```

## Generate Synthetic Data

```bash
python src/generate_data.py
```

This creates a synthetic `data/builds.csv`.

Synthetic dataset profile:

- Rows: 10,000
- Passes: 6,671
- Failures: 3,329
- Failure rate: 33.29%

## Build Dataset From Travis Logs

```bash
python src/train.py
```

`train.py` defaults to `--data-source auto`, which uses the local Travis log repository when it exists and falls back to synthetic data otherwise.

This parses real `log.txt` files, writes `data/builds.csv`, trains the models, and saves final artifacts.

Latest Travis-log dataset profile:

- Parsed logs: 20,000
- Passes: 16,005
- Failures: 3,995
- Failure rate: 19.98%

## Preprocess Data

```bash
python src/preprocess.py
```

This creates:

- `data/features.csv`
- `data/target.csv`
- `models/scaler.pkl`

## Train Models

```bash
python src/train.py
```

Training runs:

- LogisticRegression
- RandomForestClassifier
- XGBClassifier tuned with Optuna for 50 trials

Saved artifacts:

- `models/best_model.pkl`
- `models/scaler.pkl`
- `models/feature_names.pkl`
- `models/model_meta.json`
- `models/model_comparison.csv`

## Latest Model Results

| Model | Precision | Recall | F1 | AUC-ROC | Threshold | Predicted Fail Rate |
|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.9511 | 0.9975 | 0.9737 | 0.9971 | 0.60 | 20.95% |
| RandomForest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.12 | 19.98% |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.10 | 19.98% |

Best model: RandomForest.

The API uses the saved decision threshold from `models/model_meta.json` instead of the default 0.50 cutoff, so it flags risky builds instead of passing almost everything.

Best XGBoost parameters from the comparison run:

```json
{
  "max_depth": 5,
  "n_estimators": 189,
  "learning_rate": 0.041651511115571596,
  "subsample": 0.9445218281346772,
  "colsample_bytree": 0.8651250666290946
}
```

## Run SHAP Evaluation

```bash
python src/evaluate.py
```

This creates:

- `reports/shap_summary.png`
- `reports/shap_waterfall.png`

Latest top SHAP drivers:

- `exit_code`
- `error_count`
- `prior_failure_rate`
- `log_line_count`
- `files_changed`

## Start API And Web UI

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://localhost:8000/
```

Useful endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-log`
- `GET /docs`

## Demo Frontend

The web UI includes three demo paths:

- Preset Log: loads a pre-built CI failure log and sends it through the log parser and model.
- Custom Log: upload or paste a `.log` or `.txt` file, then predict from extracted metadata.
- Manual Fields: directly edit commit metadata and run the predictor.

The log demo extracts branch, language, commit time, diff size, commit message length, author ID, rough historical risk signals, and log-derived signals such as exit code, error count, warning count, and log length before calling the same trained model pipeline.

## Sample Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"author_id":7,"files_changed":18,"lines_added":450,"lines_deleted":10,"commit_msg_length":8,"hour_of_commit":3,"day_of_week":6,"branch":"hotfix","language":"Java","prior_failure_rate":0.65}'
```

## Sample Response

```json
{
  "prediction": "FAIL",
  "confidence": 0.77,
  "risk_level": "high",
  "top_risk_factors": [
    { "feature": "exit_code", "impact": 0.3863 },
    { "feature": "log_line_count", "impact": -0.108 },
    { "feature": "error_count", "impact": -0.069 }
  ]
}
```

## Sample Log Request

```bash
curl -X POST http://localhost:8000/predict-log \
  -H "Content-Type: application/json" \
  -d '{"log_text":"author_id: 7\nbranch: hotfix/payment-timeout\nlanguage: Java\ncommit_time: 2026-05-10T03:12:00+00:00\ncommit message length: 8\nfiles changed: 18\nlines added: 450\nlines deleted: 10\n[ERROR] Build failure"}'
```

## End-To-End Run

```bash
python src/train.py --data-source travis --logs-dir "C:\Users\BK\Documents\GitHub\travis-ci-build-log-dataset"
python src/evaluate.py
uvicorn api.main:app --reload
```

## Notes

The current committed model is trained on parsed Travis CI logs. Synthetic generation is still available for reproducible demos or when the external Travis log repository is not present.
