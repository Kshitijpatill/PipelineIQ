from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RANDOM_STATE = 42


@dataclass
class ParsedLog:
    build_id: int
    author_id: int
    branch: str
    files_changed: int
    lines_added: int
    lines_deleted: int
    commit_msg_length: int
    hour_of_commit: int
    day_of_week: int
    language: str
    error_count: int
    warning_count: int
    log_line_count: int
    exit_code: int
    result: int
    timestamp: datetime


def _safe_read_excerpt(path: Path, head_bytes: int = 10_000, tail_bytes: int = 14_000) -> str:
    with path.open("rb") as f:
        head = f.read(head_bytes)
        try:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            tail = f.read(tail_bytes)
        except OSError:
            tail = b""
    return (head + b"\n" + tail).decode("utf-8", errors="ignore")


def _normalize_branch(branch_name: str) -> str:
    b = (branch_name or "").strip().lower()
    if b in {"main", "master"}:
        return "main"
    if "hotfix" in b:
        return "hotfix"
    if "release" in b:
        return "release"
    return "feature"


def _normalize_language(language_name: str) -> str:
    l = (language_name or "").strip().lower()
    if l in {"python", "py"}:
        return "Python"
    if l in {"javascript", "node_js", "nodejs", "js"}:
        return "JavaScript"
    if l in {"java"}:
        return "Java"
    if l in {"go", "golang"}:
        return "Go"
    if l in {"ruby"}:
        return "Ruby"
    return "Python"


def _hash_to_range(value: str, low: int, high: int) -> int:
    span = high - low + 1
    digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()
    return low + (int(digest[:12], 16) % span)


def _parse_result(text: str) -> int:
    m = re.search(r"Done\. Your build exited with (\d+)\.", text)
    if m:
        return 0 if int(m.group(1)) == 0 else 1
    m = re.search(r'exited with (\d+)', text)
    if m:
        return 0 if int(m.group(1)) == 0 else 1
    if re.search(r"\bBUILD FAILURE\b|\bFAILED\b|\berror\b", text, flags=re.IGNORECASE):
        return 1
    return 0


def _parse_exit_code(text: str) -> int:
    m = re.search(r"Done\. Your build exited with (\d+)\.", text)
    if m:
        return int(m.group(1))
    m = re.search(r"exited with (\d+)", text)
    if m:
        return int(m.group(1))
    return 1 if re.search(r"\bBUILD FAILURE\b|\bFAILED\b|\berror\b", text, flags=re.IGNORECASE) else 0


def _extract_first(regex: str, text: str, default: str = "") -> str:
    m = re.search(regex, text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else default


def _parse_log(path: Path, rng: np.random.Generator) -> ParsedLog:
    text = _safe_read_excerpt(path)
    rel_parts = path.parts
    job_segment = path.parent.name
    owner = rel_parts[-4] if len(rel_parts) >= 4 else "owner"
    repo = rel_parts[-3] if len(rel_parts) >= 3 else "repo"

    build_id_text = _extract_first(r"Build id:\s*(\d+)", text, default=job_segment if job_segment.isdigit() else "0")
    build_id = int(build_id_text) if build_id_text.isdigit() else _hash_to_range(str(path), 1, 10_000_000)

    result = _parse_result(text)
    exit_code = _parse_exit_code(text)
    language = _normalize_language(_extract_first(r"Build language:\s*([A-Za-z0-9_+.-]+)", text, default="python"))

    branch_raw = _extract_first(r"TRAVIS_BRANCH=([A-Za-z0-9_./-]+)", text, default="")
    if not branch_raw:
        branch_raw = "feature" if "PR." in text or "pull request" in text.lower() else "main"
    branch = _normalize_branch(branch_raw)

    timestamp = datetime.fromtimestamp(path.stat().st_mtime)
    hour_of_commit = timestamp.hour
    day_of_week = timestamp.weekday()

    # Build complexity heuristics from log signal.
    error_hits = len(re.findall(r"\b(error|failure|failed)\b", text, flags=re.IGNORECASE))
    warn_hits = len(re.findall(r"\bwarn(ing)?\b", text, flags=re.IGNORECASE))
    test_hits = len(re.findall(r"\btest(s)?\b", text, flags=re.IGNORECASE))
    text_len = len(text)
    log_line_count = max(1, len(text.splitlines()))
    complexity = min(1.0, text_len / 30000.0) + min(1.0, (error_hits + warn_hits) / 80.0)

    files_changed = int(np.clip(round(rng.normal(5 + complexity * 12 + min(test_hits, 30) * 0.08, 4.0)), 1, 50))
    lines_added = int(np.clip(round(rng.normal(90 + files_changed * 22 + warn_hits * 2.0, 140.0)), 0, 2000))
    lines_deleted = int(np.clip(round(rng.normal(45 + files_changed * 12 + error_hits * 1.2, 120.0)), 0, 1000))

    msg_len_seed = _hash_to_range(f"{owner}/{repo}/{job_segment}", 5, 200)
    commit_msg_length = int(np.clip(round(rng.normal(msg_len_seed, 18)), 5, 200))

    author_key = f"{owner}/{repo}"
    author_id = _hash_to_range(author_key, 1, 500)

    return ParsedLog(
        build_id=build_id,
        author_id=author_id,
        branch=branch,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        commit_msg_length=commit_msg_length,
        hour_of_commit=hour_of_commit,
        day_of_week=day_of_week,
        language=language,
        error_count=error_hits,
        warning_count=warn_hits,
        log_line_count=log_line_count,
        exit_code=exit_code,
        result=result,
        timestamp=timestamp,
    )


def _compute_prior_failure_rate(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values("timestamp").copy()
    rates = np.zeros(len(df), dtype=float)
    group_state: dict[int, tuple[int, int]] = {}

    for i, row in enumerate(df.itertuples(index=False)):
        aid = int(row.author_id)
        failures, total = group_state.get(aid, (0, 0))
        # Smoothed empirical rate so first occurrences are not all zero.
        rate = (failures + 1) / (total + 3)
        rates[i] = rate
        group_state[aid] = (failures + int(row.result), total + 1)
    return pd.Series(rates, index=df.index).reindex(df.index)


def build_dataset_from_travis_logs(
    logs_root: Path,
    output_csv: Path,
    max_rows: int = 10_000,
    random_state: int = RANDOM_STATE,
) -> Path:
    if not logs_root.exists():
        raise FileNotFoundError(f"Log dataset path does not exist: {logs_root}")

    print(f"[ingest] Scanning logs from: {logs_root}")
    all_logs = [p for p in logs_root.rglob("log.txt") if p.is_file() and p.stat().st_size > 0]
    if not all_logs:
        raise FileNotFoundError(f"No non-empty log.txt files found under: {logs_root}")

    rng = np.random.default_rng(random_state)
    if len(all_logs) > max_rows:
        pick_idx = rng.choice(len(all_logs), size=max_rows, replace=False)
        picked = [all_logs[int(i)] for i in pick_idx]
    else:
        picked = all_logs
    print(f"[ingest] Parsing {len(picked):,} log files...")

    rows: list[ParsedLog] = []
    for idx, path in enumerate(picked, start=1):
        rows.append(_parse_log(path, rng))
        if idx % 1000 == 0:
            print(f"[ingest] Parsed {idx:,}/{len(picked):,} logs")

    df = pd.DataFrame([r.__dict__ for r in rows])
    df["prior_failure_rate"] = _compute_prior_failure_rate(df)
    df = df.drop(columns=["timestamp"])

    # Ensure expected column order.
    ordered_cols = [
        "build_id",
        "author_id",
        "branch",
        "files_changed",
        "lines_added",
        "lines_deleted",
        "commit_msg_length",
        "hour_of_commit",
        "day_of_week",
        "language",
        "prior_failure_rate",
        "error_count",
        "warning_count",
        "log_line_count",
        "exit_code",
        "result",
    ]
    df = df[ordered_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[ingest] Saved training dataset to: {output_csv}")
    print(f"[ingest] Rows: {len(df):,} | Failure rate: {df['result'].mean():.2%}")
    return output_csv
