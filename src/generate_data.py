from pathlib import Path

import numpy as np
import pandas as pd


def generate_builds_dataset(n_samples=10000, random_state=42):
    """Generate a synthetic Travis CI-style build dataset."""
    rng = np.random.default_rng(random_state)

    build_id = np.arange(1, n_samples + 1)
    author_id = rng.integers(1, 501, size=n_samples)
    branch = rng.choice(["main", "feature", "hotfix", "release"], size=n_samples, p=[0.50, 0.35, 0.10, 0.05])
    files_changed = np.maximum(1, rng.gamma(shape=2.0, scale=4.0, size=n_samples).round().astype(int))
    lines_added = np.maximum(0, rng.gamma(shape=2.0, scale=60.0, size=n_samples).round().astype(int))
    lines_deleted = np.maximum(0, rng.gamma(shape=1.8, scale=35.0, size=n_samples).round().astype(int))
    commit_msg_length = rng.integers(5, 201, size=n_samples)
    hour_of_commit = rng.integers(0, 24, size=n_samples)
    day_of_week = rng.integers(0, 7, size=n_samples)
    language = rng.choice(
        ["Python", "JavaScript", "Java", "Go", "Ruby", "C++", "TypeScript"],
        size=n_samples,
    )
    prior_failure_rate = rng.beta(1.2, 5.0, size=n_samples)

    fail_probability = np.full(n_samples, 0.25, dtype=float)
    fail_probability += 0.35 * prior_failure_rate
    fail_probability += 0.15 * (files_changed > 15)
    fail_probability += 0.12 * ((hour_of_commit >= 23) | ((hour_of_commit >= 2) & (hour_of_commit <= 5)))
    fail_probability += 0.10 * (branch == "hotfix")
    fail_probability += 0.08 * (commit_msg_length < 15)
    fail_probability -= 0.08 * (branch == "main")
    fail_probability = np.clip(fail_probability, 0, 1)

    result = rng.binomial(1, fail_probability)

    return pd.DataFrame(
        {
            "build_id": build_id,
            "author_id": author_id,
            "branch": branch,
            "files_changed": files_changed,
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "commit_msg_length": commit_msg_length,
            "hour_of_commit": hour_of_commit,
            "day_of_week": day_of_week,
            "language": language,
            "prior_failure_rate": prior_failure_rate,
            "result": result,
        }
    )


def main():
    output_path = Path(__file__).parent.parent / "data" / "builds.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_builds_dataset()
    df.to_csv(output_path, index=False)

    print("Class distribution:")
    print(df["result"].value_counts().sort_index())
    print(f"Failure rate: {df['result'].mean():.2%}")
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()
