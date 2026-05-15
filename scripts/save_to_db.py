"""
Persist NeuroSleep experiment results into PostgreSQL.

Reads the JSON metrics files produced by notebooks 03 and 04 (baselines and
CNN), along with the corresponding prediction Parquet files, and writes them
into the relational schema defined in src/db/models.py.

Usage
-----
After running notebooks 03 and 04 at least once:

    python scripts/save_to_db.py

The database connection URL is read from the DATABASE_URL environment variable
and defaults to the local docker-compose service:

    postgresql+psycopg2://neurosleep:neurosleep@localhost:5432/neurosleep

To run against the dockerized Postgres:

    docker compose up -d postgres
    python scripts/save_to_db.py

Idempotency
-----------
This script always appends new experiment rows, so it can be run multiple
times. Each call records the wall-clock time, making it easy to track how
results evolved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Make `src` importable when running this script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import Experiment, Metric, Prediction, init_db, session_scope  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "data" / "results"

# Metric fields that should be persisted (the rest is metadata)
METRIC_FIELDS = {
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "cohen_kappa",
}


def _persist_run(
    session,
    model_payload: dict,
    train_subjects: list[int] | None,
    test_subjects: list[int] | None,
    n_train: int | None,
    n_test: int | None,
    n_features: int | None,
    predictions_df: pd.DataFrame | None,
    notes: str | None = None,
) -> int:
    """Insert one experiment row + its metrics + predictions."""

    model_name = model_payload.get("model", "unknown")

    hp_keys = {"fit_seconds", "predict_seconds", "epochs_trained", "test_loss"}
    hyperparameters = {k: v for k, v in model_payload.items() if k in hp_keys}

    experiment = Experiment(
        model_name=model_name,
        dataset="sleep-edf",
        n_train_epochs=n_train,
        n_test_epochs=n_test,
        n_features=n_features,
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        hyperparameters=hyperparameters or None,
        notes=notes,
    )
    session.add(experiment)
    session.flush()  # so experiment.id is available

    for metric_name in METRIC_FIELDS:
        if metric_name in model_payload:
            session.add(
                Metric(
                    experiment_id=experiment.id,
                    name=metric_name,
                    value=float(model_payload[metric_name]),
                    split="test",
                )
            )

    if predictions_df is not None and not predictions_df.empty:
        for _, row in predictions_df.iterrows():
            session.add(
                Prediction(
                    experiment_id=experiment.id,
                    subject_id=int(row["subject"]),
                    true_label=str(row["true"]),
                    predicted_label=str(row["pred"]),
                )
            )

    return experiment.id


def persist_baselines() -> list[int]:
    """Persist Random Forest and SVM results from Notebook 03."""
    baseline_path = RESULTS_DIR / "baseline_metrics.json"
    predictions_path = RESULTS_DIR / "baseline_predictions.parquet"

    if not baseline_path.exists():
        print(f"  [!] Baseline metrics not found at {baseline_path}. "
              f"Run Notebook 03 first.")
        return []

    with open(baseline_path) as f:
        payload = json.load(f)

    predictions_df = (
        pd.read_parquet(predictions_path) if predictions_path.exists() else None
    )

    inserted_ids: list[int] = []

    with session_scope() as session:
        for slug, pred_col in [("random_forest", "rf_pred"), ("svm_rbf", "svm_pred")]:
            model_payload = payload.get(slug)
            if model_payload is None:
                print(f"  [-] Skipping {slug}: missing in payload.")
                continue

            preds_for_model = None
            if predictions_df is not None and pred_col in predictions_df.columns:
                preds_for_model = predictions_df[["subject", "true", pred_col]].rename(
                    columns={pred_col: "pred"}
                )

            exp_id = _persist_run(
                session=session,
                model_payload=model_payload,
                train_subjects=payload.get("train_subjects"),
                test_subjects=payload.get("test_subjects"),
                n_train=payload.get("n_train"),
                n_test=payload.get("n_test"),
                n_features=payload.get("n_features"),
                predictions_df=preds_for_model,
                notes=f"Classical baseline ({slug}) from Notebook 03.",
            )
            inserted_ids.append(exp_id)
            print(f"  [✓] Inserted {model_payload.get('model')} as experiment #{exp_id}")

    return inserted_ids


def persist_cnn() -> int | None:
    """Persist CNN results from Notebook 04."""
    cnn_path = RESULTS_DIR / "cnn_metrics.json"
    predictions_path = RESULTS_DIR / "cnn_predictions.parquet"

    if not cnn_path.exists():
        print(f"  [!] CNN metrics not found at {cnn_path}. Run Notebook 04 first.")
        return None

    with open(cnn_path) as f:
        payload = json.load(f)

    cnn_payload = payload.get("cnn", {})
    hyperparameters = payload.get("hyperparameters", {})
    # Merge hyperparameters into the model payload so they get stored
    cnn_payload = {**cnn_payload, **hyperparameters}

    predictions_df = None
    if predictions_path.exists():
        raw = pd.read_parquet(predictions_path)
        predictions_df = raw.rename(columns={"cnn_pred": "pred"})

    with session_scope() as session:
        exp_id = _persist_run(
            session=session,
            model_payload=cnn_payload,
            train_subjects=payload.get("train_subjects"),
            test_subjects=payload.get("test_subjects"),
            n_train=None,
            n_test=None,
            n_features=None,
            predictions_df=predictions_df,
            notes="1D CNN trained on raw EEG signals (TensorFlow/Keras).",
        )

    print(f"  [✓] Inserted CNN as experiment #{exp_id}")
    return exp_id


def print_summary() -> None:
    """Print a summary of every experiment currently in the database."""
    with session_scope() as session:
        experiments = session.query(Experiment).order_by(Experiment.id).all()
        if not experiments:
            print("  (database is empty)")
            return

        print(f"  Found {len(experiments)} experiment(s):\n")
        for exp in experiments:
            metric_lines = [
                f"      {m.name}={m.value:.4f}"
                for m in sorted(exp.metrics, key=lambda x: x.name)
            ]
            print(f"  #{exp.id} | {exp.model_name} | {exp.created_at:%Y-%m-%d %H:%M}")
            for line in metric_lines:
                print(line)
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persist NeuroSleep experiment results into PostgreSQL."
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only create tables and exit, without inserting any data.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary of stored experiments and exit.",
    )
    args = parser.parse_args()

    print("→ Initializing database schema (creating tables if missing)...")
    init_db()
    print("  [✓] Schema ready.\n")

    if args.init_only:
        return

    if args.summary:
        print("→ Stored experiments:\n")
        print_summary()
        return

    print("→ Persisting baseline results from Notebook 03...")
    persist_baselines()

    print("\n→ Persisting CNN results from Notebook 04...")
    persist_cnn()

    print("\n→ Current database contents:\n")
    print_summary()


if __name__ == "__main__":
    main()
