"""Training, evaluation and versioned artifact handling."""

import hashlib
import json
import logging
import platform
import warnings
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline

from imdb_sentiment.config import Config
from imdb_sentiment.data import audit_data, clean_review, load_data, prepare_data, split_data
from imdb_sentiment.model import build_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(model: Pipeline, data: pd.DataFrame, output: Path, name: str) -> dict[str, Any]:
    actual = data["sentiment"]
    predicted = model.predict(data["review"])
    metrics = {
        "rows": len(data),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "classification_report": classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["negative", "positive"],
            output_dict=True,
            zero_division=0,
        ),
    }
    save_json(output / f"{name}_metrics.json", metrics)
    errors = data.assign(predicted=predicted)
    errors.loc[actual.ne(predicted)].to_csv(output / f"{name}_errors.csv", index_label="row_id")
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        actual,
        predicted,
        labels=[0, 1],
        display_labels=["negative", "positive"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(name)
    figure.tight_layout()
    figure.savefig(output / f"{name}_confusion.png", dpi=150)
    plt.close(figure)
    return metrics


def train(data_path: Path, output: Path, config: Config) -> dict[str, Any]:
    """Reserve a new output directory, fit on train and evaluate validation only."""
    raw = load_data(data_path)
    frame = prepare_data(raw)
    splits = split_data(frame, config.split)
    output.mkdir(parents=True, exist_ok=False)
    LOGGER.info("Training with %d unique reviews", len(frame))
    save_json(output / "audit.json", audit_data(raw))
    manifest = {name: part.index.tolist() for name, part in splits.items()}
    save_json(output / "split_indices.json", manifest)
    model = build_model(config)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(splits["train"]["review"], splits["train"]["sentiment"])
    metadata = {
        "artifact_schema": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": fingerprint(data_path),
        "config": config.to_dict(),
        "prepared_rows": len(frame),
        "split_sizes": {name: len(part) for name, part in splits.items()},
        "labels": {"0": "negative", "1": "positive"},
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ["scikit-learn", "pandas", "joblib"]},
    }
    joblib.dump({"model": model, "metadata": metadata}, output / "model.joblib")
    save_json(output / "metadata.json", metadata)
    metrics = evaluate(model, splits["validation"], output, "validation")
    LOGGER.info("Validation accuracy=%.4f macro_f1=%.4f", metrics["accuracy"], metrics["macro_f1"])
    return metrics


def load_artifact(path: Path) -> dict[str, Any]:
    """Load only artifacts from a trusted source: joblib uses pickle internally."""
    artifact: dict[str, Any] = joblib.load(path)
    if artifact["metadata"]["artifact_schema"] != 1:
        raise ValueError("Unsupported artifact schema")
    installed = version("scikit-learn")
    if installed != artifact["metadata"]["versions"]["scikit-learn"]:
        raise ValueError("scikit-learn version differs from training; use the training lockfile")
    return artifact


def test_model(data_path: Path, artifact_path: Path, output: Path) -> dict[str, Any]:
    artifact = load_artifact(artifact_path)
    metadata = artifact["metadata"]
    if fingerprint(data_path) != metadata["dataset_sha256"]:
        raise ValueError("Dataset differs from training input; test evaluation refused")
    from imdb_sentiment.config import ModelConfig, SplitConfig

    config = Config(
        SplitConfig(**metadata["config"]["split"]), ModelConfig(**metadata["config"]["model"])
    )
    frame = prepare_data(load_data(data_path))
    held_out = split_data(frame, config.split)["test"]
    output.mkdir(parents=True, exist_ok=False)
    return evaluate(artifact["model"], held_out, output, "test")


def predict(artifact_path: Path, texts: list[str]) -> list[dict[str, Any]]:
    if not texts or any(not isinstance(t, str) or not clean_review(t) for t in texts):
        raise ValueError("Prediction requires non-empty text")
    artifact = load_artifact(artifact_path)
    model = artifact["model"]
    probabilities = model.predict_proba(texts)
    classes = model.classes_
    labels = {0: "negative", 1: "positive"}
    return [
        {
            "text": text,
            "sentiment": labels[int(classes[row.argmax()])],
            "positive_probability": float(row[list(classes).index(1)]),
        }
        for text, row in zip(texts, probabilities, strict=True)
    ]
