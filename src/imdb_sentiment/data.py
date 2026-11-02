"""Shared cleaning, data validation, auditing and leak-free splitting."""

import html
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from imdb_sentiment.config import SplitConfig

LABELS = {"negative": 0, "positive": 1}
TAG = re.compile(r"<[^>]+>")
SPACE = re.compile(r"\s+")


def clean_review(text: str) -> str:
    """Preserve case, punctuation and negations; remove HTML and extra whitespace."""
    while True:
        cleaned = SPACE.sub(" ", TAG.sub(" ", html.unescape(text))).strip()
        if cleaned == text:
            return cleaned
        text = cleaned


def duplicate_key(text: str) -> str:
    return clean_review(text).casefold()


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string")
    if not {"review", "sentiment"}.issubset(frame.columns):
        raise ValueError("CSV must contain review and sentiment columns")
    return frame[["review", "sentiment"]].copy()


def audit_data(frame: pd.DataFrame) -> dict[str, Any]:
    text = frame["review"].fillna("").astype(str)
    labels = frame["sentiment"].fillna("").str.strip().str.lower()
    keys = text.map(duplicate_key)
    valid = keys.ne("")
    groups = pd.DataFrame({"key": keys[valid], "label": labels[valid]})
    lengths = keys[valid].str.split().str.len()
    return {
        "rows": len(frame),
        "empty_reviews": int((~valid).sum()),
        "missing_labels": int(labels.eq("").sum()),
        "invalid_labels": int((~labels.isin(LABELS) & labels.ne("")).sum()),
        "duplicate_excess": int(keys[valid].duplicated().sum()),
        "conflicting_texts": int((groups.groupby("key")["label"].nunique() > 1).sum()),
        "html_reviews": int(text.str.contains(TAG).sum()),
        "class_counts": {str(k): int(v) for k, v in labels.value_counts().items()},
        "words": {str(k): float(v) for k, v in lengths.describe().items()} if len(lengths) else {},
    }


def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail on invalid rows instead of silently discarding or relabeling them."""
    if frame.empty:
        raise ValueError("Dataset is empty")
    result = frame.copy()
    if result[["review", "sentiment"]].isna().any().any():
        raise ValueError("Missing reviews or labels")
    result["review"] = result["review"].map(clean_review)
    result["sentiment"] = result["sentiment"].str.strip().str.lower()
    if result["review"].eq("").any():
        raise ValueError("Empty reviews after cleaning")
    if not result["sentiment"].isin(LABELS).all():
        raise ValueError("Only negative and positive labels are supported")
    result["key"] = result["review"].str.casefold()
    if (result.groupby("key")["sentiment"].nunique() > 1).any():
        raise ValueError("Conflicting labels for duplicate reviews")
    result = result.drop_duplicates("key").drop(columns="key").reset_index(drop=True)
    result["sentiment"] = result["sentiment"].map(LABELS).astype(int)
    if result["sentiment"].nunique() != 2:
        raise ValueError("Both sentiment classes are required")
    return result


def split_data(frame: pd.DataFrame, config: SplitConfig) -> dict[str, pd.DataFrame]:
    train, holdout = train_test_split(
        frame,
        test_size=config.validation_size + config.test_size,
        stratify=frame["sentiment"],
        random_state=config.random_state,
    )
    validation, test = train_test_split(
        holdout,
        test_size=config.test_size / (config.validation_size + config.test_size),
        stratify=holdout["sentiment"],
        random_state=config.random_state,
    )
    return {"train": train, "validation": validation, "test": test}
