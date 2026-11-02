"""Regression tests for data leakage, input contracts and artifact round trips."""

import json
from pathlib import Path

import pandas as pd
import pytest

from imdb_sentiment.config import Config, ModelConfig, SplitConfig, load_config
from imdb_sentiment.data import clean_review, duplicate_key, prepare_data, split_data
from imdb_sentiment.workflow import predict, train
from imdb_sentiment.workflow import test_model as evaluate_test


@pytest.fixture
def reviews() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "review": f"Excellent wonderful movie lovely acting story identifier{i}",
                "sentiment": "positive",
            }
            for i in range(40)
        ]
        + [
            {
                "review": f"Terrible boring movie awful acting story identifier{i}",
                "sentiment": "negative",
            }
            for i in range(40)
        ]
    )


def test_cleaning_preserves_negation() -> None:
    assert clean_review("<br /> Not good! &amp;  bad\n") == "Not good! & bad"


def test_duplicates_and_conflicts() -> None:
    frame = pd.DataFrame(
        {"review": ["Good", "<br> GOOD", "Bad"], "sentiment": ["positive", "positive", "negative"]}
    )
    assert len(prepare_data(frame)) == 2
    frame.loc[1, "sentiment"] = "negative"
    with pytest.raises(ValueError, match="Conflicting"):
        prepare_data(frame)


@pytest.mark.parametrize(
    "text,label", [(None, "positive"), ("<br>", "positive"), ("hello", "neutral"), ("hello", None)]
)
def test_invalid_rows(text: str | None, label: str | None) -> None:
    with pytest.raises(ValueError):
        prepare_data(pd.DataFrame({"review": [text], "sentiment": [label]}))


def test_splits_are_disjoint_and_reproducible(reviews: pd.DataFrame) -> None:
    frame = prepare_data(reviews)
    first = split_data(frame, SplitConfig())
    second = split_data(frame, SplitConfig())
    assert [len(part) for part in first.values()] == [64, 8, 8]
    keys = [set(part["review"].map(duplicate_key)) for part in first.values()]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    for name in first:
        assert first[name].index.tolist() == second[name].index.tolist()
        assert first[name]["sentiment"].nunique() == 2


def test_train_round_trip_and_dataset_guard(tmp_path: Path, reviews: pd.DataFrame) -> None:
    source = tmp_path / "reviews.csv"
    reviews.to_csv(source, index=False)
    output = tmp_path / "run"
    metrics = train(source, output, Config())
    assert metrics["accuracy"] > 0.9
    assert not (output / "test_metrics.json").exists()
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["split_sizes"] == {"train": 64, "validation": 8, "test": 8}
    result = predict(output / "model.joblib", ["<br>Excellent wonderful lovely movie"])
    assert result[0]["sentiment"] == "positive"
    with pytest.raises(ValueError, match="non-empty"):
        predict(output / "model.joblib", ["<br>"])
    test_metrics = evaluate_test(source, output / "model.joblib", tmp_path / "test")
    assert test_metrics["rows"] == 8
    with pytest.raises(FileExistsError):
        train(source, output, Config())
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="differs"):
        evaluate_test(source, output / "model.joblib", tmp_path / "changed")


def test_configuration_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        SplitConfig(validation_size=0.8, test_size=0.5)
    with pytest.raises(ValueError):
        ModelConfig(C=0)
    path = tmp_path / "invalid.toml"
    path.write_text("[typo]\nvalue=1\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


def test_vectorizer_does_not_learn_holdout_tokens(reviews: pd.DataFrame) -> None:
    from imdb_sentiment.model import build_model

    frame = prepare_data(reviews)
    parts = split_data(frame, SplitConfig())
    parts["test"] = parts["test"].copy()
    parts["test"]["review"] += " holdoutonlysentinel"
    config = Config(model=ModelConfig(min_df=1))
    model = build_model(config)
    model.fit(parts["train"]["review"], parts["train"]["sentiment"])
    vocabulary = model.named_steps["tfidf"].vocabulary_
    train_words = set(" ".join(parts["train"]["review"]).lower().split())
    holdout_words = set(" ".join(parts["test"]["review"]).lower().split())
    unique_words = holdout_words - train_words
    assert unique_words
    assert not unique_words.intersection(vocabulary)


def test_cli_audit(
    tmp_path: Path,
    reviews: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from imdb_sentiment.cli import main

    source = tmp_path / "data.csv"
    output = tmp_path / "audit.json"
    reviews.to_csv(source, index=False)
    monkeypatch.setattr(
        "sys.argv", ["imdb-sentiment", "audit", "--data", str(source), "--output", str(output)]
    )
    main()
    assert json.loads(capsys.readouterr().out)["rows"] == 80
    assert json.loads(output.read_text())["duplicate_excess"] == 0


@pytest.mark.parametrize("text", ["a &amp;amp; b", "&amp;lt;br&amp;gt; hello", "Not good!", ""])
def test_cleaning_is_idempotent(text: str) -> None:
    assert clean_review(clean_review(text)) == clean_review(text)
