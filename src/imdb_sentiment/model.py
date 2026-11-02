"""One preprocessing pipeline for training and inference."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from imdb_sentiment.config import Config
from imdb_sentiment.data import clean_review


def clean_texts(texts: list[str]) -> list[str]:
    return [clean_review(text) for text in texts]


def build_model(config: Config) -> Pipeline:
    options = config.model
    return Pipeline(
        [
            ("clean", FunctionTransformer(clean_texts, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(options.ngram_min, options.ngram_max),
                    min_df=options.min_df,
                    max_features=options.max_features,
                    sublinear_tf=options.sublinear_tf,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=options.C,
                    max_iter=options.max_iter,
                    random_state=config.split.random_state,
                ),
            ),
        ]
    )
