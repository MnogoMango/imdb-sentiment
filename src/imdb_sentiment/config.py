"""Validated training configuration."""

import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SplitConfig:
    validation_size: float = 0.1
    test_size: float = 0.1
    random_state: int = 42

    def __post_init__(self) -> None:
        if not (0 < self.validation_size < 1 and 0 < self.test_size < 1):
            raise ValueError("Validation and test fractions must be between 0 and 1")
        if self.validation_size + self.test_size >= 1:
            raise ValueError("Validation + test fractions must be less than 1")


@dataclass(frozen=True)
class ModelConfig:
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    max_features: int = 100_000
    sublinear_tf: bool = True
    C: float = 1.0
    max_iter: int = 1000

    def __post_init__(self) -> None:
        if not 1 <= self.ngram_min <= self.ngram_max:
            raise ValueError("Invalid ngram range")
        if min(self.min_df, self.max_features, self.max_iter) < 1:
            raise ValueError("Counts must be positive")
        if not math.isfinite(self.C) or self.C <= 0:
            raise ValueError("C must be finite and positive")


@dataclass(frozen=True)
class Config:
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path | None = None) -> Config:
    if path is None:
        return Config()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    unknown = set(raw) - {"split", "model"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    return Config(SplitConfig(**raw.get("split", {})), ModelConfig(**raw.get("model", {})))
