"""Command line interface; independent of Colab and Google Drive."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from imdb_sentiment.config import load_config
from imdb_sentiment.data import audit_data, load_data
from imdb_sentiment.workflow import predict, save_json, test_model, train


def main() -> None:
    parser = argparse.ArgumentParser(description="IMDB binary sentiment classifier")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Write a data quality report")
    audit.add_argument("--data", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    training = commands.add_parser("train", help="Train and evaluate validation only")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--config", type=Path)
    testing = commands.add_parser("test", help="Evaluate the held-out test partition")
    testing.add_argument("--data", type=Path, required=True)
    testing.add_argument("--model", type=Path, required=True)
    testing.add_argument("--output", type=Path, required=True)
    inference = commands.add_parser("predict", help="Predict English movie review sentiment")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--text", action="append", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result: dict[str, Any] | list[dict[str, Any]]
    try:
        if args.command == "audit":
            result = audit_data(load_data(args.data))
            save_json(args.output, result)
        elif args.command == "train":
            result = train(args.data, args.output, load_config(args.config))
        elif args.command == "test":
            result = test_model(args.data, args.model, args.output)
        else:
            result = predict(args.model, args.text)
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
