Test-Path ".\data\IMDB Dataset.csv"
poetry run python --version
poetry run python -m imdb_sentiment.cli --help

poetry install

poetry run python -m imdb_sentiment.cli train --data ".\data\IMDB Dataset.csv" --config ".\configs\default.toml" --output ".\artifacts\run-001"

Запуск обучения

poetry run python -m imdb_sentiment.cli test --data ".\data\IMDB Dataset.csv" --model ".\artifacts\run-001\model.joblib" --output ".\artifacts\test-001"

Проверка метрик

Get-Content ".\artifacts\test-001\test_metrics.json"

Матрица ошибок

Invoke-Item ".\artifacts\test-001\test_confusion.png"

Проверка на тестовой выборке

poetry run python -m imdb_sentiment.cli predict --model ".\artifacts\run-001\model.joblib" --text "A wonderful movie with excellent acting." --text "A boring movie and a complete waste of time."
