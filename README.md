# IMDB Sentiment

Готовый Python-проект для бинарной классификации англоязычных рецензий:
TF-IDF (слова и биграммы) + LogisticRegression. Python 3.11–3.13, Poetry 2.x.
Google Colab и подключение Google Drive для запуска не нужны.

## Быстрый старт

Установите Python и Poetry. Распакуйте архив, откройте терминал в папке проекта:

```bash
git init
poetry install
poetry run pre-commit install
git add .
poetry run pre-commit run --all-files
```

`poetry.toml` задаёт окружение `.venv` внутри проекта. Оно создаётся при установке,
не включается в Git или архив и заново восстанавливается из `poetry.lock`.
Активировать окружение вручную не требуется: используйте `poetry run`.
VS Code настроен на локальную `.venv`; на Windows при необходимости выберите
`.venv/Scripts/python.exe` через Python: Select Interpreter.

Скопируйте IMDB Dataset.csv из Google Drive в `data/IMDB Dataset.csv`.
Данные, модели и отчёты исключены из Git. CSV должен быть UTF-8 и содержать
`review` (текст) и `sentiment` (`positive` / `negative`).
Пути передаются аргументами: можно использовать любой доступный локальный CSV.

```bash
poetry run imdb-sentiment audit --data "data/IMDB Dataset.csv" --output artifacts/audit.json
poetry run imdb-sentiment train --data "data/IMDB Dataset.csv" --config configs/default.toml --output artifacts/run-001
poetry run imdb-sentiment predict --model artifacts/run-001/model.joblib --text "A wonderful movie!" --text "A terrible waste of time."
```

После выбора окончательной модели выполните тестирование:

```bash
poetry run imdb-sentiment test --data "data/IMDB Dataset.csv" --model artifacts/run-001/model.joblib --output artifacts/test-001
```

Для каждого запуска используйте новую папку вывода. Существующая папка вызывает
ошибку, чтобы не перезаписать предыдущие результаты. При прерванном обучении
может остаться неполная папка: проверьте её и выберите новую для повторного запуска.
Не подбирайте параметры по тестовой метрике.

## Структура

- `src/imdb_sentiment/data.py`: аудит, очистка, валидация, дедупликация и разбиение.
- `config.py`: типизированные параметры с проверками и загрузка TOML.
- `model.py`: единый Pipeline очистки, TF-IDF и классификатора.
- `workflow.py`: обучение, оценка, сериализация и инференс.
- `cli.py`: команды `audit`, `train`, `test`, `predict`.
- `tests/`: проверки разбиения без утечек, ошибок данных и полного цикла модели.
- `configs/default.toml`: параметры исходного эксперимента.
- `.github/workflows/ci.yml`: проверки и сборка для Python 3.11–3.13.

## Данные и воспроизводимость

HTML-сущности раскодируются, HTML-теги и лишние пробелы удаляются. Регистр и
пунктуация сохраняются в очищенном тексте; TfidfVectorizer по умолчанию приводит
слова к нижнему регистру и использует собственный токенизатор. Стоп-слова,
стемминг и лемматизация не применяются. Повторы определяются после очистки и
casefold, удаляются до разделения. При конфликтующих метках, пропусках,
пустых текстах или неизвестных классах обучение завершается понятной ошибкой.
Аудит позволяет обнаружить эти проблемы до обучения.

Разделение стратифицированное: 80% train / 10% validation / 10% test, seed=42.
TF-IDF обучается исключительно на train. `train` оценивает только validation;
`test` восстанавливает тестовую часть и проверяет SHA-256 исходного CSV.
Изменение файла, включая порядок строк, блокирует такую оценку.
Индексы в `split_indices.json` относятся к очищенной таблице после дедупликации.

В папке обучения сохраняются `model.joblib` (Pipeline + метаданные),
`metadata.json` (параметры, SHA-256 данных, версии, размеры выборок),
`split_indices.json`, `audit.json`, `validation_metrics.json`,
`validation_confusion.png` и `validation_errors.csv`.
Команда `test` сохраняет аналогичные тестовые метрики, матрицу и ошибки.
Ошибки содержат тексты рецензий: храните отчёты с учётом доступа к данным.

Загружайте joblib только из доверенного источника: формат использует pickle.
Версия scikit-learn при инференсе должна совпадать с версией обучения;
восстанавливайте окружение из того же lockfile.
Вероятности модели не следует трактовать как откалиброванную уверенность.
Модель не распознаёт нейтральный класс и не проверена на русском языке.

## Проверки и сборка

```bash
poetry check --lock
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
poetry run pytest --cov=imdb_sentiment --cov-report=term-missing
poetry build
```

Ruff заменяет отдельные Flake8, isort и Black: линтинг, порядок импортов и формат.
Mypy проверяет типы собственного кода; отсутствующие stubs некоторых ML-библиотек
исключены из проверки. Pre-commit вызывает инструменты из Poetry-окружения;
Poetry должен быть доступен в PATH при коммите. После клонирования каждый
разработчик выполняет `poetry install` и `poetry run pre-commit install`.
CI использует те же hooks. Для автоисправления: `poetry run ruff check --fix .`
и `poetry run ruff format .`.

`poetry.lock` включается в Git. Для намеренного обновления зависимостей используйте
`poetry update`, затем запустите проверки и переобучите модель.
Окружение `.venv` коммитить не нужно. Remote-репозиторий добавляется отдельно:
`git remote add origin <URL>`; проект не публикуется автоматически.

## Результаты исходного ноутбука

Исходно: 50 000 отзывов, 422 повтора, 29 202 текста с HTML. После очистки:
49 578 отзывов (24 882 positive, 24 696 negative). Валидация: accuracy=0.9040,
macro F1=0.9040. Тест: accuracy=0.9074, macro F1=0.9074.
Это результаты предоставленного ноутбука, а не нового запуска проекта:
полный датасет и обученная модель не были переданы. Зависимости и их версии
могут отличаться от Colab; точное совпадение метрик не гарантировано.
Проект проверяется на синтетических тестовых данных, не заменяющих оценку IMDB.
