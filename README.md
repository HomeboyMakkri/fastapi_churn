# Churn Prediction API

Учебный ML-сервис на FastAPI, который обучает модель бинарной классификации и
предсказывает, уйдёт ли клиент в следующем месяце. Класс `1` означает отток,
класс `0` — сохранение клиента.

Сервис загружает локальный датасет при старте, обучает `LogisticRegression` или
`RandomForestClassifier`, сохраняет весь pipeline предобработки и модели и
возвращает класс вместе с вероятностями обоих классов.

## Возможности

- загрузка и валидация `data/churn_dataset.csv` при старте;
- просмотр датасета и информации о train/test-разбиении;
- обучение логистической регрессии или случайного леса;
- сохранение модели и восстановление её после перезапуска;
- одиночные и пакетные предсказания;
- история обучения и метрики `accuracy`, `f1`, `roc_auc`;
- единый формат ошибок, health check, OpenAPI и тесты;
- запуск локально или в Docker.

## Структура проекта

```text
.
├── data/churn_dataset.csv       # обучающий датасет
├── src/
│   ├── routers/                 # HTTP API: system, dataset, model, prediction
│   ├── services/training.py     # сценарий обучения и сохранения модели
│   ├── schemas.py               # Pydantic-схемы запросов и ответов
│   ├── config.py                # пути к данным и runtime-артефактам
│   ├── preprocessing.py         # подготовка и train/test-разбиение
│   ├── model.py                 # создание и обучение ML pipeline
│   ├── evaluation.py            # расчёт метрик
│   ├── prediction.py            # инференс
│   ├── model_store.py           # сохранение и загрузка модели
│   ├── training_history.py      # история запусков обучения
│   ├── application.py           # фабрика FastAPI-приложения
│   └── main.py                  # ASGI entry point
├── tests/
├── Dockerfile
└── requirements*.txt
```

После первого успешного обучения сервис создаёт runtime-каталог `models/` с
файлами `churn_model.joblib` и `training_history.json`. Эти файлы не входят в
репозиторий.

## Формат датасета

CSV-файл должен содержать ровно следующие колонки:

| Колонка | Тип / значения | Описание |
| --- | --- | --- |
| `monthly_fee` | число ≥ 0 | Ежемесячная стоимость тарифа |
| `usage_hours` | число ≥ 0 | Часы использования за последний месяц |
| `support_requests` | целое ≥ 0 | Число обращений в поддержку |
| `account_age_months` | целое ≥ 0 | Возраст аккаунта в месяцах |
| `failed_payments` | целое ≥ 0 | Число неудачных платежей |
| `region` | `europe`, `asia`, `america`, `africa` | Регион клиента |
| `device_type` | `mobile`, `desktop`, `tablet` | Основное устройство |
| `payment_method` | `card`, `paypal`, `crypto` | Способ оплаты |
| `autopay_enabled` | `0` или `1` | Включено ли автосписание |
| `churn` | `0` или `1` | Целевой признак |

В поставляемом файле находится 2000 строк данных. При подготовке строки с
пропусками удаляются, а оставшиеся данные воспроизводимо делятся на train и
test в отношении 80/20 со стратификацией по `churn`.

## Локальный запуск

Требуется Python 3.14.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn src.main:app --reload
```

Сервис будет доступен по адресу <http://127.0.0.1:8000>, Swagger UI — по
адресу <http://127.0.0.1:8000/docs>, health check — по адресу
<http://127.0.0.1:8000/health>.

Для запуска тестов установите dev-зависимости и выполните:

```bash
python -m pip install -r requirements-dev.txt
pytest
```

## Запуск в Docker

```bash
docker build -t fastapi-churn .
docker run --rm -p 8000:8000 --name fastapi-churn fastapi-churn
```

Чтобы сохранять модель и историю обучения между пересозданиями контейнера,
подключите именованный volume:

```bash
docker volume create fastapi-churn-models
docker run --rm -p 8000:8000 \
  -v fastapi-churn-models:/app/models \
  --name fastapi-churn fastapi-churn
```

Проверка запущенного контейнера:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/docs
```

## Примеры API

### Обучение модели

Обучить логистическую регрессию с настройками по умолчанию:

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H 'Content-Type: application/json' \
  -d '{"model_type":"logreg","hyperparameters":{}}'
```

Пример ответа:

```json
{
  "accuracy": 0.8175,
  "f1": 0.7368421052631579
}
```

Для случайного леса передайте, например:

```json
{
  "model_type": "random_forest",
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 8
  }
}
```

Значения метрик зависят от выбранной модели и гиперпараметров. Статус текущей
модели доступен в `GET /model/status`, история и полные метрики — в
`GET /model/metrics`.

### Предсказание

Сначала обучите модель или запустите сервис с ранее сохранённой моделью.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "monthly_fee": 79.99,
    "usage_hours": 8.5,
    "support_requests": 4,
    "account_age_months": 6,
    "failed_payments": 2,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 0
  }'
```

Ответ содержит предсказанный класс и вероятности:

```json
{
  "predicted_class": 1,
  "class_probabilities": {
    "0": 0.23,
    "1": 0.77
  }
}
```

`POST /predict` также принимает непустой JSON-массив объектов и возвращает
предсказания в том же порядке.

## Основные эндпоинты

| Метод и путь | Назначение |
| --- | --- |
| `GET /health` | Доступность датасета и модели |
| `GET /dataset/preview?count=5` | Первые строки датасета |
| `GET /dataset/info` | Размер, колонки и распределение классов |
| `GET /dataset/split-info` | Информация о train/test-разбиении |
| `GET /model/schema` | Контракт входных признаков |
| `POST /model/train` | Обучение и сохранение модели |
| `GET /model/status` | Статус и краткие метрики текущей модели |
| `GET /model/metrics` | История обучения и полные метрики |
| `POST /predict` | Одиночное или пакетное предсказание |

