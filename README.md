<!-- 📄 Main documentation for wagon classification microservice -->
# 🏗️ microservise - Микросервисная архитектура для классификации вагонов

## 📋 Описание

Полностью переработанная микросервисная архитектура для обработки фотографий вагонов:

1. **Model-1 Классификация** → Разделение фотографий на вагоны
2. **photo_aggregate** → Хранилище фотографий по вагонам
3. **Model-2 Детекция** → YOLO детекция признаков вагонов
4. **MongoDB Агрегация** → Сохранение финальных результатов
5. **REST API** → Полное управление через API endpoints

---

## 🏭 Структура микросервисов

```
microservise/
├── services/
│   ├── classifier.py      ⭐ Model-1: Классификация и разделение вагонов
│   ├── detector.py        ⭐ Model-2: Детекция признаков YOLO
│   ├── aggregator.py      ⭐ MongoDB: Агрегация результатов
│   └── model_loader.py    (вспомогательный)
│
├── routes/
│   └── wagon.py           ⭐ REST API endpoints (/api/ml/*)
│
├── models/
│   └── schemas.py         (MongoDB Pydantic models)
│
├── db/
│   └── repository.py      (Database helpers)
│
├── core/
│   ├── config.py          (Configuration)
│   └── constants.py       (Constants)
│
├── photo_aggregate/       ⭐ Auto-generated storage for wagon photos (хранится локально)
│   ├── wagon_1/
│   ├── wagon_2/
│   └── ...
│
├── examples.py            ⭐ Usage examples
└── __init__.py
```

---

## 🔄 Основной рабочий процесс

```
                    ┌──────────────────────────────┐
                    │  Папка с фотографиями (150)  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │ [1] CLASSIFIER - Model-1     │
                    │     Классификация вагонов    │
                    └──────────────┬───────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
    ┌───▼───┐              ┌───────▼────┐          ┌───────────▼┐
    │wagon_1│              │  wagon_2   │          │ wagon_3    │
    │(50)   │              │   (60)     │          │   (40)     │
    └───┬───┘              └───────┬────┘          └───────────┬┘
        │                          │                          │
    ┌───▼──────────────────────────▼──────────────────────────▼──┐
    │ [2] DETECTOR - Model-2 (YOLO)                              │
    │     Детекция для каждого вагона                            │
    └───┬──────────────────────────┬──────────────────────────┬──┘
        │                          │                          │
    ┌───▼─────────────────────────▼──────────────────────────▼──┐
    │ [3] AGGREGATOR - MongoDB                                   │
    │     Финальная агрегация и сохранение                       │
    └──────────────────────┬─────────────────────────────────────┘
                           │
                    ┌──────▼──────────┐
                    │ MongoDB Results  │
                    │ (wagon_1/left)   │
                    │ (wagon_2/right)  │
                    │ (wagon_3/left)   │
                    └─────────────────┘
```

---

## 🔌 REST API Endpoints

### Классификация и обработка

```bash
# 1. Классифицировать папку
POST /api/ml/classify-folder?folder_path=/path/to/photos

# 2. Запустить Model-2 для вагона
POST /api/ml/detect-wagon/{wagon_id}

# 3. Агрегировать результаты в MongoDB
POST /api/ml/aggregate-wagon/{wagon_id}
```

### Запросы информации

```bash
# Получить список вагонов
GET /api/ml/wagons

# Получить фотографии вагона
GET /api/ml/wagon-photos/{wagon_id}

# Получить статус вагона
GET /api/ml/wagon-status/{wagon_id}

# Получить результат вагона из MongoDB
GET /api/ml/wagon-result/{wagon_id}

# Получить статистику photo_aggregate
GET /api/ml/aggregate-stats

# Получить статус батча
GET /api/ml/batch-status/{batch_id}

# Получить результаты батча
GET /api/ml/batch-results/{batch_id}
```

### Управление файлами

```bash
# Удалить вагон
DELETE /api/ml/wagon/{wagon_id}

# Очистить старые файлы (30+ дней)
POST /api/ml/cleanup-old?days=30
```

---

## 📦 Основные сервисы

### 1. Classifier Service (`services/classifier.py`)

**Функции:**
- `predict_model1()` - Классификация одного изображения
- `classify_and_aggregate()` - Полная обработка папки

**Логика:**
```
Фотографии → Model-1 (one_wagon/transition) → photo_aggregate/
                                                wagon_1/
                                                wagon_2/
                                                wagon_3/
```

**Пример:**
```python
result = await classifier.classify_and_aggregate(
    folder_path="/photos",
    batch_id="batch_001"
)
# Returns: {status, processed, accepted, wagons: {wagon_1: {...}, ...}}
```

---

### 2. Detector Service (`services/detector.py`)

**Функции:**
- `predict_model2()` - YOLO детекция одного изображения
- `process_wagon_photos()` - Обработка всех фото вагона

**Логика:**
```
photo_aggregate/wagon_1/ → Model-2 (YOLO) → Детекция:
                                            - brake_rod
                                            - rod_nose
                                            - crane
                                            - tank
                                            - side (L/R)
                          → MongoDB/photos
```

**Пример:**
```python
result = await detector.process_wagon_photos(
    wagon_id="wagon_1",
    photo_paths=photo_paths,
    camera_id=0,
    batch_id="batch_001"
)
# Returns: {wagon_id, final_side, left_count, right_count, ...}
```

---

### 3. Aggregator Service (`services/aggregator.py`)

**Функции:**
- `aggregate_wagon_results()` - Финальная агрегация по вагону
- `get_wagon_status()` - Статус вагона
- `get_wagon_result()` - Результат вагона
- `get_batch_status()` - Статус батча
- `get_batch_results()` - Результаты батча

**Логика:**
```
MongoDB/photos → Подсчет LEFT/RIGHT → Majority vote → MongoDB/wagon_aggregates
                                                        (final_side)
```

**Пример:**
```python
result = await aggregator.aggregate_wagon_results(
    wagon_id="wagon_1",
    batch_id="batch_001"
)
# Returns: {wagon_id, final_side, left_count, right_count, total_photos, ...}
```

---
## 💾 MongoDB Структура

### PhotoDocument (photos коллекция)
```json
{
  "_id": ObjectId,
  "wagon_id": "wagon_1",
  "file_hash": "photo_001.jpg",
  "camera_id": 0,
  "side": "left",
  "confidence": 0.95,
  "features": {
    "brake_rod": 0.92,
    "rod_nose": 0.88,
    "crane": 0.85,
    "tank": 0.90
  },
  "batch_id": "batch_001",
  "processed_at": ISODate(),
  "final_verdict": {
    "side": "left",
    "left_count": 15,
    "right_count": 10,
    "total_photos": 25
  }
}
```

### WagonAggregateDocument (wagon_aggregates)
```json
{
  "_id": ObjectId,
  "wagon_id": "wagon_1",
  "photos": [ObjectId1, ObjectId2, ...],
  "left_count": 15,
  "right_count": 10,
  "total_photos": 25,
  "final_side": "left",
  "camera_ids": [0],
  "batch_id": "batch_001",
  "created_at": ISODate(),
  "updated_at": ISODate()
}
```

### BatchDocument (batches коллекция)
```json
{
  "_id": ObjectId,
  "batch_id": "batch_001",
  "folder": "input_folder",
  "total_photos": 150,
  "total_wagons": 3,
  "processed_photos": 145,
  "status": "completed",
  "results": {
    "wagon_1": {
      "total_photos": 50,
      "final_side": "left",
      "left_count": 30,
      "right_count": 20
    }
  },
  "processed_at": ISODate()
}
```

---

# Wagon ML Pipeline

ML микросервис для классификации, детекции и агрегации данных о вагонах.

## Быстрый старт

### 1. Клонировать репозиторий
```bash
git clone <your-repo-url>
cd <project-folder>
```
2. Установить Python 3.10
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv

# Или через pyenv
pyenv install 3.10
pyenv local 3.10
```
3. Установить Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
echo "$HOME/.local/bin" >> $PATH
```
4. Установить зависимости
```bash
poetry install

# Установить PyTorch с CUDA поддержкой
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Установить остальные зависимости
pip install fastapi uvicorn pydantic pydantic-settings python-dotenv beanie motor pymongo ultralytics pillow numpy python-multipart

# Установить инструменты для разработки
pip install pytest pytest-asyncio black flake8 pytest-cov
```
5. Запустить тесты
```bash
pytest --cov=. --cov-report=xml
```
6. Запустить сервер
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
