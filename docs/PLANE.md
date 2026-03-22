# ML Service для классификации вагонов

**Версия**: 2.2.0  
**Дата**: 2026-02-24  
**Статус**: Production Ready

---

## Содержание

1. [Архитектура системы](#архитектура-системы)
2. [Компоненты](#компоненты)
3. [Pipeline обработки](#pipeline-обработки)
4. [Входные и выходные данные](#входные-и-выходные-данные)
5. [Структура папок](#структура-папок)
6. [Как работают модели](#как-работают-модели)
7. [API endpoints](#api-endpoints)
8. [Логирование и отладка](#логирование-и-отладка)
9. [Возможные ошибки](#возможные-ошибки-и-их-решения)
10. [Примеры запросов](#примеры-запросов)

---

## Архитектура системы

```
┌────────────────────────────────────────────────────────────────┐
│                    FASTAPI APPLICATION                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           HTTP API Layer (routes)                       │   │
│  │  ┌──────────────────┬──────────────────┐                │   │
│  │  │ wagon.py         │ health.py        │                │   │
│  │  │ Endpoints        │ Health Check     │                │   │
│  │  └────────┬─────────┴────────┬─────────┘                │   │
│  │           │                  │                          │   │
│  └───────────┼──────────────────┼──────────────────────────┘   │
│              │                  │                              │
│  ┌───────────▼──────────────────▼──────────────────────────┐   │
│  │           Application Core Layer                        │   │
│  │  ┌────────────────────────────────────────────────────┐ │   │
│  │  │  processor.py - Photo Processing Pipeline          │ │   │
│  │  │  • process_photo() - Single photo                  │ │   │
│  │  │  • process_folder() - batch folder                 │ │   │
│  │  │  • _predict_model1() - Model-1 inference           │ │   │
│  │  │  • _predict_model2() - Model-2 inference           │ │   │
│  │  │  • aggregation functions                           │ │   │
│  │  └────────────────────────────────────────────────────┘ │   │
│  └────────────────┬────────────────────────────────────────┘   │
│                   │                                            │
│  ┌────────────────▼────────────────────────────────────────┐   │
│  │           ML Model Layer                                │   │
│  │  ┌─────────────────────┬──────────────────────────────┐ │   │
│  │  │ model_loader.py     │ models.py (deprecated)       │ │   │
│  │  │ • Load Models       │                              │ │   │
│  │  │ • Cache Models      │ (for reference)              │ │   │
│  │  │ • Format Detection  │                              │ │   │
│  │  │ • Error Handling    │                              │ │   │
│  │  └─────────┬───────────┴──────────────────────────────┘ │   │
│  │            │                                            │   │
│  │  ┌─────────▼─────────────────────────────────────────┐  │   │
│  │  │  Cached Model Instances                           │  │   │
│  │  │  • model1: PyTorch filter (MobileNetv3)           │  │   │
│  │  │  • model2: YOLOv11 detector                       │  │   │
│  │  └─────────┬─────────────────────────────────────────┘  │   │
│  └────────────┼────────────────────────────────────────────┘   │
│               │                                                │
│  ┌────────────▼─────────────────────────────────────────────┐  │
│  │           Data Layer                                     │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  Beanie ODM + MongoDB                              │  │  │
│  │  │  Collections:                                      │  │  │
│  │  │  • photos - Photo documents                        │  │  │
│  │  │  • wagon_aggregates - Per-wagon results            │  │  │
│  │  │  • batches - Batch processing results              │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘

Configuration & Logging:
┌──────────────────────────────┐
│ config.py - Settings         │
│ • Path settings              │
│ • Model validation           │
│ • Database config            │
└──────────────────────────────┘

┌──────────────────────────────┐
│ Logging System               │
│ • DEBUG - Detailed trace     │
│ • INFO - Process flow        │
│ • ERROR - Failures           │
└──────────────────────────────┘
```

---

## Компоненты

### 1. **FastAPI Application** (`app/main.py`)
- RESTful веб-сервис
- Automatic Swagger документация
- Async обработка запросов
- CORS поддержка
- MongoDB инициализация на старте

### 2. **Model-1: Filter Model** (`../model-1/mobilenet_wagon_best.pt`)
- **Тип**: PyTorch MobileNet
- **Назначение**: Фильтрация фотографий по качеству
- **Вход**: Изображение (PIL Image)
- **Выход**: Булев флаг (валидна/невалидна) + confidence (0-1)
- **Формат**: `.pt` (PyTorch)
- **Размер**: ~70MB (примерно)

### 3. **Model-2: Classification Model** (`../model-2/Data/best.pt`)
- **Тип**: YOLOv11 Object Detector
- **Назначение**: Детекция частей вагона и определение стороны
- **Вход**: Изображение (PIL Image)
- **Выход**: Детекции (brake_rod, rod_nose, crane, tank) + side (left/right)
- **Формат**: `.pt` (PyTorch + YOLO)
- **Размер**: ~200MB (примерно)
- **Классы**: brake_rod, rod_nose, crane, tank

### 4. **MongoDB** (External)
- **База данных**: wagon_classification
- **Collections**:
  - `photos` - Фото и результаты их обработки
  - `wagon_aggregates` - Агрегированные результаты по вагонам
  - `batches` - История обработки папок
- **Индексы**: wagon_id, camera_id, batch_id, file_hash

### 5. **Configuration System** (`app/config.py`)
- **Settings**:
  - MongoDB URL
  - Model paths (проверяются при старте)
  - Timeout для inference
  - Log level
  - API settings
- **Валидация путей моделей** на инициализацию

### 6. **Logging System**
- **Levels**:
  - **DEBUG**: Детальная информация о каждом шаге
  - **INFO**: Основной процесс (загрузка, обработка, результаты)
  - **ERROR**: Ошибки и сбои
- **Output**: Console + структурированные логи
- **Форматирование**: Timestamp, Level, Message, Context

---

## Pipeline обработки

### Однофотография (Single Photo)

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. API Request Arrives                                           │
│    POST /api/upload                                              │
│    - wagon_id: str                                               │
│    - camera_id: int                                              │
│    - file: binary (image)                                        │
└────────┬─────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│ 2. Request Validation                                            │
│    - Check parameters exist                                      │
│    - Validate file is image                                      │
│    - Check max file size (10MB)                                  │
└────────┬──────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│ 3. Image Processing                                              │
│    - Convert bytes to PIL Image                                  │
│    - Get image size & format                                     │
│    - Log: "Image loaded: size, mode, format"                     │
└────────┬──────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│ 4. Model-1 Inference (Filter)                                    │
│                                                                  │
│    4.1 Load Model (cached if already loaded)                     │
│        └─ Log: "Model-1 loaded from cache" or "Loading Model-1"  │
│                                                                  │
│    4.2 Convert image to tensor                                   │
│        └─ Log image dimensions and conversion steps              │
│                                                                  │
│    4.3 Run inference                                             │
│        └─ Output: (is_valid: bool, confidence: float)            │
│                                                                  │
│    4.4 Decision: Valid?                                          │
│        ├─ YES → Continue to Model-2                              │
│        └─ NO  → Return REJECTED                                  │
└────────┬──────────────────────────────────────────────────────────┘
         │ (if passed)
┌────────▼──────────────────────────────────────────────────────────┐
│ 5. Model-2 Inference (Classification)                            │
│                                                                  │
│    5.1 Load Model (cached if already loaded)                     │
│        └─ Log: "Model-2 loaded from cache" or "Loading Model-2"  │
│                                                                  │
│    5.2 Run YOLO inference                                        │
│        └─ Output: List of detected objects with confidence       │
│                                                                  │
│    5.3 Parse detections                                          │
│        └─ Extract confidence for each class:                     │
│           • brake_rod, rod_nose, crane, tank                     │
│                                                                  │
│    5.4 Determine wagon side                                      │
│        └─ Heuristic or detection-based logic                     │
└────────┬──────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│ 6. Create Database Document                                      │
│    PhotoDocument:                                                │
│    - wagon_id                                                    │
│    - file_hash (filename)                                        │
│    - camera_id                                                   │
│    - side (left/right)                                           │
│    - confidence                                                  │
│    - features (brake_rod, rod_nose, crane, tank)                 │
│    - processed_at (timestamp)                                    │
│    - batch_id (if from batch)                                    │
└────────┬──────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│ 7. Store in MongoDB                                              │
│    - Insert PhotoDocument                                        │
│    - Get auto-generated ID                                       │
│    - Log: "Saved to MongoDB: photo_id"                            │
└────────┬──────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│ 8. Return Response                                               │
│    {                                                             │
│      "status": "accepted" | "rejected" | "error",                │
│      "message": str,                                             │
│      "data": {                                                   │
│        "photo_id": "507f1f77bcf86cd799439011",                   │
│        "side": "left",                                           │
│        "confidence": 0.87,                                       │
│        "features": {                                             │
│          "brake_rod": 0.95,                                      │
│          "rod_nose": 0.05,                                       │
│          "crane": 0.87,                                          │
│          "tank": 0.02                                            │
│        }                                                         │
│      }                                                           │
│    }                                                             │
└──────────────────────────────────────────────────────────────────┘
```

### Папка с фото (Batch Processing)

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. API Request: Batch Upload                                     │
│    POST /api/upload-folder                                       │
│    {                                                             │
│      "folder_path": "/path/to/PZD_test",                         │
│      "mapping": {                                                │
│        "2": {                                                    │
│          "photo_1.jpg": "wagon_12",                              │
│          "photo_2.jpg": "wagon_13"                               │
│        },                                                        │
│        "5": {...}                                                │
│      }                                                           │
│    }                                                             │
└────────┬──────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│ 2. Validate Folder                                               │
│    - Check folder exists                                         │
│    - Check it's a directory                                      │
│    - Log folder size and structure                               │
└────────┬──────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│ 3. Create Batch ID                                               │
│    Format: {folder_name}_{timestamp}                             │
│    Example: PZD_test_20260224_105230                              │
└────────┬──────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│ 4. Scan Folder Structure                                         │
│    /PZD_test/                                                    │
│      /2/                  (camera_id = 2)                        │
│        photo_1.jpg                                               │
│        photo_2.jpg                                               │
│      /5/                  (camera_id = 5)                        │
│        photo_3.jpg                                               │
│        photo_4.jpg                                               │
└────────┬──────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│ 5. For Each Photo                                                │
│    - Get file path                                               │
│    - Look up wagon_id in mapping[camera_id][filename]            │
│    - Read file bytes                                             │
│    - Process as single photo (steps 3-6 from single photo flow)  │
│    - Collect results per wagon                                   │
└────────┬──────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│ 6. Aggregate Results                                             │
│    For each wagon_id:                                           │
│    - Count left/right detections                                │
│    - Determine final side (majority vote)                        │
│    - List participating cameras                                 │
│    - Count processed photos                                     │
└────────┬──────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│ 7. Store Batch Results                                           │
│    BatchDocument:                                                │
│    - batch_id                                                   │
│    - folder name                                                │
│    - results (per wagon)                                        │
│    - total_photos, total_wagons, processed_count                │
│    - status (completed, error)                                  │
└────────┬──────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────┐
│ 8. Return Batch Response                                         │
│    {                                                             │
│      "batch_id": "PZD_test_20260224_105230",                      │
│      "folder": "PZD_test",                                       │
│      "total_wagons": 3,                                          │
│      "total_photos": 10,                                         │
│      "results": {                                                │
│        "wagon_12": {                                             │
│          "final_side": "left",                                   │
│          "left_count": 3,                                        │
│          "right_count": 0,                                       │
│          "cameras": [2, 5],                                      │
│          "photos_processed": 3                                   │
│        },                                                        │
│        ...                                                       │
│      },                                                          │
│      "status": "completed"                                       │
│    }                                                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Входные и выходные данные

### Request: Single Photo Upload

```bash
POST /api/upload
Content-Type: multipart/form-data

wagon_id: wagon_12
camera_id: 2
file: <binary image data>
```

**Параметры**:
| Параметр | Тип | Описание | Примеры |
|----------|-----|---------|---------|
| `wagon_id` | string | Идентификатор вагона | "wagon_12", "wagon_001" |
| `camera_id` | integer | ID камеры | 1, 2, 3, 5, 10 |
| `file` | file | Изображение (JPG, PNG, BMP, GIF) | photo.jpg |

### Request: Batch Folder Upload

```json
POST /api/upload-folder
Content-Type: application/json

{
  "folder_path": "/path/to/PZD_test",
  "mapping": {
    "2": {
      "photo_1.jpg": "wagon_12",
      "photo_2.jpg": "wagon_13",
      "photo_3.jpg": "wagon_12"
    },
    "5": {
      "photo_4.jpg": "wagon_14",
      "photo_5.jpg": "wagon_13"
    }
  }
}
```

### Response: Success (Accepted)

```json
HTTP 200 OK
{
  "status": "accepted",
  "message": "Photo processed successfully",
  "data": {
    "photo_id": "507f1f77bcf86cd799439011",
    "side": "left",
    "confidence": 0.87,
    "features": {
      "brake_rod": 0.95,
      "rod_nose": 0.05,
      "crane": 0.87,
      "tank": 0.02
    }
  }
}
```

### Response: Rejected (Poor Quality)

```json
HTTP 200 OK
{
  "status": "rejected",
  "message": "Photo doesn't meet quality criteria",
  "data": null
}
```

### Response: Error

```json
HTTP 500 Internal Server Error
{
  "status": "error",
  "message": "Processing error: Model inference failed",
  "data": null
}
```

### Response: Batch Upload

```json
HTTP 200 OK
{
  "batch_id": "PZD_test_20260224_105230",
  "folder": "PZD_test",
  "total_wagons": 3,
  "total_photos": 10,
  "results": {
    "wagon_12": {
      "final_side": "left",
      "left_count": 3,
      "right_count": 0,
      "cameras": [2, 5],
      "photos_processed": 3
    },
    "wagon_13": {
      "final_side": "right",
      "left_count": 0,
      "right_count": 2,
      "cameras": [2, 5],
      "photos_processed": 2
    },
    "wagon_14": {
      "final_side": "left",
      "left_count": 1,
      "right_count": 0,
      "cameras": [5],
      "photos_processed": 1
    }
  },
  "processed_at": "2026-02-24T10:52:30.123456",
  "status": "completed"
}
```

---

## Структура папок

### Папка проекта

```
wagon_api/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Settings + validation
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic models
│   ├── ml_models/
│   │   ├── __init__.py
│   │   └── models.py            # Deprecated (for reference)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── wagon.py             # Main endpoints
│   │   └── health.py            # Health check
│   └── services/
│       ├── __init__.py
│       ├── model_loader.py      # Model loading service
│       └── processor.py         # Core processing logic
├── .env                         # Configuration (model paths, etc)
├── .env.example                 # Example configuration
├── run.py                       # Dev server entry point
├── test_api.py                  # API integration tests
├── test_batch_api.py            # Batch processing tests
├── test_models.py               # Model integration tests
├── requirements.txt             # Dependencies
├── PLANE.md                     # This file
├── QUICK_REFERENCE.md           # Quick start guide
├── REAL_MODELS.md               # Model details
└── IMPLEMENTATION_SUMMARY.md    # Technical details

../
├── model-1/
│   ├── mobilenet_wagon_best.pt  # Filter model
│   ├── train.py
│   ├── wagon.yaml
│   └── dataset/
├── model-2/
│   ├── Data/
│   │   ├── best.pt              # Classification model
│   │   ├── last.pt
│   │   └── images/
│   ├── main.py
│   ├── requirements.txt
│   └── README.txt
```

### Папка с фото для обработки

```
PZD_test/                         # Root folder
├── 2/                            # Camera ID = 2
│   ├── 1747758424903_uuid1.jpg
│   ├── 1747758424923_uuid2.jpg
│   ├── 1747758424945_uuid3.jpg
│   └── ...
├── 5/                            # Camera ID = 5
│   ├── 1747758424967_uuid4.jpg
│   ├── 1747758424989_uuid5.jpg
│   └── ...
└── 7/                            # Camera ID = 7
    ├── 1747758425011_uuid6.jpg
    └── ...

Условие: Каждый файл должен быть сопоставлен в mapping!

mapping = {
  "2": {
    "1747758424903_uuid1.jpg": "wagon_12",
    "1747758424923_uuid2.jpg": "wagon_13",
    ...
  },
  "5": {
    "1747758424967_uuid4.jpg": "wagon_12",
    ...
  },
  ...
}
```

---

## Как работают модели

### Model-1: Фильтра качества

**Архитектура**: MobileNet (PyTorch)

**Процесс**:
```
1. Вход: PIL Image (RGB)
   └─ Размер: любой (автоматически масштабируется)
   
2. Предпроцессинг:
   └─ Конвертация в RGB (если需要)
   └─ Нормализация: pixel values → [0, 1]
   └─ Resize: к размеру модели (обычно 224×224)
   └─ Permute: (H, W, C) → (C, H, W)
   └─ Batch dimension: (C, H, W) → (1, C, H, W)
   
3. Forward pass:
   └─ model(tensor)
   
4. Постпроцессинг:
   └─ Softmax если 2 класса
   └─ Берем вероятность позитивного класса (suitable)
   
5. Выход:
   ├─ is_valid: True если confidence > threshold (0.5)
   └─ confidence: float [0.0, 1.0]
```

**Threshold**: 0.5 (можно настроить в коде)

**Интерпретация**:
- `is_valid=True, confidence=0.95` → Фото отличного качества
- `is_valid=True, confidence=0.52` → Фото приемлемого качества
- `is_valid=False, confidence=0.3` → Фото плохого качества (отклонено)

### Model-2: Классификация

**Архитектура**: YOLOv11 Object Detector

**Процесс**:
```
1. Вход: PIL Image (RGB)
   └─ Размер: любой (обычно ≥640×640)
   
2. Предпроцессинг (выполняется YOLOv11 автоматически):
   └─ Конвертация в RGB
   └─ Нормализация
   └─ Resize: безопасный
   
3. Forward pass:
   └─ model(image)
   
4. Postprocessing (выполняется YOLOv11 автоматически):
   └─ NMS (Non-Maximum Suppression)
   └─ Фильтрация по confidence (default 0.25)
   
5. Выход:
   ├─ results list
   ├─ каждый результат содержит:
   │  ├─ boxes: array of detections
   │  ├─ cls: class IDs
   │  ├─ conf: confidence scores
   │  └─ names: class names mapping
   
6. Наша обработка:
   ├─ Перебираем все detections
   ├─ Извлекаем class_name и confidence
   ├─ Собираем max confidence для каждого класса:
   │  ├─ brake_rod: float
   │  ├─ rod_nose: float
   │  ├─ crane: float
   │  └─ tank: float
   └─ Определяем side (left/right) - логика в коде
```

**Классы, которые детектирует**:
- `brake_rod` - Тормозной стержень
- `rod_nose` - Нос стержня
- `crane` - Кран
- `tank` - Резервуар/цистерна

**Определение стороны**:
Текущая логика: Если есть детекции → "left", иначе используется heuristic

### Fallback Режим

Если модель не загрузилась (файл не найден, не установлены зависимости):

**Model-1 fallback**:
```python
Model1Output(is_valid=True, confidence=0.5)
# Все фото пропускаются
```

**Model-2 fallback**:
```python
Model2Output(
    brake_rod=0.5,
    rod_nose=0.5,
    crane=0.5,
    tank=0.5,
    side="left",
    confidence=0.5
)
# Возвращаются нейтральные значения
```

---

## API Endpoints

### 1. Health Check

```bash
GET /health
```

**Response**: Просто проверка, что API живой

---

### 2. Single Photo Upload

```bash
POST /api/upload?wagon_id=wagon_12&camera_id=2
Content-Type: multipart/form-data

file: <image>
```

**Возвращает**: `PhotoUploadResponse`

---

### 3. Batch Folder Upload

```bash
POST /api/upload-folder
Content-Type: application/json

{
  "folder_path": "/path/to/PZD_test",
  "mapping": {
    "2": {
      "filename1.jpg": "wagon_12",
      ...
    }
  }
}
```

**Возвращает**: `BatchResultResponse`

---

### 4. Get Wagon Result

```bash
GET /api/wagon/{wagon_id}
```

Пример: `GET /api/wagon/wagon_12`

**Возвращает**: Финальный результат для вагона (side + статистика)

---

### 5. Get Batch Status

```bash
GET /api/batch-status/{batch_id}
```

Пример: `GET /api/batch-status/PZD_test_20260224_105230`

**Возвращает**: Статус обработки batch (processing/completed/error)

---

### 6. Get Batch Result

```bash
GET /api/batch-result/{batch_id}
```

Пример: `GET /api/batch-result/PZD_test_20260224_105230`

**Возвращает**: Полный результат batch со всеми вагонами

---

## Логирование и отладка

### Уровни логирования

```
DEBUG (10)   - Детальная информация
  Examples:
  - Image sizes, tensor shapes
  - Model paths and loading steps
  - Each detection in YOLO
  - File operations

INFO (20)    - Основная информация
  Examples:
  - "Model-1 loaded successfully"
  - "Photo passed filter model"
  - "Classification complete"
  - "PhotoDocument saved"

WARNING (30) - Предупреждения
  Examples:
  - "Model not found, using fallback"
  - "Model path not readable"

ERROR (40)   - Ошибки
  Examples:
  - "Failed to load model"
  - "Image processing error"
  - "Model inference failed"

CRITICAL (50) - Критические ошибки
  (не используется в текущем коде)
```

### Как смотреть логи

#### Файл конфигурации (`.env`):
```dotenv
LOG_LEVEL=DEBUG    # Для разработки
LOG_LEVEL=INFO     # Для production
```

#### Запуск с логированием:

```bash
# Разработка (DEBUG)
export LOG_LEVEL=DEBUG
python3 -m uvicorn app.main:app --reload

# Production (INFO)
LOG_LEVEL=INFO python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### Пример логов при обработке фото:

```
2026-02-24 10:52:30,123 - INFO - ▶️ Processing photo: wagon_id=wagon_12, camera_id=2, file_hash=photo.jpg
2026-02-24 10:52:30,124 - DEBUG - Model path resolved to: /home/max_jalo/ml_project/wagon_api/../model-1/mobilenet_wagon_best.pt
2026-02-24 10:52:30,125 - DEBUG - Loading image from bytes (125000 bytes)
2026-02-24 10:52:30,126 - INFO - Image loaded: size=(1024, 768), mode=RGB, format=JPEG
2026-02-24 10:52:30,127 - INFO - 📍 Step 1: Filter model (Model-1)
2026-02-24 10:52:30,128 - DEBUG - ▶️ Starting Model-1 inference
2026-02-24 10:52:30,129 - DEBUG - Image size: (1024, 768), Mode: RGB, Format: JPEG
2026-02-24 10:52:30,130 - DEBUG - 🔄 Initializing Model-1 loader from ../model-1/mobilenet_wagon_best.pt
2026-02-24 10:52:30,131 - DEBUG - 📦 Loading model from cache: ../model-1/mobilenet_wagon_best.pt
2026-02-24 10:52:30,132 - DEBUG - Converting image to RGB...
2026-02-24 10:52:30,133 - DEBUG - Converting image to tensor...
2026-02-24 10:52:30,134 - DEBUG - Tensor shape before permute: torch.Size([1024, 768, 3])
2026-02-24 10:52:30,135 - DEBUG - Tensor shape for inference: torch.Size([1, 3, 1024, 768])
2026-02-24 10:52:30,136 - DEBUG - Running Model-1 inference...
2026-02-24 10:52:30,456 - DEBUG - Model output type: <class 'torch.Tensor'>, shape: torch.Size([1, 2])
2026-02-24 10:52:30,457 - DEBUG - Binary classification: is_valid=True, confidence=0.9521
2026-02-24 10:52:30,458 - INFO - ✅ Model-1 result: valid=True, confidence=0.9521
2026-02-24 10:52:30,459 - INFO - ✅ Photo photo.jpg passed filter model (confidence: 0.9521)
2026-02-24 10:52:30,460 - INFO - 📍 Step 2: Classification model (Model-2)
2026-02-24 10:52:30,461 - DEBUG - ▶️ Starting Model-2 inference
2026-02-24 10:52:30,462 - DEBUG - Image size: (1024, 768), Mode: RGB, Format: JPEG
2026-02-24 10:52:30,463 - DEBUG - 🔄 Initializing Model-2 loader from ../model-2/Data/best.pt
2026-02-24 10:52:30,465 - DEBUG - 📦 Loading model from cache: ../model-2/Data/best.pt
2026-02-24 10:52:30,466 - DEBUG - Converting image to RGB...
2026-02-24 10:52:30,467 - DEBUG - Running YOLO inference...
2026-02-24 10:52:30,789 - DEBUG - YOLO returned 1 result(s)
2026-02-24 10:52:30,790 - DEBUG - Processing result 0
2026-02-24 10:52:30,791 - DEBUG - Found 4 bounding box(es)
2026-02-24 10:52:30,792 - DEBUG - Box 0: brake_rod (confidence: 0.9521)
2026-02-24 10:52:30,793 - DEBUG - Box 1: tank (confidence: 0.0234)
2026-02-24 10:52:30,794 - DEBUG - Box 2: crane (confidence: 0.8765)
2026-02-24 10:52:30,795 - DEBUG - Box 3: rod_nose (confidence: 0.0567)
2026-02-24 10:52:30,796 - DEBUG - Total detections: 4
2026-02-24 10:52:30,797 - DEBUG - Detected features: {'brake_rod': 0.9521, 'rod_nose': 0.0567, 'crane': 0.8765, 'tank': 0.0234}
2026-02-24 10:52:30,798 - DEBUG - Determining wagon side
2026-02-24 10:52:30,799 - DEBUG - Detected classes found, defaulting to 'left'
2026-02-24 10:52:30,800 - DEBUG - Determined side: left
2026-02-24 10:52:30,801 - INFO - ✅ Model-2 result: side=left, confidence=0.4772, features={'brake_rod': 0.9521, 'rod_nose': 0.0567, 'crane': 0.8765, 'tank': 0.0234}
2026-02-24 10:52:30,802 - INFO - Creating PhotoDocument for MongoDB
2026-02-24 10:52:30,803 - DEBUG - PhotoDocument saved: 507f1f77bcf86cd799439011
2026-02-24 10:52:30,804 - INFO - ✅ Successfully processed photo photo.jpg: side=left, confidence=0.4772
```

### Отладка ошибок

**Ошибка**: `Model file not found: ../model-1/mobilenet_wagon_best.pt`

**Решение**:
```
1. Проверить путь в .env
2. Проверить что файл существует:
   ls -la ../model-1/mobilenet_wagon_best.pt
3. Проверить права на чтение:
   file ../model-1/mobilenet_wagon_best.pt
```

**Ошибка**: `PyTorch not installed`

**Решение**:
```bash
pip install torch ultralytics
```

**Ошибка**: `Image processing error: cannot identify image file`

**Решение**:
```
1. Проверить что файл - это валидное изображение
2. Проверить поддерживаемые форматы: JPG, PNG, BMP, GIF
3. Смотреть логи DEBUG для размера файла
```

---

## Возможные ошибки и их решения

### Ошибки загрузки моделей

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `Model file not found: ../model-1/mobilenet_wagon_best.pt` | Файл не существует | Проверить путь в .env, убедиться что файл скопирован |
| `PyTorch not installed` | Зависимость не установлена | `pip install torch ultralytics` |
| `YOLO not available, loading as generic torch model` | Ultralytics не установлен | `pip install ultralytics` |
| `Failed to load Keras model` | TensorFlow не установлен | `pip install tensorflow` |

### Ошибки при обработке фото

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `cannot identify image file` | Файл не изображение | Проверить формат (JPG, PNG, BMP, GIF) |
| `Image size too large` | Память исчерпана | Уменьшить разрешение или размер batch |
| `Model inference timed out` | Слишком долгий инференс | Увеличить MODEL_TIMEOUT в .env |
| `Tensor shape mismatch` | Неправильный размер входа | Автоматически переводится, проверить логи |

### Ошибки MongoDB

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `Connection refused` | MongoDB не запущена | `mongod` или проверить MONGODB_URL |
| `Authentication failed` | Неверные credentials | Проверить MONGODB_URL в .env |
| `Database disconnected` | Сетевая ошибка | Перезагрузить MongoDB или проверить сеть |

### Ошибки API

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `405 Method Not Allowed` | Неверный HTTP метод | Использовать POST для upload, GET для query |
| `400 Bad Request` | Неверные параметры | Проверить формат JSON или multipart data |
| `413 Payload Too Large` | Файл больше 10MB | Сжать изображение или увеличить MAX_UPLOAD_SIZE |
| `422 Unprocessable Entity` | Неверный тип данных | Проверить типы параметров (int для camera_id, string для wagon_id) |

---

## Примеры запросов

### Пример 1: Загрузить одно фото

```bash
#!/bin/bash

# Установить переменные
WAGON_ID="wagon_12"
CAMERA_ID="2"
IMAGE_FILE="./test_wagon.jpg"
API_URL="http://localhost:8000"

# Отправить запрос
curl -X POST "${API_URL}/api/upload" \
  -F "wagon_id=${WAGON_ID}" \
  -F "camera_id=${CAMERA_ID}" \
  -F "file=@${IMAGE_FILE}" \
  | jq .
```

**Ожидаемый результат**:
```json
{
  "status": "accepted",
  "message": "Photo processed successfully",
  "data": {
    "photo_id": "507f1f77bcf86cd799439011",
    "side": "left",
    "confidence": 0.87,
    "features": {
      "brake_rod": 0.95,
      "rod_nose": 0.05,
      "crane": 0.87,
      "tank": 0.02
    }
  }
}
```

### Пример 2: Загрузить папку с фото

```bash
#!/bin/bash

API_URL="http://localhost:8000"
FOLDER_PATH="/path/to/PZD_test"

# Создать JSON с mapping
MAPPING=$(cat <<'EOF'
{
  "folder_path": "/path/to/PZD_test",
  "mapping": {
    "2": {
      "photo1_uuid.jpg": "wagon_12",
      "photo2_uuid.jpg": "wagon_13",
      "photo3_uuid.jpg": "wagon_12"
    },
    "5": {
      "photo4_uuid.jpg": "wagon_14",
      "photo5_uuid.jpg": "wagon_13"
    }
  }
}
EOF
)

# Отправить запрос
curl -X POST "${API_URL}/api/upload-folder" \
  -H "Content-Type: application/json" \
  -d "${MAPPING}" \
  | jq .
```

### Пример 3: Python скрипт для обработки

```python
import requests
import json
from pathlib import Path

API_URL = "http://localhost:8000"

# Single photo
def upload_photo(wagon_id, camera_id, image_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        params = {'wagon_id': wagon_id, 'camera_id': camera_id}
        resp = requests.post(f"{API_URL}/api/upload", 
                            files=files, params=params)
    return resp.json()

# Batch processing
def upload_folder(folder_path, mapping):
    payload = {
        "folder_path": str(folder_path),
        "mapping": mapping
    }
    resp = requests.post(f"{API_URL}/api/upload-folder",
                        json=payload)
    return resp.json()

# Query results
def get_wagon_result(wagon_id):
    resp = requests.get(f"{API_URL}/api/wagon/{wagon_id}")
    return resp.json()

# Stream photos
def process_folder(folder_path, mapping):
    result = upload_folder(folder_path, mapping)
    batch_id = result['batch_id']
    
    # Wait for processing
    while True:
        status = get_batch_status(batch_id)
        if status['status'] == 'completed':
            break
        time.sleep(1)
    
    return get_batch_result(batch_id)

# Usage
result = upload_photo("wagon_12", 2, "./test.jpg")
print(json.dumps(result, indent=2))
```

### Пример 4: С помощью Python requests (одна фото)

```python
import requests

image_path = "/path/to/photo.jpg"
wagon_id = "wagon_12"
camera_id = 2

with open(image_path, 'rb') as f:
    files = {'file': (image_path, f)}
    params = {
        'wagon_id': wagon_id,
        'camera_id': camera_id
    }
    
    response = requests.post(
        "http://localhost:8000/api/upload",
        files=files,
        params=params
    )

print(response.json())
```

### Пример 5: Monitoring с логированием

```bash
#!/bin/bash

# Запустить API с логированием
export LOG_LEVEL=DEBUG
python3 -m uvicorn app.main:app --reload 2>&1 | tee api.log

# В другом терминале - смотреть логи real-time
tail -f api.log | grep -E "(ERROR|WARNING|INFO|✅|❌)"
```

---

## Рекомендации по использованию

### Development
```bash
# Выставить DEBUG логирование
export LOG_LEVEL=DEBUG

# Запустить с автоперезагрузкой
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Смотреть логи
tail -f api.log
```

### Production
```bash
# Использовать INFO логирование
export LOG_LEVEL=INFO

# Запустить с gunicorn (или в Docker)
gunicorn \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  app.main:app

# Используть systemd или supervisor для управления
```

### Testing
```bash
# Unit tests
python3 test_models.py

# API tests
python3 test_api.py
python3 test_batch_api.py

# Integration tests
# Запустить API, затем:
./run_integration_tests.sh
```

---

## Заключение

Эта архитектура обеспечивает:

✅ **Надежность**: 
  - Fallback режимы если модели не загруженhty
  - Comprehensive error handling
  - Graceful degradation

✅ **Производительность**:
  - Model caching (загрузка один раз)
  - Async processing
  - Batch operations

✅ **Отладка**:
  - Многоуровневое логирование
  - Детальные логи на каждом шаге
  - Трассировка ошибок

✅ **Масштабируемость**:
  - MongoDB для масштабируемого хранилища
  - REST API для легкой интеграции
  - Async поддержка для большого количества запросов

Для дополнительной информации см.:
- `QUICK_REFERENCE.md` - Быстрый старт
- `REAL_MODELS.md` - Детали моделей
- `IMPLEMENTATION_SUMMARY.md` - Технические детали
