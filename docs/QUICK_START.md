# Wagon Classification API - Quick Start Guide

## 1. Быстрый старт (5 минут)

### Требования
- Python 3.10+
- MongoDB (локально или удаленно)
- Интернет (для установки зависимостей)

### Установка и запуск

```bash
# 1. Перейти в папку проекта
cd /home/max_jalo/ml_project

# 2. Активировать виртуальное окружение
source ml_env/bin/activate

# 3. Установить зависимости
pip install -r wagon_api/requirements.txt

# 4. Убедиться, что MongoDB запущена
# Linux:
sudo systemctl status mongod
# Если не запущена:
sudo systemctl start mongod

# 5. Запустить API
cd wagon_api
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API будет доступен по адресу: `http://localhost:8000`

## 2. API Документация

### Swagger UI (интерактивная документация)
```
http://localhost:8000/docs
```

### ReDoc (альтернативная документация)
```
http://localhost:8000/redoc
```

## 3. Основные эндпоинты

### Главный эндпоинт: Обработка папки с фотографиями

```bash
POST /api/upload-folder
```

**Payload (JSON):**
```json
{
  "folder_path": "/absolute/path/to/PZD_test"
}
```

**Структура папки:**
```
PZD_test/
├── camera_2/
│   ├── wagon_12_001.jpg
│   ├── wagon_12_002.jpg
│   └── wagon_13_001.jpg
├── camera_5/
│   ├── wagon_12_003.jpg
│   └── wagon_13_002.jpg
└── camera_7/
    └── wagon_12_004.jpg
```

**Ответ:**
```json
{
  "batch_id": "PZD_test_20240224_103000",
  "folder": "PZD_test",
  "total_wagons": 2,
  "total_photos": 6,
  "results": {
    "wagon_12": {
      "final_side": "right",
      "left_count": 2,
      "right_count": 2,
      "cameras": [2, 5, 7],
      "photos_processed": 4
    },
    "wagon_13": {
      "final_side": "left",
      "left_count": 2,
      "right_count": 0,
      "cameras": [2, 5],
      "photos_processed": 2
    }
  },
  "processed_at": "2024-02-24T10:30:00",
  "status": "completed"
}
```

### Другие эндпоинты

```bash
# Health check
GET /api/health

# Статус батча
GET /api/batch-status/{batch_id}

# Результаты батча
GET /api/batch-result/{batch_id}

# Информация по вагону
GET /api/wagon/{wagon_id}

# Статус вагона
GET /api/status/{wagon_id}

# Результат вагона
GET /api/result/{wagon_id}
```

## 4. Примеры использования

### Через curl

```bash
# Обработать папку
curl -X POST http://localhost:8000/api/upload-folder \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "/absolute/path/to/PZD_test"}' | jq .

# Получить результаты
curl http://localhost:8000/api/batch-result/PZD_test_20240224_103000 | jq .

# Получить информацию по вагону
curl http://localhost:8000/api/wagon/wagon_12 | jq .
```

### Через Python

```python
import requests
import json

# Обработать папку
response = requests.post(
    "http://localhost:8000/api/upload-folder",
    json={"folder_path": "/absolute/path/to/PZD_test"}
)
batch_id = response.json()["batch_id"]
print(f"Batch ID: {batch_id}")

# Получить результаты
response = requests.get(f"http://localhost:8000/api/batch-result/{batch_id}")
results = response.json()
print(json.dumps(results, indent=2, default=str))
```

## 5. Создание тестовой папки

Используйте скрипт для создания тестовой папки с примерами изображений:

```bash
cd wagon_api

# Создать тестовую папку с изображениями из dataset
./setup_test_folder.sh PZD_test ../model-1/dataset

# Запустить тесты
./batch_examples.sh $(pwd)/PZD_test
```

## 6. Логика обработки

1. **Сканирование**: Приложение сканирует папку `folder/{camera_id}/photo_files`
2. **Извлечение**: Из имени файла извлекается `wagon_id` (regex: `wagon_(\d+)`)
3. **Фильтрация**: Каждое фото проходит Model-1 (True/False - 50/50)
4. **Классификация**: Model-2 извлекает признаки и определяет сторону (left/right)
5. **Агрегация**: Для каждого вагона считается голосование (majority vote)
6. **Сохранение**: Результаты сохраняются в MongoDB

## 7. MongoDB

### Проверить подключение

```bash
# Подключиться к MongoDB
mongosh

# Выбрать базу
use wagon_classification

# Посмотреть коллекции
show collections

# Посчитать документы
db.photos.countDocuments()
db.batches.countDocuments()
db.wagon_aggregates.countDocuments()
```

### Структура данных

**photos** коллекция:
```json
{
  "wagon_id": "wagon_12",
  "camera_id": 2,
  "side": "left/right",
  "confidence": 0.95,
  "features": {...},
  "batch_id": "folder_20240224_103000",
  "processed_at": "2024-02-24T..."
}
```

**batches** коллекция:
```json
{
  "batch_id": "PZD_test_20240224_103000",
  "folder": "PZD_test",
  "results": {
    "wagon_12": {
      "final_side": "right",
      "left_count": 2,
      "right_count": 2,
      "cameras": [2, 5, 7],
      "photos_processed": 4
    }
  },
  "status": "completed"
}
```

## 8. Отладка

### Проверить logs приложения

```bash
# Если запущено с --reload, logs будут в терминале
# Для production просмотреть файл логов:
tail -f /var/log/wagon-api.log
```

### Common Errors

**"Folder not found"**
- Убедитесь, что путь абсолютный (начинается с `/`)
- Проверьте существование папки: `ls -la /path/to/folder`

**"Connection refused" для MongoDB**
- Проверьте, что MongoDB запущена: `sudo systemctl status mongod`
- Проверьте URL в `.env`: `MONGODB_URL=mongodb://localhost:27017`

**"No module named 'app'"**
- Убедитесь, что вы находитесь в папке `wagon_api`
- Активируйте виртуальное окружение: `source ../ml_env/bin/activate`

**API не отвечает**
- Проверьте, что сервер запущен на порту 8000
- Проверьте firewall: `sudo ufw allow 8000`

## 9. Полезные команды

```bash
# Проверить здоровье API
curl http://localhost:8000/api/health

# Просмотреть документацию
open http://localhost:8000/docs

# Проверить структуру папки перед обработкой
find /path/to/folder -type f | head -10

# Посчитать файлы по камерам
for dir in /path/to/folder/camera_*; do
  count=$(find "$dir" -type f | wc -l)
  echo "$(basename $dir): $count files"
done
```

## 10. Структура проекта

```
wagon_api/
├── app/
│   ├── main.py              # FastAPI приложение
│   ├── config.py            # Конфигурация
│   ├── models/
│   │   └── schemas.py       # Pydantic & Beanie модели
│   ├── ml_models/
│   │   └── models.py        # Mock ML модели
│   ├── services/
│   │   └── processor.py     # Логика обработки
│   └── routes/
│       ├── health.py        # Health endpoint
│       └── wagon.py         # Wagon endpoints
├── requirements.txt
├── .env                     # Конфигурация
├── README.md                # Полная документация
├── DEPLOYMENT.md            # Deployment guide
└── batch_examples.sh        # Примеры для batch
```

## 11. Production Deployment

Для развертывания на production сервере:

1. Установить Gunicorn
2. Настроить NGINX как reverse proxy
3. Использовать systemd service
4. Настроить SSL/TLS
5. Включить логирование и мониторинг

Смотрите `DEPLOYMENT.md` для полных инструкций.

## 12. Контакт и поддержка

- Документация: [README.md](README.md)
- Deployment гайд: [DEPLOYMENT.md](DEPLOYMENT.md)
- Swagger UI: http://localhost:8000/docs
