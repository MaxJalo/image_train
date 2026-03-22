<!-- 🔄 ML Pipeline usage documentation and examples -->
# Использование новых функций ML Pipeline

## Быстрый старт

### 1. Полная обработка папки (все 3 этапа)

```python
from app.services.processor import process_folder

# Обработать всю папку с трехэтапным pipeline
batch_id, result = await process_folder(
    folder_path="/path/to/folder",
    mapping=None  # или {"2": {"wagon_1_001.jpg": "wagon_1"}, ...}
)

# Результат содержит
print(result["summary"])
# {
#     "total_photos": 25,
#     "accepted_photos": 20,      # После ЭТАПА 1
#     "rejected_photos": 5,        # Отбраковано
#     "processed_photos": 20,      # Обработано Model-2
#     "total_wagons": 2,
#     "stages": {
#         "stage1_filter": "Фото отфильтровано Model-1: 20/25",
#         "stage2_classify": "Фото обработано Model-2 по вагонам: 20",
#         "stage3_aggregate": "Агрегация завершена для 2 вагонов"
#     }
# }

print(result["results"])
# {
#     "wagon_1": {
#         "total_photos": 12,
#         "processed_photos": 12,
#         "left_count": 7,
#         "right_count": 5,
#         "final_side": "left",
#         "cameras": [2, 3, 5]
#     },
#     ...
# }
```

---

## Раздельные функции для кастомных сценариев

### 2. Только ЭТАП 1 - Фильтрация

```python
from app.services.processor import filter_photo

# Проверить, прошло ли фото фильтр Model-1
passed, confidence = await filter_photo(
    image_bytes=image_data,
    camera_id=2,
    file_hash="wagon_1_001.jpg"
)

if passed:
    print(f"✅ Фото принято (confidence={confidence:.4f})")
else:
    print(f"🚫 Фото отбраковано (confidence={confidence:.4f})")
```

**Когда использовать:**
- Нужно отфильтровать фото перед обработкой
- Нужно построить пользовательский pipeline
- Нужна кастомная группировка по wagon_id

---

### 3. Только ЭТАП 2 - Классификация

```python
from app.services.processor import classify_photo

# Классифицировать фото через Model-2
features = await classify_photo(
    image_bytes=image_data,
    camera_id=2,
    file_hash="wagon_1_001.jpg",
    wagon_id="wagon_1"
)

print(features)
# {
#     "brake_rod": 0.85,
#     "rod_nose": 0.92,
#     "crane": 0.0,
#     "tank": 0.55,
#     "side": "left",           # Определена сторона
#     "confidence": 0.75
# }

# Использовать для агрегации
if features.side == "left":
    left_count += 1
else:
    right_count += 1
```

**Когда использовать:**
- Фото уже прошло фильтр (или это обязательно "хорошее" фото)
- Нужна только классификация
- Нужна информация о признаках и стороне

---

### 4. Быстрая обработка одного фото

```python
from app.services.processor import process_photo

# Для отдельно загруженных фото (пропускает ЭТАП 1)
result = await process_photo(
    image_bytes=image_data,
    camera_id=2,
    file_hash="wagon_1_001.jpg",
    wagon_id="wagon_1",  # Должен быть известен!
    batch_id="batch_123"
)

print(result)
# {
#     "status": "accepted",
#     "message": "Photo processed successfully",
#     "data": {
#         "photo_id": "...",
#         "wagon_id": "wagon_1",
#         "camera_id": 2,
#         "side": "left",
#         "confidence": 0.85,
#         "features": {...}
#     }
# }
```

**Когда использовать:**
- Загрузка отдельного фото через API `/upload`
- wagon_id уже определен (известен)
- Не нужна фильтрация или группировка

---

## Сценарии использования

### Сценарий 1: Обработка папки из веб-интерфейса

```python
from app.services.processor import process_folder

@router.post("/api/upload-folder")
async def upload_folder(request: FolderUploadRequest):
    try:
        batch_id, result = await process_folder(
            request.folder_path, 
            request.mapping
        )
        
        return BatchResultResponse(
            batch_id=batch_id,
            folder=result["folder"],
            summary=result["summary"],
            results=result["results"],
            status="completed"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

### Сценарий 2: Кастомная обработка с контролем этапов

```python
from app.services.processor import filter_photo, classify_photo

async def custom_process_folder(folder_path: str):
    """Кастомный pipeline с дополнительной обработкой."""
    
    from pathlib import Path
    from PIL import Image
    from io import BytesIO
    
    # ЭТАП 1: Фильтрация и группировка
    wagon_photos = {}
    
    for file_path in Path(folder_path).rglob("*/*.jpg"):
        camera_id = int(file_path.parent.name)  # "camera_2" → 2
        
        with open(file_path, "rb") as f:
            image_bytes = f.read()
        
        # Фильтруем через Model-1
        passed, confidence = await filter_photo(
            image_bytes, 
            camera_id, 
            file_path.name
        )
        
        if not passed:
            print(f"❌ {file_path.name} не прошел фильтр")
            continue
        
        # Определяем wagon_id из имени файла
        wagon_id = extract_wagon_id_from_filename(file_path.name)
        
        if wagon_id not in wagon_photos:
            wagon_photos[wagon_id] = []
        
        wagon_photos[wagon_id].append((file_path, camera_id, image_bytes))
    
    # ЭТАП 2: Классификация с кастомной логикой
    results = {}
    
    for wagon_id, photos in wagon_photos.items():
        wagon_result = {
            "left": [],
            "right": [],
        }
        
        for file_path, camera_id, image_bytes in photos:
            features = await classify_photo(
                image_bytes,
                camera_id,
                file_path.name,
                wagon_id
            )
            
            # Кастомная обработка
            if features.confidence > 0.8:  # Только высокая уверенность
                if features.side == "left":
                    wagon_result["left"].append(features)
                else:
                    wagon_result["right"].append(features)
        
        # ЭТАП 3: Кастомная агрегация
        results[wagon_id] = {
            "final_side": "left" if len(wagon_result["left"]) > len(wagon_result["right"]) else "right",
            "confidence_scores": {
                "left": [f.confidence for f in wagon_result["left"]],
                "right": [f.confidence for f in wagon_result["right"]]
            }
        }
    
    return results
```

---

### Сценарий 3: Batch обработка в фоновой задаче

```python
from app.services.processor import process_folder
from celery import shared_task

@shared_task
def process_folder_task(folder_path: str, mapping: dict = None):
    """Фоновая задача обработки папки."""
    import asyncio
    
    async def run():
        batch_id, result = await process_folder(folder_path, mapping)
        
        # Сохраняем результаты в кеш или БД
        cache.set(f"batch:{batch_id}", result, timeout=3600)
        
        # Отправляем уведомление
        notify_user(f"Batch {batch_id} completed", result["summary"])
        
        return batch_id
    
    return asyncio.run(run())

# Запуск в background
process_folder_task.delay(folder_path, mapping)
```

---

## Обработка ошибок

### Обработка фильтрации

```python
from app.services.processor import filter_photo

try:
    passed, confidence = await filter_photo(image_bytes, camera_id, filename)
    
    if not passed:
        # Фото не прошло фильтр - это нормально
        logger.info(f"Photo {filename} rejected by filter")
        handle_rejected_photo(filename, confidence)
    else:
        # Фото прошло фильтр
        process_accepted_photo(filename)
        
except Exception as e:
    # Ошибка при обработке (не связана с фильтром)
    logger.error(f"Error filtering {filename}: {e}")
    handle_processing_error(filename, e)
```

### Обработка классификации

```python
from app.services.processor import classify_photo

try:
    features = await classify_photo(image_bytes, camera_id, filename, wagon_id)
    
    if features.confidence < 0.5:
        # Низкая уверенность - можно игнорировать
        logger.warning(f"Low confidence for {filename}: {features.confidence}")
    
    # Использовать features
    use_classification(wagon_id, features)
    
except Exception as e:
    # Ошибка при классификации
    logger.error(f"Error classifying {filename}: {e}")
    handle_classification_error(filename, e)
```

---

## Performance Optimization

### Параллельная обработка ЭТАПА 2

```python
import asyncio
from app.services.processor import classify_photo

async def parallel_classify_wagon(wagon_id: str, photos: list):
    """Классифицировать фото вагона параллельно."""
    
    tasks = [
        classify_photo(image_bytes, camera_id, filename, wagon_id)
        for file_path, camera_id, filename, image_bytes in photos
    ]
    
    # Обрабатываем до 10 фото одновременно
    results = []
    for i in range(0, len(tasks), 10):
        batch = tasks[i:i+10]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
    
    return results
```

---

## Логирование и мониторинг

### Трассировка обработки

```python
from app.services.processor import process_folder
import logging

logger = logging.getLogger(__name__)

async def monitored_process_folder(folder_path: str):
    """Обработка с детальным логированием."""
    
    logger.info(f"🚀 Starting folder processing: {folder_path}")
    
    try:
        batch_id, result = await process_folder(folder_path)
        
        summary = result["summary"]
        logger.info(f"✅ Processing completed")
        logger.info(f"   Total photos: {summary['total_photos']}")
        logger.info(f"   Accepted: {summary['accepted_photos']}")
        logger.info(f"   Rejected: {summary['rejected_photos']}")
        logger.info(f"   Wagons: {summary['total_wagons']}")
        
        for stage, desc in summary["stages"].items():
            logger.info(f"   {stage}: {desc}")
        
        return batch_id, result
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        raise
```

---

## Структура данных Model2Output

```python
from app.models.schemas import Model2Output

features: Model2Output = await classify_photo(...)

# Доступные поля:
print(features.brake_rod)      # float 0-1
print(features.rod_nose)       # float 0-1
print(features.crane)          # float 0-1
print(features.tank)           # float 0-1
print(features.side)           # "left" или "right"
print(features.confidence)     # float 0-1 (overall confidence)

# Использование для агрегации
left_presence = features.brake_rod > 0.5  # Наличие детали слева?
side_vote = "left" if features.side == "left" else "right"
```

---

## Миграция со старого кода

### Старый код
```python
# Раньше нужно было обрабатывать все этапы вместе
result = await process_something(...)
```

### Новый код (полностью совместимый)
```python
# Работает как раньше
batch_id, result = await process_folder(folder_path)
wagons = result["wagons"]  # Или result["results"]
```

### Или используйте новые функции
```python
# Финтовка - только Model-1
passed, conf = await filter_photo(...)

# Классификация - только Model-2
features = await classify_photo(...)

# Полная обработка папки
batch_id, result = await process_folder(...)
```

---

## FAQ

**Q: Зачем разделять filter_photo и classify_photo?**  
A: Чтобы не тратить ресурсы Model-2 на фото, которые не прошли фильтр Model-1.

**Q: Что если wagon_id не определен?**  
A: Фото будет отбраковано на ЭТАПЕ 1 с ошибкой "wagon_id_error".

**Q: Где хранятся результаты?**  
A: В MongoDB (PhotoDocument, WagonAggregateDocument) и в папке session (файлы).

**Q: Можно ли пропустить фильтрацию?**  
A: Да, используйте `process_photo()` для отдельных фото с известным wagon_id.

**Q: Как отслеживать прогресс?**  
A: Логи в `/var/log/app.log` + batch статус в БД (BatchDocument).
