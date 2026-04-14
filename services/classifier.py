# (classifier)
import logging
import shutil
from collections import Counter, deque
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

from core.config import settings
from core.constants import CONFIDENCE_THRESHOLD, IMAGE_SIZE, MODEL1_BUFFER_SIZE
from services.model_loader import load_model

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger(__name__)

# Cache для модели Model-1
_model1_cache: Optional[Any] = None


def _get_model1():
    """Загрузить Model-1 с кешированием"""
    global _model1_cache
    if _model1_cache is None:
        logger.debug(f"🔄 Инициализация Model-1 из {settings.model1_path}")
        try:
            _model1_cache = load_model(settings.model1_path)
            logger.info("✅ Model-1 успешно загружена")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки Model-1: {str(e)}")
            logger.warning("⚠️ Model-1 используется в режиме заглушки")
            _model1_cache = "FALLBACK"
    return _model1_cache


def predict_model1(image: Image.Image) -> Tuple[bool, float]:
    logger.debug("▶️ Начало вывода Model-1")
    logger.debug(f"   Размер изображения: {image.size}, Режим: {image.mode}")

    try:
        model = _get_model1()

        # Проверка доступности модели
        if model == "FALLBACK":
            logger.warning("📌 Использование заглушки Model-1 (модель не загружена)")
            return True, 0.5

        # Конвертирование в RGB если нужно
        if image.mode != "RGB":
            image = image.convert("RGB")

        logger.debug(
            f"   Масштабирование к {IMAGE_SIZE}x{IMAGE_SIZE} и применение нормализации..."
        )

        # Применение трансформации
        transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        img_tensor = transform(image).unsqueeze(0)

        # Получаем устройство из модели
        params = model.parameters()
        if hasattr(params, "__iter__") and not hasattr(params, "__next__"):
            params = iter(params)
        device = next(params).device
        img_tensor = img_tensor.to(device)
        logger.debug(f"   Форма тензора: {img_tensor.shape}, устройство: {device}")

        logger.debug("   Запуск вывода Model-1...")
        # Запуск вывода
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)

        # Ожидаем выходной тензор с logits/probs размера [1, C]
        class_map = {0: "one_wagon", 1: "no_wagon"}
        if not isinstance(output, torch.Tensor):
            logger.warning("⚠️ Model-1 вернул не-tensor, используем заглушку")
            return True, 0.5

        # Softmax и проверка числа классов
        probs = torch.softmax(output, dim=1)

        pred_idx = int(probs.argmax(dim=1).item())
        class_name = class_map.get(pred_idx)
        confidence = float(probs[0, pred_idx].item())
        logger.debug(f"   Классификация: {class_name} (confidence={confidence:.4f})")
        logger.info(f"✅ Model-1 result: {class_name} (confidence={confidence:.4f})")
        return pred_idx == 0, confidence

    except Exception as e:
        logger.error(f"❌ Ошибка вывода Model-1: {type(e).__name__}: {str(e)}")
        logger.debug(f"   Traceback: {repr(e)}")
        return False, 0.0


async def classify_and_group_wagons(
    folder_path: str,
) -> Dict[str, List[Tuple[Path, int]]]:

    logger.info("🚂 Начало классификации и группирования по вагонам")
    logger.info(f"📂 Путь к папке: {folder_path}")

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder_path}")

    logger.debug(f"📁 Содержимое папки {folder}:")
    for item in sorted(folder.iterdir())[:10]:
        logger.debug(f"   {'📁' if item.is_dir() else '📄'} {item.name}")

    wagon_groups: Dict[str, List[Tuple[Path, int]]] = {}
    wagon_count = 1
    current_wagon_id = None
    current_wagon_photos: List[Tuple[Path, int]] = []

    # Буфер для сглаживания предсказаний (последние N)
    buffer: deque = deque(maxlen=MODEL1_BUFFER_SIZE)

    processed_count = 0
    rejected_count = 0

    # Найти все изображения рекурсивно и отсортировать по имени
    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(folder.rglob(pattern))
    image_files = sorted(list(set(image_files)))  # удалить дубликаты если есть

    logger.info(f"📸 Найдено {len(image_files)} изображений (рекурсивный поиск)")
    if image_files:
        logger.debug("   Примеры найденных файлов:")
        for img in image_files[:5]:
            logger.debug(f"      {img.relative_to(folder)}")

    try:
        for idx, img_path in enumerate(image_files):
            try:
                # Загрузить изображение
                with open(img_path, "rb") as f:
                    image_bytes = f.read()
                image = Image.open(BytesIO(image_bytes))

                # Предсказание Model-1
                logger.debug(
                    f"🔍 Классификация {idx+1}/{len(image_files)}: {img_path.name}"
                )
                pred_class, pred_conf = predict_model1(image)

                # Добавляем текущее предсказание в буфер и делаем сглаживание простым большинством
                buffer.append((pred_class, pred_conf))
                # Majority vote in buffer
                counts = Counter([p for p, _ in buffer])
                if counts:
                    smoothed_label = counts.most_common(1)[0][0]
                else:
                    smoothed_label = pred_class

                # For logging confidence, take average confidence of items matching smoothed label
                confidences = [c for (p, c) in buffer if p == smoothed_label]
                smoothed_conf = (
                    float(sum(confidences) / len(confidences))
                    if confidences
                    else pred_conf
                )

                class_name, confidence = smoothed_label, smoothed_conf

                processed_count += 1

                # Фильтрация по уверенности
                if confidence < CONFIDENCE_THRESHOLD:
                    logger.warning(f"⚠️ ОТБРАКОВКА:\
                     {img_path.name} (confidence={confidence:.4f}\
                     < {CONFIDENCE_THRESHOLD})")
                    rejected_count += 1
                    continue
                # Логирование предсказания и текущего wagon_id
                logger.debug(f"📸 {img_path.name} → {class_name}\
                     (confidence={confidence:.3f}), current_wagon={current_wagon_id}")

                if class_name:
                    # Если нет текущего вагона, начинаем новый
                    if current_wagon_id is None:
                        wagon_count += 1
                        current_wagon_id = f"wagon_{wagon_count}"
                        current_wagon_photos = []
                        logger.info(f"🚂 Новый вагон #{wagon_count} начат")

                    current_wagon_photos.append((img_path, 0))
                    logger.debug(
                        f"   ➕ Добавлено к {current_wagon_id} ({len(current_wagon_photos)})"
                    )
                    wagon_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_wagon_dir = (
                        Path.cwd()
                        / settings.output_model1_path
                        / f"{wagon_timestamp}"
                        / f"{current_wagon_id}"
                    )
                    output_wagon_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, output_wagon_dir)
                    logger.info(f"📁 Фото {img_path} сохраненно в {output_wagon_dir}")

                else:
                    # Завершаем текущий вагон и начинаем новый (edge separates)
                    if current_wagon_id is not None:
                        # Проверка длины перед сохранением
                        count_photos = len(current_wagon_photos)
                        if count_photos < 3:
                            logger.warning(f"⚠️ Вагон #{current_wagon_id}\
                                 слишком короткий ({count_photos}<3), отбрасывается")
                            # Отбраковываем
                        else:
                            wagon_groups[current_wagon_id] = list(current_wagon_photos)
                            logger.info(
                                f"✅ Вагон #{current_wagon_id} завершен, фото: {count_photos}"
                            )

                    # Начать новый вагон (новый id), but wait for next wagon_body to add photos
                    if len(current_wagon_photos) > 2:
                        wagon_count += 1
                    current_wagon_id = f"wagon_{wagon_count}"
                    current_wagon_photos = []
                    logger.info(f"🔄 Граница обнаружена — начинаем {current_wagon_id}")

            except Exception as e:
                logger.error(
                    f"❌ Ошибка обработки {img_path.name}: {type(e).__name__}: {str(e)}"
                )
                rejected_count += 1
                continue

        # После обработки всех фото — финализировать текущий вагон, если есть
        if current_wagon_id is not None and len(current_wagon_photos) > 0:
            count_photos = len(current_wagon_photos)
            if count_photos < 3:
                logger.warning(f"⚠️ Вагон #{current_wagon_id}\
                     слишком короткий ({count_photos}<3), отбрасывается")
            else:
                wagon_groups[current_wagon_id] = list(current_wagon_photos)
                logger.info(
                    f"✅ Вагон #{current_wagon_id} завершен, фото: {count_photos}"
                )

        logger.info("✅ Группирование завершено:")
        logger.info(f"   Обработано: {processed_count}")
        logger.info(f"   Отбракована: {rejected_count}")
        logger.info(f"   Вагонов: {len(wagon_groups)}")
        for wagon_id, photos in wagon_groups.items():
            logger.info(f"      {wagon_id}: {len(photos)} фото")

        return wagon_groups

    except Exception as e:
        logger.error(
            f"❌ Ошибка в classify_and_group_wagons: {type(e).__name__}: {str(e)}"
        )
        raise
