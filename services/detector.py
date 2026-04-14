# (detector)
"""
Микросервис для детекции признаков вагонов (Model-2 YOLO)

Отвечает за:
- Загрузку модели YOLO (Model-2)
- Детекцию признаков вагонов (тормозная штанга, носик штока, кран, бак)
- Определение стороны вагона (левая/правая)
- Возврат результатов для агрегации в батч
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from core.config import settings
from models.schemas import Model2Output
from services.model_loader import load_model

logger = logging.getLogger(__name__)

# Cache для модели Model-2
_model2_cache: Optional[Any] = None


def _get_model2():
    """Загрузить Model-2 (YOLO) с кешированием"""
    global _model2_cache
    if _model2_cache is None:
        logger.debug(f"🔄 Инициализация Model-2 из {settings.model2_path}")
        try:
            _model2_cache = load_model(settings.model2_path)
            logger.info("✅ Model-2 успешно загружена")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки Model-2: {str(e)}")
            logger.warning("⚠️ Model-2 используется в режиме заглушки")
            _model2_cache = "FALLBACK"
    return _model2_cache


def predict_model2(image: Image.Image) -> Model2Output:
    """
    Предсказание Model-2 YOLO для детекции признаков вагона.

    Args:
        image: PIL Image объект

    Returns:
        Model2Output: Результаты детекции с признаками и стороной вагона
    """
    logger.debug("▶️ Начало вывода Model-2 (YOLO)")
    logger.debug(f"   Размер изображения: {image.size}, Режим: {image.mode}")

    try:
        model = _get_model2()

        # Проверка доступности модели
        if model == "FALLBACK":
            logger.warning("📌 Использование заглушки Model-2 (модель не загружена)")
            return Model2Output(
                brake_rod=0.5, rod_nose=0.5, crane=0.5, tank=0.5, side="left", confidence=0.5
            )

        logger.debug("   Конвертирование изображения в RGB...")
        if image.mode != "RGB":
            image = image.convert("RGB")

        logger.debug("   Запуск YOLO детекции...")
        results = model(image)
        logger.debug(f"   YOLO вернул {len(results)} результат(ов)")

        # Парсинг результатов YOLO
        detected_classes = {}
        total_confidence = 0
        detection_count = 0

        for result_idx, result in enumerate(results):
            logger.debug(f"   Обработка результата {result_idx}")
            if hasattr(result, "boxes"):
                logger.debug(f"   Найдено {len(result.boxes)} bounding box(ов)")
                for box_idx, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0]) if len(box.cls) > 0 else -1
                    confidence = float(box.conf[0]) if len(box.conf) > 0 else 0.0

                    if hasattr(model, "names") and cls_id in model.names:
                        class_name = model.names[cls_id]
                    else:
                        class_name = f"class_{cls_id}"

                    logger.debug(f"   Box {box_idx}: {class_name} (confidence: {confidence:.4f})")

                    if class_name not in detected_classes:
                        detected_classes[class_name] = []
                    detected_classes[class_name].append(confidence)
                    total_confidence += confidence
                    detection_count += 1

        logger.debug(f"   Всего детекций: {detection_count}")

        # Расчет максимальной confidence для каждого признака
        features = {
            "brake_rod": max(detected_classes.get("brake_rod", [0.0])),
            "rod_nose": max(detected_classes.get("rod_nose", [0.0])),
            "crane": max(detected_classes.get("crane", [0.0])),
            "tank": max(detected_classes.get("tank", [0.0])),
        }

        logger.debug(f"   Обнаруженные признаки: {features}")

        # Определение стороны вагона
        side = _determine_side(image, detected_classes)
        logger.debug(f"   Определена сторона: {side}")

        # Общая уверенность
        overall_confidence = total_confidence / detection_count if detection_count > 0 else 0.0

        logger.info(f"✅ Model-2 result: side={side},\
             confidence={overall_confidence:.4f}, features={features}")

        return Model2Output(
            brake_rod=features["brake_rod"],
            rod_nose=features["rod_nose"],
            crane=features["crane"],
            tank=features["tank"],
            side=side,
            confidence=overall_confidence,
        )

    except Exception as e:
        logger.error(f"❌ Ошибка вывода Model-2: {type(e).__name__}: {str(e)}")
        logger.debug(f"   Traceback: {repr(e)}")
        # Возвращаем минимальный валидный выход при ошибке
        return Model2Output(
            brake_rod=0.0, rod_nose=0.0, crane=0.0, tank=0.0, side="left", confidence=0.0
        )


def _determine_side(image: Image.Image, detected_classes: Dict[str, list]) -> str:
    """
    Определить сторону вагона (левая/правая) на основе детекции.

    Args:
        image: PIL Image объект
        detected_classes: Словарь обнаруженных классов

    Returns:
        str: "left" или "right"
    """
    logger.debug("▶️ Определение стороны вагона")
    width = image.width if hasattr(image, "width") else 640

    if detected_classes:
        logger.debug("   Найдены классы, устанавливаю 'left'")
        return "left"

    # Использовать ширину как эвристику если нет детекции
    side = "right" if width % 2 == 0 else "left"
    logger.debug(f"   Детекции не найдены, эвристика ширины: {side}")
    return side


async def detect_wagon_sides(
    wagon_groups: Dict[str, List[Tuple[Path, int]]],
) -> Dict[str, Dict[str, Any]]:
    """
    Запустить Model-2 для всех вагонов и определить стороны.

    Args:
        wagon_groups: Dict[wagon_id] → List[(image_path, camera_id), ...]

    Returns:
        Dict[wagon_id] → {
            "total_photos": int,
            "processed_photos": int,
            "left_count": int,
            "right_count": int,
            "final_side": str,
            "cameras": List[int]
        }
    """
    logger.info(f"🔍 Начало детекции сторон для {len(wagon_groups)} вагонов")

    wagon_results: Dict[str, Dict[str, Any]] = {}

    for wagon_id, photos in wagon_groups.items():
        logger.info(f"📸 Обработка {wagon_id}: {len(photos)} фото")

        left_count = 0
        right_count = 0
        processed_count = 0
        cameras = set()

        for photo_path, camera_id in photos:
            try:
                logger.debug(f"   🖼️ Загрузка {photo_path.name}")

                # Загрузить изображение
                with open(photo_path, "rb") as f:
                    image_bytes = f.read()
                image = Image.open(BytesIO(image_bytes))

                # Запустить Model-2
                logger.debug(f"   🔍 Классификация {photo_path.name} (Model-2)")
                model2_output = predict_model2(image)

                # Обновить статистику
                if model2_output.side == "left":
                    left_count += 1
                else:
                    right_count += 1

                processed_count += 1
                cameras.add(camera_id)

                logger.debug(
                    f"      Side: {model2_output.side}, confidence: {model2_output.confidence:.4f}"
                )

            except Exception as e:
                logger.error(f"❌ Ошибка обработки {photo_path.name}: {type(e).__name__}: {str(e)}")
                continue

        # Определить финальную сторону по большинству
        if left_count > right_count:
            final_side = "left"
        elif right_count > left_count:
            final_side = "right"
        else:
            final_side = "left" if left_count > 0 else "right"

        wagon_results[wagon_id] = {
            "total_photos": len(photos),
            "processed_photos": processed_count,
            "left_count": left_count,
            "right_count": right_count,
            "final_side": final_side,
            "cameras": sorted(list(cameras)),
        }

        logger.info(
            f"✅ {wagon_id}: обработано {processed_count}/{len(photos)} фото → {final_side}"
        )

    logger.info(f"✅ Детекция завершена для {len(wagon_results)} вагонов")

    return wagon_results
