# (model_loader)
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_model_cache: dict = {}


def load_model(model_path: str) -> Any:
    logger.debug(f"load_model() called with path: {model_path}")
    logger.info(f"Loading model from {model_path}")

    # Возвращаем кэшированную модель если существует
    if model_path in _model_cache:
        logger.debug(f"📦 Загрузка модели из кэша: {model_path}")
        return _model_cache[model_path]

    path = Path(model_path)
    logger.debug(f"Model path resolved to: {path.resolve()}")

    if not path.exists():
        logger.error(f"Файл модели не существует: {model_path}")
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    logger.info(f"📂 Файл модели найден: {path.name} ({path.stat().st_size} байт)")

    suffix = path.suffix.lower()
    logger.debug(f"Обнаружен формат модели: {suffix}")

    try:
        if suffix == ".pt":
            logger.info(f"Загрузка PyTorch модели из {path.name}...")
            model = _load_pytorch_model(model_path)
        elif suffix == ".pkl":
            logger.info(f"Загрузка Pickle модели из {path.name}...")
            model = _load_pickle_model(model_path)
        else:
            logger.error(f"Неподдерживаемый формат модели: {suffix}")
            raise ValueError(f"Неподдерживаемый формат модели: {suffix}")

        # Кэшируем загруженную модель
        _model_cache[model_path] = model
        logger.info(f"✅ Модель успешно загружена и закэширована: {path.name}")
        return model

    except Exception as e:
        logger.error(
            f"❌ Ошибка загрузки модели {path.name}: {type(e).__name__}: {str(e)}"
        )
        raise


def _load_mobilenet_model(model_path: str) -> Any:
    try:
        import torch
        import torch.nn as nn
        from torchvision import models

        logger.debug(f"Загрузка MobileNet V3 Small из state_dict: {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Загрузка state_dict из {model_path}")

        state_dict = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(state_dict, dict):
            classifier_outputs = []
            for key in state_dict.keys():
                if "classifier" in key and "weight" in key:
                    weight = state_dict[key]
                    if len(weight.shape) >= 1:
                        classifier_outputs.append((weight.shape[0], key))

            if classifier_outputs:
                classifier_outputs.sort()
                num_classes = classifier_outputs[0][0]
                logger.debug(
                    f"Обнаружено num_classes={num_classes} из {classifier_outputs[0][1]}"
                )

        if isinstance(state_dict, dict):
            if any(key.startswith("module.") for key in state_dict.keys()):
                logger.debug(
                    "Обнаружен префикс 'module.' в ключах state_dict, удаляю..."
                )
                state_dict = {
                    key.replace("module.", "", 1): value
                    for key, value in state_dict.items()
                }

        logger.info(
            f"Создание MobileNet V3 Small на device: {device} (num_classes={num_classes})"
        )
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        model = model.to(device)

        logger.debug("Загрузка весов state_dict в модель")
        result = model.load_state_dict(state_dict, strict=False)
        logger.debug(f"State dict загружен: {len(result.missing_keys)}\
             недостающих ключей, {len(result.unexpected_keys)} неожиданных ключей")

        model.eval()
        logger.info(f"✅ MobileNet V3 Small успешно загружена на device: {device}")

        return model

    except ImportError as e:
        logger.error(f"PyTorch или torchvision не установлены: {str(e)}")
        raise ImportError(
            "PyTorch и torchvision требуются. Установить с: pip install torch torchvision"
        )
    except Exception as e:
        logger.error(
            f"Ошибка загрузки MobileNet V3 Small: {type(e).__name__}: {str(e)}"
        )
        raise Exception(f"Не удалось загрузить модель MobileNet V3 Small: {str(e)}")


def _load_pytorch_model(model_path: str) -> Any:
    """Загружает PyTorch модель (.pt файл)"""
    try:
        import torch

        logger.debug(f"Версия PyTorch: {torch.__version__}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Используемый device: {device}")

        try:
            from ultralytics import YOLO

            logger.debug("Попытка загрузки как YOLO модель...")
            model = YOLO(model_path)
            logger.info(f"✅ YOLO модель успешно загружена на device: {device}")
            return model
        except Exception as yolo_error:
            logger.debug(f"Загрузчик YOLO не удался\
                 ({type(yolo_error).__name__}), попытка generic torch.load")
            loaded = torch.load(model_path, map_location=device, weights_only=False)

            if isinstance(loaded, dict):
                logger.info(
                    "PyTorch файл похож на state_dict, попытка загрузки MobileNet..."
                )
                try:
                    return _load_mobilenet_model(model_path)
                except Exception as mobilenet_error:
                    logger.error(f"Ошибка загрузки MobileNet:\
                         {type(mobilenet_error).__name__}: {mobilenet_error}")
                    raise Exception(
                        f"Не удалось загрузить как state_dict MobileNet: {str(mobilenet_error)}"
                    )

            try:
                loaded.eval()
            except Exception:
                logger.debug(
                    "Загруженный объект не имеет eval(), пропускаю вызов eval()"
                )
            logger.info(
                f"✅ Общая PyTorch модель успешно загружена на device: {device}"
            )
            return loaded

    except ImportError as e:
        logger.error(f"PyTorch или YOLO не установлены: {str(e)}")
        raise ImportError(
            "PyTorch не установлен. Установить с: pip install torch ultralytics"
        )
    except Exception as e:
        logger.error(f"Ошибка загрузки PyTorch модели: {type(e).__name__}: {str(e)}")
        raise Exception(f"Не удалось загрузить PyTorch модель: {str(e)}")


def _load_pickle_model(model_path: str) -> Any:
    try:
        logger.debug(f"Открытие pickle файла: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("✅ Pickle модель успешно загружена")
        return model
    except Exception as e:
        logger.error(f"Ошибка загрузки Pickle модели: {type(e).__name__}: {str(e)}")
        raise Exception(f"Не удалось загрузить Pickle модель: {str(e)}")


def clear_cache():
    count = len(_model_cache)
    _model_cache.clear()
    logger.info(f"🗑️ Кэш модели очищен ({count} модель(й) удалено)")


def get_cached_models() -> dict:
    logger.debug(f"Кэшированные модели: {len(_model_cache)} модель(й)")
    return dict(_model_cache)
