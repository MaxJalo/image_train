# (upload_handler)

import logging 
import tempfile
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict, NamedTuple
from fastapi import UploadFile
from PIL import Image
import io
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExtractedImageFile(NamedTuple):
    """Информация об извлеченном файле изображения"""
    path: str  # Полный путь относительно корня распакованного архива
    relative_path: str  # Путь на диске
    depth: int  # Глубина вложенности (количество папок)
    camera_id: Optional[str]  # ID камеры (числовая папка в пути)
    train_hash: Optional[str]  # Hash поезда (первая буквенно-цифровая папка)
    filename: str  # Имя файла

# Поддерживаемые расширения изображений
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_FILES = 1000

# Временная директория для всех загрузок
TEMP_BASE_DIR = Path(tempfile.gettempdir()) / "wagon_uploads"
TEMP_BASE_DIR.mkdir(parents=True, exist_ok=True)


class UploadHandler:
    """Обработчик загрузок файлов"""
    
    @staticmethod
    def _extract_metadata_from_path(file_path: Path, extract_root: Path) -> Tuple[Optional[str], Optional[str], int]:
        """
        Извлечь метаданные из полного пути файла.
        
        Возвращает: (camera_id, train_hash, depth)
        - camera_id: Числовая папка в пути (если есть)
        - train_hash: Первая буквенно-цифровая папка в пути (если есть)
        - depth: Глубина вложенности (количество папок от корня до файла)
        """
        try:
            # Получить относительный путь от корня распакованного архива
            rel_path = file_path.relative_to(extract_root)
            parts = rel_path.parts[:-1]  # Исключить сам файл
            
            depth = len(parts)
            camera_id = None
            train_hash = None
            
            # Ищем camera_id (чистое число)
            for part in parts:
                if part.isdigit():
                    camera_id = part
                    break
            
            # Ищем train_hash (первая папка которая не число)
            for part in parts:
                # Проверяем что это похоже на hash (буквы и цифры, не основная структура)
                if not part.isdigit() and len(part) > 0:
                    # Исключаем типичные системные папки
                    if part.lower() not in ['extracted', '__pycache__', '.git', '.vscode']:
                        train_hash = part
                        break
            
            return camera_id, train_hash, depth
        except Exception as e:
            logger.debug(f"Ошибка при извлечении метаданных из {file_path}: {str(e)}")
            return None, None, 0
    
    @staticmethod
    def _log_zip_structure(extract_root: Path):
        """Логировать полную структуру распакованного архива"""
        logger.info("=" * 80)
        logger.info("📁 СТРУКТУРА РАСПАКОВАННОГО АРХИВА")
        logger.info("=" * 80)
        
        def log_tree(path: Path, prefix: str = "", is_last: bool = True):
            """Рекурсивно вывести структуру как дерево"""
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "
            
            if path.is_file():
                logger.info(f"{prefix}{connector}{path.name}")
            else:
                logger.info(f"{prefix}{connector}{path.name}/")
                items = sorted(path.iterdir())
                for i, item in enumerate(items):
                    is_last_item = (i == len(items) - 1)
                    new_prefix = prefix + extension
                    log_tree(item, new_prefix, is_last_item)
        
        if extract_root.exists():
            items = sorted(extract_root.iterdir())
            for i, item in enumerate(items):
                is_last_item = (i == len(items) - 1)
                log_tree(item, "", is_last_item)
    
    @staticmethod
    def _log_extracted_files_summary(extracted_files: List[ExtractedImageFile]):
        """Логировать подробную информацию о извлеченных файлах"""
        logger.info("=" * 80)
        logger.info(f"📸 ИНФОРМАЦИЯ О ИЗВЛЕЧЕННЫХ ФАЙЛАХ ({len(extracted_files)} всего)")
        logger.info("=" * 80)
        
        for idx, file_info in enumerate(extracted_files, 1):
            logger.info(f"\n{idx}. {file_info.filename}")
            logger.info(f"   📂 Путь: {file_info.path}")
            logger.info(f"   ├─ Глубина: {file_info.depth}")
            logger.info(f"   ├─ Camera ID: {file_info.camera_id or 'не найден'}")
            logger.info(f"   └─ Train Hash: {file_info.train_hash or 'не найден'}")
    
    @staticmethod
    def _log_statistics(extracted_files: List[ExtractedImageFile]):
        """Логировать статистику по извлеченным файлам"""
        if not extracted_files:
            logger.info("⚠️ Нет файлов для анализа статистики")
            return
        
        logger.info("=" * 80)
        logger.info("📊 СТАТИСТИКА ПО ИЗВЛЕЧЕННЫМ ФАЙЛАМ")
        logger.info("=" * 80)
        
        # Общая статистика
        logger.info(f"\n✅ Общее количество файлов: {len(extracted_files)}")
        
        # Распределение по глубине
        depth_distribution = defaultdict(int)
        for file_info in extracted_files:
            depth_distribution[file_info.depth] += 1
        
        logger.info(f"\n📊 Распределение по глубине вложенности:")
        for depth in sorted(depth_distribution.keys()):
            count = depth_distribution[depth]
            percentage = (count / len(extracted_files)) * 100
            logger.info(f"   Глубина {depth}: {count} файлов ({percentage:.1f}%)")
        
        # Найденные camera_id
        camera_ids = defaultdict(int)
        for file_info in extracted_files:
            if file_info.camera_id:
                camera_ids[file_info.camera_id] += 1
        
        if camera_ids:
            logger.info(f"\n🎥 Найденные Camera ID ({len(camera_ids)} уникальных):")
            for cam_id in sorted(camera_ids.keys()):
                count = camera_ids[cam_id]
                logger.info(f"   Camera {cam_id}: {count} файлов")
        else:
            logger.info(f"\n🎥 Camera ID не найдены ни в одном пути")
        
        # Найденные train_hash
        train_hashes = defaultdict(int)
        for file_info in extracted_files:
            if file_info.train_hash:
                train_hashes[file_info.train_hash] += 1
        
        if train_hashes:
            logger.info(f"\n🚂 Найденные Train Hash ({len(train_hashes)} уникальных):")
            for hash_id in sorted(train_hashes.keys()):
                count = train_hashes[hash_id]
                logger.info(f"   {hash_id}: {count} файлов")
        else:
            logger.info(f"\n🚂 Train Hash не найдены ни в одном пути")
    
    @staticmethod
    async def validate_image_file(file: UploadFile) -> Tuple[bool, Optional[str]]:
        """Валидировать загруженный файл как изображение"""
        # Проверка расширения
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return False, f"Тип файла не поддерживается: {file_ext}. Поддерживаемые: {', '.join(ALLOWED_EXTENSIONS)}"
        
        # Проверка размера
        file.file.seek(0, io.SEEK_END)
        file_size = file.file.tell()
        await file.seek(0)
        if file_size > MAX_FILE_SIZE:
            return False, f"Размер файла превышает лимит: {file_size / 1024 / 1024:.1f} MB > {MAX_FILE_SIZE / 1024 / 1024:.0f} MB"
        
        # Проверка что это реально изображение
        try:
            content = await file.read()
            img = Image.open(io.BytesIO(content))
            img.verify()
            await file.seek(0)
            return True, None
        except Exception as e:
            return False, f"Файл поврежден или не является изображением: {str(e)}"
    
    @staticmethod
    async def save_single_file(
        file: UploadFile,
        job_id: str,
        camera_id: Optional[int] = None
    ) -> Tuple[bool, Path, Optional[str]]:
        """Сохранить один загруженный файл"""
        logger.info(f"💾 Сохранение одного файла для задания {job_id}")
        
        # Валидация
        is_valid, error = await UploadHandler.validate_image_file(file)
        if not is_valid:
            logger.error(f"❌ Валидация не пройдена: {error}")
            return False, None, error
        
        # Создать временную директорию для задания
        job_dir = TEMP_BASE_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Определить путь сохранения
        if camera_id is not None:
            camera_dir = job_dir / f"camera_{camera_id}"
            camera_dir.mkdir(parents=True, exist_ok=True)
            file_path = camera_dir / file.filename
        else:
            file_path = job_dir / file.filename
        
        # Сохранить файл
        try:
            content = await file.read()
            file_path.write_bytes(content)
            logger.info(f"✅ Файл сохранен: {file_path}")
            return True, job_dir, None
        except Exception as e:
            error_msg = f"Ошибка сохранения файла: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, None, error_msg
    
    @staticmethod
    async def save_multiple_files(
        files: List[UploadFile],
        job_id: str,
        camera_id: Optional[int] = None
    ) -> Tuple[bool, Path, Optional[str], int]:
        """Сохранить несколько загруженных файлов"""
        logger.info(f"💾 Сохранение {len(files)} файлов для задания {job_id}")
        
        # Проверка количества файлов
        if len(files) > MAX_FILES:
            error = f"Превышено максимальное количество файлов: {len(files)} > {MAX_FILES}"
            logger.error(f"❌ {error}")
            return False, None, error, 0
        
        # Создать временную директорию для задания
        job_dir = TEMP_BASE_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        last_error = None
        
        for file in files:
            # Валидация
            is_valid, error = await UploadHandler.validate_image_file(file)
            if not is_valid:
                logger.warning(f"⚠️ Файл {file.filename} не валидирован: {error}")
                last_error = error
                continue
            
            # Определить путь сохранения
            if camera_id is not None:
                camera_dir = job_dir / f"camera_{camera_id}"
                camera_dir.mkdir(parents=True, exist_ok=True)
                file_path = camera_dir / file.filename
            else:
                file_path = job_dir / file.filename
            
            # Сохранить файл
            try:
                content = await file.read()
                file_path.write_bytes(content)
                saved_count += 1
                logger.debug(f"✅ Файл сохранен: {file_path}")
            except Exception as e:
                error_msg = f"Ошибка сохранения файла {file.filename}: {str(e)}"
                logger.warning(f"⚠️ {error_msg}")
                last_error = error_msg
        
        if saved_count == 0:
            error = f"Не удалось сохранить ни один файл. {last_error or ''}"
            logger.error(f"❌ {error}")
            return False, None, error, 0
        
        logger.info(f"✅ Сохранено {saved_count} файлов из {len(files)}")
        return True, job_dir, last_error if saved_count < len(files) else None, saved_count
    
    @staticmethod
    async def extract_and_save_zip(
        file: UploadFile,
        job_id: str,
        camera_id: Optional[int] = None
    ) -> Tuple[bool, Path, Optional[str], List[ExtractedImageFile]]:
        """
        Распаковать и сохранить ZIP архив со сложной структурой.
        
        Поддерживает произвольная глубина вложенности и любые варианты структуры:
        - {hash}/{camera}/{hash}/{фото}.jpg
        - {camera}/{hash}/{фото}.jpg
        - {hash}/{camera}/{фото}.jpg
        - {camera}/{фото}.jpg
        - {hash}/{фото}.jpg
        - {фото}.jpg (в корне)
        
        Возвращает: (success, job_dir, error, extracted_files_info)
        """
        logger.info(f"💾 Обработка ZIP архива для задания {job_id}")
        logger.info(f"📦 Имя файла: {file.filename}")
        
        # Проверка что это ZIP
        if not file.filename.lower().endswith('.zip'):
            error = "Файл должен быть ZIP архивом"
            logger.error(f"❌ {error}")
            return False, None, error, []
        
        # Создать временную директорию для задания
        job_dir = TEMP_BASE_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        extract_dir = job_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Сохранить ZIP во временный файл
            zip_path = job_dir / "temp_archive.zip"
            content = await file.read()
            zip_path.write_bytes(content)
            
            logger.info(f"📦 Размер архива: {len(content) / 1024 / 1024:.2f} MB")
            
            # Распаковать архив
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            logger.info(f"✅ ZIP архив распакован в {extract_dir}")
            
            # Логировать структуру архива
            UploadHandler._log_zip_structure(extract_dir)
            
            # Найти все изображения рекурсивно
            image_files = []
            for ext in ALLOWED_EXTENSIONS:
                image_files.extend(extract_dir.rglob(f"*{ext}"))
            
            logger.info(f"\n📸 Найдено {len(image_files)} изображений(й) в архиве")
            
            if not image_files:
                error = "В ZIP архиве не найдено поддерживаемых изображений (jpg, jpeg, png)"
                logger.warning(f"⚠️ {error}")
                shutil.rmtree(extract_dir)
                zip_path.unlink()
                return False, job_dir, error, []
            
            # Обработать каждое найденное изображение
            extracted_files = []
            validation_errors = []
            
            logger.info("\n🔍 Обработка найденных изображений...")
            
            for img_path in sorted(image_files):
                try:
                    # Валидация по размеру
                    file_size = img_path.stat().st_size
                    if file_size > MAX_FILE_SIZE:
                        msg = f"Файл слишком большой: {img_path.name} ({file_size / 1024 / 1024:.1f} MB)"
                        logger.warning(f"⚠️ {msg}")
                        validation_errors.append(msg)
                        continue
                    
                    # Валидация что это реально изображение
                    img = Image.open(img_path)
                    img.verify()
                    
                    # Извлечь метаданные из пути
                    camera_id_from_path, train_hash, depth = UploadHandler._extract_metadata_from_path(
                        img_path, extract_dir
                    )
                    
                    # Получить путь относительно корня распакованного архива
                    relative_path = img_path.relative_to(extract_dir)
                    
                    # Создать информацию о файле
                    file_info = ExtractedImageFile(
                        path=str(relative_path).replace('\\', '/'),
                        relative_path=str(relative_path),
                        depth=depth,
                        camera_id=camera_id_from_path,
                        train_hash=train_hash,
                        filename=img_path.name
                    )
                    
                    extracted_files.append(file_info)
                    logger.debug(f"✅ Обработан: {file_info.path}")
                    
                except Exception as e:
                    msg = f"Не удалось обработать {img_path.name}: {str(e)}"
                    logger.warning(f"⚠️ {msg}")
                    validation_errors.append(msg)
                    continue
            
            if not extracted_files:
                error = f"Не удалось обработать ни одно изображение из архива"
                if validation_errors:
                    error += f". Ошибки: {', '.join(validation_errors[:3])}"
                logger.error(f"❌ {error}")
                shutil.rmtree(extract_dir)
                zip_path.unlink()
                return False, job_dir, error, []
            
            # Логировать подробную информацию о файлах
            UploadHandler._log_extracted_files_summary(extracted_files)
            
            # Логировать статистику
            UploadHandler._log_statistics(extracted_files)
            
            # Сохранить оригинальную распакованную структуру для дальнейшей обработки
            logger.info(f"\n✅ Распакованная структура сохранена в: {extract_dir}")
            logger.info(f"✅ Успешно обработано {len(extracted_files)} изображений из {len(image_files)}")
            
            # Удалить временный ZIP файл (но сохранить распакованные содержимое)
            zip_path.unlink()
            
            warning = None
            if validation_errors and len(extracted_files) < len(image_files):
                warning = f"Успешно обработано {len(extracted_files)} из {len(image_files)} файлов. " \
                          f"Ошибки при обработке {len(image_files) - len(extracted_files)} файлов"
                logger.warning(f"⚠️ {warning}")
            
            logger.info("=" * 80)
            
            return True, job_dir, warning, extracted_files
            
        except zipfile.BadZipFile as e:
            error = f"Неверный или поврежденный ZIP архив: {str(e)}"
            logger.error(f"❌ {error}")
            return False, job_dir, error, []
        except Exception as e:
            error = f"Ошибка обработки ZIP архива: {str(e)}"
            logger.error(f"❌ {error}")
            return False, job_dir, error, []
    

    @staticmethod
    def cleanup_job_files(job_id: str) -> bool:
        """Удалить все временные файлы задания"""
        logger.info(f"🗑️ Очистка файлов задания {job_id}")
        
        job_dir = TEMP_BASE_DIR / job_id
        
        if not job_dir.exists():
            logger.debug(f"⚠️ Директория задания не найдена: {job_dir}")
            return False
        
        try:
            shutil.rmtree(job_dir)
            logger.info(f"✅ Директория удалена: {job_dir}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка удаления директории: {str(e)}")
            return False
    
    @staticmethod
    def get_job_files(job_id: str) -> List[Path]:
        """Получить список файлов в директории задания"""
        job_dir = TEMP_BASE_DIR / job_id
        
        if not job_dir.exists():
            logger.debug(f"⚠️ Директория задания не найдена: {job_dir}")
            return []
        
        image_files = []
        for ext in ALLOWED_EXTENSIONS:
            image_files.extend(job_dir.rglob(f"*{ext}"))
        
        return image_files

