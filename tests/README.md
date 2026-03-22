# Testing Guide / Руководство по тестированию

## Структура тестов / Test Structure

```
tests/
├── conftest.py          # Общие fixtures и конфигурация
├── unit/                # Unit тесты
│   ├── core/            # Тесты модуля core
│   │   └── test_config.py
│   ├── services/        # Тесты для services
│   │   ├── test_classifier.py
│   │   ├── test_detector.py
│   │   ├── test_storage.py
│   │   ├── test_job_manager.py
│   │   ├── test_aggregator.py
│   │   ├── test_upload_handler.py
│   │   └── test_model_loader.py
│   └── routes/          # Тесты для API endpoints
│       ├── test_health.py
│       └── test_wagon.py
```

## Установка зависимостей / Installation

```bash
# Установить все зависимости включая тестовые
pip install -r requirements.txt

# Или установить только основные (без tests)
pip install -r requirements.txt --no-deps
```

## Запуск тестов / Running Tests

### Запустить все тесты / Run all tests
```bash
pytest
```

### Запустить с выводом покрытия кода / Run with coverage report
```bash
pytest --cov=. --cov-report=html
```

### Запустить только unit тесты / Run only unit tests
```bash
pytest -m unit
```

### Запустить конкретный файл тестов / Run specific test file
```bash
pytest tests/unit/services/test_storage.py
```

### Запустить конкретный тест / Run specific test
```bash
pytest tests/unit/services/test_storage.py::TestStorageFunctions::test_get_aggregate_dir
```

### Запустить с подробным выводом / Run with verbose output
```bash
pytest -v
```

### Запустить и остановиться на первой ошибке / Run and stop on first failure
```bash
pytest -x
```

### Запустить последние N неудачных тестов / Run last N failed tests
```bash
pytest --lf  # Run last failed
pytest --ff  # Run failed first
```

## Написание новых тестов / Writing New Tests

### Структура теста / Test structure
```python
import pytest
from unittest.mock import patch, MagicMock

class TestMyFeature:
    """Документируйте класс тестов"""
    
    def test_something_works(self):
        """Каждый тест должен иметь описание"""
        # Arrange - подготовка
        data = "test"
        
        # Act - действие
        result = process(data)
        
        # Assert - проверка
        assert result is not None
    
    @pytest.fixture
    def my_fixture(self):
        """Используйте fixtures для подготовки данных"""
        return "fixture_data"
    
    def test_with_fixture(self, my_fixture):
        """Fixtures автоматически подставляются в параметры"""
        assert my_fixture == "fixture_data"
```

### Использование мок-объектов / Using mocks
```python
from unittest.mock import patch, MagicMock

@patch('module.to_mock')
def test_with_mock(self, mock_obj):
    mock_obj.return_value = "mocked"
    # Ваш код
    assert mock_obj.called
```

## Доступные Fixtures

- `sample_image` - тестовое RGB изображение (224x224)
- `sample_image_as_bytes` - изображение в формате байтов
- `sample_grayscale_image` - граyscale изображение
- `sample_rgba_image` - RGBA изображение
- `mock_settings` - мок конфигурации
- `temp_upload_dir` - временная директория загрузок
- `temp_aggregate_dir` - временная директория photo_aggregate
- `mock_model1` - мок PyTorch модели Model-1
- `mock_model2` - мок YOLO модели Model-2
- `mock_device` - мок torch device
- `fastapi_client` - TestClient для FastAPI тестов
- `mock_mongodb_client` - мок MongoDB клиента
- `mock_upload_file` - мок UploadFile объекта

## Лучшие практики / Best Practices

1. **Изоляция тестов** - Каждый тест должен быть независим
2. **AAA паттерн** - Arrange, Act, Assert
3. **Понятные имена** - `test_function_with_specific_condition`
4. **Используйте fixtures** - Для подготовки тестовых данных
5. **Мокируйте внешние зависимости** - DB, API calls, модели
6. **Одна ассерция на тест** - Когда возможно
7. **Документация** - Используйте docstrings

## CI/CD Integration

### GitHub Actions пример / Example GitHub Actions
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest --cov
```

## Отладка тестов / Debugging Tests

```bash
# Добавить точки останова в тесте
pytest --pdb  # Откроет debugger при ошибке

# Показать локальные переменные при ошибке
pytest -l

# Увеличенный вывод для диагностики
pytest --tb=long
```

## Проблемы и решения / Troubleshooting

### ImportError при запуске тестов
- Убедитесь что вы в правильной директории
- Проверьте что все `__init__.py` файлы на месте
- Используйте `PYTHONPATH`

### Async тесты не работают
- Убедитесь что установлен `pytest-asyncio`
- Используйте `@pytest.mark.asyncio` декоратор

### Модели не загружаются в тестах
- Мокируйте загрузку моделей
- Используйте фиксчуры для создания мок-объектов

## Метрики покрытия / Coverage Metrics

Целевое покрытие кода: **80%+**

```bash
# Генерировать HTML отчёт о покрытии
pytest --cov=. --cov-report=html

# Открыть отчёт
# htmlcov/index.html
```

---

**Последнее обновление** / Last updated: 2026-03-20
