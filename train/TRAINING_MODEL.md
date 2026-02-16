# YOLO training helper (YOLOv11-compatible script)

Этот репозиторий содержит скрипт `train_yolov11.py`, который:
- сканирует датасет в `C:\Users\Lenovo\Documents\vagon_yolo11` для изображений и соответствующих `.txt` разметок (YOLO TXT)
- генерирует `yolo_data/data.yaml`, `yolo_data/train.txt` и `yolo_data/val.txt`
- запускает обучение через API `ultralytics.YOLO.train()` (поддерживает передачу произвольных весов, включая ваши YOLOv11, если они совместимы)

Файлы в проекте:
- `train_yolov11.py` — основной скрипт
- `requirements.txt` — зависимости (см. заметку про PyTorch)

Пример быстрого запуска (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Установите PyTorch согласно официальной инструкции: https://pytorch.org/get-started/locally/
# Например (пример, подберите свою команду на сайте PyTorch):
# pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Запустить обучение (по умолчанию скрипт смотрит в C:\Users\Lenovo\Documents\vagon_yolo11)
python train_yolov11.py --model yolov8n.pt --epochs 50 --batch 16 --imgsz 640
```

Пара полезных опций `train_yolov11.py`:
- `--data_dir` — путь к вашей папке с данными
- `--model` или `-m` — путь к файлу весов (можно указать вашу модель YOLOv11.weights/pt)
- `--epochs`, `--batch`, `--imgsz` — стандартные параметры обучения
- `--out_dir` — куда записать `data.yaml`, `train.txt`, `val.txt`

Советы:
- Убедитесь, что для каждого изображения есть соответствующий `.txt` с разметкой YOLO (в той же папке или в `labels/`)
- Если dataset уже содержит `val` набор в отдельной папке, вы можете указать `--val_split 0.0` и положить готовые списки вручную
- Если у вас действительно YOLOv11 (нестандартная/локальная реализация), передавайте путь к весам в `--model`.

Если хотите, могу:
- адаптировать скрипт под конкретную структуру вашего датасета (покажите дерево папок)
- добавить поддержку кастомных имен классов (файл `.names` или `classes.txt`)
- подготовить примеры конфигурации тренировки (увеличение/уменьшение lr, scheduler и т.д.)
