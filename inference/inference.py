from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os

app = FastAPI(title="YOLOv11 Folder Processing API")

# Пути
MODEL_PATH = "Data/best.pt"
IMAGES_DIR = "Data/images"
OUTPUT_DIR = "Data/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Подключаем папку output как статику для отображения изображений
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Загружаем модель
model = YOLO(MODEL_PATH)
class_names = model.names

def process_images(max_images=100):
    all_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(all_files) == 0:
        return []

    results_list = []

    for filename in all_files:
        image_path = os.path.join(IMAGES_DIR, filename)
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        results = model(img)

        detected_classes = set()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = class_names[cls_id]
                detected_classes.add(cls_name)

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1 - 15), cls_name, fill="red", font=font)

        # Перезапись файла
        save_filename = f"{filename.split('.')[0]}_result.jpg"
        save_path = os.path.join(OUTPUT_DIR, save_filename)
        img.save(save_path)

        results_list.append({
            "filename": filename,
            "output_image": save_filename,
            "classes": list(detected_classes)
        })

    return results_list

@app.get("/", response_class=HTMLResponse)
def index():
    results = process_images()
    if not results:
        return "<h2>No images found in Data/images</h2>"

    html_content = "<h1>YOLOv11 Detection Results</h1>"
    for r in results:
        html_content += "<div style='margin-bottom:30px;'>"
        html_content += f"<h3>{r['filename']}</h3>"
        html_content += f"<img src='/output/{r['output_image']}' width='500'><br>"
        html_content += f"<b>Detected classes:</b> {', '.join(r['classes'])}"
        html_content += "</div>"
    return html_content
