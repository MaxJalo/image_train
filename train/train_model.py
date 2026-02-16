import argparse
import glob
import os
import random
import sys
import yaml

try:
    from ultralytics import YOLO
except Exception:
    print("Error importing ultralytics. Install it with: pip install ultralytics")
    raise

try:
    import torch
except Exception:
    torch = None


def find_images(root):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    imgs = []
    for e in exts:
        imgs.extend(glob.glob(os.path.join(root, "**", e), recursive=True))
    return sorted(set(imgs))


def find_label_for_image(img_path):
    base, _ = os.path.splitext(img_path)
    txt = base + ".txt"
    if os.path.exists(txt):
        return txt
    
    dirn, name = os.path.split(base)
    
    labels_dir = os.path.join(dirn, "labels")
    alt = os.path.join(labels_dir, name + ".txt")
    if os.path.exists(alt):
        return alt

    
    parent = os.path.dirname(dirn)
    if parent:
        sibling = os.path.join(parent, "labels", name + ".txt")
        if os.path.exists(sibling):
            return sibling

    
    grandparent = os.path.dirname(parent)
    if grandparent:
        sibling2 = os.path.join(grandparent, "labels", name + ".txt")
        if os.path.exists(sibling2):
            return sibling2

    return None


def build_data_files(data_dir, out_dir, val_split=0.2, seed=42):
    images = find_images(data_dir)
    items = []
    for img in images:
        lbl = find_label_for_image(img)
        if lbl is None:
            continue
        items.append((img, lbl))

    if not items:
        raise RuntimeError(f"No labeled images found in {data_dir}. Expected image files + corresponding .txt labels in same folder or a 'labels' subfolder.")

    random.Random(seed).shuffle(items)
    n_val = max(1, int(len(items) * val_split)) if 0.0 < val_split < 1.0 else 0
    val_items = items[:n_val]
    train_items = items[n_val:] if n_val > 0 else items

    os.makedirs(out_dir, exist_ok=True)
    train_txt = os.path.join(out_dir, "train.txt")
    val_txt = os.path.join(out_dir, "val.txt")

    def write_list(lst, path):
        with open(path, "w", encoding="utf-8") as f:
            for img, _ in lst:
                
                f.write(img.replace('\\', '/') + "\n")

    write_list(train_items, train_txt)
    if n_val > 0:
        write_list(val_items, val_txt)

    classes = set()
    for _, lbl in items:
        with open(lbl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cls = int(parts[0])
                except Exception:
                    continue
                classes.add(cls)

    if classes:
        max_cls = max(classes)
        names = [f'class{i}' for i in range(max_cls + 1)]
    else:
        names = []

    data_yaml = os.path.join(out_dir, 'data.yaml')
    data = {
        'train': train_txt.replace('\\', '/'),
        'val': val_txt.replace('\\', '/') if n_val > 0 else train_txt.replace('\\', '/'),
        'names': names,
    }
    with open(data_yaml, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return data_yaml


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset and train YOLO (Ultralytics API).')
    parser.add_argument('--data_dir', type=str, default=r'C:\Users\Lenovo\go\train_rail\vagon_yolo11', help='Path to dataset root')
    parser.add_argument('--out_dir', type=str, default='yolo_data', help='Where to write data.yaml/train.txt/val.txt')
    parser.add_argument('--model', '-m', type=str, default='yolov8n.pt', help='Model weights or model name (you can pass YOLOv11 weights path)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data used for validation (used if no val set present)')
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='exp')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto', help="Device to use: 'auto'|'cpu'|'cuda'|'cuda:0'|`0` (GPU index)")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Data directory does not exist: {data_dir}")
        sys.exit(1)

    print(f"Scanning dataset in: {data_dir}")

    candidate_yaml = os.path.join(data_dir, 'data.yaml')
    out_arg_is_yaml = str(args.out_dir).lower().endswith('.yaml')
    if out_arg_is_yaml and os.path.exists(args.out_dir):
        data_yaml = args.out_dir
        print(f"Using user-provided data.yaml: {data_yaml}")
    elif os.path.exists(candidate_yaml):
        data_yaml = candidate_yaml
        print(f"Found existing data.yaml in data_dir, using: {data_yaml}")
    else:
        data_yaml = build_data_files(data_dir, args.out_dir, val_split=args.val_split, seed=args.seed)
        print(f"Wrote data yaml: {data_yaml}")

    print(f"Initializing model: {args.model}")

    model_path = args.model
    is_path_like = ('/' in model_path) or ('\\' in model_path) or (len(model_path) > 1 and model_path[1] == ':')
    if is_path_like and not os.path.exists(model_path):
        print(f"Model file not found at path: {model_path}")
        print("If you meant to use a built-in ultralytics model name (for example 'yolov8n.pt'), pass that name without a path.")
        print("If you have local weights, provide the full absolute path to the .pt file. Example:")
        print(r"  --model C:\\models\\yolov11.pt")
        sys.exit(1)

    model = YOLO(args.model)

    def _resolve_device(dev_arg):
        if dev_arg is None:
            return 'cpu'
        a = str(dev_arg).lower()
        if a == 'auto':
            if torch is not None and torch.cuda.is_available():
                cnt = torch.cuda.device_count()
                names = []
                try:
                    for i in range(cnt):
                        names.append(torch.cuda.get_device_name(i))
                except Exception:
                    names = []
                print(f"CUDA available: {cnt} device(s).", "Names:", names)
                return 0
            else:
                print("CUDA not available, using CPU.")
                return 'cpu'
        if a == 'cpu':
            return 'cpu'
        if a == 'cuda':
            return 0
        if a.startswith('cuda:'):
            try:
                return int(a.split(':', 1)[1])
            except Exception:
                return a
        if a.isdigit():
            return int(a)
        return dev_arg

    device_param = _resolve_device(args.device)
    print(f"Using device: {device_param}")

    print("Starting training with the following params:")
    print(f"  epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}")

    model.train(data=data_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=args.project, name=args.name, device=device_param)


if __name__ == '__main__':
    main()
