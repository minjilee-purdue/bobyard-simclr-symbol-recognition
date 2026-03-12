# create_symbol_crops.py

import os
import cv2
import yaml
from pathlib import Path

# =========================
# 1. Dataset paths
# =========================
DATASET_ROOT = Path("/home/minjilee/Downloads/Firefighting Device Detection.v6i.yolov8")
OUTPUT_ROOT = DATASET_ROOT / "cropped_symbols"

SPLITS = ["train", "valid", "test"]

PADDING_RATIO = 0.08
MIN_SIZE = 5

# =========================
# 2. Load class names
# =========================
yaml_path = DATASET_ROOT / "data.yaml"

with open(yaml_path, "r") as f:
    data_cfg = yaml.safe_load(f)

class_names = data_cfg["names"]

if isinstance(class_names, dict):
    class_names = [class_names[i] for i in sorted(class_names.keys())]

print("Number of classes:", len(class_names))

# =========================
# 3. YOLO bbox conversion
# =========================
def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h

    x1 = int(xc - w / 2)
    y1 = int(yc - h / 2)
    x2 = int(xc + w / 2)
    y2 = int(yc + h / 2)

    return x1, y1, x2, y2

def add_padding(x1, y1, x2, y2, img_w, img_h, ratio):
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * ratio)
    pad_y = int(bh * ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)

    return x1, y1, x2, y2

# =========================
# 4. Crop objects
# =========================
total_crops = 0

for split in SPLITS:

    image_dir = DATASET_ROOT / split / "images"
    label_dir = DATASET_ROOT / split / "labels"

    if not image_dir.exists():
        continue

    image_files = sorted(list(image_dir.glob("*")))

    print(f"Processing split: {split} ({len(image_files)} images)")

    for img_path in image_files:

        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label_path = label_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        with open(label_path, "r") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
            x1, y1, x2, y2 = add_padding(x1, y1, x2, y2, img_w, img_h, PADDING_RATIO)

            if (x2 - x1) < MIN_SIZE or (y2 - y1) < MIN_SIZE:
                continue

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            class_name = class_names[class_id]
            class_name = str(class_name).replace(" ", "_").replace("/", "_")

            save_dir = OUTPUT_ROOT / split / class_name
            save_dir.mkdir(parents=True, exist_ok=True)

            save_name = f"{img_path.stem}_{idx}.png"
            save_path = save_dir / save_name

            cv2.imwrite(str(save_path), crop)

            total_crops += 1

print("Total crops saved:", total_crops)
print("Saved to:", OUTPUT_ROOT)