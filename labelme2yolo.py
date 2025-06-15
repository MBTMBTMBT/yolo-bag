import os
import json
import shutil
import sys
from PIL import Image
from tqdm import tqdm

def polygon_to_bbox(polygon):
    x_coords = [pt[0] for pt in polygon]
    y_coords = [pt[1] for pt in polygon]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

def scan_class_names(json_dir):
    class_set = set()
    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            with open(os.path.join(json_dir, fname), "r") as f:
                data = json.load(f)
            for shape in data["shapes"]:
                class_set.add(shape["label"])
    return sorted(list(class_set))  # alphabetically sorted

def convert_annotation(json_path, class_names, image_size):
    width, height = image_size
    with open(json_path, "r") as f:
        data = json.load(f)

    yolo_lines = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in class_names:
            continue
        class_id = class_names.index(label)
        points = shape["points"]
        xmin, ymin, xmax, ymax = polygon_to_bbox(points)
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return yolo_lines

def convert_labelme_to_yolo(input_dir, output_dir):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    class_names = scan_class_names(input_dir)

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_exts)]

    for fname in tqdm(image_files):
        image_path = os.path.join(input_dir, fname)
        json_path = os.path.splitext(image_path)[0] + ".json"
        output_image_path = os.path.join(output_dir, "images", fname)
        output_label_path = os.path.join(output_dir, "labels", os.path.splitext(fname)[0] + ".txt")

        shutil.copy(image_path, output_image_path)

        if os.path.exists(json_path):
            image = Image.open(image_path)
            yolo_lines = convert_annotation(json_path, class_names, image.size)
            with open(output_label_path, "w") as f:
                f.write("\n".join(yolo_lines))
        else:
            with open(output_label_path, "w") as f:
                pass  # empty file

    return class_names

def generate_data_yaml(output_dir, class_names):
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write("train: images\n")
        f.write("val: images  # or split manually\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names: [" + ", ".join(f"'{name}'" for name in class_names) + "]\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python labelme2yolo.py <input_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = input_folder.rstrip("/").rstrip("\\") + "_yolo"

    print(f"📁 Input:  {input_folder}")
    print(f"📂 Output: {output_folder}")

    classes = convert_labelme_to_yolo(input_folder, output_folder)
    generate_data_yaml(output_folder, classes)

    print("✅ Done! YOLO dataset ready.")
    print("📋 Classes (alphabetical):", classes)
