import os
import glob
import shutil
import random
import argparse
from tqdm import tqdm
import yaml

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp"]


def collect_image_label_pairs(dataset_root):
    pairs = []
    subdirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    for subdir in subdirs:
        image_dir = os.path.join(dataset_root, subdir, "images")
        label_dir = os.path.join(dataset_root, subdir, "labels")

        if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
            continue

        for ext in IMAGE_EXTENSIONS:
            for img_path in glob.glob(os.path.join(image_dir, f"*{ext}")):
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(label_dir, img_name + ".txt")
                if os.path.exists(label_path):
                    pairs.append((img_path, label_path))
    return pairs


def split_dataset(pairs, train_ratio, seed):
    random.seed(seed)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * train_ratio)
    return pairs[:split_idx], pairs[split_idx:]


def copy_dataset(pairs, output_dir):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    for img_path, label_path in tqdm(pairs, desc=f"Copying to {output_dir}"):
        shutil.copy(img_path, os.path.join(output_dir, "images", os.path.basename(img_path)))
        shutil.copy(label_path, os.path.join(output_dir, "labels", os.path.basename(label_path)))


def find_max_class_id(pairs):
    max_id = 0
    for _, label_path in pairs:
        with open(label_path, "r") as f:
            for line in f:
                if line.strip() == "":
                    continue
                class_id = int(line.split()[0])
                if class_id > max_id:
                    max_id = class_id
    return max_id


def export_yaml(save_dir, nc):
    yaml_path = os.path.join(save_dir, "dataset.yaml")
    data = {
        "train": os.path.abspath(os.path.join(save_dir, "train", "images")),
        "val": os.path.abspath(os.path.join(save_dir, "valid", "images")),
        "nc": nc,
        "names": [str(i) for i in range(nc)]
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    print(f"[✔] Exported Ultralytics-compatible YAML to: {yaml_path}")


def main(args):
    dataset_root = os.path.join(args.root_path, "datasets")
    output_root = args.output_path

    print(f"[INFO] Scanning: {dataset_root}")
    all_pairs = collect_image_label_pairs(dataset_root)
    print(f"[INFO] Found {len(all_pairs)} image-label pairs.")

    train_pairs, valid_pairs = split_dataset(all_pairs, args.train_ratio, args.seed)
    print(f"[INFO] Train / Valid split: {len(train_pairs)} / {len(valid_pairs)}")

    copy_dataset(train_pairs, os.path.join(output_root, "train"))
    copy_dataset(valid_pairs, os.path.join(output_root, "valid"))

    nc = find_max_class_id(all_pairs) + 1
    export_yaml(output_root, nc)

    print(f"[✔] Dataset preprocessing complete. Output at: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 Dataset Preprocessing Pipeline")
    parser.add_argument("root_path", type=str, help="Root path above datasets/")
    parser.add_argument("--output_path", type=str, default="processed_dataset", help="Output path to save processed dataset")
    parser.add_argument("--train_ratio", type=float, default=0.95, help="Proportion of training data (0 < train_ratio < 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset shuffling")

    args = parser.parse_args()
    main(args)
