#!/usr/bin/env python3
"""
Consolidate multiple Ultralytics-style (.yolov11) exports that each
already contain train/valid(/test) splits into a single dataset folder
plus a compatible dataset.yaml.

Layout expected under <root_path> (any number of .yolov11 exports):

    paper bag.v1i.yolov11/
    ├─ train/
    │   ├─ images/
    │   └─ labels/
    ├─ valid/
    │   ├─ images/
    │   └─ labels/
    └─ test/            # optional
        ├─ images/
        └─ labels/

The script **does not reshuffle**; it simply merges everything that
already exists.
"""

import argparse
import glob
import os
import shutil
from tqdm import tqdm
import yaml

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp"}
SPLITS = ["train", "valid", "test"]  # test is optional


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def collect_image_label_pairs(dataset_root):
    """
    Return a dict {split: [(img_path, label_path), ...]}.
    Any split (train/valid/test) that isn't present remains an empty list.
    """
    pairs_by_split = {s: [] for s in SPLITS}

    for export in os.listdir(dataset_root):
        export_dir = os.path.join(dataset_root, export)
        if not os.path.isdir(export_dir):
            continue

        for split in SPLITS:
            img_dir = os.path.join(export_dir, split, "images")
            lbl_dir = os.path.join(export_dir, split, "labels")
            if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
                continue  # split not present in this export

            for file in os.listdir(img_dir):
                if os.path.splitext(file)[1].lower() not in IMAGE_EXTENSIONS:
                    continue
                img_path = os.path.join(img_dir, file)
                label_path = os.path.join(
                    lbl_dir, os.path.splitext(file)[0] + ".txt"
                )
                if os.path.exists(label_path):
                    pairs_by_split[split].append((img_path, label_path))

    return pairs_by_split


def copy_dataset(pairs, out_dir):
    if not pairs:
        return
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)

    for img_path, lbl_path in tqdm(
        pairs, desc=f"Copy → {os.path.basename(out_dir)}", ncols=80
    ):
        shutil.copy(img_path, os.path.join(out_dir, "images", os.path.basename(img_path)))
        shutil.copy(lbl_path, os.path.join(out_dir, "labels", os.path.basename(lbl_path)))


def find_max_class_id(pairs):
    max_id = -1
    for _, lbl_path in pairs:
        with open(lbl_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                cid = int(line.split()[0])
                if cid > max_id:
                    max_id = cid
    return max_id


def export_yaml(save_dir, nc, splits_present):
    data = {"nc": nc, "names": [str(i) for i in range(nc)]}

    if "train" in splits_present:
        data["train"] = os.path.abspath(os.path.join(save_dir, "train", "images"))
    if "valid" in splits_present:
        data["val"] = os.path.abspath(os.path.join(save_dir, "valid", "images"))
    if "test" in splits_present:
        data["test"] = os.path.abspath(os.path.join(save_dir, "test", "images"))

    yaml_path = os.path.join(save_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    print(f"[✔] Exported Ultralytics YAML → {yaml_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(args):
    dataset_root = args.root_path  # *direct* path to the exports folder
    output_root = args.output_path

    print(f"[INFO] Scanning: {dataset_root}")
    pairs_by_split = collect_image_label_pairs(dataset_root)

    # copy each split that actually exists
    for split, pairs in pairs_by_split.items():
        if not pairs:
            continue
        print(f"[INFO] {split.capitalize():5}: {len(pairs)} pairs")
        copy_dataset(pairs, os.path.join(output_root, split))

    all_pairs = [p for plist in pairs_by_split.values() for p in plist]
    if not all_pairs:
        raise RuntimeError("No image/label pairs found. Check directory layout.")

    nc = find_max_class_id(all_pairs) + 1
    export_yaml(output_root, nc, [s for s, p in pairs_by_split.items() if p])

    print(f"[✔] Dataset consolidation complete → {output_root}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolidate YOLOv11 exports with existing train/valid(/test) splits."
    )
    parser.add_argument(
        "root_path",
        type=str,
        help="Directory containing one or more '*.yolov11' export folders",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="processed_dataset",
        help="Destination folder for the unified dataset",
    )

    args = parser.parse_args()
    main(args)
