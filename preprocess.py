#!/usr/bin/env python3
"""
Merge multiple Ultralytics (.yolov11) exports into one raw_dataset.
Any `test/` split that exists inside an export is appended to `valid/`
so the final structure only has `train/` and `valid/`.

Source layout under <root_path> (one or more exports):

    paper bag.v1i.yolov11/
    ├─ train/images, train/labels
    ├─ valid/images, valid/labels
    └─ test/images,  test/labels   # optional, will be merged into valid/
"""

import argparse
import os
import shutil
from tqdm import tqdm
import yaml

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp"}
SPLITS = ["train", "valid", "test"]  # we still recognise "test"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def collect_pairs(dataset_root):
    """
    Return a dict:  {split: [(img, lbl), ...]}.
    """
    split_pairs = {s: [] for s in SPLITS}

    for export in os.listdir(dataset_root):
        exp_dir = os.path.join(dataset_root, export)
        if not os.path.isdir(exp_dir):
            continue

        for split in SPLITS:
            img_dir = os.path.join(exp_dir, split, "images")
            lbl_dir = os.path.join(exp_dir, split, "labels")
            if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
                continue

            for fname in os.listdir(img_dir):
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
                    continue
                img_path = os.path.join(img_dir, fname)
                lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
                if os.path.exists(lbl_path):
                    split_pairs[split].append((img_path, lbl_path))

    return split_pairs


def copy_pairs(pairs, out_split_dir):
    if not pairs:
        return
    img_out = os.path.join(out_split_dir, "images")
    lbl_out = os.path.join(out_split_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for img, lbl in tqdm(
        pairs, desc=f"Copy → {os.path.basename(out_split_dir)}", ncols=80
    ):
        shutil.copy(img, os.path.join(img_out, os.path.basename(img)))
        shutil.copy(lbl, os.path.join(lbl_out, os.path.basename(lbl)))


def max_class_id(all_pairs):
    m = -1
    for _, lbl in all_pairs:
        with open(lbl, "r") as f:
            for line in f:
                if line.strip():
                    cid = int(line.split()[0])
                    m = max(m, cid)
    return m


def write_yaml(save_dir, nc):
    data = {
        "train": os.path.abspath(os.path.join(save_dir, "train", "images")),
        "val": os.path.abspath(os.path.join(save_dir, "valid", "images")),
        "nc": nc,
        "names": [str(i) for i in range(nc)],
    }
    yaml_path = os.path.join(save_dir, "raw_dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f)
    print(f"[✔] Wrote YAML → {yaml_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(args):
    dataset_root = args.root_path
    out_root = args.output_path

    print(f"[INFO] Scanning {dataset_root}")
    pairs = collect_pairs(dataset_root)

    # merge test into valid
    pairs["valid"].extend(pairs["test"])
    del pairs["test"]

    # copy train & valid
    for split in ["train", "valid"]:
        copy_pairs(pairs[split], os.path.join(out_root, split))
        print(f"[INFO] {split.capitalize():5}: {len(pairs[split])} pairs")

    all_pairs = pairs["train"] + pairs["valid"]
    if not all_pairs:
        raise RuntimeError("No images/labels found – check directory layout.")

    nc = max_class_id(all_pairs) + 1
    write_yaml(out_root, nc)

    print(f"[✔] Dataset ready at {out_root}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge YOLOv11 exports; fold test/ into valid/."
    )
    parser.add_argument(
        "root_path",
        nargs="?",
        default=".",
        help="Folder containing *.yolov11 exports (default: current dir)",
    )
    parser.add_argument(
        "--output_path",
        default="processed_dataset",
        help="Destination folder for merged raw_dataset",
    )
    main(parser.parse_args())
