#!/usr/bin/env python3
"""
Merge multiple Ultralytics (.yolov11) exports into one raw_dataset.
Collect all data from various split names (train/val/valid/validation/eval/test)
and redistribute them according to specified train/valid ratio.

Source layout under <root_path> (one or more exports):

    paper bag.v1i.yolov11/
    ├─ train/images, train/labels
    ├─ valid/images, valid/labels
    ├─ test/images,  test/labels
    ├─ validation/images, validation/labels  # any combination possible
    └─ data.yaml  # YAML configuration file (optional)
"""

import argparse
import os
import random
import shutil
from tqdm import tqdm
import yaml

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp"}
# All possible split names we want to recognize
SPLIT_VARIANTS = ["train", "training", "val", "valid", "validation", "eval", "evaluate", "test", "testing"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def find_and_read_yaml_files(dataset_root):
    """
    Find and read all data.yaml files in the dataset exports.
    Returns merged class names and metadata, prioritizing existing YAML files.
    """
    all_class_names = {}
    yaml_metadata = {}
    yaml_files_found = []

    print("[INFO] Searching for existing data.yaml files...")

    for export in os.listdir(dataset_root):
        exp_dir = os.path.join(dataset_root, export)
        if not os.path.isdir(exp_dir):
            continue

        # Look for data.yaml or dataset.yaml files
        yaml_candidates = ["data.yaml", "dataset.yaml", "config.yaml"]
        yaml_path = None

        for yaml_name in yaml_candidates:
            candidate_path = os.path.join(exp_dir, yaml_name)
            if os.path.exists(candidate_path):
                yaml_path = candidate_path
                break

        if yaml_path:
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f)

                yaml_files_found.append(f"{export}/{os.path.basename(yaml_path)}")
                print(f"[INFO] Reading existing YAML: {export}/{os.path.basename(yaml_path)}")

                # Extract class names if available
                if 'names' in yaml_data:
                    names = yaml_data['names']
                    if isinstance(names, list):
                        # Convert list to dict with indices
                        for idx, name in enumerate(names):
                            if idx not in all_class_names:  # Don't overwrite existing names
                                all_class_names[idx] = name
                    elif isinstance(names, dict):
                        # Merge dict-style names
                        for idx, name in names.items():
                            key = int(idx)
                            if key not in all_class_names:  # Don't overwrite existing names
                                all_class_names[key] = name

                # Store other metadata from first YAML file
                if not yaml_metadata:
                    for key in ['nc', 'path', 'description']:
                        if key in yaml_data:
                            yaml_metadata[key] = yaml_data[key]

            except Exception as e:
                print(f"[WARNING] Error reading {yaml_path}: {e}")

    if yaml_files_found:
        print(f"[INFO] Successfully read {len(yaml_files_found)} YAML files")
        print(f"[INFO] Found {len(all_class_names)} class definitions")
    else:
        print("[INFO] No existing YAML files found, will generate default class names")

    return all_class_names, yaml_metadata, len(yaml_files_found) > 0


def collect_all_pairs(dataset_root):
    """
    Collect all image-label pairs from all possible split directories.
    Returns a list of (img_path, lbl_path) tuples.
    """
    all_pairs = []

    for export in os.listdir(dataset_root):
        exp_dir = os.path.join(dataset_root, export)
        if not os.path.isdir(exp_dir):
            continue

        # Check all possible split variant names
        for split_name in SPLIT_VARIANTS:
            img_dir = os.path.join(exp_dir, split_name, "images")
            lbl_dir = os.path.join(exp_dir, split_name, "labels")

            if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
                continue

            print(f"[INFO] Found split: {export}/{split_name}")

            for fname in os.listdir(img_dir):
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
                    continue

                img_path = os.path.join(img_dir, fname)
                lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")

                if os.path.exists(lbl_path):
                    all_pairs.append((img_path, lbl_path))

    return all_pairs


def split_pairs_by_ratio(all_pairs, train_ratio, seed=42):
    """
    Split all pairs into train and valid sets according to the given ratio.

    Args:
        all_pairs: List of (img_path, lbl_path) tuples
        train_ratio: Float between 0 and 1, proportion for training set
        seed: Random seed for reproducible splits

    Returns:
        dict: {"train": [...], "valid": [...]}
    """
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1")

    # Shuffle pairs for random distribution
    random.seed(seed)
    shuffled_pairs = all_pairs.copy()
    random.shuffle(shuffled_pairs)

    # Calculate split point
    total_count = len(shuffled_pairs)
    train_count = int(total_count * train_ratio)

    return {
        "train": shuffled_pairs[:train_count],
        "valid": shuffled_pairs[train_count:]
    }


def copy_pairs(pairs, out_split_dir):
    """
    Copy image-label pairs to the output directory structure.
    """
    if not pairs:
        return

    img_out = os.path.join(out_split_dir, "images")
    lbl_out = os.path.join(out_split_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for img, lbl in tqdm(
            pairs, desc=f"Copy → {os.path.basename(out_split_dir)}", ncols=80
    ):
        # Generate unique filename to avoid conflicts
        img_basename = os.path.basename(img)
        lbl_basename = os.path.basename(lbl)

        # Handle filename conflicts by adding suffix
        img_out_path = os.path.join(img_out, img_basename)
        lbl_out_path = os.path.join(lbl_out, lbl_basename)

        counter = 1
        while os.path.exists(img_out_path):
            name, ext = os.path.splitext(img_basename)
            img_out_path = os.path.join(img_out, f"{name}_{counter}{ext}")
            lbl_out_path = os.path.join(lbl_out, f"{os.path.splitext(lbl_basename)[0]}_{counter}.txt")
            counter += 1

        shutil.copy(img, img_out_path)
        shutil.copy(lbl, lbl_out_path)


def max_class_id(all_pairs):
    """
    Find the maximum class ID across all label files.
    """
    max_id = -1
    for _, lbl_path in all_pairs:
        try:
            with open(lbl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        max_id = max(max_id, class_id)
        except (ValueError, IndexError, IOError) as e:
            print(f"[WARNING] Error reading {lbl_path}: {e}")
            continue
    return max_id


def generate_default_class_names(nc):
    """
    Generate default class names when no YAML files are found.
    """
    print("[INFO] Generating default class names...")
    return {i: f"class_{i}" for i in range(nc)}


def write_yaml(save_dir, nc, class_names=None, metadata=None, has_existing_yaml=False):
    """
    Write the dataset configuration YAML file.

    Args:
        save_dir: Output directory path
        nc: Number of classes
        class_names: Dict mapping class indices to names (None to generate defaults)
        metadata: Additional metadata from original YAML files
        has_existing_yaml: Whether we found existing YAML files
    """
    # Determine class names strategy
    if class_names and has_existing_yaml:
        print("[INFO] Using class names from existing YAML files")
        # Use provided class names, fill missing indices with defaults
        names = {}
        for i in range(nc):
            if i in class_names:
                names[i] = class_names[i]
            else:
                names[i] = f"class_{i}"  # Default for missing classes
                print(f"[INFO] Generated default name for missing class {i}: class_{i}")
    else:
        print("[INFO] No existing YAML found, generating default class names")
        # Generate all default class names
        names = generate_default_class_names(nc)

    # Build the YAML data structure
    data = {
        "path": os.path.abspath(save_dir),  # Root directory
        "train": os.path.join("train", "images"),  # Relative to path
        "val": os.path.join("valid", "images"),  # Relative to path
        "nc": nc,
        "names": names,
    }

    # Add metadata if available
    if metadata:
        if 'description' in metadata:
            data['description'] = metadata['description']
        else:
            data['description'] = 'Merged dataset from multiple YOLO exports'
    else:
        data['description'] = 'Merged dataset from multiple YOLO exports'

    # Add creation info
    data['created_by'] = 'Dataset Merger Script'

    # Write YAML file
    yaml_path = os.path.join(save_dir, "data.yaml")
    with open(yaml_path, "w", encoding='utf-8') as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"[✔] Wrote YAML → {yaml_path}")

    # Also write a legacy format for backward compatibility
    legacy_data = {
        "train": os.path.abspath(os.path.join(save_dir, "train", "images")),
        "val": os.path.abspath(os.path.join(save_dir, "valid", "images")),
        "nc": nc,
        "names": list(names.values()),  # Convert to list format
    }

    legacy_yaml_path = os.path.join(save_dir, "raw_dataset.yaml")
    with open(legacy_yaml_path, "w", encoding='utf-8') as f:
        yaml.safe_dump(legacy_data, f, default_flow_style=False, allow_unicode=True)

    print(f"[✔] Wrote legacy YAML → {legacy_yaml_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(args):
    dataset_root = args.root_path
    out_root = args.output_path
    train_ratio = args.train_ratio
    seed = args.seed

    print(f"[INFO] Scanning {dataset_root} for all split variants")
    print(f"[INFO] Target train/valid ratio: {train_ratio:.2f}/{1 - train_ratio:.2f}")

    # Try to read existing YAML configurations first
    class_names, yaml_metadata, has_existing_yaml = find_and_read_yaml_files(dataset_root)

    if has_existing_yaml and class_names:
        print(f"[INFO] Loaded {len(class_names)} class names from existing YAML files:")
        for idx, name in sorted(class_names.items()):
            print(f"  {idx}: {name}")

    # Collect all pairs from all possible split directories
    all_pairs = collect_all_pairs(dataset_root)

    if not all_pairs:
        raise RuntimeError("No valid image-label pairs found. Check directory structure.")

    print(f"[INFO] Found {len(all_pairs)} total image-label pairs")

    # Split pairs according to specified ratio
    split_pairs = split_pairs_by_ratio(all_pairs, train_ratio, seed)

    # Create output directory structure and copy files
    os.makedirs(out_root, exist_ok=True)

    for split_name, pairs in split_pairs.items():
        split_dir = os.path.join(out_root, split_name)
        copy_pairs(pairs, split_dir)
        print(f"[INFO] {split_name.capitalize():5}: {len(pairs)} pairs ({len(pairs) / len(all_pairs) * 100:.1f}%)")

    # Calculate number of classes
    nc = max_class_id(all_pairs) + 1

    # Validate class names coverage if we have existing YAML
    if has_existing_yaml and class_names:
        missing_classes = set(range(nc)) - set(class_names.keys())
        if missing_classes:
            print(f"[WARNING] Missing class names for IDs: {sorted(missing_classes)}")
            print("[INFO] Will generate default names for missing classes")

    write_yaml(out_root, nc, class_names, yaml_metadata, has_existing_yaml)

    print(f"[✔] Dataset ready at {out_root}")
    print(f"[✔] Total classes detected: {nc}")

    if has_existing_yaml:
        print(f"[✔] Used existing YAML configurations from source datasets")
    else:
        print(f"[✔] Generated default class names (no existing YAML found)")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge YOLOv11 exports from various split names and redistribute by ratio."
    )
    parser.add_argument(
        "root_path",
        nargs="?",
        default=".",
        help="Folder containing dataset exports (default: current dir)",
    )
    parser.add_argument(
        "--output_path",
        default="processed_dataset",
        help="Destination folder for merged dataset (default: processed_dataset)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training set (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )

    args = parser.parse_args()

    # Validate train_ratio
    if not (0 < args.train_ratio < 1):
        parser.error("train_ratio must be between 0 and 1")

    main(args)
