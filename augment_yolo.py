import os
import cv2
import numpy as np
import random
import argparse
from pathlib import Path
import shutil
from typing import List, Tuple, Dict, Optional
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from tqdm import tqdm
import yaml


class YOLODataAugmentation:
    """
    YOLO data augmentation pipeline that applies various transformations
    while maintaining proper bounding box coordinates
    """

    def __init__(self, output_dir: str, augment_count: int = 3):
        """
        Initialize the augmentation pipeline

        Args:
            output_dir: Directory to save augmented dataset
            augment_count: Number of augmented versions per original image
        """
        self.output_dir = Path(output_dir)
        self.augment_count = augment_count
        self.class_names = {}
        self.yaml_metadata = {}
        self.has_existing_yaml = False

        # Create output directory structure
        self.setup_output_dirs()

        # Define augmentation pipeline with enhanced parameters for better small object detection
        self.transform = A.Compose([
            # Geometric transformations with increased distortion
            A.ShiftScaleRotate(
                shift_limit=0.15,  # 15% shift (increased from 10%)
                scale_limit=(-0.50, 0.0),  # 50% scale change (increased from 20%)
                rotate_limit=25,  # 25 degree rotation (increased from 15)
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.9  # Increased probability to 90%
            ),

            # Random perspective distortion to simulate different viewing angles
            A.Perspective(
                scale=(0.05, 0.1),  # Add perspective transformation
                p=0.3
            ),

            # Force resize to exact 480x640 resolution
            A.Resize(height=480, width=640, p=1.0),  # Always resize to exact dimensions

            # Random crops to simulate objects at different distances
            A.RandomSizedBBoxSafeCrop(
                height=480,
                width=640,
                erosion_rate=0.2,  # Helps maintain small objects
                p=0.3
            ),

            # Apply zoom-out effect to simulate faraway objects
            # This will make the bag appear smaller in some augmentations
            A.RandomScale(
                scale_limit=(-0.3, 0),  # Scale down by up to 30%
                p=0.4
            ),

            # Pad to maintain dimensions
            A.PadIfNeeded(
                min_height=480,
                min_width=640,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),

            # Enhanced color and brightness adjustments
            A.RandomBrightnessContrast(
                brightness_limit=0.35,  # 35% brightness change (increased from 20%)
                contrast_limit=0.35,  # 35% contrast change (increased from 20%)
                p=0.7  # Increased probability
            ),

            A.HueSaturationValue(
                hue_shift_limit=20,  # Stronger hue shift (increased from 10)
                sat_shift_limit=30,  # Stronger saturation change (increased from 15)
                val_shift_limit=20,  # Stronger value change (increased from 10)
                p=0.6  # Increased probability
            ),

            # Simulate different lighting conditions
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.3
            ),

            # Increased noise and blur effects
            A.GaussNoise(
                var_limit=(10.0, 40.0),  # Stronger noise (increased from 5.0-15.0)
                p=0.4  # Increased probability
            ),

            A.OneOf([
                A.MotionBlur(blur_limit=5, p=0.4),  # Increased blur limit
                A.GaussianBlur(blur_limit=5, p=0.4),  # Increased blur limit
                A.MedianBlur(blur_limit=5, p=0.2),  # Added median blur
            ], p=0.3),  # Increased overall blur probability

            # Enhanced weather effects
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,  # More shadows possible
                shadow_dimension=7,  # Larger shadow dimension
                p=0.3  # Increased probability
            ),

            # Add occasional fog/haze to simulate distance
            A.RandomFog(
                fog_coef_lower=0.1,
                fog_coef_upper=0.3,
                alpha_coef=0.1,
                p=0.2
            ),

            # Occasionally reduce image quality to simulate poor conditions
            A.ImageCompression(
                quality_lower=70,
                quality_upper=99,
                p=0.3
            ),

        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.2  # Reduced from 0.3 to keep more partially visible boxes
        ))

    def find_and_read_yaml_file(self, input_dir: str) -> bool:
        """
        Find and read existing data.yaml file from input directory.
        Returns True if YAML was found and read successfully.

        Args:
            input_dir: Path to input dataset directory

        Returns:
            bool: True if existing YAML was found and loaded
        """
        input_path = Path(input_dir)

        print("Searching for existing data.yaml file...")

        # Look for YAML files in the input directory
        yaml_candidates = ["data.yaml", "dataset.yaml", "config.yaml"]
        yaml_path = None

        for yaml_name in yaml_candidates:
            candidate_path = input_path / yaml_name
            if candidate_path.exists():
                yaml_path = candidate_path
                break

        if yaml_path:
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f)

                print(f"Found and reading existing YAML: {yaml_path.name}")

                # Extract class names if available
                if 'names' in yaml_data:
                    names = yaml_data['names']
                    if isinstance(names, list):
                        # Convert list to dict with indices
                        for idx, name in enumerate(names):
                            self.class_names[idx] = name
                    elif isinstance(names, dict):
                        # Use dict-style names directly
                        for idx, name in names.items():
                            self.class_names[int(idx)] = name

                # Store other metadata
                for key in ['nc', 'path', 'description']:
                    if key in yaml_data:
                        self.yaml_metadata[key] = yaml_data[key]

                print(f"Loaded {len(self.class_names)} class names from existing YAML")
                self.has_existing_yaml = True
                return True

            except Exception as e:
                print(f"Warning: Error reading {yaml_path}: {e}")

        print("No existing YAML file found, will generate default class names")
        return False

    def generate_default_class_names(self, nc: int) -> Dict[int, str]:
        """
        Generate default class names when no YAML file is found.

        Args:
            nc: Number of classes

        Returns:
            Dict mapping class indices to default names
        """
        print("Generating default class names...")
        return {i: f"class_{i}" for i in range(nc)}

    def max_class_id(self, input_dir: str) -> int:
        """
        Find the maximum class ID across all label files in the dataset.

        Args:
            input_dir: Path to input dataset directory

        Returns:
            Maximum class ID found
        """
        input_path = Path(input_dir)
        max_id = -1

        for split in ['train', 'valid']:
            split_path = input_path / split
            if not split_path.exists():
                continue

            labels_dir = split_path / 'labels'
            if not labels_dir.exists():
                continue

            # Check all label files in this split
            for label_file in labels_dir.glob('*.txt'):
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                class_id = int(line.split()[0])
                                max_id = max(max_id, class_id)
                except (ValueError, IndexError, IOError) as e:
                    print(f"Warning: Error reading {label_file}: {e}")
                    continue

        return max_id

    def write_yaml_config(self, nc: int):
        """
        Write the dataset configuration YAML file.

        Args:
            nc: Number of classes detected in the dataset
        """
        # Determine class names strategy
        if self.class_names and self.has_existing_yaml:
            print("Using class names from existing YAML file")
            # Use existing class names, fill missing indices with defaults
            names = {}
            for i in range(nc):
                if i in self.class_names:
                    names[i] = self.class_names[i]
                else:
                    names[i] = f"class_{i}"  # Default for missing classes
                    print(f"Generated default name for missing class {i}: class_{i}")
        else:
            print("No existing YAML found, generating default class names")
            # Generate all default class names
            names = self.generate_default_class_names(nc)

        # Build the YAML data structure
        data = {
            "path": str(self.output_dir.absolute()),  # Root directory
            "train": os.path.join("train", "images"),  # Relative to path
            "val": os.path.join("valid", "images"),  # Relative to path
            "nc": nc,
            "names": names,
        }

        # Add metadata if available
        if self.yaml_metadata:
            if 'description' in self.yaml_metadata:
                data['description'] = self.yaml_metadata['description']
            else:
                data['description'] = 'Augmented dataset from YOLO data augmentation pipeline'
        else:
            data['description'] = 'Augmented dataset from YOLO data augmentation pipeline'

        # Add creation info
        data['created_by'] = 'YOLO Data Augmentation Pipeline'

        # Write main YAML file
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w", encoding='utf-8') as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"Wrote YAML config → {yaml_path}")

        # Also write a legacy format for backward compatibility
        legacy_data = {
            "train": str((self.output_dir / "train" / "images").absolute()),
            "val": str((self.output_dir / "valid" / "images").absolute()),
            "nc": nc,
            "names": list(names.values()),  # Convert to list format
        }

        legacy_yaml_path = self.output_dir / "augmented_dataset.yaml"
        with open(legacy_yaml_path, "w", encoding='utf-8') as f:
            yaml.safe_dump(legacy_data, f, default_flow_style=False, allow_unicode=True)

        print(f"Wrote legacy YAML → {legacy_yaml_path}")

    def setup_output_dirs(self):
        """Create output directory structure"""
        for split in ['train', 'valid']:
            for subdir in ['images', 'labels']:
                (self.output_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    def parse_yolo_label(self, label_path: str) -> Tuple[List[int], List[List[float]]]:
        """
        Parse YOLO format label file

        Args:
            label_path: Path to label file

        Returns:
            Tuple of (class_ids, bboxes) where bboxes are in YOLO format [x_center, y_center, width, height]
        """
        class_ids = []
        bboxes = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        class_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        class_ids.append(class_id)
                        bboxes.append(bbox)

        return class_ids, bboxes

    def save_yolo_label(self, label_path: str, class_ids: List[int], bboxes: List[List[float]]):
        """
        Save YOLO format label file

        Args:
            label_path: Path to save label file
            class_ids: List of class IDs
            bboxes: List of bounding boxes in YOLO format
        """
        with open(label_path, 'w') as f:
            for class_id, bbox in zip(class_ids, bboxes):
                class_id_int = int(class_id)
                # Ensure bbox coordinates are within [0, 1] range
                bbox = [max(0, min(1, coord)) for coord in bbox]
                f.write(f"{class_id_int} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

    def augment_image_and_labels(self, image: np.ndarray, class_ids: List[int],
                                 bboxes: List[List[float]]) -> Tuple[np.ndarray, List[int], List[List[float]]]:
        """
        Apply augmentation to image and corresponding labels

        Args:
            image: Input image
            class_ids: List of class IDs
            bboxes: List of bounding boxes in YOLO format

        Returns:
            Tuple of (augmented_image, augmented_class_ids, augmented_bboxes)
        """
        if len(bboxes) == 0:
            # If no bounding boxes, apply image-only transformations
            image_transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
            ])
            transformed = image_transform(image=image)
            return transformed['image'], class_ids, bboxes

        try:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_ids
            )

            int_labels = [int(c) for c in transformed['class_labels']]

            return (
                transformed['image'],
                int_labels,
                transformed['bboxes']
            )
        except Exception as e:
            print(f"Augmentation failed: {e}, returning original")
            return image, class_ids, bboxes

    def get_total_operations(self, input_dir: str) -> int:
        """
        Calculate total number of operations for progress tracking

        Args:
            input_dir: Path to input dataset directory

        Returns:
            Total number of operations (original copies + augmentations)
        """
        input_path = Path(input_dir)
        total_operations = 0

        for split in ['train', 'valid']:
            split_path = input_path / split
            if not split_path.exists():
                continue

            images_dir = split_path / 'images'
            if not images_dir.exists():
                continue

            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(images_dir.glob(f'*{ext}')))
                image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))

            # Each image: 1 original copy + N augmentations
            total_operations += len(image_files) * (1 + self.augment_count)

        return total_operations

    def process_dataset(self, input_dir: str):
        """
        Process the entire dataset and create augmented versions

        Args:
            input_dir: Path to input dataset directory
        """
        input_path = Path(input_dir)

        # Try to read existing YAML configuration first
        self.find_and_read_yaml_file(input_dir)

        # Calculate total operations for progress tracking
        total_operations = self.get_total_operations(input_dir)

        # Initialize global progress bar
        with tqdm(total=total_operations, desc="Processing dataset", unit="images") as pbar:

            for split in ['train', 'valid']:
                split_path = input_path / split
                if not split_path.exists():
                    print(f"Warning: {split} directory not found, skipping...")
                    continue

                images_dir = split_path / 'images'
                labels_dir = split_path / 'labels'

                if not images_dir.exists():
                    print(f"Warning: {images_dir} not found, skipping {split}...")
                    continue

                # Get all image files
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(list(images_dir.glob(f'*{ext}')))
                    image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))

                print(f"\nProcessing {len(image_files)} images in {split} split...")

                # Process each image with detailed progress
                for img_path in image_files:
                    # Update progress bar description with current file
                    pbar.set_description(f"Processing {split}: {img_path.name}")

                    # Copy original image and label
                    self.copy_original(img_path, labels_dir, split)
                    pbar.update(1)  # Update progress for original copy

                    # Create augmented versions
                    for aug_idx in range(self.augment_count):
                        pbar.set_description(
                            f"Augmenting {split}: {img_path.name} ({aug_idx + 1}/{self.augment_count})")
                        self.create_augmented_version(img_path, labels_dir, split, aug_idx)
                        pbar.update(1)  # Update progress for each augmentation

                # Update description after completing each split
                pbar.set_description(f"Completed {split} split")

        # Calculate number of classes and write YAML config
        nc = self.max_class_id(input_dir) + 1

        # Validate class names coverage if we have existing YAML
        if self.has_existing_yaml and self.class_names:
            missing_classes = set(range(nc)) - set(self.class_names.keys())
            if missing_classes:
                print(f"Warning: Missing class names for IDs: {sorted(missing_classes)}")
                print("Will generate default names for missing classes")

        self.write_yaml_config(nc)

        print(f"\nTotal classes detected: {nc}")
        if self.has_existing_yaml:
            print("Used existing YAML configuration from source dataset")
        else:
            print("Generated default class names (no existing YAML found)")

    def copy_original(self, img_path: Path, labels_dir: Path, split: str):
        """Copy original image and label to output directory"""
        # Copy image
        output_img_path = self.output_dir / split / 'images' / img_path.name
        shutil.copy2(img_path, output_img_path)

        # Copy label if exists
        label_path = labels_dir / f"{img_path.stem}.txt"
        output_label_path = self.output_dir / split / 'labels' / f"{img_path.stem}.txt"

        if label_path.exists():
            shutil.copy2(label_path, output_label_path)
        else:
            # Create empty label file
            output_label_path.touch()

    def create_augmented_version(self, img_path: Path, labels_dir: Path, split: str, aug_idx: int):
        """Create one augmented version of the image and label"""
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load labels
        label_path = labels_dir / f"{img_path.stem}.txt"
        class_ids, bboxes = self.parse_yolo_label(str(label_path))

        # Apply augmentation
        aug_image, aug_class_ids, aug_bboxes = self.augment_image_and_labels(
            image, class_ids, bboxes
        )

        # Save augmented image
        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
        aug_img_name = f"{img_path.stem}_aug_{aug_idx}{img_path.suffix}"
        output_img_path = self.output_dir / split / 'images' / aug_img_name
        cv2.imwrite(str(output_img_path), aug_image_bgr)

        # Save augmented labels
        aug_label_name = f"{img_path.stem}_aug_{aug_idx}.txt"
        output_label_path = self.output_dir / split / 'labels' / aug_label_name
        self.save_yolo_label(str(output_label_path), aug_class_ids, aug_bboxes)


def main():
    """Main function to run the augmentation pipeline"""
    parser = argparse.ArgumentParser(description='YOLO Dataset Augmentation Pipeline')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to input dataset directory (containing train and valid folders)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output augmented dataset directory')
    parser.add_argument('--augment_count', type=int, default=3,
                        help='Number of augmented versions per original image (default: 3)')

    args = parser.parse_args()

    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {args.input_dir} does not exist!")
        return

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Augmentations per image: {args.augment_count}")
    print("-" * 50)

    # Initialize and run augmentation
    augmenter = YOLODataAugmentation(args.output_dir, args.augment_count)

    # Calculate and display total expected files
    total_ops = augmenter.get_total_operations(args.input_dir)
    print(f"Total operations to perform: {total_ops}")
    print("Starting augmentation process...\n")

    augmenter.process_dataset(args.input_dir)

    print("\nAugmentation completed successfully!")
    print(f"Augmented dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

# Usage example:
# python augment_yolo.py --input_dir ./dataset/merged_dataset --output_dir ./dataset/augmented_dataset --augment_count 20
