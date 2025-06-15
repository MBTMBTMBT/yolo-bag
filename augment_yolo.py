import os
import cv2
import numpy as np
import random
import argparse
from pathlib import Path
import shutil
from typing import List, Tuple
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from tqdm import tqdm


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

        # Create output directory structure
        self.setup_output_dirs()

        # Define augmentation pipeline with moderate parameters
        self.transform = A.Compose([
            # Geometric transformations
            A.ShiftScaleRotate(
                shift_limit=0.1,  # 10% shift
                scale_limit=0.2,  # 20% scale change
                rotate_limit=15,  # 15 degree rotation
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.8
            ),

            # Resize with aspect ratio preservation
            A.LongestMaxSize(max_size=640, p=0.3),
            A.PadIfNeeded(min_height=640, min_width=640,
                          border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3),

            # Color and brightness adjustments
            A.RandomBrightnessContrast(
                brightness_limit=0.2,  # 20% brightness change
                contrast_limit=0.2,  # 20% contrast change
                p=0.6
            ),

            A.HueSaturationValue(
                hue_shift_limit=10,  # Slight hue shift
                sat_shift_limit=15,  # Slight saturation change
                val_shift_limit=10,  # Slight value change
                p=0.5
            ),

            # Noise and blur
            A.GaussNoise(
                var_limit=(5.0, 15.0),  # Light gaussian noise
                p=0.3
            ),

            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
            ], p=0.2),

            # Weather effects (light)
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.2
            ),

        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3  # Keep boxes with at least 30% visibility
        ))

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
                # Ensure bbox coordinates are within [0, 1] range
                bbox = [max(0, min(1, coord)) for coord in bbox]
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

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

            return (
                transformed['image'],
                transformed['class_labels'],
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
