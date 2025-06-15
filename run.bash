python merge_and_split.py ./dataset/raw_dataset --output_path ./dataset/merged_dataset --train_ratio 0.8 --seed 8888
python augment_yolo.py --input_dir ./dataset/merged_dataset --output_dir ./dataset/augmented_dataset --augment_count 20
