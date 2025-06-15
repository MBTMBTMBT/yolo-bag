python merge_and_split.py ./dataset/raw_dataset --output_path ./dataset/merged_dataset --train_ratio 0.8 --seed 8888
python augment_yolo.py --input_dir ./dataset/merged_dataset --output_dir ./dataset/augmented_dataset --augment_count 20
yolo task=detect mode=train \
  model=yolov8s.pt \
  data=./dataset/augmented_dataset/data.yaml \
  epochs=100 imgsz=640 batch=32 device=0 \
  name=yolo11_bag_exp seed=42 \
  lr0=0.0001 weight_decay=0.0001 \
  mosaic=1.0 fliplr=0.5
