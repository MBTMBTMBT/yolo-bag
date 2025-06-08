import os
import glob
import argparse

def convert_polygon_to_bbox(polygon_coords):
    """
    Convert polygon points to YOLO-format bbox (x_center, y_center, w, h)

    :param polygon_coords: List of float numbers [x1, y1, x2, y2, ..., xn, yn]
    :return: x_center, y_center, width, height (all normalized)
    """
    x_coords = polygon_coords[0::2]
    y_coords = polygon_coords[1::2]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height

def convert_mask_txt_to_yolo_bbox(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

    if not txt_files:
        print(f"[!] No .txt files found in '{input_dir}'")
        return

    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            lines = f.readlines()

        converted_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # skip invalid lines

            class_id = parts[0]
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                continue  # skip malformed polygon

            x_center, y_center, width, height = convert_polygon_to_bbox(coords)
            converted_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            converted_lines.append(converted_line)

        output_path = os.path.join(output_dir, os.path.basename(txt_file))
        with open(output_path, "w") as f_out:
            for line in converted_lines:
                f_out.write(line + "\n")

    print(f"[✔] Converted {len(txt_files)} files to YOLO bbox format in '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert polygon-format YOLO masks to YOLO bbox format")
    parser.add_argument("input_dir", type=str, help="Input folder containing polygon .txt files")
    parser.add_argument("output_dir", type=str, help="Output folder to save bbox .txt files")
    args = parser.parse_args()

    convert_mask_txt_to_yolo_bbox(args.input_dir, args.output_dir)
