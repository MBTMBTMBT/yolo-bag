import cv2
import argparse
import time
from ultralytics import YOLO


class UltralyticsWebcamTester:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def run(self, camera_id: int = 0, save_detections: bool = False):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        print("Starting YOLO webcam detection...")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")

        frame_count = 0
        fps_counter = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)[0]

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                name = self.model.names.get(cls_id, str(cls_id))
                color = (0, 255, 0)

                label = f"{name}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Overlay info
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_counter)
                fps_counter = time.time()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Ultralytics YOLO Webcam", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and save_detections:
                filename = f"yolo_detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved frame as {filename}")

        cap.release()
        cv2.destroyAllWindows()
        print("Webcam test completed.")


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLOv8/v11 Webcam Tester")
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv8/v11 .pt model')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--save', action='store_true', help='Save detected frames')
    parser.add_argument('--conf', type=float, default=0.5, help='Initial confidence threshold')
    args = parser.parse_args()

    tester = UltralyticsWebcamTester(args.model, conf_threshold=args.conf)
    tester.run(camera_id=args.camera, save_detections=args.save)


if __name__ == "__main__":
    main()
