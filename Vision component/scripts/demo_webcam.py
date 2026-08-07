import cv2
import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)

from engine.component import GenderDetectionComponent
from engine.schemas import GenderLabel


def draw_background(frame, x, y, w, h, color=(0, 0, 0), alpha=0.6):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def main():
    config_path = os.path.join(root_dir, "configs/config.yaml")

    print("Initializing Gender Detection Component...")
    try:
        component = GenderDetectionComponent.from_config(path=config_path)
    except Exception as e:
        print(f"Error initializing component: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam demo. Press 'q' to quit.")

    frame_idx = 0
    fps_start = time.time()
    fps_counter = 0
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_start = time.time()

        start_time = time.time()
        result = component.process_frame(frame, frame_idx)
        latency = (time.time() - start_time) * 1000

        h, w = frame.shape[:2]

        # Right-side info panel background
        panel_x = w - 380
        panel_y = 10
        draw_background(frame, panel_x, panel_y, 370, 250, (0, 0, 0), 0.65)

        y_offset = panel_y + 25
        cv2.putText(frame, "GENDER DETECTION", (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30

        cv2.putText(frame, f"Frame: {frame_idx}", (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 22
        cv2.putText(frame, f"FPS:   {fps_display}", (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 22
        cv2.putText(frame, f"Latency: {latency:.1f}ms", (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 30

        # Separator
        cv2.line(frame, (panel_x + 10, y_offset), (panel_x + 360, y_offset),
                 (100, 100, 100), 1)
        y_offset += 15

        if result.label != GenderLabel.NO_FACE:
            bbox = result.face_bbox
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

            color = (255, 180, 0) if result.label == GenderLabel.MALE else (255, 80, 120)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label above box
            label_text = f"{result.label.value.upper()}  {result.confidence:.2%}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Gender prediction
            cv2.putText(frame, "Prediction:", (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            gender_text = f"{result.label.value.upper()}  ({result.confidence:.2%})"
            cv2.putText(frame, gender_text, (panel_x + 130, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y_offset += 25

            # Face quality
            detection_str = f"{result.detection_score:.2%}" if result.detection_score else "N/A"
            cv2.putText(frame, "Face Quality:", (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            quality_color = (0, 255, 0) if result.detection_score and result.detection_score > 0.9 else \
                            (0, 255, 255) if result.detection_score and result.detection_score > 0.8 else \
                            (0, 165, 255)
            cv2.putText(frame, detection_str, (panel_x + 130, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, quality_color, 2)
            y_offset += 25

            # Source
            source_label = "CACHE" if "cache" in result.source else \
                           "SKIPPED" if "skip" in result.source else \
                           "INFERENCE"
            source_color = (255, 200, 0) if "cache" in result.source else \
                           (100, 100, 100) if "skip" in result.source else \
                           (0, 255, 0)
            cv2.putText(frame, "Source:", (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.putText(frame, source_label, (panel_x + 130, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, source_color, 2)
            y_offset += 25

            # Smoothed info
            cv2.putText(frame, "Smoothed:", (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.putText(frame, "YES" if result.is_smoothed else "NO",
                        (panel_x + 130, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if result.is_smoothed else (100, 100, 100), 1)

        else:
            cv2.putText(frame, "Prediction:", (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.putText(frame, "NO FACE", (panel_x + 130, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 2)

        cv2.imshow("Gender Detection Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
