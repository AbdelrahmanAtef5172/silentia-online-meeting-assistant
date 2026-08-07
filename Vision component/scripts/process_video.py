"""
scripts/process_video.py
────────────────────────
Offline batch processor for video files.
Extracts gender predictions for every frame and saves them to a JSON file.
Optionally exports an MP4 with the detection overlay drawn on each frame.
"""

import cv2
import os
import sys
import json
import time
import imageio
from tqdm import tqdm
from argparse import ArgumentParser

# Add root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)

from engine.component import GenderDetectionComponent
from engine.schemas import GenderLabel


def draw_overlay(frame, result, frame_idx, latency_ms, fps_display):
    h, w = frame.shape[:2]
    panel_x = w - 380
    panel_y = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + 370, panel_y + 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

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
    cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (panel_x + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += 30

    cv2.line(frame, (panel_x + 10, y_offset), (panel_x + 360, y_offset),
             (100, 100, 100), 1)
    y_offset += 15

    if result.label != GenderLabel.NO_FACE:
        bbox = result.face_bbox
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

        color = (255, 180, 0) if result.label == GenderLabel.MALE else (255, 80, 120)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text = f"{result.label.value.upper()}  {result.confidence:.2%}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.putText(frame, "Prediction:", (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        gender_text = f"{result.label.value.upper()}  ({result.confidence:.2%})"
        cv2.putText(frame, gender_text, (panel_x + 130, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y_offset += 25

        detection_str = f"{result.detection_score:.2%}" if result.detection_score else "N/A"
        cv2.putText(frame, "Face Quality:", (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        quality_color = (0, 255, 0) if result.detection_score and result.detection_score > 0.9 else \
                        (0, 255, 255) if result.detection_score and result.detection_score > 0.8 else \
                        (0, 165, 255)
        cv2.putText(frame, detection_str, (panel_x + 130, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, quality_color, 2)
        y_offset += 25

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


def main():
    parser = ArgumentParser(description="Process a video file for gender detection.")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, help="Path to output JSON file (default: input_base.json)")
    parser.add_argument("--output-video", type=str, help="Path to output MP4 with overlay (optional, H.264)")
    parser.add_argument("--config", type=str, help="Path to config yaml")
    parser.add_argument("--env", type=str, default="production", help="Environment (development|production)")
    args = parser.parse_args()

    if not args.output:
        args.output = os.path.splitext(args.input)[0] + "_gender.json"

    print(f"Initializing Gender Detection Component (env={args.env})...")
    try:
        component = GenderDetectionComponent.from_config(path=args.config, env=args.env)
    except Exception as e:
        print(f"Error initializing component: {e}")
        return

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.input}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Processing {total_frames} frames at {fps:.2f} FPS ({frame_w}x{frame_h})...")

    writer = None
    vid_path = None
    if args.output_video:
        vid_path = args.output_video
        if not vid_path.lower().endswith(".mp4"):
            vid_path += ".mp4"
            print(f"Output video path adjusted to: {vid_path}")
        writer = imageio.get_writer(
            vid_path, fps=fps, codec="libx264", pixelformat="yuv420p",
        )
        print(f"Video output enabled: {vid_path}")

    results = []
    start_time = time.time()
    proc_start = time.time()
    proc_counter = 0

    try:
        for frame_idx in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            t0 = time.time()
            result = component.process_frame(frame, frame_idx, timestamp=timestamp)
            latency = (time.time() - t0) * 1000

            results.append(result.to_dict())

            proc_counter += 1
            if time.time() - proc_start >= 1.0:
                fps_display = proc_counter
                proc_counter = 0
                proc_start = time.time()
            else:
                fps_display = 0

            if writer is not None:
                draw_overlay(frame, result, frame_idx, latency, fps_display)
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
        if writer is not None:
            writer.close()

    total_time = time.time() - start_time
    print(f"\nProcessing complete in {total_time:.2f}s ({total_frames/total_time:.2f} FPS)")

    output_data = {
        "metadata": {
            "input_file": args.input,
            "total_frames": total_frames,
            "fps": fps,
            "processing_time_sec": total_time,
            "component_version": "1.0.0"
        },
        "frames": results
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"JSON results saved to {args.output}")
    if writer is not None:
        print(f"Video with overlay saved to {vid_path}")

    stats = component.get_stats()
    print("\nPipeline Statistics:")
    print(f"  Qualified for inference: {stats['qualified']}")
    print(f"  Cache hits (pHash):      {stats['cache_hits']}")
    print(f"  Stride skips:            {stats['stride_skips']}")


if __name__ == "__main__":
    main()
