from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from common import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and track players, goalkeepers, referees and the ball in a video.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", default="yolo11n.pt", help="Path to the model weights (.pt).")
    parser.add_argument("--track-class-names", nargs="+", default=["person"], help="Class names to track, e.g. player goalkeeper referee.")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--low-confidence", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input video not found: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    names = model.names
    name_to_id = {str(name): int(index) for index, name in names.items()}
    unknown = set(args.track_class_names) - set(name_to_id)
    if unknown:
        raise ValueError(f"Track class names not in this model: {sorted(unknown)}. Available: {sorted(name_to_id)}")
    track_class_ids = [name_to_id[name] for name in args.track_class_names]
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Video metadata is invalid.")
    writer = cv2.VideoWriter(str(args.output_dir / "annotated_tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not create annotated MP4 output.")
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    frame_index, detections, low_count = 0, 0, 0
    class_counts: Counter[str] = Counter()
    start = time.perf_counter()
    with (args.output_dir / "detections.csv").open("w", newline="", encoding="utf-8") as detections_file, (args.output_dir / "tracks.csv").open("w", newline="", encoding="utf-8") as tracks_file, (args.output_dir / "low_confidence_detections.csv").open("w", newline="", encoding="utf-8") as low_file:
        fields = ["frame_index", "timestamp_seconds", "class_id", "class_name", "track_id", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y", "center_x_normalized", "center_y_normalized"]
        detection_writer, track_writer, low_writer = (csv.DictWriter(file, fieldnames=fields) for file in (detections_file, tracks_file, low_file))
        for writer_csv in (detection_writer, track_writer, low_writer):
            writer_csv.writeheader()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.track(frame, persist=True, classes=track_class_ids, tracker=args.tracker, conf=args.confidence, imgsz=args.imgsz, device=device, verbose=False)[0]
            writer.write(result.plot())
            boxes = result.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
                for bounds, confidence, class_id, track_id in zip(xyxy, confidences, class_ids, track_ids):
                    x1, y1, x2, y2 = bounds
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    row = {"frame_index": frame_index, "timestamp_seconds": round(frame_index / fps, 3), "class_id": class_id, "class_name": names[class_id], "track_id": "" if track_id is None else track_id, "confidence": round(confidence, 5), "x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2), "center_x": round(center_x, 2), "center_y": round(center_y, 2), "center_x_normalized": round(center_x / width, 6), "center_y_normalized": round(center_y / height, 6)}
                    detection_writer.writerow(row)
                    if track_id is not None:
                        track_writer.writerow(row)
                    if confidence < args.low_confidence:
                        low_writer.writerow(row)
                        low_count += 1
                    class_counts[str(names[class_id])] += 1
                    detections += 1
            frame_index += 1
    elapsed = time.perf_counter() - start
    capture.release()
    writer.release()
    inference_fps = frame_index / elapsed if elapsed else 0.0
    summary = {"input_video": str(args.input.resolve()), "weights": str(args.weights), "device": str(device), "frame_count": frame_index, "source_fps": fps, "resolution": [width, height], "elapsed_seconds": round(elapsed, 3), "pipeline_fps": round(inference_fps, 3), "detection_count": detections, "class_detection_counts": dict(class_counts), "low_confidence_detection_count": low_count, "track_classes": args.track_class_names}
    save_json(args.output_dir / "run_summary.json", summary)
    print(f"Processed {frame_index} frames at {inference_fps:.2f} pipeline FPS.")
    print(f"Outputs saved under: {args.output_dir}")


if __name__ == "__main__":
    main()

