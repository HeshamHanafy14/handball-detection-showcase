"""Detect and track players, goalkeepers, referees and the ball in a short video segment and save it as a GIF."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
from PIL import Image

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from ultralytics import YOLO

# BGR colours per class id: player blue, goalkeeper cyan, referee white, ball orange.
COLORS = {0: (220, 110, 30), 1: (220, 200, 40), 2: (235, 235, 235), 3: (0, 140, 255)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .gif path.")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--seconds", type=float, default=None, help="Length of the segment (default: until the end of the video).")
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=640, help="Output width in pixels.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for people.")
    parser.add_argument("--ball-conf", type=float, default=None, help="Confidence threshold for the ball (default: same as --conf).")
    parser.add_argument("--imgsz", type=int, default=1280)
    return parser.parse_args()


def draw(frame, boxes, names):
    for box, cls, conf, track_id in boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        color = COLORS.get(cls, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        if cls == 3:
            text = "ball"
        else:
            text = f"{names[cls]} {track_id}" if track_id is not None else names[cls]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ty = max(y1 - 3, th + 3)
        cv2.rectangle(frame, (x1, ty - th - 3), (x1 + tw + 4, ty + 2), color, -1)
        text_color = (0, 0, 0) if sum(color) > 450 else (255, 255, 255)
        cv2.putText(frame, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
    return frame


def main() -> None:
    args = parse_args()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(args.start * fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if args.seconds is None:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame
    else:
        total = int(args.seconds * fps)
    step = max(1, round(fps / args.gif_fps))

    model = YOLO(str(args.weights))
    ball_conf = args.ball_conf if args.ball_conf is not None else args.conf
    frames = []
    for index in range(total):
        ok, frame = capture.read()
        if not ok:
            break
        # Every frame goes through the tracker so IDs stay stable; only some are kept for the GIF.
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=min(args.conf, ball_conf), imgsz=args.imgsz, verbose=False)[0]
        if index % step:
            continue
        boxes = result.boxes
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        items = list(zip(boxes.xyxy.tolist(), boxes.cls.int().tolist(), boxes.conf.tolist(), ids))
        items = [item for item in items if item[2] >= (ball_conf if item[1] == 3 else args.conf)]
        scale = args.width / frame.shape[1]
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        items = [([v * scale for v in box], cls, conf, track_id) for box, cls, conf, track_id in items]
        small = draw(small, items, result.names)
        frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    capture.release()
    if not frames:
        raise RuntimeError("No frames were read. Check --start and the video path.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    palette = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    palette[0].save(args.output, save_all=True, append_images=palette[1:], duration=int(1000 * step / fps), loop=0, optimize=True)
    print(f"Saved {len(frames)} frames to {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
