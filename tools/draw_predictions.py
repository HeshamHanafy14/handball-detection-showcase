"""Draw model predictions on a single image with thin boxes and small labels.

The image is enlarged by --scale before drawing so the labels stay readable.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import cv2

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from ultralytics import YOLO

# BGR colours per class id: player blue, goalkeeper cyan, referee white, ball orange.
COLORS = {0: (220, 110, 30), 1: (220, 200, 40), 2: (235, 235, 235), 3: (0, 140, 255)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="One image file.")
    parser.add_argument("--output", type=Path, required=True, help="Output image path (.jpg/.png).")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--scale", type=float, default=2.0, help="Enlarge image before drawing.")
    parser.add_argument("--font-scale", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.source))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.source}")
    result = YOLO(str(args.weights)).predict(str(args.source), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
    canvas = cv2.resize(image, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
    counts: Counter[str] = Counter()
    for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        x1, y1, x2, y2 = (int(v * args.scale) for v in box)
        name = result.names[int(cls)]
        color = COLORS.get(int(cls), (0, 255, 0))
        counts[name] += 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        text = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, 1)
        ty = max(y1 - 3, th + 3)
        cv2.rectangle(canvas, (x1, ty - th - 3), (x1 + tw + 4, ty + 2), color, -1)
        text_color = (0, 0, 0) if sum(color) > 450 else (255, 255, 255)
        cv2.putText(canvas, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, text_color, 1, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), canvas)
    print(f"Detections (conf >= {args.conf}): {dict(counts) or 'none'}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
