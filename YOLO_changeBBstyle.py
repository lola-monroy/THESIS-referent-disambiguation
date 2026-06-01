#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from typing import Tuple

import cv2
from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


STYLES = [
    {"name": "green_t2", "color": (0, 255, 0), "thickness": 2},
    {"name": "green_t4", "color": (0, 255, 0), "thickness": 4},
    {"name": "red_t2", "color": (0, 0, 255), "thickness": 2},
    {"name": "red_t4", "color": (0, 0, 255), "thickness": 4},
    {"name": "blue_t2", "color": (255, 0, 0), "thickness": 2},
    {"name": "yellow_t3", "color": (0, 255, 255), "thickness": 3},
    {"name": "white_t3", "color": (255, 255, 255), "thickness": 3},
]


def make_writer(out_path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")
    return writer


def annotate_frame(frame, model, conf, color, thickness):
    results = model.predict(frame, conf=conf, verbose=False)
    annotated = frame.copy()

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy
        if boxes is None:
            continue

        boxes = boxes.cpu().numpy()

        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                thickness,
            )

    return annotated


def process_video(video_path, model, out_path, conf, every_k, max_frames, fps_override, color, thickness):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    in_fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_fps = in_fps if in_fps and in_fps > 0 else 25.0
    out_fps = fps_override if fps_override > 0 else max(base_fps / max(every_k, 1), 1.0)

    writer = make_writer(out_path, out_fps, (w, h))

    processed = 0
    read_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if every_k > 1 and read_idx % every_k != 0:
            read_idx += 1
            continue

        annotated = annotate_frame(
            frame=frame,
            model=model,
            conf=conf,
            color=color,
            thickness=thickness,
        )

        writer.write(annotated)

        processed += 1
        read_idx += 1

        if max_frames > 0 and processed >= max_frames:
            break

    cap.release()
    writer.release()

    return processed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--yolo-model", default="yolov11m-face.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--every-k", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--video-fps", type=float, default=0.0)
    p.add_argument("--limit-videos", type=int, default=0)
    args = p.parse_args()

    video_dir = Path(args.video_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading YOLO model: {args.yolo_model}")
    model = YOLO(args.yolo_model)

    video_files = sorted(video_dir.glob("*.mp4"))

    if args.limit_videos > 0:
        video_files = video_files[:args.limit_videos]

    log.info(f"Found {len(video_files)} videos")

    for style in STYLES:
        style_name = style["name"]
        style_dir = output_root / style_name
        style_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"=== STYLE: {style_name} ===")

        for idx, video_path in enumerate(video_files):
            vid_id = video_path.stem
            out_path = style_dir / f"{vid_id}_faces.mp4"

            if out_path.exists():
                log.info(f"SKIP existing: {out_path}")
                continue

            log.info(f"[{idx + 1}/{len(video_files)}] {vid_id}")

            try:
                processed = process_video(
                    video_path=video_path,
                    model=model,
                    out_path=out_path,
                    conf=args.conf,
                    every_k=max(1, args.every_k),
                    max_frames=max(0, args.max_frames),
                    fps_override=args.video_fps,
                    color=style["color"],
                    thickness=style["thickness"],
                )
                log.info(f"Wrote {out_path} ({processed} frames)")
            except Exception as e:
                log.error(f"Failed on {video_path}: {e}")

    log.info("DONE")


if __name__ == "__main__":
    main()