#!/usr/bin/env python3
"""
YOLO_ON_VIDEOS_EMO.py
--------------------------
Full video mode ONLY:
Reads emotion videos, detects faces using YOLO, and writes a full-length annotated .mp4
for each input video.

Output:
- Annotated video:  <output-dir>/<video_stem>_faces.mp4

Usage:
  python YOLO_ON_VIDEOS_EMO.py --save-video
  python YOLO_ON_VIDEOS_EMO.py --video-dir ... --output-dir ... --yolo-model yolov11m-face.pt --conf 0.25
  python YOLO_ON_VIDEOS_EMO.py --every-k 2               # faster (process every 2nd frame)
  python YOLO_ON_VIDEOS_EMO.py --max-frames 500          # debug (stop early)
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def annotate_faces(frame: np.ndarray, model: YOLO, conf: float = 0.25) -> np.ndarray:
    """Run YOLO face detection and draw bounding boxes on the frame."""
    results = model.predict(frame, conf=conf, verbose=False)
    annotated = frame.copy()

    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy
        if boxes is None:
            continue
        boxes = boxes.cpu().numpy()
        for (x1, y1, x2, y2) in boxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return annotated


def make_writer(out_path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """Create an mp4 writer (mp4v)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")
    return writer


def process_full_video(
    video_path: Path,
    model: YOLO,
    out_path: Path,
    conf: float,
    every_k: int,
    max_frames: int,
    out_fps_override: float,
) -> int:
    """
    Process the full video stream frame-by-frame and write a full-length annotated video.
    Returns number of frames written (processed frames).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    in_fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_fps = in_fps if in_fps and in_fps > 0 else 25.0

    # If skipping frames, reduce fps to keep perceived speed roughly consistent.
    # If you want to keep original fps no matter what, set out_fps_override to base_fps.
    out_fps = out_fps_override if out_fps_override > 0 else max(base_fps / max(every_k, 1), 1.0)

    writer = make_writer(out_path, out_fps, (w, h))

    processed = 0
    read_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if every_k > 1 and (read_idx % every_k != 0):
            read_idx += 1
            continue

        annotated = annotate_faces(frame, model, conf=conf)

        # Ensure size matches writer size
        if annotated.shape[1] != w or annotated.shape[0] != h:
            annotated = cv2.resize(annotated, (w, h))

        writer.write(annotated)

        processed += 1
        read_idx += 1

        if max_frames > 0 and processed >= max_frames:
            break

    cap.release()
    writer.release()
    return processed


def main():
    p = argparse.ArgumentParser(description="Annotate faces on MoMentS emotion videos using YOLO (full video mode).")
    p.add_argument("--video-dir", type=str, default="/scratch/monroy/Playground/datasets/MoMentS_val_videos_emo")
    p.add_argument(
        "--output-dir",
        type=str,
        default="/scratch/monroy/Playground/datasets/yolov11mface_video2video",
        help="Directory to save annotated videos",
    )
    p.add_argument("--yolo-model", type=str, default="yolov11m-face.pt", help="Path to YOLO .pt model")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")

    # Speed / debug controls
    p.add_argument("--every-k", type=int, default=1, help="Process every k-th frame (>=1). Higher = faster, less smooth.")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after this many processed frames (0=all).")

    # FPS control
    p.add_argument("--video-fps", type=float, default=0.0, help="Override output FPS (0 = auto).")

    args = p.parse_args()

    # Load Model
    log.info(f"Loading YOLO model: {args.yolo_model}")
    try:
        model = YOLO(args.yolo_model)
    except Exception as e:
        log.error(f"Failed to load YOLO model '{args.yolo_model}': {e}")
        return

    # Source Dir
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        log.error(f"Video directory not found: {video_dir}")
        return

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    video_files = sorted(video_dir.glob("*.mp4"))
    log.info(f"Processing {len(video_files)} videos from {video_dir}...")

    for v_idx, video_path in enumerate(video_files):
        vid_id = video_path.stem
        out_video_path = output_base / f"{vid_id}_faces.mp4"

        log.info(f"[{v_idx + 1}/{len(video_files)}] Processing {vid_id}")

        try:
            processed = process_full_video(
                video_path=video_path,
                model=model,
                out_path=out_video_path,
                conf=args.conf,
                every_k=max(1, int(args.every_k)),
                max_frames=max(0, int(args.max_frames)),
                out_fps_override=float(args.video_fps),
            )
            log.info(f"    Wrote annotated video: {out_video_path} ({processed} frames)")

        except Exception as e:
            log.error(f"    Failed to process video {video_path}: {e}")

    log.info("Processing complete.")


if __name__ == "__main__":
    main()