#!/usr/bin/env python3
"""
YOLO_video2video_filtercrop.py
------------------------------
Reads videos, detects faces using YOLO, and writes an annotated .mp4.

NEW:
- Drop frames with no bbox OR bbox area < threshold.
- Optional spatial crop around bbox (pad), then resize to original size (constant output size).

Output:
- <output-dir>/<video_stem><suffix>.mp4   (default suffix: _faces_filtered.mp4)

Examples:
  # Keep only frames with a face bbox >= 2% of frame area
  python YOLO_video2video_filtercrop.py --min-area-frac 0.02

  # Keep frames only if bbox area >= 50000 pixels
  python YOLO_video2video_filtercrop.py --min-area-px 50000

  # Also crop to bbox region (pad 15%), resize back to original size
  python YOLO_video2video_filtercrop.py --min-area-frac 0.02 --spatial-crop --pad 0.15
"""

import logging
import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def detect_faces_xyxy(frame: np.ndarray, model: YOLO, conf: float = 0.25) -> np.ndarray:
    """
    Return Nx4 float array of face boxes in xyxy (x1,y1,x2,y2).
    If none, returns shape (0,4).
    """
    results = model.predict(frame, conf=conf, verbose=False)
    boxes_all = []
    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy
        if boxes is None:
            continue
        b = boxes.detach().cpu().numpy()
        if b.size > 0:
            boxes_all.append(b)
    if not boxes_all:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(boxes_all, axis=0).astype(np.float32)


def draw_boxes(frame: np.ndarray, boxes_xyxy: np.ndarray, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    out = frame.copy()
    for (x1, y1, x2, y2) in boxes_xyxy:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def make_writer(out_path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")
    return writer


def clip_box(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(1, min(W, x2))
    y2 = max(1, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def process_full_video(
    video_path: Path,
    model: YOLO,
    out_path: Path,
    conf: float,
    every_k: int,
    max_frames: int,
    out_fps_override: float,
    min_area_px: int,
    min_area_frac: float,
    drop_no_box: bool,
    spatial_crop: bool,
    pad: float,
    draw: bool,
    box_thickness: int,
) -> Tuple[int, int, int]:
    """
    Returns (frames_read, frames_kept, frames_dropped)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    in_fps = float(cap.get(cv2.CAP_PROP_FPS))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = W * H

    base_fps = in_fps if in_fps and in_fps > 0 else 25.0
    out_fps = out_fps_override if out_fps_override > 0 else max(base_fps / max(every_k, 1), 1.0)

    writer = make_writer(out_path, out_fps, (W, H))

    # threshold area
    area_thr = min_area_px if min_area_px > 0 else int(min_area_frac * frame_area)

    frames_read = 0
    frames_kept = 0
    frames_dropped = 0
    processed = 0
    read_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1

        if every_k > 1 and (read_idx % every_k != 0):
            read_idx += 1
            continue

        # YOLO detect on this frame
        boxes = detect_faces_xyxy(frame, model, conf=conf)

        if boxes.shape[0] == 0:
            if drop_no_box:
                frames_dropped += 1
                read_idx += 1
                processed += 1
                if max_frames > 0 and processed >= max_frames:
                    break
                continue
            # else keep even without box (rare use-case)

        # choose largest box, compute area
        chosen_box = None
        if boxes.shape[0] > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_i = int(np.argmax(areas))
            best_area = float(areas[best_i])
            if best_area < area_thr:
                frames_dropped += 1
                read_idx += 1
                processed += 1
                if max_frames > 0 and processed >= max_frames:
                    break
                continue
            chosen_box = boxes[best_i].copy()

        out_frame = frame
        out_boxes = boxes

        # optional spatial crop around largest box
        if spatial_crop and chosen_box is not None:
            x1, y1, x2, y2 = map(int, chosen_box.tolist())
            pad_x = int((x2 - x1) * pad)
            pad_y = int((y2 - y1) * pad)
            bb = clip_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, W, H)
            if bb is not None:
                cx1, cy1, cx2, cy2 = bb
                crop = frame[cy1:cy2, cx1:cx2]

                # adjust ALL boxes into crop coordinates (so rectangles remain correct)
                if boxes.shape[0] > 0:
                    adj = boxes.copy()
                    adj[:, 0] -= cx1
                    adj[:, 2] -= cx1
                    adj[:, 1] -= cy1
                    adj[:, 3] -= cy1
                    # clip to crop
                    cw = cx2 - cx1
                    ch = cy2 - cy1
                    adj[:, 0] = np.clip(adj[:, 0], 0, cw - 1)
                    adj[:, 2] = np.clip(adj[:, 2], 1, cw)
                    adj[:, 1] = np.clip(adj[:, 1], 0, ch - 1)
                    adj[:, 3] = np.clip(adj[:, 3], 1, ch)
                    out_boxes = adj

                # draw on crop (optional), then resize back to original size
                if draw:
                    crop_drawn = draw_boxes(crop, out_boxes, thickness=box_thickness)
                else:
                    crop_drawn = crop

                out_frame = cv2.resize(crop_drawn, (W, H), interpolation=cv2.INTER_LINEAR)

                # IMPORTANT: after resize, the drawn boxes are already baked in.
                # We don't redraw after resizing.

            else:
                # fallback: no valid crop, just keep original
                if draw and boxes.shape[0] > 0:
                    out_frame = draw_boxes(frame, boxes, thickness=box_thickness)

        else:
            # no spatial crop: just annotate and write full frame
            if draw and boxes.shape[0] > 0:
                out_frame = draw_boxes(frame, boxes, thickness=box_thickness)

        # Ensure size matches writer size
        if out_frame.shape[1] != W or out_frame.shape[0] != H:
            out_frame = cv2.resize(out_frame, (W, H))

        writer.write(out_frame)
        frames_kept += 1

        processed += 1
        read_idx += 1
        if max_frames > 0 and processed >= max_frames:
            break

    cap.release()
    writer.release()
    return frames_read, frames_kept, frames_dropped


def main():
    p = argparse.ArgumentParser(description="Annotate faces on videos using YOLO + drop/crop frames by bbox size.")
    p.add_argument("--video-dir", type=str, default="/scratch/monroy/Playground/datasets/MoMentS_val_videos_emo")
    p.add_argument("--output-dir", type=str, default="/scratch/monroy/Playground/datasets/yolov11mface_video2video",
                   help="Directory to save output videos")
    p.add_argument("--output-suffix", type=str, default="_faces_filtered.mp4",
                   help="Suffix for output filename (default: _faces_filtered.mp4)")
    p.add_argument("--yolo-model", type=str, default="yolov11m-face.pt", help="Path to YOLO .pt model")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")

    # Speed / debug controls
    p.add_argument("--every-k", type=int, default=1, help="Process every k-th frame (>=1). Higher=faster.")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after this many processed frames (0=all).")

    # FPS control
    p.add_argument("--video-fps", type=float, default=0.0, help="Override output FPS (0 = auto).")

    # NEW: filtering thresholds
    p.add_argument("--drop-no-box", action="store_true", help="Drop frames with no detected bbox (recommended).")
    p.add_argument("--min-area-frac", type=float, default=0.0,
                   help="Drop frames where largest bbox area < frac * frame_area (e.g. 0.02 = 2%).")
    p.add_argument("--min-area-px", type=int, default=0,
                   help="Drop frames where largest bbox area < this many pixels (overrides frac if >0).")

    # NEW: optional spatial crop
    p.add_argument("--spatial-crop", action="store_true",
                   help="Crop to the largest bbox (with padding) and resize back to original size.")
    p.add_argument("--pad", type=float, default=0.10, help="Padding fraction around bbox when spatial-crop is on.")
    p.add_argument("--no-draw", action="store_true", help="Do not draw boxes on output (just filter/crop).")
    p.add_argument("--box-thickness", type=int, default=2, help="Rectangle thickness if drawing.")

    args = p.parse_args()

    log.info(f"Loading YOLO model: {args.yolo_model}")
    model = YOLO(args.yolo_model)

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        log.error(f"Video directory not found: {video_dir}")
        return

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    video_files = sorted(video_dir.glob("*.mp4"))
    log.info(f"Processing {len(video_files)} videos from {video_dir}...")

    # Default behavior: if user sets a min-area threshold, also drop no-box frames.
    drop_no_box = bool(args.drop_no_box or args.min_area_px > 0 or args.min_area_frac > 0.0)

    for v_idx, video_path in enumerate(video_files):
        vid_id = video_path.stem
        out_video_path = output_base / f"{vid_id}{args.output_suffix}"

        log.info(f"[{v_idx + 1}/{len(video_files)}] Processing {vid_id}")

        try:
            frames_read, kept, dropped = process_full_video(
                video_path=video_path,
                model=model,
                out_path=out_video_path,
                conf=float(args.conf),
                every_k=max(1, int(args.every_k)),
                max_frames=max(0, int(args.max_frames)),
                out_fps_override=float(args.video_fps),
                min_area_px=max(0, int(args.min_area_px)),
                min_area_frac=float(args.min_area_frac),
                drop_no_box=drop_no_box,
                spatial_crop=bool(args.spatial_crop),
                pad=float(args.pad),
                draw=(not args.no_draw),
                box_thickness=int(args.box_thickness),
            )
            log.info(f"    Wrote: {out_video_path}")
            log.info(f"    Frames read: {frames_read} | kept: {kept} | dropped: {dropped}")

        except Exception as e:
            log.error(f"    Failed to process video {video_path}: {e}")

    log.info("Processing complete.")


if __name__ == "__main__":
    main()