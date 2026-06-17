#!/usr/bin/env python3
"""
apply_yolo_boxes.py
-------------------
Apply YOLOv8 and YOLOv11 person detection to every clip in a video folder and
write two separate output datasets with bounding boxes burned into the frames.
Original audio is preserved by muxing it back in with ffmpeg.

Run:
  CUDA_VISIBLE_DEVICES=4 python -u apply_yolo_boxes.py \
    --video-dir /scratch/monroy/Playground/datasets/MoMentS_val_videos_emo \
    --out-root  /scratch/monroy/Playground/datasets/YOLO_datasets

Use `python -u` (unbuffered) so progress prints appear immediately under nice.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
from ultralytics import YOLO

PERSON_CLASS = 0

MODEL_WEIGHTS = {"yolov8": "yolov8x.pt", "yolov11": "yolo11x.pt"}
OUT_SUBDIR    = {"yolov8": "yolov8_boxes", "yolov11": "yolov11_boxes"}

# Accept several extensions, case-insensitive
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def log(*a):
    print(*a, flush=True)


def find_clips(video_dir: Path):
    return sorted(p for p in video_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def has_audio_stream(video_path: Path) -> bool:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True,
        )
        return bool(out.stdout.strip())
    except FileNotFoundError:
        return False


def annotate_video(model, in_path: Path, out_path: Path, conf: float):
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        log(f"  !! could not open {in_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w == 0 or h == 0:
        log(f"  !! zero dimensions for {in_path} (w={w} h={h}) -- skipping")
        cap.release()
        return False

    tmp_silent = Path(tempfile.mktemp(suffix=".mp4"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_silent), fourcc, fps, (w, h))

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, classes=[PERSON_CLASS], conf=conf, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                score = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"person {score:.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        writer.write(frame)
        n += 1

    cap.release()
    writer.release()
    log(f"     {n} frames processed")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if has_audio_stream(in_path):
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-i", str(tmp_silent), "-i", str(in_path),
               "-c:v", "copy", "-c:a", "aac",
               "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(out_path)]
        subprocess.run(cmd, check=False)
        tmp_silent.unlink(missing_ok=True)
    else:
        shutil.move(str(tmp_silent), str(out_path))
    return out_path.exists()


def run_model(model_key: str, video_dir: Path, out_root: Path, conf: float):
    weights = MODEL_WEIGHTS[model_key]
    out_dir = out_root / OUT_SUBDIR[model_key]
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"\n=== {model_key} ({weights}) -> {out_dir} ===")
    model = YOLO(weights)

    clips = find_clips(video_dir)
    log(f"Found {len(clips)} clips in {video_dir}")
    if not clips:
        log("  (nothing to do -- check the path and file extensions)")
        return

    for i, clip in enumerate(clips, 1):
        out_path = out_dir / (clip.stem + ".mp4")
        if out_path.exists():
            log(f"[{i}/{len(clips)}] skip (exists) {clip.name}")
            continue
        log(f"[{i}/{len(clips)}] {clip.name}")
        annotate_video(model, clip, out_path, conf)
    log(f"=== {model_key} done: {out_dir} ===")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--models", nargs="+", default=["yolov8", "yolov11"],
                   choices=["yolov8", "yolov11"])
    p.add_argument("--conf", type=float, default=0.35)
    return p.parse_args()


def main():
    args = parse_args()
    log(f"video-dir: {args.video_dir}")
    log(f"out-root:  {args.out_root}")
    log(f"models:    {args.models}")
    if not args.video_dir.exists():
        log(f"ERROR: video dir not found: {args.video_dir}")
        sys.exit(1)
    if shutil.which("ffmpeg") is None:
        log("WARNING: ffmpeg not on PATH; audio will not be preserved.")
    for model_key in args.models:
        run_model(model_key, args.video_dir, args.out_root, args.conf)
    log("\nALL DONE.")


if __name__ == "__main__":
    main()