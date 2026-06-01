#!/usr/bin/env python3
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger(__name__)


def make_writer(out_path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")
    return writer


def load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_question_index(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for r in records:
        qid = str(r.get("question_id", "")).strip()
        if qid:
            idx[qid] = r
    return idx


def process_clip(
    video_path: Path,
    model: YOLO,
    out_json_path: Path,
    out_video_path: Path | None,
    conf: float,
    iou: float,
    tracker: str,
    every_k: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if out_video_path is not None:
        out_fps = max(fps / max(every_k, 1), 1.0)
        writer = make_writer(out_video_path, out_fps, (w, h))

    read_idx = 0
    processed = 0
    tracks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if every_k > 1 and (read_idx % every_k != 0):
            read_idx += 1
            continue

        results = model.track(
            source=frame,
            persist=True,
            conf=conf,
            iou=iou,
            tracker=tracker,
            verbose=False,
        )

        annotated = frame.copy()

        if results and len(results) > 0:
            r = results[0]
            boxes = r.boxes

            if boxes is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy() if boxes.cls is not None else None
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                ids = boxes.id.cpu().numpy() if boxes.id is not None else None

                for det_i in range(len(xyxy)):
                    class_id = int(cls[det_i]) if cls is not None else -1
                    if class_id != 0:
                        continue  # person only

                    x1, y1, x2, y2 = [int(v) for v in xyxy[det_i]]
                    track_id = int(ids[det_i]) if ids is not None else -1
                    score = float(confs[det_i]) if confs is not None else None

                    tracks.append({
                        "frame_idx": int(read_idx),
                        "time_sec": float(read_idx / fps),
                        "track_id": track_id,
                        "cls": class_id,
                        "conf": score,
                        "bbox_xyxy": [x1, y1, x2, y2],
                    })

                    if writer is not None:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"id={track_id} conf={score:.2f}" if score is not None else f"id={track_id}"
                        cv2.putText(
                            annotated,
                            label,
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

        if writer is not None:
            writer.write(annotated)

        processed += 1
        read_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    payload = {
        "clip_path": str(video_path),
        "fps": fps,
        "width": w,
        "height": h,
        "num_processed_frames": processed,
        "tracks": tracks,
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=str, required=True)
    ap.add_argument("--questions", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--yolo-model", type=str, default="yolo11m.pt")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml")
    ap.add_argument("--every-k", type=int, default=1)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--save-video", action="store_true")
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    tracks_dir = output_dir / "tracks_json"
    videos_dir = output_dir / "tracked_videos"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    if args.save_video:
        videos_dir.mkdir(parents=True, exist_ok=True)

    records = load_questions(args.questions)
    q_index = build_question_index(records)

    log.info(f"Loading model: {args.yolo_model}")
    model = YOLO(args.yolo_model)

    video_files = sorted(video_dir.glob("*.mp4"))
    log.info(f"Found {len(video_files)} clips in {video_dir}")

    if args.max_videos > 0:
        video_files = video_files[:args.max_videos]

    for i, video_path in enumerate(video_files, start=1):
        stem = video_path.stem.strip()
        meta = q_index.get(stem)

        log.info(f"[{i}/{len(video_files)}] Processing {stem}")

        out_json = tracks_dir / f"{stem}.json"
        out_video = (videos_dir / f"{stem}_tracked.mp4") if args.save_video else None

        try:
            processed = process_clip(
                video_path=video_path,
                model=model,
                out_json_path=out_json,
                out_video_path=out_video,
                conf=args.conf,
                iou=args.iou,
                tracker=args.tracker,
                every_k=max(1, args.every_k),
            )

            if meta:
                with open(out_json, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                payload["question_id"] = meta.get("question_id")
                payload["question"] = meta.get("question")
                payload["assigned_categories"] = meta.get("assigned_categories", [])
                payload["multimodal_cues"] = meta.get("multimodal_cues")
                payload["t_i"] = meta.get("t_i")
                payload["t_j"] = meta.get("t_j")
                payload["source_video_path"] = meta.get("video_path")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)

            log.info(f"    wrote {processed} frames")
        except Exception as e:
            log.error(f"    failed on {video_path}: {e}")

    log.info("Done.")


if __name__ == "__main__":
    main()