#!/usr/bin/env python3
"""
iou_v1_v2.py
------------
Compute per-frame IoU between two referent-box datasets:

  v1 (grounded_videos_v1):     PURPLE box burned into the annotated video
                               -> recovered from pixels
  v2 (dataset_REFERENT_v2):    GREEN box, exact coords already stored in
                               each sample's result.json ("chosen_boxes")
                               -> read directly, no pixel work needed

For each video present in both datasets, the script:
  1. reads v2 boxes from result.json (list of [x1,y1,x2,y2] or null per frame)
  2. extracts the v1 purple box per frame from the v1 annotated video
  3. aligns the two per frame and computes IoU
  4. reports mean IoU per video and overall

Usage
=====
CUDA_VISIBLE_DEVICES=7 nice -n 15 taskset -c 0-6 python iou_v1_v2.py \
  --v1-dir /scratch/monroy/Playground/datasets/grounded_videos_v1 \
  --v2-root /scratch/monroy/Playground/datasets/dataset_REFERENT_v2_outputs \
  --out /scratch/monroy/Playground/iou_v1_v2_results.json

Notes
-----
* --v2-root should be the folder that contains one subfolder per sample, each
  with a result.json (the dir holding <question_id>/result.json). Adjust to
  wherever your v2 result.json files live.
* v1 videos are matched to v2 samples by video stem (e.g. "_paES").
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Purple-box detection (v1)
# ---------------------------------------------------------------------------

# The v1 box is a saturated purple/blue. In HSV (OpenCV: H in 0-179) purple/
# violet sits roughly H 120-160 with high saturation and value. We detect those
# pixels and take their bounding extent. Corner-bracket style boxes are handled
# fine because we use the min/max of all matching pixels, not continuous lines.
PURPLE_HSV_LO = np.array([120,  80,  80])
PURPLE_HSV_HI = np.array([165, 255, 255])

MIN_PURPLE_PIXELS = 40   # below this, treat frame as "no box found"


def extract_purple_box(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return [x1,y1,x2,y2] bounding the purple annotation, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, PURPLE_HSV_LO, PURPLE_HSV_HI)
    ys, xs = np.where(mask > 0)
    if len(xs) < MIN_PURPLE_PIXELS:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def extract_v1_boxes(video_path: Path) -> List[Optional[Tuple[int, int, int, int]]]:
    """Extract the purple box for every frame of a v1 annotated video."""
    cap = cv2.VideoCapture(str(video_path))
    boxes: List[Optional[Tuple[int, int, int, int]]] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes.append(extract_purple_box(frame))
    cap.release()
    return boxes


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou(a, b) -> float:
    """IoU of two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# v2 loading
# ---------------------------------------------------------------------------

def find_v2_results(v2_root: Path):
    """Map video_stem -> result.json path."""
    out = {}
    for rj in v2_root.rglob("result.json"):
        try:
            data = json.loads(rj.read_text())
        except Exception:
            continue
        stem = Path(data.get("video_path", "")).stem
        if stem:
            out[stem] = rj
    return out


def load_v2_boxes(result_json: Path) -> List[Optional[Tuple[int, int, int, int]]]:
    data = json.loads(result_json.read_text())
    boxes = []
    for b in data.get("chosen_boxes", []):
        if b is None:
            boxes.append(None)
        else:
            boxes.append(tuple(int(v) for v in b))
    return boxes


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare_video(v1_video: Path, v2_result: Path):
    v2_boxes = load_v2_boxes(v2_result)
    v1_boxes = extract_v1_boxes(v1_video)

    n = min(len(v1_boxes), len(v2_boxes))
    if n == 0:
        return None

    ious = []
    both = v1_only = v2_only = neither = 0
    for i in range(n):
        b1, b2 = v1_boxes[i], v2_boxes[i]
        if b1 is not None and b2 is not None:
            ious.append(iou(b1, b2))
            both += 1
        elif b1 is not None:
            v1_only += 1
        elif b2 is not None:
            v2_only += 1
        else:
            neither += 1

    return {
        "v1_frames": len(v1_boxes),
        "v2_frames": len(v2_boxes),
        "compared_frames": n,
        "frames_both_have_box": both,
        "frames_only_v1": v1_only,
        "frames_only_v2": v2_only,
        "frames_neither": neither,
        "mean_iou_over_both": float(np.mean(ious)) if ious else 0.0,
        "median_iou_over_both": float(np.median(ious)) if ious else 0.0,
        # IoU treating frames where only one has a box as IoU=0:
        "mean_iou_over_union_frames": (
            float(np.sum(ious) / (both + v1_only + v2_only))
            if (both + v1_only + v2_only) > 0 else 0.0
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", required=True, type=Path,
                    help="Folder of v1 annotated videos (purple boxes).")
    ap.add_argument("--v2-root", required=True, type=Path,
                    help="Root holding per-sample result.json files.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--video-ext", default=".mp4")
    args = ap.parse_args()

    v2_map = find_v2_results(args.v2_root)
    print(f"Found {len(v2_map)} v2 result.json files", flush=True)

    results = {}
    all_mean_ious = []

    v1_videos = sorted(p for p in args.v1_dir.iterdir()
                       if p.suffix.lower() == args.video_ext)
    print(f"Found {len(v1_videos)} v1 videos", flush=True)

    for v1_video in v1_videos:
        stem = v1_video.stem
        if stem not in v2_map:
            print(f"  skip {stem}: no v2 result.json", flush=True)
            continue
        print(f"  comparing {stem}", flush=True)
        res = compare_video(v1_video, v2_map[stem])
        if res is None:
            print(f"    !! no frames", flush=True)
            continue
        results[stem] = res
        all_mean_ious.append(res["mean_iou_over_both"])
        print(f"    mean IoU (both-box frames) = {res['mean_iou_over_both']:.3f} "
              f"over {res['frames_both_have_box']} frames", flush=True)

    summary = {
        "n_videos_compared": len(results),
        "overall_mean_of_per_video_mean_iou": (
            float(np.mean(all_mean_ious)) if all_mean_ious else 0.0
        ),
        "per_video": results,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nCompared {len(results)} videos", flush=True)
    print(f"Overall mean IoU = {summary['overall_mean_of_per_video_mean_iou']:.3f}", flush=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()