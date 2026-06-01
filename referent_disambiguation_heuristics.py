#!/usr/bin/env python3
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


def bbox_area_xyxy(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_center_xyxy(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def face_inside_person(face_box, person_box):
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box
    fcx = (fx1 + fx2) / 2.0
    fcy = (fy1 + fy2) / 2.0
    return px1 <= fcx <= px2 and py1 <= fcy <= py2


def minmax_normalize_dict(d, invert=False):
    keys = list(d.keys())
    vals = np.array([d[k] for k in keys], dtype=float)
    if len(vals) == 0:
        return {}
    vmin, vmax = vals.min(), vals.max()
    if math.isclose(vmin, vmax):
        norm = {k: 1.0 for k in keys}
    else:
        norm = {k: (d[k] - vmin) / (vmax - vmin) for k in keys}
    if invert:
        norm = {k: 1.0 - v for k, v in norm.items()}
    return norm


def load_tracking_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_tracks_by_frame(tracks):
    by_frame = defaultdict(list)
    for det in tracks:
        by_frame[int(det["frame_idx"])].append(det)
    return by_frame


def build_tracks_by_id(tracks):
    by_id = defaultdict(list)
    for det in tracks:
        by_id[int(det["track_id"])].append(det)
    return by_id


def detect_faces_in_frame(face_model, frame, conf=0.25):
    results = face_model.predict(frame, conf=conf, verbose=False)
    faces = []
    for r in results:
        if r.boxes is None or r.boxes.xyxy is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            faces.append([x1, y1, x2, y2])
    return faces


def compute_track_scores(track_json, clip_path, face_model, sample_every=5, face_conf=0.25):
    tracks = track_json["tracks"]
    W = int(track_json["width"])
    H = int(track_json["height"])
    diag = math.sqrt(W * W + H * H)

    by_frame = build_tracks_by_frame(tracks)
    by_id = build_tracks_by_id(tracks)

    # ---------- size ----------
    mean_area = {}
    for tid, dets in by_id.items():
        areas = [bbox_area_xyxy(d["bbox_xyxy"]) for d in dets]
        mean_area[tid] = float(np.mean(areas)) if areas else 0.0

    # ---------- center ----------
    mean_center_dist = {}
    frame_cx, frame_cy = W / 2.0, H / 2.0
    for tid, dets in by_id.items():
        dists = []
        for d in dets:
            cx, cy = bbox_center_xyxy(d["bbox_xyxy"])
            dist = math.sqrt((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2)
            dists.append(dist / diag)
        mean_center_dist[tid] = float(np.mean(dists)) if dists else 1.0

    # ---------- motion ----------
    mean_motion = {}
    for tid, dets in by_id.items():
        dets = sorted(dets, key=lambda x: x["frame_idx"])
        centers = [bbox_center_xyxy(d["bbox_xyxy"]) for d in dets]
        if len(centers) < 2:
            mean_motion[tid] = 0.0
        else:
            disps = []
            for i in range(1, len(centers)):
                x0, y0 = centers[i - 1]
                x1, y1 = centers[i]
                disps.append(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / diag)
            mean_motion[tid] = float(np.mean(disps))

    # ---------- face visibility ----------
    face_hits = defaultdict(int)
    face_total_sampled_frames = defaultdict(int)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open clip: {clip_path}")

    frame_idx = 0
    max_frame = max(by_frame.keys()) if by_frame else -1

    while frame_idx <= max_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every == 0 and frame_idx in by_frame:
            faces = detect_faces_in_frame(face_model, frame, conf=face_conf)
            person_dets = by_frame[frame_idx]

            for d in person_dets:
                tid = int(d["track_id"])
                pbox = d["bbox_xyxy"]
                face_total_sampled_frames[tid] += 1

                has_face = any(face_inside_person(fbox, pbox) for fbox in faces)
                if has_face:
                    face_hits[tid] += 1

        frame_idx += 1

    cap.release()

    face_ratio = {}
    for tid in by_id.keys():
        denom = face_total_sampled_frames[tid]
        face_ratio[tid] = (face_hits[tid] / denom) if denom > 0 else 0.0

    # ---------- normalize ----------
    size_score = minmax_normalize_dict(mean_area, invert=False)
    center_score = minmax_normalize_dict(mean_center_dist, invert=True)  # closer center = better
    face_score = minmax_normalize_dict(face_ratio, invert=False)
    motion_score = minmax_normalize_dict(mean_motion, invert=False)

    # ---------- final ----------
    final_score = {}
    for tid in by_id.keys():
        final_score[tid] = (
            0.25 * size_score.get(tid, 0.0) +
            0.30 * center_score.get(tid, 0.0) +
            0.35 * face_score.get(tid, 0.0) +
            0.10 * motion_score.get(tid, 0.0)
        )

    # track summary for inspection
    summary = {}
    for tid, dets in by_id.items():
        frames = sorted(int(d["frame_idx"]) for d in dets)
        summary[tid] = {
            "num_detections": len(dets),
            "first_frame": frames[0],
            "last_frame": frames[-1],
            "mean_area": mean_area[tid],
            "mean_center_dist": mean_center_dist[tid],
            "face_ratio": face_ratio[tid],
            "mean_motion": mean_motion[tid],
            "size_score": size_score.get(tid, 0.0),
            "center_score": center_score.get(tid, 0.0),
            "face_score": face_score.get(tid, 0.0),
            "motion_score": motion_score.get(tid, 0.0),
            "final_score": final_score.get(tid, 0.0),
        }

    ranked = sorted(summary.items(), key=lambda kv: kv[1]["final_score"], reverse=True)
    return ranked, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--face-model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sample-every", type=int, default=5)
    ap.add_argument("--face-conf", type=float, default=0.25)
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    tracks_dir = Path(args.tracks_dir)
    clips_dir = Path(args.clips_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    face_model = YOLO(args.face_model)

    json_files = sorted(tracks_dir.glob("*.json"))
    if args.max_files > 0:
        json_files = json_files[:args.max_files]

    for jf in json_files:
        stem = jf.stem
        clip_path = clips_dir / f"{stem}.mp4"
        if not clip_path.exists():
            print(f"[WARN] Missing clip for {stem}: {clip_path}")
            continue

        track_json = load_tracking_json(jf)
        ranked, summary = compute_track_scores(
            track_json=track_json,
            clip_path=clip_path,
            face_model=face_model,
            sample_every=args.sample_every,
            face_conf=args.face_conf,
        )

        out = {
            "question_id": track_json.get("question_id"),
            "question": track_json.get("question"),
            "clip_path": str(clip_path),
            "suggested_referent_track_id": int(ranked[0][0]) if ranked else None,
            "ranked_tracks": [
                {"track_id": int(tid), **stats}
                for tid, stats in ranked
            ],
        }

        with open(output_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        if ranked:
            print(f"[OK] {stem}: suggested track {ranked[0][0]}")

if __name__ == "__main__":
    main()