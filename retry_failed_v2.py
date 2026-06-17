#!/usr/bin/env python3
"""
Retry failed samples from REFERENT_ANCHOR_FACE_v2.py.

Reads referent_v2_outputs/manifests/failed.jsonl, extracts the question IDs,
filters the full dataset JSON to those samples, and re-runs the v2 script with:
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (reduces fragmentation OOM)
  - --yolo-batch-size 4                               (smaller peak GPU allocations)
  - --n-anchor-frames 15                              (fewer Qwen calls)
  - --skip-existing                                   (don't redo successful ones)

Samples that previously failed with "no visible face" are retried a second time
with a lower --face-threshold (0.30 instead of 0.45).
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
FAILED_JSONL     = Path("/scratch/monroy/Playground/referent_v2_outputs/manifests/failed.jsonl")
DATASET_JSON     = Path("/scratch/monroy/Playground/datasets/MoMentS/data/moments_questions_updated.json")
VIDEO_ROOT       = Path("/scratch/monroy/Playground/datasets/MoMentS_val_videos_emo")
OUTPUT_ROOT      = Path("/scratch/monroy/Playground/referent_v2_outputs")
SCRIPT           = Path("/scratch/monroy/Playground/REFERENT_ANCHOR_FACE_v2.py")

QWEN_MODEL  = "Qwen/Qwen2.5-VL-3B-Instruct"
YOLO_MODEL  = "yolo11x.pt"
DEVICE      = "cuda"
CONF        = "0.35"
N_ANCHOR    = "15"
BATCH_SIZE  = "4"
# ───────────────────────────────────────────────────────────────────────────


def load_failed(path: Path):
    seen, ids, no_face_ids = set(), [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec["sample"].get("question_id")
            if qid is None or qid in seen:
                continue
            seen.add(qid)
            if "no visible face" in rec.get("error", "") or "not in this frame" in rec.get("error", ""):
                no_face_ids.append(qid)
            else:
                ids.append(qid)
    return ids, no_face_ids


def filter_dataset(dataset_path: Path, qids: list[str]) -> list[dict]:
    with open(dataset_path) as f:
        data = json.load(f)
    samples = data if isinstance(data, list) else list(data.values())
    qid_set = set(qids)
    return [s for s in samples if s.get("question_id") in qid_set]


def run_v2(samples: list[dict], extra_args: list[str], label: str):
    if not samples:
        print(f"[{label}] No samples — skipping.")
        return

    print(f"\n[{label}] Running {len(samples)} sample(s): "
          f"{[s['question_id'] for s in samples]}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="retry_subset_"
    ) as tf:
        json.dump(samples, tf, ensure_ascii=False)
        tmp_path = tf.name

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Pin to GPU 0 which has ~32 GiB free (the original run hit a congested GPU)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    python = "/scratch/monroy/.conda-envs/qwen-lola/bin/python3"
    cmd = [
        python, "-u", str(SCRIPT),
        "--dataset-json",        tmp_path,
        "--video-root",          str(VIDEO_ROOT),
        "--dataset-output-root", str(OUTPUT_ROOT),
        "--qwen-model",          QWEN_MODEL,
        "--yolo-model",          YOLO_MODEL,
        "--device",              DEVICE,
        "--conf",                CONF,
        "--n-anchor-frames",     N_ANCHOR,
        "--yolo-batch-size",     BATCH_SIZE,
        "--skip-existing",
    ] + extra_args

    print("CMD:", " ".join(cmd))
    try:
        subprocess.run(cmd, env=env, check=True)
    finally:
        os.unlink(tmp_path)


def main():
    oom_ids, no_face_ids = load_failed(FAILED_JSONL)
    print(f"OOM failures:     {oom_ids}")
    print(f"No-face failures: {no_face_ids}")

    # Retry OOM samples with memory-friendly settings
    oom_samples = filter_dataset(DATASET_JSON, oom_ids)
    run_v2(oom_samples, extra_args=[], label="OOM retry")

    # Retry "no face found" samples: lower YOLO conf and face threshold so
    # partially-visible people (e.g. bathroom scene) get detected.
    # The video files are pre-trimmed clips so no window restriction needed.
    no_face_samples = filter_dataset(DATASET_JSON, no_face_ids)
    for s in no_face_samples:
        extra = [
            "--face-threshold", "0.25",
            "--n-anchor-frames", "30",
            "--conf", "0.15",       # much lower to catch partial/obscured detections
        ]
        run_v2([s], extra_args=extra, label=f"no-face retry {s['question_id']}")


if __name__ == "__main__":
    main()
