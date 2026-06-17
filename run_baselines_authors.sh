#!/usr/bin/env bash
# =============================================================================
# run_baselines.sh
# -----------------------------------------------------------------------------
# Runs all baseline modality experiments for the thesis using the authors-style
# evaluation script. Each configuration writes its own out-dir with
# predictions.jsonl + metrics.json. A summary table is printed at the end.
#
# Adjust DATASETS_ROOT, OUT_ROOT, and SCRIPT to match your paths.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4 nice -n 15 taskset -c 0-6 bash run_baselines.sh
# =============================================================================

set -euo pipefail
 
# ---- paths (edit these) -----------------------------------------------------
SCRIPT="/scratch/monroy/Playground/AUTHORS_eval.py"
 
# Video folders
ORIGINAL_DIR="/scratch/monroy/Playground/datasets/MoMentS_val_videos_emo"
AUDIO_ONLY_DIR="/scratch/monroy/Playground/Experiments_Baseline/audio_only/dataset_audio_only"
VIDEO_ONLY_DIR="/scratch/monroy/Playground/Experiments_Baseline/video_only/dataset_video_only"
 
# Transcripts JSON (video_id -> string)
TRANSCRIPTS_JSON="/scratch/monroy/Playground/Experiments_Baseline/transcripts/transcripts_by_videoid.json"
TRANSCRIPT_ONLY_DIR="/scratch/monroy/Playground/Experiments_Baseline/transcript_only/dataset_transcript_only"
# Output root
OUT_ROOT="/scratch/monroy/Playground/Experiments_AUTHORS"
 
mkdir -p "$OUT_ROOT"
 
# ---- helper -----------------------------------------------------------------
run_one() {
    local name="$1"
    local video_dir="$2"
    local transcripts_arg="${3:-}"
 
    local out_dir="$OUT_ROOT/$name"
    mkdir -p "$out_dir"
 
    echo
    echo "==================================================================="
    echo "[$(date +%H:%M:%S)] RUN: $name"
    echo "  video-dir:   $video_dir"
    if [[ -n "$transcripts_arg" ]]; then
        echo "  transcripts: $TRANSCRIPTS_JSON"
    else
        echo "  transcripts: (none)"
    fi
    echo "  out-dir:     $out_dir"
    echo "==================================================================="
 
    if [[ ! -d "$video_dir" ]]; then
        echo "  !! video-dir not found, skipping"
        return
    fi
 
    # shellcheck disable=SC2086
    python "$SCRIPT" \
        --video-dir "$video_dir" \
        --out-dir   "$out_dir" \
        $transcripts_arg \
        2>&1 | tee "$out_dir/run.log" || true
}
 
TRANS_FLAG="--transcripts-json $TRANSCRIPTS_JSON"
 
# ---- the 6 baseline conditions ---------------------------------------------
# run_one  frames_audio          "$ORIGINAL_DIR"      ""
# run_one  audio_only            "$AUDIO_ONLY_DIR"    ""
# run_one  video_only            "$VIDEO_ONLY_DIR"    ""
 
run_one  frames_audio_trans    "$ORIGINAL_DIR"      "$TRANS_FLAG"
# run_one  frames_trans          "$VIDEO_ONLY_DIR"    "$TRANS_FLAG"
# run_one  audio_trans           "$AUDIO_ONLY_DIR"    "$TRANS_FLAG"
run_one  transcript_only       "$TRANSCRIPT_ONLY_DIR"  "$TRANS_FLAG"
 
# ---- summary table ----------------------------------------------------------
echo
echo "==================================================================="
echo "SUMMARY"
echo "==================================================================="
printf "%-22s  %8s  %8s  %10s  %8s  %8s\n" \
       "condition" "total" "correct" "accuracy" "f1" "C-pred"
 
for d in "$OUT_ROOT"/*/; do
    name=$(basename "$d")
    metrics="$d/metrics.json"
    if [[ -f "$metrics" ]]; then
        python - <<PY "$name" "$metrics"
import json, sys
name, path = sys.argv[1], sys.argv[2]
m = json.load(open(path))
c_pred = m.get("pred_dist", {}).get("C", 0)
print(f"{name:<22}  {m['total']:>8}  {m['correct']:>8}  "
      f"{m['accuracy']:>10.3f}  {m['f1']:>8.3f}  {c_pred:>8}")
PY
    else
        printf "%-22s  %8s\n" "$name" "(no metrics)"
    fi
done
 