#!/usr/bin/env bash
# Clean, comparable evaluation sweep: one script, temperature 0.1 + seed 42
# (matches the protocol of all reference runs in Experiments_Baseline_RERUN,
# e.g. audio_video original baseline = 25/68 = 36.8%).
# All results go to EXPERIMENTS_CLEAN/ — only report numbers from there.
#
# Usage:
#   GPU=4 bash /scratch/monroy/Playground/run_all_clean.sh            # run everything
#   GPU=4 bash /scratch/monroy/Playground/run_all_clean.sh yolo       # run one condition
#
# Conditions: original audio_only video_only yolo grounded

set -uo pipefail

GPU="${GPU:?Set GPU, e.g. GPU=4 bash run_all_clean.sh}"
PG=/scratch/monroy/Playground
SCRIPT="$PG/EVALUATE_MOMENTS3.py"
LLAMA_ROOT=/scratch/monroy/Emotion-LLaMA

# cfg-path (eval_configs/demo.yaml) is relative to the cwd, so run from the repo root
cd "$LLAMA_ROOT"
OUT_ROOT="$PG/EXPERIMENTS_CLEAN"
COMMON=(--temperature 0.1 --seed 42 --overwrite --no-timestamp)

declare -A DATASETS=(
  [original]="$PG/datasets/MoMentS_val_videos_emo"
  [audio_only]="$PG/Experiments_Baseline/audio_only/dataset_audio_only"
  [video_only]="$PG/Experiments_Baseline/video_only/dataset_video_only"
  [yolo]="$PG/datasets/YOLO_datasets/yolov11mface_video2video_audio"
  [grounded]="$PG/datasets/grounded_videos"
)
ORDER=(original audio_only video_only yolo grounded)

# Optional: run only the conditions named on the command line
if [ "$#" -gt 0 ]; then ORDER=("$@"); fi

mkdir -p "$OUT_ROOT"

for name in "${ORDER[@]}"; do
  vdir="${DATASETS[$name]:?Unknown condition: $name}"
  out="$OUT_ROOT/$name"
  echo "=============================================="
  echo "== $name  ($vdir)"
  echo "=============================================="
  CUDA_VISIBLE_DEVICES="$GPU" nice -n 15 taskset -c 0-6 \
    python "$SCRIPT" "${COMMON[@]}" \
      --video-dir "$vdir" \
      --out-dir "$out" \
    2>&1 | tee "$OUT_ROOT/${name}.log"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "!! $name FAILED (exit $status) — continuing with next condition" >&2
  fi
done

echo
echo "== Summary =="
for name in "${ORDER[@]}"; do
  m="$OUT_ROOT/$name/metrics.json"
  if [ -f "$m" ]; then
    python3 -c "import json;d=json.load(open('$m'));print(f\"{'$name':12s} total={d['total']:3d} correct={d['correct']:3d} acc={d['accuracy']:.3f} skipped={d.get('skipped','-')}\")"
  else
    echo "$name: no metrics.json (run failed?)"
  fi
done
