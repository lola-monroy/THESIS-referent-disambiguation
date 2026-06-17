#!/usr/bin/env bash
# Baseline sweep using EVALUATE_MOMENTS_yolo.py inference logic
# (via EVALUATE_MOMENTS_yolo_baselines.py: same code + CLI args + transcripts).
# Conditions: original, video_only (no audio), audio_only, original + transcript.
#
# Usage:
#   GPU=5 bash /scratch/monroy/Playground/run_baselines_yoloeval.sh             # run everything
#   GPU=5 bash /scratch/monroy/Playground/run_baselines_yoloeval.sh audio_only  # run one condition

set -uo pipefail

GPU="${GPU:?Set GPU, e.g. GPU=5 bash run_baselines_yoloeval.sh}"
PG=/scratch/monroy/Playground
SCRIPT="$PG/EVALUATE_MOMENTS_yolo_baselines.py"
LLAMA_ROOT=/scratch/monroy/Emotion-LLaMA
TRANSCRIPTS="$PG/Experiments_Baseline/transcripts/transcripts_by_videoid.json"
OUT_ROOT="$PG/EXPERIMENTS_BASELINE_yoloeval"

# cfg-path (eval_configs/demo.yaml) is relative to the cwd, so run from the repo root
cd "$LLAMA_ROOT"

declare -A DATASETS=(
  [original]="$PG/datasets/MoMentS_val_videos_emo"
  [video_only]="$PG/Experiments_Baseline/video_only/dataset_video_only"
  [audio_only]="$PG/Experiments_Baseline/audio_only/dataset_audio_only"
  [original_transcript]="$PG/datasets/MoMentS_val_videos_emo"
)
ORDER=(original video_only audio_only original_transcript)

if [ "$#" -gt 0 ]; then ORDER=("$@"); fi

mkdir -p "$OUT_ROOT"

for name in "${ORDER[@]}"; do
  vdir="${DATASETS[$name]:?Unknown condition: $name}"
  out="$OUT_ROOT/$name"
  extra=()
  if [ "$name" = "original_transcript" ]; then
    extra=(--transcripts "$TRANSCRIPTS")
  fi
  # predictions.jsonl is append-only: start from a clean output dir
  rm -rf "$out"
  echo "=============================================="
  echo "== $name  ($vdir)"
  echo "=============================================="
  CUDA_VISIBLE_DEVICES="$GPU" nice -n 15 taskset -c 7-13 \
    python "$SCRIPT" \
      --video-dir "$vdir" \
      --out-dir "$out" \
      "${extra[@]}" \
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
    python3 -c "import json;d=json.load(open('$m'));print(f\"{'$name':22s} total={d['total']:3d} correct={d['correct']:3d} acc={d['accuracy']:.3f}\")"
  else
    echo "$name: no metrics.json (run failed?)"
  fi
done
