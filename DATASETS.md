# Experiment datasets

All datasets are derived from the 68 MoMentS validation videos with emotion questions
(`datasets/MoMentS/data/MoMentS_emotions_only.json`). Evaluation protocol:
`EVALUATE_MOMENTS3.py --temperature 0.1 --seed 42` via `run_all_clean.sh`,
results in `EXPERIMENTS_CLEAN/`.

Layout is `flat` unless noted: one `<video_id>.mp4` per video directly in the folder.

## Baselines

| Name | Path | Videos | Notes |
|---|---|---|---|
| original | `datasets/MoMentS_val_videos_emo` | 68 | Untouched videos (audio + video). Reference baseline: 25/68 = 36.8% |
| audio_only | `Experiments_Baseline/audio_only/dataset_audio_only` | 68 | Audio track only |
| video_only | `Experiments_Baseline/video_only/dataset_video_only` | 70 | Video only, no audio (2 extra files vs. the 68 questions) |

## Intervention datasets

| Name | Path | Videos | Notes |
|---|---|---|---|
| yolo | `datasets/YOLO_datasets/yolov11mface_video2video_audio` | 68 | YOLOv11m-face boxes drawn on video, audio kept |
| grounded | `datasets/grounded_videos` | 52 | Referent-grounded videos — **16 videos missing** |
| referent_v2 | `datasets/dataset_referent_v2_outputs` | 65 | **Nested layout**: `samples/<id>/<id>_referent.mp4` (plus `result.json`, `anchor_candidates/` per sample). Failures listed in `manifests/failed.jsonl` |
| rga3_redshade | `datasets/rga3_dataset_redshade` | 67 | RGA3 referent highlighting, red shade style |
| rga3_blackout | `datasets/rga3_dataset_blackout` | 67 | RGA3 referent highlighting, blackout style |
| saliency | `datasets/SALIENCY_MAP_WITH_AUDIO` | 68 | **Nested layout**: `<id>/<id>.mp4` (one subfolder per video), saliency-map overlay with audio |

## Caveats for comparison

- Conditions with fewer than 68 videos (grounded: 52, referent_v2: 65, rga3: 67) are
  not directly comparable on raw accuracy — compare on the intersection of answered
  question ids, or report accuracy only over common videos.
- The two nested-layout datasets (referent_v2, saliency) need either flattening/symlinks
  or a recursive video glob before they can run through `run_all_clean.sh`.
