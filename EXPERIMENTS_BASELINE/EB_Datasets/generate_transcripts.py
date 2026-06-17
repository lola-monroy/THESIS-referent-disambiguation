#!/usr/bin/env python3
import json
from pathlib import Path
import torch
import whisper

VIDEO_DIR = Path("/scratch/monroy/Playground/datasets/MoMentS_val_videos_emo")
OUT_PATH = Path("/scratch/monroy/Playground/baseline_experiments/transcripts/transcripts_by_videoid.json")

MODEL_NAME = "base"   # tiny / base / small
LANGUAGE = "en"       # set to None to auto-detect

def main():
    model = whisper.load_model(MODEL_NAME)

    if OUT_PATH.exists():
        data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    else:
        data = {}

    video_files = sorted(VIDEO_DIR.glob("*.mp4"))
    print(f"Found {len(video_files)} videos")

    for vp in video_files:
        stem = vp.stem
        if stem in data and str(data[stem]).strip():
            print(f"[skip] {stem}")
            continue

        print(f"[transcribe] {stem}")
        try:
            if LANGUAGE is None:
                result = model.transcribe(str(vp), fp16=torch.cuda.is_available())
            else:
                result = model.transcribe(str(vp), language=LANGUAGE, fp16=torch.cuda.is_available())

            text = (result.get("text") or "").strip()
        except Exception as e:
            print(f"[error] {stem}: {e}")
            text = ""

        data[stem] = text
        OUT_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    print(f"Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()