# 26/05 to add audio to the processed videos 
from pathlib import Path
import subprocess

ORIG_DIR = Path("/scratch/monroy/Playground/datasets/MoMentS_val_videos_emo")
PROC_DIR = Path("/scratch/monroy/Playground/yolo_bbox_style/red_t2")
OUT_DIR  = Path("/scratch/monroy/Playground/yolo_bbox_style/red_t2_audio")

OUT_DIR.mkdir(parents=True, exist_ok=True)

videos = sorted(PROC_DIR.glob("*.mp4"))

for proc_video in videos:

    name = proc_video.stem

    if name.endswith("_faces"):
        orig_name = name.replace("_faces", "")
    else:
        orig_name = name

    orig_video = ORIG_DIR / f"{orig_name}.mp4"

    if not orig_video.exists():
        print("Missing original:", name)
        continue

    out_video = OUT_DIR / f"{name}.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(proc_video),
        "-i", str(orig_video),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(out_video)
    ]

    print("Processing:", name)

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

print("DONE")
