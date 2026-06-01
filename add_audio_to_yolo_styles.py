# 1/6/26 loops through every style subfolder and reattaches audio from the original videos.

from pathlib import Path
import subprocess

ORIG_DIR = Path("/scratch/monroy/Playground/datasets/MoMentS_val_videos_emo")
STYLE_ROOT = Path("/scratch/monroy/Playground/yolo_bbox_style")

for style_dir in sorted(STYLE_ROOT.iterdir()):
    if not style_dir.is_dir():
        continue
    if style_dir.name.endswith("_audio"):
        continue

    out_dir = STYLE_ROOT / f"{style_dir.name}_audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSTYLE: {style_dir.name}")

    for proc_video in sorted(style_dir.glob("*.mp4")):
        name = proc_video.stem

        if name.endswith("_faces"):
            orig_name = name.replace("_faces", "")
        else:
            orig_name = name

        orig_video = ORIG_DIR / f"{orig_name}.mp4"
        out_video = out_dir / f"{orig_name}.mp4"

        if not orig_video.exists():
            print("Missing original:", orig_name)
            continue

        cmd = [
            "ffmpeg", "-y",
            "-i", str(proc_video),
            "-i", str(orig_video),
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(out_video),
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("Saved to:", out_dir)

print("\nDONE")