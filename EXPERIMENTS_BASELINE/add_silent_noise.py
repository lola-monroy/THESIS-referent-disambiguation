#!/usr/bin/env python3
"""
Finds all *.mp4 in a directory that have no audio stream and patches them
in-place by adding a silent audio track (AAC, 44100 Hz, mono).

Used to create the 'No audio' dataset, for the baseline experiments. 
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def has_audio(path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def add_white_noise(path: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(path),
                "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(tmp_path),
            ],
            check=True,
            capture_output=True,
        )
        tmp_path.replace(path)
        print(f"  patched: {path.name}")
    except subprocess.CalledProcessError as e:
        tmp_path.unlink(missing_ok=True)
        print(f"  FAILED:  {path.name}\n{e.stderr.decode()}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python add_white_noise.py <video_dir>")
        sys.exit(1)

    video_dir = Path(sys.argv[1])
    if not video_dir.is_dir():
        print(f"Not a directory: {video_dir}")
        sys.exit(1)

    videos = sorted(video_dir.glob("*.mp4"))
    print(f"Scanning {len(videos)} videos in {video_dir} ...")

    to_patch = [v for v in videos if not has_audio(v)]
    if not to_patch:
        print("All videos already have audio. Nothing to do.")
        return

    print(f"Found {len(to_patch)} video(s) without audio:")
    for v in to_patch:
        print(f"  {v.name}")
    print("Patching ...")
    for v in to_patch:
        add_white_noise(v)
    print("Done.")


if __name__ == "__main__":
    main()
