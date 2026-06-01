import json
import subprocess
from pathlib import Path
import shutil


VIDEO_DIR = Path("/scratch/monroy/Playground/datasets/multiple_people")
QUESTIONS_JSON = Path("/scratch/monroy/Playground/datasets/MoMentS/data/moments_questions_updated.json")
SCRIPT = Path("/scratch/monroy/Playground/referent_anchor_face.py")
OUTPUT_BASE = Path("/scratch/monroy/Playground/referent_batch_outputs")
TMP_VIDEO_DIR = Path("/scratch/monroy/Playground/referent_batch_outputs/_tmp_lowres_videos")

QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
YOLO_MODEL = "yolo11x.pt"
DEVICE = "cuda"
CONF = "0.20"

# low-memory settings
N_ANCHOR_FRAMES = 4
MAX_CANDIDATES_PER_FRAME = 2
YOLO_IMGSZ = 640
LOWRES_WIDTH = 1280   # change to 960 if still too heavy

BAD_STATUSES = {
    "failed_no_anchor",
    "failed_no_propagation",
    "fatal_error",
}

TMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

with open(QUESTIONS_JSON, "r", encoding="utf-8") as f:
    questions = json.load(f)

by_qid = {}
for q in questions:
    qid = q.get("question_id")
    question = q.get("question")
    cats = q.get("assigned_categories", [])
    if qid and question and "Emotions" in cats:
        by_qid[str(qid)] = question


def should_rerun(out_dir: Path) -> bool:
    existing_videos = list(out_dir.glob("*_referent.mp4"))
    result_json = out_dir / "result.json"

    if not existing_videos:
        return True

    if result_json.exists():
        try:
            with open(result_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") in BAD_STATUSES:
                return True
        except Exception:
            return True

    return False



def make_lowres_copy(src: Path, dst: Path, width: int = 1280):
    if dst.exists():
        return

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(
            "ffmpeg was not found in PATH. Install it with:\n"
            "conda install -c conda-forge ffmpeg -y"
        )

    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(src),
        "-vf", f"scale={width}:-2",
        "-c:v", "libopenh264",
        "-b:v", "2500k",
        "-c:a", "copy",
        str(dst),
    ]

    print(f"[FFMPEG] Creating low-res copy: {dst.name}")
    subprocess.run(cmd, check=True)


for out_dir in sorted(OUTPUT_BASE.iterdir()):
    if not out_dir.is_dir():
        continue

    qid = out_dir.name
    video_path = VIDEO_DIR / f"{qid}.mp4"

    if not should_rerun(out_dir):
        continue

    if qid not in by_qid:
        print(f"[SKIP] No question for {qid}")
        continue

    if not video_path.exists():
        print(f"[SKIP] Missing source video for {qid}")
        continue

    question = by_qid[qid]
    lowres_video = TMP_VIDEO_DIR / f"{qid}_w{LOWRES_WIDTH}.mp4"

    try:
        make_lowres_copy(video_path, lowres_video, width=LOWRES_WIDTH)
    except subprocess.CalledProcessError:
        print(f"[ERROR] ffmpeg failed for {qid}")
        continue

    cmd = [
        "nice", "-n", "5",
        "python", "-u", str(SCRIPT),
        "--video", str(lowres_video),
        "--question", question,
        "--output-dir", str(out_dir),
        "--qwen-model", QWEN_MODEL,
        "--yolo-model", YOLO_MODEL,
        "--device", DEVICE,
        "--conf", CONF,
        "--n-anchor-frames", str(N_ANCHOR_FRAMES),
        "--max-candidates-per-frame", str(MAX_CANDIDATES_PER_FRAME),
    ]

    print(f"\n[RUN LOWMEM] {qid}")

    print(f"    original video: {video_path}")
    print(f"    lowres video  : {lowres_video}")
    print(f"    question      : {question}")

    subprocess.run(cmd)