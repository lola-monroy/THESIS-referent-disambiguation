import json
import subprocess
from pathlib import Path

VIDEO_DIR = Path("/scratch/monroy/Playground/datasets/MoMentS_val_videos_emo")
QUESTIONS_JSON = Path("/scratch/monroy/Playground/datasets/MoMentS/data/moments_questions_updated.json")
SCRIPT = Path("/scratch/monroy/Playground/referent_anchor_face.py")
BASE_OUTPUT_DIR = Path("/scratch/monroy/Playground/datasets/MoMentS_val_videos_emo_REFERENT")

QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
YOLO_MODEL = "yolo11x.pt"
DEVICE = "cuda"
CONF = "0.20"


def load_questions(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_questions(QUESTIONS_JSON)

    # map question_id -> question dict
    by_qid = {}
    for q in questions:
        qid = q.get("question_id")
        question_text = q.get("question")
        cats = q.get("assigned_categories", [])

        if not qid or not question_text:
            continue

        if "Emotions" not in cats:
            continue

        by_qid[str(qid)] = q

    videos = sorted(VIDEO_DIR.glob("*.mp4"))

    print(f"Loaded {len(by_qid)} Emotions questions")
    print(f"Found {len(videos)} videos")

    for video_path in videos:
        stem = video_path.stem   # e.g. kREcQ
        q = by_qid.get(stem)

        if q is None:
            print(f"[SKIP] No matching question_id for {video_path.name}")
            continue

        question_text = q["question"]
        out_dir = BASE_OUTPUT_DIR / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "nice", "-n", "5",
            "python", str(SCRIPT),
            "--video", str(video_path),
            "--question", question_text,
            "--output-dir", str(out_dir),
            "--qwen-model", QWEN_MODEL,
            "--yolo-model", YOLO_MODEL,
            "--device", DEVICE,
            "--conf", CONF,
        ]

        print(f"\n[RUN] {video_path.name}")
        print(f"      question_id: {stem}")
        print(f"      question: {question_text}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed on {video_path.name} | returncode={e.returncode}")


if __name__ == "__main__":
    main()