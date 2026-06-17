#!/usr/bin/env python3
import os, re, json, random, logging, argparse
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


DEFAULT_VIDEO_DIR = "/scratch/monroy/Playground/rga3_dataset_redshade"
DEFAULT_OUT_DIR = "/scratch/monroy/Playground/qwen_eval_2/05_06_rga3_redshade2"
DEFAULT_QUESTIONS = "/scratch/monroy/Playground/datasets/MoMentS/data/moments_questions_updated.json"
DEFAULT_GT = "/scratch/monroy/Playground/datasets/MoMentS/data/validation/moments_validation_keys.json"
EMOTIONS_ONLY = True
MIN_CLIP_SIZE_BYTES = 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, obj: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_mcq_prompt(q: Dict) -> str:
    question = (q.get("question") or "").strip()
    opts = q.get("options") or {}

    A = str(opts.get("A", "")).strip()
    B = str(opts.get("B", "")).strip()
    C = str(opts.get("C", "")).strip()
    D = str(opts.get("D", "")).strip()

    return (
        f"{question}\n\n"
        f"Options:\n"
        f"A. {A}\n"
        f"B. {B}\n"
        f"C. {C}\n"
        f"D. {D}\n\n"
        "Task: Analyze the video and choose the single best answer (A, B, C, or D).\n"
        "Instructions:\n"
        "1. IMPORTANT: First, provide a very brief one-sentence reason for EACH option (A, B, C, and D).\n"
        "2. Finally, output a new line exactly in this format: FINAL_ANSWER: [LETTER]\n"
    )


def extract_choice_letter(text: str) -> Optional[str]:
    if not text:
        return None

    t = text.strip()

    m = re.search(r"FINAL_ANSWER\s*:\s*[\[\(]?\s*([A-D])\s*[\]\)]?", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b(?:answer(?:\s+is)?|option)\s*[:\s]\s*([A-D])\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    if t and t[0].upper() in "ABCD" and (len(t) == 1 or t[1] in (".", ")", " ", ":")):
        return t[0].upper()

    return None


def load_gt_map(gt_path: str) -> Dict[str, str]:
    gt_data = load_json(gt_path)
    gt_map = {}

    for x in gt_data:
        if "question_id" not in x:
            continue

        qid = str(x["question_id"]).strip()

        ans = (
            x.get("correct_answer_key")
            or x.get("answer")
            or x.get("correct_answer")
            or x.get("label")
        )

        if ans is not None:
            gt_map[qid] = str(ans).strip().upper()

    return gt_map


def index_questions(all_questions: List[Dict]):
    qid_to_qs: Dict[str, List[Dict]] = {}
    vid_to_qs: Dict[str, List[Dict]] = {}

    for q in all_questions:
        qid = str(q.get("question_id", "")).strip()
        vid = str(q.get("video_id", "")).strip()

        if qid:
            qid_to_qs.setdefault(qid, []).append(q)
        if vid:
            vid_to_qs.setdefault(vid, []).append(q)

    return qid_to_qs, vid_to_qs


def load_qwen_model(model_name: str):
    log.info(f"Loading Qwen model: {model_name}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


@torch.inference_mode()
def run_qwen_video_inference(
    model,
    processor,
    video_path: str,
    prompt: str,
    fps: float,
    max_pixels: int,
    max_new_tokens: int,
) -> str:

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--gt", default=DEFAULT_GT)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=151200)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_jsonl = out_dir / "predictions.jsonl"
    fail_jsonl = out_dir / "failed.jsonl"

    # Clear old files
    pred_jsonl.write_text("", encoding="utf-8")
    fail_jsonl.write_text("", encoding="utf-8")

    video_dir = Path(args.video_dir)

    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    all_questions = load_json(args.questions)
    gt_map = load_gt_map(args.gt)
    qid_to_qs, vid_to_qs = index_questions(all_questions)

    model, processor = load_qwen_model(args.model)

    video_files = sorted(video_dir.glob("*.mp4"))

    if args.limit is not None:
        video_files = video_files[: args.limit]

    log.info(f"Found {len(video_files)} videos in {video_dir}")

    total = 0
    correct = 0
    pred_none = 0
    failed = 0
    skipped_no_question = 0
    skipped_no_gt = 0
    skipped_non_emotion = 0

    for clip_path in tqdm(video_files):
        stem = clip_path.stem.strip()

        qrecs = qid_to_qs.get(stem) or vid_to_qs.get(stem)

        if not qrecs:
            skipped_no_question += 1
            log.warning(f"ID '{stem}' not found in questions JSON. Skipping.")
            continue

        for qrec in qrecs:
            qid = str(qrec.get("question_id", "")).strip()

            if not qid:
                continue

            cats = qrec.get("assigned_categories") or []
            if EMOTIONS_ONLY and "Emotions" not in cats:
                skipped_non_emotion += 1
                continue

            gt = gt_map.get(qid)
            if not gt:
                skipped_no_gt += 1
                log.warning(f"No GT found for question_id {qid}. Skipping.")
                continue

            if (not clip_path.exists()) or (clip_path.stat().st_size < MIN_CLIP_SIZE_BYTES):
                failed += 1
                write_jsonl(
                    fail_jsonl,
                    {
                        "question_id": qid,
                        "video_id": stem,
                        "error": "missing_or_too_small",
                    },
                )
                continue

            prompt = build_mcq_prompt(qrec)

            log.info(f"Video {stem} | Question {qid}: Running Qwen-VL inference")

            try:
                raw = run_qwen_video_inference(
                    model=model,
                    processor=processor,
                    video_path=str(clip_path),
                    prompt=prompt,
                    fps=args.fps,
                    max_pixels=args.max_pixels,
                    max_new_tokens=args.max_new_tokens,
                )

                pred = extract_choice_letter(raw)

                rec = {
                    "question_id": qid,
                    "video_id": stem,
                    "gt": gt,
                    "pred": pred,
                    "correct": pred == gt,
                    "raw_response": raw,
                    "video_path": str(clip_path),
                    "model": args.model,
                }

                write_jsonl(pred_jsonl, rec)

                total += 1
                if pred == gt:
                    correct += 1
                if pred is None:
                    pred_none += 1

            except Exception as e:
                failed += 1
                log.exception(f"Error processing question {qid} video {stem}")
                write_jsonl(
                    fail_jsonl,
                    {
                        "question_id": qid,
                        "video_id": stem,
                        "error": repr(e),
                    },
                )

    metrics = {
        "model": args.model,
        "video_dir": str(video_dir),
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "accuracy_percent": 100 * correct / total if total else 0.0,
        "pred_none": pred_none,
        "failed": failed,
        "skipped_no_question": skipped_no_question,
        "skipped_no_gt": skipped_no_gt,
        "skipped_non_emotion": skipped_non_emotion,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Evaluation Complete. Accuracy: {metrics['accuracy']:.2%}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()