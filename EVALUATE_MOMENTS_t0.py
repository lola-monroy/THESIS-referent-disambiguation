#!/usr/bin/env python3
"""
t = 0.2"""

import os
import re
import sys
import json
import random
import logging
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch

# ====== CONFIG / PATHS ======
DEFAULT_VIDEO_DIR = "/scratch/monroy/Playground/datasets/MoMentS_val_videos_emo_REFERENT_v2"  # where the original unsegmented videos are stored
DEFAULT_OUT_DIR = "/scratch/monroy/Playground/referent_v2_evaluate/script_t1_REFERENT"
DEFAULT_QUESTIONS = "/scratch/monroy/Playground/datasets/MoMentS/data/moments_questions_updated.json"
DEFAULT_GT        = "/scratch/monroy/Playground/datasets/MoMentS/data/validation/moments_validation_keys.json"


EMOTIONS_ONLY = True
MIN_CLIP_SIZE_BYTES = 1024

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── seeds ────────────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── Model Loading & Inference ────────────────────────────────────────────────

def load_official_model(cfg_path: str = "eval_configs/demo.yaml", gpu_id: int = 0):
    """Load Emotion-LLaMA model exactly like cli_inference."""
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.conversation.conversation import Chat

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    class _Args:
        def __init__(self, cfg_path):
            self.cfg_path = cfg_path
            self.options  = None

    cfg = Config(_Args(cfg_path))
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    vis_processor = None
    for key in ["feature_face_caption", "webvid"]:
        try:
            vis_cfg = getattr(cfg.datasets_cfg, key).vis_processor.train
            vis_processor = registry.get_processor_class(vis_cfg.name).from_config(vis_cfg)
            break
        except AttributeError:
            pass

    if vis_processor is None:
        raise RuntimeError("Could not find a valid vis_processor in model config.")

    return Chat(model, vis_processor, device=device), device

def run_inference(chat, video_path: str, question: str) -> str:
    """Run clip inference mirrors cli_inference.py."""
    from minigpt4.conversation.conversation import Conversation, SeparatorStyle
    chat_state = Conversation(
        system="", roles=(r"<s>[INST] ", r" [/INST]"), messages=[], offset=2,
        sep_style=SeparatorStyle.SINGLE, sep="",
    )

    full_prompt = f"<video><VideoHere></video> <feature><FeatureHere></feature> {question}"
    chat.ask(full_prompt, chat_state)

    img_list = [video_path]
    if img_list and not isinstance(img_list[0], torch.Tensor):
        chat.encode_img(img_list)

    response = chat.answer(
        conv=chat_state,
        img_list=img_list,
        temperature=0.1,
        max_new_tokens=500,
        max_length=2000
    )[0]
    return response

def extract_choice_letter(text: str) -> Optional[str]:
    """Extract A/B/C/D from model response."""
    if not text:
        return None
    t = text.strip()

    m = re.search(r"FINAL_ANSWER\s*:\s*([A-D])", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b(?:answer(?:\s+is)?|option)\s*[:\s]\s*([A-D])\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    if t and t[0].upper() in "ABCD" and (len(t) == 1 or t[1] in (".", ")", " ", ":")):
        return t[0].upper()

    return None

# ── IO helpers ───────────────────────────────────────────────────────────────

def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_jsonl(path: Path, obj: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def build_mcq_prompt(q: Dict) -> str:
    question = (q.get("question") or "").strip()
    opts = q.get("options") or {}
    A, B, C, D = [opts.get(k, "").strip() for k in "ABCD"]
    return (
        f"{question}\n\nOptions:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
        "Task: Analyze the video and choose the single best answer (A, B, C, or D).\n"
        "Instructions:\n"
        "1. IMPORTANT: First, provide a very brief one-sentence reason for EACH option (A, B, C, and D).\n"
        "2. Finally, output a new line exactly in this format: FINAL_ANSWER: [LETTER]\n"
    )

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path(DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_jsonl = out_dir / "predictions.jsonl"
    fail_jsonl = out_dir / "failed.jsonl"

    video_dir = Path(DEFAULT_VIDEO_DIR)
    if not video_dir.exists():
        log.error(f"Video directory not found: {video_dir}")
        return

    q_path = Path(DEFAULT_QUESTIONS)
    if not q_path.exists():
        log.error(f"Questions file not found: {q_path}")
        return

    all_questions = load_json(str(q_path))

    # index questions by question_id and video_id
    qid_to_qs: Dict[str, List[Dict]] = {}
    vid_to_qs: Dict[str, List[Dict]] = {}
    for q in all_questions:
        qid = str(q.get("question_id", "")).strip()
        vid = str(q.get("video_id", "")).strip()
        if qid:
            qid_to_qs.setdefault(qid, []).append(q)
        if vid:
            vid_to_qs.setdefault(vid, []).append(q)

    gt_map = {
        str(x["question_id"]).strip(): str(x["correct_answer_key"]).upper()
        for x in load_json(DEFAULT_GT)
        if "question_id" in x
    }

    # model load
    llama_root = "/scratch/monroy/Emotion-LLaMA"
    if llama_root not in sys.path:
        sys.path.insert(0, llama_root)

    log.info("Loading model...")
    chat, device = load_official_model()
    log.info(f"Model ready on {device}")

    total = correct = pred_none = failed = 0

    video_files = list(video_dir.glob("*.mp4"))
    log.info(f"Found {len(video_files)} videos in {video_dir}")

    for clip_path in video_files:
        stem = clip_path.stem.strip()
        qrecs = qid_to_qs.get(stem) or vid_to_qs.get(stem)
        if not qrecs:
            log.warning(f"ID '{stem}' not found in questions JSON (question_id or video_id). Skipping.")
            continue

        for qrec in qrecs:
            qid = str(qrec.get("question_id", "")).strip()
            if not qid:
                continue

            cats = qrec.get("assigned_categories") or []
            if EMOTIONS_ONLY and "Emotions" not in cats:
                continue

            gt = gt_map.get(qid)
            if not gt:
                log.warning(f"No GT found for question_id {qid}. Skipping.")
                continue

            if (not clip_path.exists()) or (clip_path.stat().st_size < MIN_CLIP_SIZE_BYTES):
                write_jsonl(fail_jsonl, {"question_id": qid, "video_id": stem, "error": "missing_or_too_small"})
                failed += 1
                continue

            prompt = build_mcq_prompt(qrec)
            log.info(f"Video {stem} | Question {qid}: Running inference")

            try:
                raw = run_inference(chat, str(clip_path), prompt)
                pred = extract_choice_letter(raw)

                rec = {
                    "question_id": qid,
                    "video_id": stem,
                    "gt": gt,
                    "pred": pred,
                    "correct": (pred == gt),
                    "raw_response": raw,
                }
                write_jsonl(pred_jsonl, rec)

                total += 1
                if pred == gt:
                    correct += 1
                if pred is None:
                    pred_none += 1

            except Exception as e:
                failed += 1
                log.exception(f"Error processing question {qid} (video {stem})")
                write_jsonl(fail_jsonl, {"question_id": qid, "video_id": stem, "error": str(e)})

    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "pred_none": pred_none,
        "failed": failed,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Evaluation Complete. Accuracy: {metrics['accuracy']:.2%}")

if __name__ == "__main__":
    main()