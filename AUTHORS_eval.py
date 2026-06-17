#!/usr/bin/env python3
"""
EVALUATE_MOMENTS_authors_style.py
---------------------------------
Batch evaluation of Emotion-LLaMA on the MoMentS emotion subset, using the
same metrics and decoding conventions as the original Emotion-LLaMA evaluation
script (deterministic greedy decoding, sklearn accuracy/precision/recall/F1,
confusion matrix, fallback on invalid output).

Run examples
============

# Original videos (baseline)
CUDA_VISIBLE_DEVICES=3 nice -n 15 taskset -c 0-6 \
  python /scratch/monroy/Playground/AUTHORS_eval.py \
  --video-dir /scratch/monroy/Playground/datasets/MoMentS_val_videos_emo \
  --out-dir   /scratch/monroy/Playground/Experiments_AUTHORS/original

# YOLO face-highlighted videos
CUDA_VISIBLE_DEVICES=3 nice -n 15 taskset -c 0-6 \
  python /scratch/monroy/Playground/AUTHORS_eval.py \
  --video-dir /scratch/monroy/Playground/datasets/YOLO_datasets/yolov11mface_video2video_audio \
  --out-dir   /scratch/monroy/Playground/Experiments_AUTHORS/yolo_face_v2v

# Referent-grounded videos
CUDA_VISIBLE_DEVICES=4 nice -n 15 taskset -c 0-6 \
  python /scratch/monroy/Playground/AUTHORS_eval.py \
  --video-dir /scratch/monroy/Playground/datasets/dataset_REFERENT_v2_annotated_audio \
  --out-dir   /scratch/monroy/Playground/Experiments_AUTHORS/grounded
"""

#!/usr/bin/env python3
"""
EVALUATE_MOMENTS_authors_style.py
---------------------------------
Batch evaluation of Emotion-LLaMA on the MoMentS emotion subset, using the
same metrics and decoding conventions as the original Emotion-LLaMA evaluation
script (deterministic greedy decoding, sklearn accuracy/precision/recall/F1,
confusion matrix, fallback on invalid output).

Run examples
============

# Original videos (baseline)
CUDA_VISIBLE_DEVICES=4 nice -n 15 taskset -c 0-6 \
  python /scratch/monroy/Playground/AUTHORS_eval.py \
  --video-dir /scratch/monroy/Playground/datasets/MoMentS_val_videos_emo \
  --out-dir   /scratch/monroy/Playground/Experiments_AUTHORS/original

# YOLO face-highlighted videos
CUDA_VISIBLE_DEVICES=1 nice -n 15 taskset -c 0-6 \
  python /scratch/monroy/Playground/AUTHORS_eval.py \
  --video-dir /scratch/monroy/Playground/datasets/YOLO_datasets/yolov11mface_video2video_audio \
  --out-dir   /scratch/monroy/Playground/Experiments_AUTHORS/yolo_face_v2v

# Referent-grounded videos
CUDA_VISIBLE_DEVICES=4 nice -n 15 taskset -c 0-6 \
  python /scratch/monroy/Playground/AUTHORS_eval.py \
  --video-dir /scratch/monroy/Playground/datasets/grounded_videos \
  --out-dir   /scratch/monroy/Playground/Experiments_AUTHORS/grounded
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)


# ============================================================================
# DEFAULTS  (override on the CLI)
# ============================================================================

DEFAULT_QUESTIONS  = "/scratch/monroy/Playground/datasets/MoMentS/data/moments_questions_updated.json"
DEFAULT_GT         = "/scratch/monroy/Playground/datasets/MoMentS/data/validation/moments_validation_keys.json"
DEFAULT_LLAMA_ROOT = "/scratch/monroy/Emotion-LLaMA"
DEFAULT_CFG_PATH   = "eval_configs/demo.yaml"

EMOTIONS_ONLY       = True
MIN_CLIP_SIZE_BYTES = 1024


# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# Model loading
# ============================================================================

def load_official_model(cfg_path: str, gpu_id: int = 0):
    """Load Emotion-LLaMA using its own machinery (same imports as authors)."""
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.conversation.conversation import Chat

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    class _Args:
        def __init__(self, cfg_path):
            self.cfg_path = cfg_path
            self.options  = None

    cfg = Config(_Args(cfg_path))
    model_cls = registry.get_model_class(cfg.model_cfg.arch)
    model = model_cls.from_config(cfg.model_cfg).to(device)
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


# ============================================================================
# Inference  (deterministic greedy, mirroring authors' do_sample=False)
# ============================================================================

def run_inference(chat, video_path: str, question: str) -> str:
    """Single deterministic greedy generation -- no temperature, no sampling."""
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

    # Greedy decoding -- equivalent to authors' model.generate(..., do_sample=False)
    generation_dict = chat.answer_prepare(
        conv=chat_state, img_list=img_list,
        temperature=1.0, max_new_tokens=500, max_length=2000,
    )
    generation_dict["do_sample"] = False
    generation_dict.pop("top_p", None)
    generation_dict.pop("temperature", None)
    output_token = chat.model_generate(**generation_dict)[0]
    output_text  = chat.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
    output_text  = output_text.split("###")[0].split("Assistant:")[-1].strip()
    chat_state.messages[-1][1] = output_text
    return output_text


# ============================================================================
# Answer extraction  (closed label set, fallback on invalid)
# ============================================================================

VALID_LETTERS    = {"A", "B", "C", "D"}
FALLBACK_LETTER  = "A"     # authors fall back to 'neutral'; we fall back to A

_re_final  = re.compile(r"FINAL_ANSWER\s*:?\s*[\[\(]?\s*([A-D])\s*[\]\)]?", re.IGNORECASE)
_re_phrase = re.compile(r"\b(?:answer(?:\s+is)?|option)\s*[:\s]\s*([A-D])\b", re.IGNORECASE)


def extract_letter(text: str) -> str:
    if not text:
        return FALLBACK_LETTER
    t = text.strip()
    m = _re_final.search(t)
    if m:
        return m.group(1).upper()
    m = _re_phrase.search(t)
    if m:
        return m.group(1).upper()
    if t and t[0].upper() in VALID_LETTERS and (len(t) == 1 or t[1] in (".", ")", " ", ":")):
        return t[0].upper()
    m = re.search(r"\b([A-D])\b", t[:200])
    if m:
        return m.group(1).upper()
    log.warning("Could not parse letter from response -- falling back to %s", FALLBACK_LETTER)
    return FALLBACK_LETTER


# ============================================================================
# Helpers
# ============================================================================

def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_mcq_prompt(q: Dict, transcript: Optional[str] = None) -> str:
    question = (q.get("question") or "").strip()
    opts = q.get("options") or {}
    A, B, C, D = [opts.get(k, "").strip() for k in "ABCD"]

    transcript_block = ""
    if transcript is not None and transcript.strip():
        transcript_block = (
            f"Transcript of the spoken dialogue in the video:\n"
            f"\"{transcript.strip()}\"\n\n"
        )

    return (
        f"{transcript_block}"
        f"{question}\n\nOptions:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
        "Task: Analyze the video and choose the single best answer (A, B, C, or D).\n"
        "Instructions:\n"
        "1. IMPORTANT: First, provide a very brief one-sentence reason for EACH option (A, B, C, and D).\n"
        "2. Finally, output a new line exactly in this format: FINAL_ANSWER: [LETTER]\n"
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Authors-style evaluation of Emotion-LLaMA on MoMentS.")
    p.add_argument("--video-dir", required=True, help="Folder of .mp4 clips to evaluate.")
    p.add_argument("--out-dir",   required=True, help="Folder to write predictions + metrics.")
    p.add_argument("--questions", default=DEFAULT_QUESTIONS)
    p.add_argument("--gt",        default=DEFAULT_GT)
    p.add_argument("--llama-root", default=DEFAULT_LLAMA_ROOT)
    p.add_argument("--cfg-path",   default=DEFAULT_CFG_PATH)
    p.add_argument("--transcripts-json", default=None,
                   help="Optional JSON mapping video_id -> transcript string. "
                        "When provided, the transcript is injected into the prompt.")
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_jsonl = out_dir / "predictions.jsonl"
    fail_jsonl = out_dir / "failed.jsonl"
    for f in [pred_jsonl, fail_jsonl, out_dir / "metrics.json"]:
        if f.exists():
            f.unlink()

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        log.error("Video directory not found: %s", video_dir)
        return

    # ---- load questions + GT ----
    all_questions = load_json(args.questions)
    qid_to_qs: Dict[str, List[Dict]] = {}
    vid_to_qs: Dict[str, List[Dict]] = {}
    for q in all_questions:
        qid = str(q.get("question_id", "")).strip()
        vid = str(q.get("video_id", "")).strip()
        if qid: qid_to_qs.setdefault(qid, []).append(q)
        if vid: vid_to_qs.setdefault(vid, []).append(q)

    gt_map = {
        str(x["question_id"]).strip(): str(x["correct_answer_key"]).upper()
        for x in load_json(args.gt) if "question_id" in x
    }

    # ---- optional transcripts (video_id -> string) ----
    transcripts: Dict[str, str] = {}
    if args.transcripts_json:
        transcripts = load_json(args.transcripts_json)
        log.info("Loaded %d transcripts from %s", len(transcripts), args.transcripts_json)

    # ---- bring Emotion-LLaMA on the path ----
    if args.llama_root not in sys.path:
        sys.path.insert(0, args.llama_root)

    log.info("Loading model...")
    chat, device = load_official_model(args.cfg_path)
    log.info("Model ready on %s", device)

    # ---- iterate videos ----
    targets_list: List[str] = []
    answers_list: List[str] = []
    names_list:   List[str] = []
    pred_none = failed = 0

    video_files = sorted(video_dir.glob("*.mp4"))
    log.info("Found %d videos in %s", len(video_files), video_dir)

    for clip_path in video_files:
        stem = clip_path.stem.strip()
        qrecs = qid_to_qs.get(stem) or vid_to_qs.get(stem)
        if not qrecs:
            log.warning("ID '%s' not found in questions JSON. Skipping.", stem)
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
                log.warning("No GT for %s. Skipping.", qid)
                continue
            if (not clip_path.exists()) or (clip_path.stat().st_size < MIN_CLIP_SIZE_BYTES):
                with open(fail_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"question_id": qid, "video_id": stem,
                                        "error": "missing_or_too_small"}) + "\n")
                failed += 1
                continue

            prompt = build_mcq_prompt(qrec, transcript=transcripts.get(stem))
            log.info("%s / %s", stem, qid)

            try:
                raw  = run_inference(chat, str(clip_path), prompt)
                pred = extract_letter(raw)
                if pred not in VALID_LETTERS:
                    log.warning("Error: %s  Target: %s", pred, gt)
                    pred_none += 1
                    pred = FALLBACK_LETTER

                rec = {
                    "question_id": qid, "video_id": stem,
                    "gt": gt, "pred": pred,
                    "correct": (pred == gt),
                    "raw_response": raw,
                }
                with open(pred_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                targets_list.append(gt)
                answers_list.append(pred)
                names_list.append(qid)

            except Exception as e:
                failed += 1
                log.exception("Error on %s (%s)", qid, stem)
                with open(fail_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"question_id": qid, "video_id": stem,
                                        "error": str(e)}) + "\n")

    # ========================================================================
    # METRICS  (same block style as the authors' eval_emotion.py)
    # ========================================================================

    if not targets_list:
        log.error("No predictions were produced -- nothing to score.")
        return

    accuracy  = accuracy_score(targets_list, answers_list)
    precision = precision_score(targets_list, answers_list, average='weighted', zero_division=0)
    recall    = recall_score(targets_list, answers_list, average='weighted', zero_division=0)
    f1        = f1_score(targets_list, answers_list, average='weighted', zero_division=0)

    print("Accuracy:",  accuracy)
    print("Precision:", precision)
    print("Recall:",    recall)
    print("F1 Score:",  f1)

    cm = confusion_matrix(targets_list, answers_list, labels=["A", "B", "C", "D"])
    print(cm)

    # Bias diagnostic (useful given the model's C-bias)
    pred_dist = dict(Counter(answers_list))
    gt_dist   = dict(Counter(targets_list))
    print("Predicted letter distribution:", pred_dist)
    print("Ground-truth letter distribution:", gt_dist)

    metrics = {
        "video_dir":  str(video_dir),
        "out_dir":    str(out_dir),
        "total":      len(targets_list),
        "correct":    int(sum(1 for g, p in zip(targets_list, answers_list) if g == p)),
        "accuracy":   accuracy,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "confusion_matrix": cm.tolist(),
        "labels":     ["A", "B", "C", "D"],
        "pred_dist":  pred_dist,
        "gt_dist":    gt_dist,
        "pred_none":  pred_none,
        "failed":     failed,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Done. %d/%d = %.2f%%",
             metrics["correct"], metrics["total"], 100 * metrics["accuracy"])


if __name__ == "__main__":
    main()

# import argparse
# import json
# import logging
# import os
# import re
# import sys
# from collections import Counter
# from pathlib import Path
# from typing import Dict, List, Optional

# import numpy as np
# import torch
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score,
#     f1_score, confusion_matrix,
# )


# # ============================================================================
# # DEFAULTS  (override on the CLI)
# # ============================================================================

# DEFAULT_QUESTIONS  = "/scratch/monroy/Playground/datasets/MoMentS/data/moments_questions_updated.json"
# DEFAULT_GT         = "/scratch/monroy/Playground/datasets/MoMentS/data/validation/moments_validation_keys.json"
# DEFAULT_LLAMA_ROOT = "/scratch/monroy/Emotion-LLaMA"
# DEFAULT_CFG_PATH   = "eval_configs/demo.yaml"

# EMOTIONS_ONLY       = True
# MIN_CLIP_SIZE_BYTES = 1024


# # ============================================================================
# # Logging
# # ============================================================================

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s  %(levelname)-8s  %(message)s",
#     datefmt="%H:%M:%S",
# )
# log = logging.getLogger(__name__)


# # ============================================================================
# # Model loading
# # ============================================================================

# def load_official_model(cfg_path: str, gpu_id: int = 0):
#     """Load Emotion-LLaMA using its own machinery (same imports as authors)."""
#     from minigpt4.common.config import Config
#     from minigpt4.common.registry import registry
#     from minigpt4.conversation.conversation import Chat

#     device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

#     class _Args:
#         def __init__(self, cfg_path):
#             self.cfg_path = cfg_path
#             self.options  = None

#     cfg = Config(_Args(cfg_path))
#     model_cls = registry.get_model_class(cfg.model_cfg.arch)
#     model = model_cls.from_config(cfg.model_cfg).to(device)
#     model.eval()

#     vis_processor = None
#     for key in ["feature_face_caption", "webvid"]:
#         try:
#             vis_cfg = getattr(cfg.datasets_cfg, key).vis_processor.train
#             vis_processor = registry.get_processor_class(vis_cfg.name).from_config(vis_cfg)
#             break
#         except AttributeError:
#             pass
#     if vis_processor is None:
#         raise RuntimeError("Could not find a valid vis_processor in model config.")

#     return Chat(model, vis_processor, device=device), device


# # ============================================================================
# # Inference  (deterministic greedy, mirroring authors' do_sample=False)
# # ============================================================================

# def run_inference(chat, video_path: str, question: str) -> str:
#     """Single deterministic greedy generation -- no temperature, no sampling."""
#     from minigpt4.conversation.conversation import Conversation, SeparatorStyle

#     chat_state = Conversation(
#         system="", roles=(r"<s>[INST] ", r" [/INST]"), messages=[], offset=2,
#         sep_style=SeparatorStyle.SINGLE, sep="",
#     )
#     full_prompt = f"<video><VideoHere></video> <feature><FeatureHere></feature> {question}"
#     chat.ask(full_prompt, chat_state)

#     img_list = [video_path]
#     if img_list and not isinstance(img_list[0], torch.Tensor):
#         chat.encode_img(img_list)

#     # Greedy decoding -- equivalent to authors' model.generate(..., do_sample=False)
#     generation_dict = chat.answer_prepare(
#         conv=chat_state, img_list=img_list,
#         temperature=1.0, max_new_tokens=500, max_length=2000,
#     )
#     generation_dict["do_sample"] = False
#     generation_dict.pop("top_p", None)
#     generation_dict.pop("temperature", None)
#     output_token = chat.model_generate(**generation_dict)[0]
#     output_text  = chat.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
#     output_text  = output_text.split("###")[0].split("Assistant:")[-1].strip()
#     chat_state.messages[-1][1] = output_text
#     return output_text


# # ============================================================================
# # Answer extraction  (closed label set, fallback on invalid)
# # ============================================================================

# VALID_LETTERS    = {"A", "B", "C", "D"}
# FALLBACK_LETTER  = "A"     # authors fall back to 'neutral'; we fall back to A

# _re_final  = re.compile(r"FINAL_ANSWER\s*:?\s*[\[\(]?\s*([A-D])\s*[\]\)]?", re.IGNORECASE)
# _re_phrase = re.compile(r"\b(?:answer(?:\s+is)?|option)\s*[:\s]\s*([A-D])\b", re.IGNORECASE)


# def extract_letter(text: str) -> str:
#     if not text:
#         return FALLBACK_LETTER
#     t = text.strip()
#     m = _re_final.search(t)
#     if m:
#         return m.group(1).upper()
#     m = _re_phrase.search(t)
#     if m:
#         return m.group(1).upper()
#     if t and t[0].upper() in VALID_LETTERS and (len(t) == 1 or t[1] in (".", ")", " ", ":")):
#         return t[0].upper()
#     m = re.search(r"\b([A-D])\b", t[:200])
#     if m:
#         return m.group(1).upper()
#     log.warning("Could not parse letter from response -- falling back to %s", FALLBACK_LETTER)
#     return FALLBACK_LETTER


# # ============================================================================
# # Helpers
# # ============================================================================

# def load_json(path: str):
#     return json.loads(Path(path).read_text(encoding="utf-8"))


# def build_mcq_prompt(q: Dict) -> str:
#     question = (q.get("question") or "").strip()
#     opts = q.get("options") or {}
#     A, B, C, D = [opts.get(k, "").strip() for k in "ABCD"]
#     return (
#         f"{question}\n\nOptions:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
#         "Task: Analyze the video and choose the single best answer (A, B, C, or D).\n"
#         "Instructions:\n"
#         "1. IMPORTANT: First, provide a very brief one-sentence reason for EACH option (A, B, C, and D).\n"
#         "2. Finally, output a new line exactly in this format: FINAL_ANSWER: [LETTER]\n"
#     )


# # ============================================================================
# # CLI
# # ============================================================================

# def parse_args():
#     p = argparse.ArgumentParser(description="Authors-style evaluation of Emotion-LLaMA on MoMentS.")
#     p.add_argument("--video-dir", required=True, help="Folder of .mp4 clips to evaluate.")
#     p.add_argument("--out-dir",   required=True, help="Folder to write predictions + metrics.")
#     p.add_argument("--questions", default=DEFAULT_QUESTIONS)
#     p.add_argument("--gt",        default=DEFAULT_GT)
#     p.add_argument("--llama-root", default=DEFAULT_LLAMA_ROOT)
#     p.add_argument("--cfg-path",   default=DEFAULT_CFG_PATH)
#     return p.parse_args()


# # ============================================================================
# # Main
# # ============================================================================

# def main():
#     args = parse_args()

#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     pred_jsonl = out_dir / "predictions.jsonl"
#     fail_jsonl = out_dir / "failed.jsonl"
#     for f in [pred_jsonl, fail_jsonl, out_dir / "metrics.json"]:
#         if f.exists():
#             f.unlink()

#     video_dir = Path(args.video_dir)
#     if not video_dir.exists():
#         log.error("Video directory not found: %s", video_dir)
#         return

#     # ---- load questions + GT ----
#     all_questions = load_json(args.questions)
#     qid_to_qs: Dict[str, List[Dict]] = {}
#     vid_to_qs: Dict[str, List[Dict]] = {}
#     for q in all_questions:
#         qid = str(q.get("question_id", "")).strip()
#         vid = str(q.get("video_id", "")).strip()
#         if qid: qid_to_qs.setdefault(qid, []).append(q)
#         if vid: vid_to_qs.setdefault(vid, []).append(q)

#     gt_map = {
#         str(x["question_id"]).strip(): str(x["correct_answer_key"]).upper()
#         for x in load_json(args.gt) if "question_id" in x
#     }

#     # ---- bring Emotion-LLaMA on the path ----
#     if args.llama_root not in sys.path:
#         sys.path.insert(0, args.llama_root)

#     log.info("Loading model...")
#     chat, device = load_official_model(args.cfg_path)
#     log.info("Model ready on %s", device)

#     # ---- iterate videos ----
#     targets_list: List[str] = []
#     answers_list: List[str] = []
#     names_list:   List[str] = []
#     pred_none = failed = 0

#     video_files = sorted(video_dir.glob("*.mp4"))
#     log.info("Found %d videos in %s", len(video_files), video_dir)

#     for clip_path in video_files:
#         stem = clip_path.stem.strip()
#         qrecs = qid_to_qs.get(stem) or vid_to_qs.get(stem)
#         if not qrecs:
#             log.warning("ID '%s' not found in questions JSON. Skipping.", stem)
#             continue

#         for qrec in qrecs:
#             qid = str(qrec.get("question_id", "")).strip()
#             if not qid:
#                 continue
#             cats = qrec.get("assigned_categories") or []
#             if EMOTIONS_ONLY and "Emotions" not in cats:
#                 continue
#             gt = gt_map.get(qid)
#             if not gt:
#                 log.warning("No GT for %s. Skipping.", qid)
#                 continue
#             if (not clip_path.exists()) or (clip_path.stat().st_size < MIN_CLIP_SIZE_BYTES):
#                 with open(fail_jsonl, "a", encoding="utf-8") as f:
#                     f.write(json.dumps({"question_id": qid, "video_id": stem,
#                                         "error": "missing_or_too_small"}) + "\n")
#                 failed += 1
#                 continue

#             prompt = build_mcq_prompt(qrec)
#             log.info("%s / %s", stem, qid)

#             try:
#                 raw  = run_inference(chat, str(clip_path), prompt)
#                 pred = extract_letter(raw)
#                 if pred not in VALID_LETTERS:
#                     log.warning("Error: %s  Target: %s", pred, gt)
#                     pred_none += 1
#                     pred = FALLBACK_LETTER

#                 rec = {
#                     "question_id": qid, "video_id": stem,
#                     "gt": gt, "pred": pred,
#                     "correct": (pred == gt),
#                     "raw_response": raw,
#                 }
#                 with open(pred_jsonl, "a", encoding="utf-8") as f:
#                     f.write(json.dumps(rec, ensure_ascii=False) + "\n")

#                 targets_list.append(gt)
#                 answers_list.append(pred)
#                 names_list.append(qid)

#             except Exception as e:
#                 failed += 1
#                 log.exception("Error on %s (%s)", qid, stem)
#                 with open(fail_jsonl, "a", encoding="utf-8") as f:
#                     f.write(json.dumps({"question_id": qid, "video_id": stem,
#                                         "error": str(e)}) + "\n")

#     # ========================================================================
#     # METRICS  (same block style as the authors' eval_emotion.py)
#     # ========================================================================

#     if not targets_list:
#         log.error("No predictions were produced -- nothing to score.")
#         return

#     accuracy  = accuracy_score(targets_list, answers_list)
#     precision = precision_score(targets_list, answers_list, average='weighted', zero_division=0)
#     recall    = recall_score(targets_list, answers_list, average='weighted', zero_division=0)
#     f1        = f1_score(targets_list, answers_list, average='weighted', zero_division=0)

#     print("Accuracy:",  accuracy)
#     print("Precision:", precision)
#     print("Recall:",    recall)
#     print("F1 Score:",  f1)

#     cm = confusion_matrix(targets_list, answers_list, labels=["A", "B", "C", "D"])
#     print(cm)

#     # Bias diagnostic (useful given the model's C-bias)
#     pred_dist = dict(Counter(answers_list))
#     gt_dist   = dict(Counter(targets_list))
#     print("Predicted letter distribution:", pred_dist)
#     print("Ground-truth letter distribution:", gt_dist)

#     metrics = {
#         "video_dir":  str(video_dir),
#         "out_dir":    str(out_dir),
#         "total":      len(targets_list),
#         "correct":    int(sum(1 for g, p in zip(targets_list, answers_list) if g == p)),
#         "accuracy":   accuracy,
#         "precision":  precision,
#         "recall":     recall,
#         "f1":         f1,
#         "confusion_matrix": cm.tolist(),
#         "labels":     ["A", "B", "C", "D"],
#         "pred_dist":  pred_dist,
#         "gt_dist":    gt_dist,
#         "pred_none":  pred_none,
#         "failed":     failed,
#     }
#     (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
#     log.info("Done. %d/%d = %.2f%%",
#              metrics["correct"], metrics["total"], 100 * metrics["accuracy"])


# if __name__ == "__main__":
#     main()