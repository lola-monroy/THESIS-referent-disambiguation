#!/usr/bin/env python3
# 10 de junio: much less biased 
"""

evaluate_moments_onvideo_antibias.py
------------------------------------
Emotion-LLaMA MoMentS evaluation with:
- restored aggregation modes
- text fallback for null predictions
- anti-"always A" safeguards

Why:
Your logs show many raw_response values are literally:
    "a"
or:
    "option a reason"

That means the model/prompt is collapsing to the first option, not genuinely choosing.
This script reduces that problem by:
1. Deterministically shuffling option positions per question.
2. Mapping displayed answer letters back to the original MoMentS letters.
3. Refusing to parse "option a reason" as answer A.
4. Keeping raw_response unchanged for thesis/debugging.
"""

import os
import re
import sys
import json
import random
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np
import torch

# ==============================================================================
# CONFIG
# ==============================================================================

DEFAULT_VIDEO_DIR = "/scratch/monroy/Playground/datasets/YOLO_datasets/yolov11mface_video2video_audio"
DEFAULT_QUESTIONS = "/scratch/monroy/Playground/datasets/MoMentS/data/moments_questions_updated.json"
DEFAULT_GT = "/scratch/monroy/Playground/datasets/MoMentS/data/validation/moments_validation_keys.json"
DEFAULT_OUT_DIR = "/scratch/monroy/Playground/Experiments_Antibias/yolo"

EMOTIONS_ONLY = True
MIN_CLIP_SIZE_BYTES = 1024

AGG_MODE = "top1"       # top1, vote, weighted, fallback
WEIGHT_POWER = 1.0
FALLBACK_MARGIN = 0.05
APPEND_OUTPUTS = False

TEMPERATURE = 0.1
MAX_NEW_TOKENS = 500
MAX_LENGTH = 2000

# Critical anti-bias option.
SHUFFLE_OPTIONS = False

# If True, free-text responses are matched to the option texts.
USE_TEXT_OPTION_FALLBACK = True

# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ==============================================================================
# Model loading / inference
# ==============================================================================

def load_official_model(cfg_path: str = "eval_configs/demo.yaml", gpu_id: int = 0):
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.conversation.conversation import Chat

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    class _Args:
        def __init__(self, cfg_path):
            self.cfg_path = cfg_path
            self.options = None

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


def run_inference(chat, video_path: str, prompt: str, device: str) -> str:
    from minigpt4.conversation.conversation import Conversation, SeparatorStyle

    chat_state = Conversation(
        system="",
        roles=(r"<s>[INST] ", r" [/INST]"),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="",
    )

    full_prompt = f"<video><VideoHere></video> <feature><FeatureHere></feature> {prompt}"
    chat.ask(full_prompt, chat_state)

    img_list = [video_path]
    if img_list and not isinstance(img_list[0], torch.Tensor):
        chat.encode_img(img_list)

    response = chat.answer(
        conv=chat_state,
        img_list=img_list,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
        max_length=MAX_LENGTH,
    )[0]

    return response.strip() if isinstance(response, str) else str(response)


# ==============================================================================
# Prompt, shuffling, extraction
# ==============================================================================

def stable_shuffle_letters(question_id: str) -> List[str]:
    """
    Deterministic shuffle, so repeated runs use the same option order.
    """
    letters = list("ABCD")
    h = hashlib.md5(question_id.encode("utf-8")).hexdigest()
    seed = int(h[:8], 16)
    rng = random.Random(seed)
    rng.shuffle(letters)
    return letters


def get_original_options(q: Dict) -> Dict[str, str]:
    opts = q.get("options") or {}
    return {k: (opts.get(k, "") or "").strip() for k in "ABCD"}


def make_display_options(q: Dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      display_options: displayed A-D -> option text
      display_to_original: displayed A-D -> original A-D
    """
    qid = str(q.get("question_id", "")).strip()
    original = get_original_options(q)

    if not SHUFFLE_OPTIONS:
        return original, {k: k for k in "ABCD"}

    original_order = stable_shuffle_letters(qid)
    display_letters = list("ABCD")

    display_options = {}
    display_to_original = {}

    for display_letter, original_letter in zip(display_letters, original_order):
        display_options[display_letter] = original[original_letter]
        display_to_original[display_letter] = original_letter

    return display_options, display_to_original


def build_mcq_prompt(q: Dict) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    question = (q.get("question") or "").strip()
    display_options, display_to_original = make_display_options(q)

    a, b, c, d = [display_options[k] for k in "ABCD"]

    prompt = (
        f"{question}\n\n"
        f"Options:\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
        f"Instructions:\n"
        f"1. Analyze the video and choose the single best answer (A, B, C, or D).\n"
        f"2. Finally, output a new line exactly in this format: FINAL_ANSWER: [LETTER]\n"
    )
    return prompt, display_options, display_to_original

# def build_mcq_prompt(q: Dict) -> str:
#     question = (q.get("question") or "").strip()
#     opts = q.get("options") or {}
#     A, B, C, D = [opts.get(k, "").strip() for k in "ABCD"]
#     return (
#         f"{question}\n\n"
#         f"Options:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
#         f"Instructions:\n"
#         f"1. Analyze the video and choose the single best answer (A, B, C, or D).\n"
#         f"2. Finally, output a new line exactly in this format: FINAL_ANSWER: [LETTER]\n"
#     )

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


EMOTION_SYNONYMS = {
    "happy": {"happy", "happiness", "glad", "joy", "joyful", "pleased", "delighted", "excited", "amused", "smiling", "cheerful"},
    "sad": {"sad", "sadness", "unhappy", "upset", "depressed", "down", "crying", "tearful", "sorrow", "heartbroken", "regret"},
    "angry": {"angry", "anger", "mad", "furious", "annoyed", "irritated", "frustrated", "rage"},
    "afraid": {"afraid", "fear", "fearful", "scared", "terrified", "worried", "anxious", "nervous", "panic"},
    "surprised": {"surprised", "surprise", "shocked", "astonished", "amazed", "startled", "taken aback"},
    "disgusted": {"disgusted", "disgust", "repulsed", "grossed", "revulsion"},
    "confused": {"confused", "confusion", "puzzled", "uncertain", "unsure", "bewildered"},
    "concerned": {"concerned", "concern", "worried", "anxious", "troubled"},
    "embarrassed": {"embarrassed", "embarrassment", "ashamed", "awkward", "humiliated"},
    "disappointed": {"disappointed", "disappointment", "let down", "sad", "upset"},
    "neutral": {"neutral", "calm", "indifferent", "unemotional", "serious"},
}


def option_keyword_score(raw_norm: str, option_norm: str) -> float:
    if not raw_norm or not option_norm:
        return 0.0

    score = 0.0

    if option_norm in raw_norm:
        score += 10.0
    if raw_norm in option_norm and len(raw_norm.split()) <= 6:
        score += 5.0

    raw_words = set(raw_norm.split())
    opt_words = set(option_norm.split())

    stop = {
        "the", "a", "an", "is", "are", "to", "of", "and", "with", "because",
        "he", "she", "they", "his", "her", "their", "feels", "feeling", "feel",
        "option", "reason", "answer", "final"
    }

    raw_content = {w for w in raw_words if len(w) > 2 and w not in stop}
    opt_content = {w for w in opt_words if len(w) > 2 and w not in stop}

    score += len(raw_content & opt_content)

    for _, syns in EMOTION_SYNONYMS.items():
        raw_has = any(normalize_text(x) in raw_norm for x in syns)
        opt_has = any(normalize_text(x) in option_norm for x in syns)
        if raw_has and opt_has:
            score += 4.0

    return score


def fallback_match_option_from_text(raw_response: str, display_options: Dict[str, str]) -> Tuple[Optional[str], Dict[str, Any]]:
    raw_norm = normalize_text(raw_response)
    scores = {}

    for display_letter in "ABCD":
        opt_norm = normalize_text(display_options.get(display_letter, ""))
        scores[display_letter] = option_keyword_score(raw_norm, opt_norm)

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_letter, best_score = ranked[0]
    second_letter, second_score = ranked[1]

    info = {
        "fallback_used": True,
        "fallback_type": "text_option_match",
        "fallback_scores_display": scores,
        "fallback_best_display": best_letter,
        "fallback_best_score": best_score,
        "fallback_second_display": second_letter,
        "fallback_second_score": second_score,
    }

    if best_score <= 0:
        info["fallback_accepted"] = False
        info["reason"] = "no_positive_score"
        return None, info

    if best_score == second_score:
        info["fallback_accepted"] = False
        info["reason"] = "tie"
        return None, info

    info["fallback_accepted"] = True
    return best_letter, info


def extract_display_letter(text: str, display_options: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Extracts the DISPLAY letter A-D.
    Later we map display letter back to original MoMentS letter.

    Important:
    - Does NOT parse "option a reason" as answer A.
    - Bare "a" is allowed, but because options are shuffled, this no longer always means original A.
    """
    info: Dict[str, Any] = {"method": None, "fallback_used": False}

    if not text:
        return None, info

    t = text.strip()

    explicit_patterns = [
        ("final_answer", r"FINAL_ANSWER\s*:?\s*\[?\s*([A-Da-d])\s*\]?"),
        ("final answer", r"FINAL\s+ANSWER\s*:?\s*\[?\s*([A-Da-d])\s*\]?"),
        ("answer_is", r"\banswer(?:\s+is)?\s*[:\s]\s*([A-Da-d])\b"),
    ]

    for method, pat in explicit_patterns:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            info["method"] = method
            return m.group(1).upper(), info

    # Parse "option C" only when it looks like a final choice, not "option C reason".
    m = re.search(r"\boption\s+([A-Da-d])\b", t, re.IGNORECASE)
    if m:
        after = t[m.end():].strip().lower()
        if not after.startswith("reason"):
            info["method"] = "option_letter"
            return m.group(1).upper(), info

    # Bare one-letter answer.
    if t.lower() in {"a", "b", "c", "d"}:
        info["method"] = "bare_letter"
        return t.upper(), info

    if USE_TEXT_OPTION_FALLBACK and display_options is not None:
        pred, fb_info = fallback_match_option_from_text(t, display_options)
        info.update(fb_info)
        if pred is not None:
            info["method"] = "fallback_text_option_match"
        return pred, info

    return None, info


def map_display_to_original(display_pred: Optional[str], display_to_original: Dict[str, str]) -> Optional[str]:
    if display_pred not in {"A", "B", "C", "D"}:
        return None
    return display_to_original.get(display_pred)


# ==============================================================================
# IO helpers
# ==============================================================================

def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, obj: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def clear_old_outputs(*paths: Path):
    if APPEND_OUTPUTS:
        return
    for p in paths:
        if p.exists():
            p.unlink()


# ==============================================================================
# Video grouping / aggregation
# ==============================================================================

def strip_chunk_suffix(stem: str) -> str:
    s = stem.strip()
    patterns = [
        r"(.+?)[_-]chunk[_-]?\d+$",
        r"(.+?)[_-]seg(?:ment)?[_-]?\d+$",
        r"(.+?)[_-]part[_-]?\d+$",
        r"(.+?)[_-]clip[_-]?\d+$",
        r"(.+?)[_-]\d{1,4}$",
    ]
    for pat in patterns:
        m = re.match(pat, s, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return s


def extract_chunk_index(stem: str) -> int:
    matches = re.findall(r"(\d+)", stem)
    if not matches:
        return 10**9
    return int(matches[-1])


def build_video_groups(video_dir: Path, qid_to_qs: Dict[str, List[Dict]], vid_to_qs: Dict[str, List[Dict]]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = defaultdict(list)

    for path in sorted(video_dir.glob("*.mp4")):
        stem = path.stem.strip()
        exact_key = stem if (stem in qid_to_qs or stem in vid_to_qs) else None
        base = strip_chunk_suffix(stem)
        base_key = base if (base in qid_to_qs or base in vid_to_qs) else None

        key = exact_key or base_key
        if key is None:
            log.warning(f"ID '{stem}' not found as exact id or chunked id. Skipping.")
            continue

        groups[key].append(path)

    for key in groups:
        groups[key] = sorted(groups[key], key=lambda p: (extract_chunk_index(p.stem), p.name))

    return dict(groups)


def get_qrecs_for_group(group_id: str, qid_to_qs: Dict[str, List[Dict]], vid_to_qs: Dict[str, List[Dict]]) -> List[Dict]:
    return qid_to_qs.get(group_id) or vid_to_qs.get(group_id) or []


def clip_weight(path: Path) -> float:
    if WEIGHT_POWER == 0:
        return 1.0
    try:
        size = max(path.stat().st_size, 1)
    except OSError:
        size = 1
    return float(size) ** float(WEIGHT_POWER)


def aggregate_predictions(clip_records: List[Dict], mode: str = AGG_MODE) -> Tuple[Optional[str], Dict[str, Any]]:
    valid = [r for r in clip_records if r.get("pred") in {"A", "B", "C", "D"}]

    info: Dict[str, Any] = {
        "agg_mode": mode,
        "num_clips": len(clip_records),
        "num_valid_preds": len(valid),
        "scores": {},
        "top1_pred": None,
    }

    if not valid:
        return None, info

    top1_pred = valid[0]["pred"]
    info["top1_pred"] = top1_pred

    if mode == "top1":
        return top1_pred, info

    if mode == "vote":
        counts = Counter(r["pred"] for r in valid)
        winner, _ = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        info["scores"] = dict(counts)
        return winner, info

    scores = {k: 0.0 for k in "ABCD"}
    for r in valid:
        scores[r["pred"]] += float(r.get("weight", 1.0))

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    winner, winner_score = ranked[0]
    runner_up, runner_score = ranked[1]

    info["scores"] = scores
    info["winner_score"] = winner_score
    info["runner_up"] = runner_up
    info["runner_up_score"] = runner_score

    if mode == "weighted":
        return winner, info

    if mode == "fallback":
        total_score = sum(scores.values()) or 1.0
        margin = (winner_score - runner_score) / total_score
        info["margin"] = margin
        info["fallback_margin"] = FALLBACK_MARGIN
        if margin < FALLBACK_MARGIN:
            info["used_fallback_to_top1"] = True
            return top1_pred, info
        info["used_fallback_to_top1"] = False
        return winner, info

    raise ValueError(f"Unknown AGG_MODE: {mode}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    out_dir = Path(DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_jsonl = out_dir / "predictions.jsonl"
    clip_jsonl = out_dir / "clip_predictions.jsonl"
    fail_jsonl = out_dir / "failed.jsonl"
    metrics_path = out_dir / "metrics.json"

    clear_old_outputs(pred_jsonl, clip_jsonl, fail_jsonl, metrics_path)

    video_dir = Path(DEFAULT_VIDEO_DIR)
    if not video_dir.exists():
        log.error(f"Video directory not found: {video_dir}")
        return

    q_path = Path(DEFAULT_QUESTIONS)
    if not q_path.exists():
        log.error(f"Questions file not found: {q_path}")
        return

    all_questions = load_json(str(q_path))

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
        if "question_id" in x and "correct_answer_key" in x
    }

    llama_root = "/scratch/monroy/Emotion-LLaMA"
    if llama_root not in sys.path:
        sys.path.insert(0, llama_root)

    log.info("Loading model...")
    chat, device = load_official_model()
    log.info(f"Model ready on {device}")

    video_groups = build_video_groups(video_dir, qid_to_qs, vid_to_qs)
    num_video_files = sum(len(v) for v in video_groups.values())

    log.info(f"Found {num_video_files} matched video files in {len(video_groups)} groups")
    log.info(f"Aggregation mode: {AGG_MODE}")
    log.info(f"Shuffle options: {SHUFFLE_OPTIONS}")
    log.info(f"Text fallback enabled: {USE_TEXT_OPTION_FALLBACK}")

    total = 0
    correct = 0
    pred_none = 0
    failed = 0
    total_clip_inferences = 0

    display_letter_counts = Counter()
    original_pred_counts = Counter()
    raw_one_letter_count = 0
    fallback_accepted_count = 0

    for group_id, clip_paths in video_groups.items():
        qrecs = get_qrecs_for_group(group_id, qid_to_qs, vid_to_qs)

        for qrec in qrecs:
            qid = str(qrec.get("question_id", "")).strip()
            if not qid:
                continue

            cats = qrec.get("assigned_categories") or []
            if EMOTIONS_ONLY and "Emotions" not in cats:
                continue

            gt = gt_map.get(qid)
            if not gt:
                log.warning(f"No GT found for question_id: {qid}. Skipping.")
                continue

            prompt, display_options, display_to_original = build_mcq_prompt(qrec)

            clip_records: List[Dict] = []

            for clip_path in clip_paths:
                if not clip_path.exists() or clip_path.stat().st_size < MIN_CLIP_SIZE_BYTES:
                    write_jsonl(
                        fail_jsonl,
                        {"question_id": qid, "video_group_id": group_id, "clip": str(clip_path), "error": "missing_or_too_small"},
                    )
                    failed += 1
                    continue

                log.info(f"Group {group_id} | Question {qid} | Clip {clip_path.name}: Running inference")

                try:
                    raw = run_inference(chat, str(clip_path), prompt, device)

                    display_pred, extraction_info = extract_display_letter(raw, display_options=display_options)
                    original_pred = map_display_to_original(display_pred, display_to_original)

                    if raw.strip().lower() in {"a", "b", "c", "d"}:
                        raw_one_letter_count += 1

                    if display_pred:
                        display_letter_counts[display_pred] += 1
                    if original_pred:
                        original_pred_counts[original_pred] += 1
                    if extraction_info.get("fallback_accepted"):
                        fallback_accepted_count += 1

                    weight = clip_weight(clip_path)

                    clip_rec = {
                        "question_id": qid,
                        "video_group_id": group_id,
                        "clip_name": clip_path.name,
                        "clip_path": str(clip_path),
                        "clip_index": extract_chunk_index(clip_path.stem),
                        "display_pred": display_pred,
                        "pred": original_pred,
                        "weight": weight,
                        "display_options": display_options,
                        "display_to_original": display_to_original,
                        "extraction_info": extraction_info,
                        "raw_response": raw,
                    }

                    clip_records.append(clip_rec)
                    write_jsonl(clip_jsonl, clip_rec)
                    total_clip_inferences += 1

                except Exception as e:
                    failed += 1
                    log.exception(f"Error processing question {qid} clip {clip_path}")
                    write_jsonl(
                        fail_jsonl,
                        {"question_id": qid, "video_group_id": group_id, "clip": str(clip_path), "error": str(e)},
                    )

            final_pred, agg_info = aggregate_predictions(clip_records, AGG_MODE)
            is_correct = final_pred == gt

            rec = {
                "question_id": qid,
                "video_id": group_id,
                "gt": gt,
                "pred": final_pred,
                "correct": is_correct,
                "agg_mode": AGG_MODE,
                "num_clips": len(clip_records),
                "display_to_original": clip_records[0]["display_to_original"] if clip_records else {},
                "clip_preds": [
                    {
                        "clip_name": r["clip_name"],
                        "display_pred": r["display_pred"],
                        "pred": r["pred"],
                        "weight": r["weight"],
                        "extraction_info": r.get("extraction_info", {}),
                    }
                    for r in clip_records
                ],
                "agg_info": agg_info,
                "raw_response": "\n\n--- CLIP RESPONSE ---\n\n".join([r.get("raw_response", "") for r in clip_records]),
            }

            write_jsonl(pred_jsonl, rec)

            total += 1
            if is_correct:
                correct += 1
            if final_pred is None:
                pred_none += 1

    metrics = {
        "video_dir": str(video_dir),
        "questions_path": str(q_path),
        "gt_path": str(DEFAULT_GT),
        "out_dir": str(out_dir),
        "agg_mode": AGG_MODE,
        "shuffle_options": SHUFFLE_OPTIONS,
        "use_text_option_fallback": USE_TEXT_OPTION_FALLBACK,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "accuracy_percent": (correct / total * 100) if total > 0 else 0.0,
        "pred_none": pred_none,
        "failed": failed,
        "total_video_groups": len(video_groups),
        "total_clip_inferences": total_clip_inferences,
        "raw_one_letter_count": raw_one_letter_count,
        "fallback_accepted_count": fallback_accepted_count,
        "display_letter_counts": dict(display_letter_counts),
        "original_pred_counts": dict(original_pred_counts),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    log.info(f"Evaluation Complete. Accuracy: {metrics['accuracy']:.2%}")
    log.info(f"Null predictions: {pred_none}/{total}")
    log.info(f"Raw one-letter outputs: {raw_one_letter_count}/{total_clip_inferences}")
    log.info(f"Display letter counts: {dict(display_letter_counts)}")
    log.info(f"Original prediction counts after mapping: {dict(original_pred_counts)}")
    log.info(f"Predictions: {pred_jsonl}")
    log.info(f"Clip predictions: {clip_jsonl}")
    log.info(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
