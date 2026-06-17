from pathlib import Path
import py_compile

"""
AUTHORS_eval_temperature.py
---------------------------
Authors-style MoMentS evaluation for Emotion-LLaMA, with:

- optional transcript injection
- configurable decoding temperature
- configurable seed
- sklearn accuracy / precision / recall / F1
- confusion matrix
- predicted-letter distribution
- ground-truth distribution

Run from the Emotion-LLaMA repo root so eval_configs/demo.yaml resolves:

cd /scratch/monroy/Emotion-LLaMA

Example: frames + audio + transcript, T=0.1

CUDA_VISIBLE_DEVICES=4 nice -n 15 taskset -c 0-6 \
python /scratch/monroy/Playground/AUTHORS_eval_temperature.py \
  --video-dir /scratch/monroy/Playground/datasets/MoMentS_val_videos_emo \
  --out-dir /scratch/monroy/Playground/Experiments_AUTHORS/frames_audio_trans_t01 \
  --transcripts-json /scratch/monroy/Playground/Experiments_Baseline/transcripts/transcripts_by_videoid.json \
  --temperature 0.1 \
  --seed 42 \
  --overwrite


python /scratch/monroy/Playground/AUTHORS_eval_temperature.py \
  --video-dir /scratch/monroy/Playground/Experiments_Baseline/audio_only/dataset_audio_only \
  --out-dir /scratch/monroy/Playground/Experiments_AUTHORS/audio_trans_t0 \
  --transcripts-json /scratch/monroy/Playground/Experiments_Baseline/transcripts/transcripts_by_videoid.json \
  --temperature 0 \
  --seed 42 \
  --overwrite

CUDA_VISIBLE_DEVICES=5 nice -n 15 taskset -c 0-6 \
python /scratch/monroy/Playground/AUTHORS_eval_temperature.py \
  --video-dir /scratch/monroy/Playground/Experiments_Baseline/audio_only/dataset_audio_only \
  --out-dir /scratch/monroy/Playground/Experiments_AUTHORS/audio_only_t0 \
  --temperature 0

====== YOLO ==========
CUDA_VISIBLE_DEVICES=5 nice -n 15 taskset -c 0-6 \
python /scratch/monroy/Playground/AUTHORS_eval_temperature.py \
  --video-dir /scratch/monroy/Playground/datasets/YOLO_datasets/yolov11mface_video2video_audio \
  --out-dir /scratch/monroy/Playground/Experiments_AUTHORS/Experiments_YOLO/yolov11mface_video2video_audio \
  --temperature 0
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================================
# DEFAULTS
# ============================================================================

DEFAULT_QUESTIONS = "/scratch/monroy/Playground/datasets/MoMentS/data/moments_questions_updated.json"
DEFAULT_GT = "/scratch/monroy/Playground/datasets/MoMentS/data/validation/moments_validation_keys.json"
DEFAULT_LLAMA_ROOT = "/scratch/monroy/Emotion-LLaMA"
DEFAULT_CFG_PATH = "eval_configs/demo.yaml"

DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.0

EMOTIONS_ONLY = True
MIN_CLIP_SIZE_BYTES = 1024

VALID_LETTERS = {"A", "B", "C", "D"}
FALLBACK_LETTER = "A"


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Authors-style Emotion-LLaMA evaluation on MoMentS with optional "
            "transcripts and configurable decoding temperature."
        )
    )

    parser.add_argument(
        "--video-dir",
        required=True,
        type=Path,
        help="Folder containing .mp4 clips to evaluate.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Folder where predictions.jsonl, failed.jsonl, and metrics.json are written.",
    )
    parser.add_argument(
        "--questions",
        default=DEFAULT_QUESTIONS,
        help="MoMentS questions JSON path.",
    )
    parser.add_argument(
        "--gt",
        default=DEFAULT_GT,
        help="MoMentS validation keys JSON path.",
    )
    parser.add_argument(
        "--llama-root",
        default=DEFAULT_LLAMA_ROOT,
        help="Emotion-LLaMA repo root to add to sys.path.",
    )
    parser.add_argument(
        "--cfg-path",
        default=DEFAULT_CFG_PATH,
        help="Emotion-LLaMA eval config path, relative to the current working directory.",
    )
    parser.add_argument(
        "--transcripts-json",
        default=None,
        help=(
            "Optional JSON mapping video_id -> transcript string. "
            "When provided, the transcript is injected into the prompt."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            "Decoding temperature. Use 0.0 for greedy deterministic decoding. "
            "Values > 0 enable sampling through chat.answer."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used for Python, NumPy, and Torch sampling.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "Visible CUDA GPU id. If CUDA_VISIBLE_DEVICES is set to one physical GPU, "
            "this should usually stay 0."
        ),
    )
    parser.add_argument(
        "--physical-gpu",
        default=None,
        help=(
            "Optional physical GPU id. If provided, the script sets CUDA_VISIBLE_DEVICES "
            "before loading torch/model code."
        ),
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Evaluate all questions instead of only assigned_categories containing Emotions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing predictions.jsonl, failed.jsonl, and metrics.json in out-dir.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help=(
            "Allow writing into an existing out-dir without deleting previous files. "
            "Not recommended for final runs."
        ),
    )

    return parser.parse_args()


# ============================================================================
# SEEDING
# ============================================================================

def set_run_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# LOADING HELPERS
# ============================================================================

def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_transcripts(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transcripts file not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(k).strip(): str(v).strip() for k, v in data.items()}


def build_question_indexes(all_questions: List[Dict]):
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


def write_jsonl(path: Path, obj: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_official_model(cfg_path: str, gpu_id: int = 0):
    """Load Emotion-LLaMA using its own MiniGPT-4 registry/config machinery."""
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.conversation.conversation import Chat

    log.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    log.info("torch.cuda.is_available()=%s", torch.cuda.is_available())
    log.info("torch.cuda.device_count()=%s", torch.cuda.device_count())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot see CUDA in this process. Check CUDA_VISIBLE_DEVICES "
            "and the active conda environment."
        )

    if gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"--gpu-id {gpu_id} is out of range for visible CUDA device count "
            f"{torch.cuda.device_count()}. If CUDA_VISIBLE_DEVICES contains one GPU, "
            "use --gpu-id 0."
        )

    device = f"cuda:{gpu_id}"

    class _Args:
        def __init__(self, cfg_path: str):
            self.cfg_path = cfg_path
            self.options = None

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
# PROMPTING / INFERENCE
# ============================================================================

def build_mcq_prompt(q: Dict, transcript: Optional[str] = None) -> str:
    question = (q.get("question") or "").strip()
    opts = q.get("options") or {}

    a, b, c, d = [opts.get(k, "").strip() for k in "ABCD"]

    transcript_block = ""
    if transcript is not None and transcript.strip():
        transcript_block = (
            "Transcript of the spoken dialogue in the video:\n"
            f"\"{transcript.strip()}\"\n\n"
        )

    return (
        f"{transcript_block}"
        f"{question}\n\n"
        f"Options:\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
        "Task: Analyze the video and choose the single best answer (A, B, C, or D).\n"
        "Instructions:\n"
        "1. IMPORTANT: First, provide a very brief one-sentence reason for EACH option (A, B, C, and D).\n"
        "2. Finally, output a new line exactly in this format: FINAL_ANSWER: [LETTER]\n"
    )


def run_inference(chat, video_path: str, question: str, temperature: float) -> str:
    """
    Run one Emotion-LLaMA inference.

    temperature <= 0.0:
        greedy decoding with do_sample=False

    temperature > 0.0:
        sampled decoding through chat.answer(...)
    """
    from minigpt4.conversation.conversation import Conversation, SeparatorStyle

    chat_state = Conversation(
        system="",
        roles=(r"<s>[INST] ", r" [/INST]"),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="",
    )

    full_prompt = f"<video><VideoHere></video> <feature><FeatureHere></feature> {question}"
    chat.ask(full_prompt, chat_state)

    img_list = [video_path]
    if img_list and not isinstance(img_list[0], torch.Tensor):
        chat.encode_img(img_list)

    if temperature <= 0.0:
        generation_dict = chat.answer_prepare(
            conv=chat_state,
            img_list=img_list,
            temperature=1.0,
            max_new_tokens=500,
            max_length=2000,
        )
        generation_dict["do_sample"] = False
        generation_dict.pop("top_p", None)
        generation_dict.pop("temperature", None)

        output_token = chat.model_generate(**generation_dict)[0]
        output_text = chat.model.llama_tokenizer.decode(
            output_token,
            skip_special_tokens=True,
        )
        output_text = output_text.split("###")[0]
        output_text = output_text.split("Assistant:")[-1].strip()
        chat_state.messages[-1][1] = output_text
        return output_text

    return chat.answer(
        conv=chat_state,
        img_list=img_list,
        temperature=temperature,
        max_new_tokens=500,
        max_length=2000,
    )[0]


# ============================================================================
# ANSWER EXTRACTION
# ============================================================================

_re_final = re.compile(
    r"FINAL_ANSWER\s*:?\s*[\[\(]?\s*([A-D])\s*[\]\)]?",
    re.IGNORECASE,
)
_re_phrase = re.compile(
    r"\b(?:answer(?:\s+is)?|option)\s*[:\s]\s*([A-D])\b",
    re.IGNORECASE,
)


def extract_letter(text: str) -> str:
    """
    Return one of A/B/C/D.

    This keeps the authors-style closed-label scoring idea: if the model does
    not produce a parseable answer, fall back to A rather than returning None.
    """
    if not text:
        return FALLBACK_LETTER

    t = text.strip()

    m = _re_final.search(t)
    if m:
        return m.group(1).upper()

    m = _re_phrase.search(t)
    if m:
        return m.group(1).upper()

    if t and t[0].upper() in VALID_LETTERS and (
        len(t) == 1 or t[1] in (".", ")", " ", ":")
    ):
        return t[0].upper()

    m = re.search(r"\b([A-D])\b", t[:200])
    if m:
        return m.group(1).upper()

    log.warning("Could not parse letter from response; falling back to %s", FALLBACK_LETTER)
    return FALLBACK_LETTER


# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

def prepare_out_dir(out_dir: Path, overwrite: bool, allow_existing: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_outputs = [
        out_dir / "predictions.jsonl",
        out_dir / "failed.jsonl",
        out_dir / "metrics.json",
    ]

    if overwrite:
        for p in existing_outputs:
            if p.exists():
                p.unlink()
        return

    if not allow_existing:
        existing_present = [p for p in existing_outputs if p.exists()]
        if existing_present:
            raise FileExistsError(
                "Output files already exist. Use --overwrite to delete them or "
                "--allow-existing to append/reuse. Existing files: "
                + ", ".join(str(p) for p in existing_present)
            )


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(
    video_dir: Path,
    out_dir: Path,
    chat,
    qid_to_qs: Dict[str, List[Dict]],
    vid_to_qs: Dict[str, List[Dict]],
    gt_map: Dict[str, str],
    transcripts: Dict[str, str],
    emotions_only: bool,
    temperature: float,
    seed: int,
) -> Dict:
    set_run_seed(seed)

    pred_jsonl = out_dir / "predictions.jsonl"
    fail_jsonl = out_dir / "failed.jsonl"

    targets_list: List[str] = []
    answers_list: List[str] = []

    pred_none = 0
    failed = 0
    skipped = 0
    transcript_hits = 0

    video_files = sorted(video_dir.glob("*.mp4"))
    log.info("Found %d videos in %s", len(video_files), video_dir)

    for clip_path in video_files:
        stem = clip_path.stem.strip()
        qrecs = qid_to_qs.get(stem) or vid_to_qs.get(stem)

        if not qrecs:
            skipped += 1
            log.warning("ID '%s' not found in questions JSON. Skipping.", stem)
            continue

        for qrec in qrecs:
            qid = str(qrec.get("question_id", "")).strip()
            if not qid:
                skipped += 1
                continue

            cats = qrec.get("assigned_categories") or []
            if emotions_only and "Emotions" not in cats:
                skipped += 1
                continue

            gt = gt_map.get(qid)
            if not gt:
                skipped += 1
                log.warning("No GT found for %s. Skipping.", qid)
                continue

            if not clip_path.exists() or clip_path.stat().st_size < MIN_CLIP_SIZE_BYTES:
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

            transcript = transcripts.get(stem, "")
            transcript_found = bool(transcript.strip())
            if transcript_found:
                transcript_hits += 1

            prompt = build_mcq_prompt(
                qrec,
                transcript=transcript if transcript_found else None,
            )

            log.info(
                "Video %s | Question %s | transcript_found=%s",
                stem,
                qid,
                transcript_found,
            )

            try:
                raw = run_inference(
                    chat=chat,
                    video_path=str(clip_path),
                    question=prompt,
                    temperature=temperature,
                )
                pred = extract_letter(raw)

                if pred not in VALID_LETTERS:
                    pred_none += 1
                    pred = FALLBACK_LETTER

                rec = {
                    "question_id": qid,
                    "video_id": stem,
                    "gt": gt,
                    "pred": pred,
                    "correct": pred == gt,
                    "temperature": temperature,
                    "seed": seed,
                    "transcript_used": transcript_found,
                    "transcript": transcript if transcript_found else "",
                    "raw_response": raw,
                }
                write_jsonl(pred_jsonl, rec)

                targets_list.append(gt)
                answers_list.append(pred)

            except Exception as exc:
                failed += 1
                log.exception("Error on %s (%s)", qid, stem)
                write_jsonl(
                    fail_jsonl,
                    {
                        "question_id": qid,
                        "video_id": stem,
                        "error": str(exc),
                    },
                )

    if not targets_list:
        raise RuntimeError("No predictions were produced; nothing to score.")

    labels = ["A", "B", "C", "D"]

    accuracy = accuracy_score(targets_list, answers_list)
    precision = precision_score(
        targets_list,
        answers_list,
        average="weighted",
        zero_division=0,
    )
    recall = recall_score(
        targets_list,
        answers_list,
        average="weighted",
        zero_division=0,
    )
    f1 = f1_score(
        targets_list,
        answers_list,
        average="weighted",
        zero_division=0,
    )
    cm = confusion_matrix(targets_list, answers_list, labels=labels)

    pred_dist = dict(Counter(answers_list))
    gt_dist = dict(Counter(targets_list))

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion matrix labels:", labels)
    print(cm)
    print("Predicted letter distribution:", pred_dist)
    print("Ground-truth letter distribution:", gt_dist)

    metrics = {
        "video_dir": str(video_dir),
        "out_dir": str(out_dir),
        "temperature": float(temperature),
        "seed": int(seed),
        "total": len(targets_list),
        "correct": int(sum(1 for g, p in zip(targets_list, answers_list) if g == p)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "pred_dist": pred_dist,
        "gt_dist": gt_dist,
        "pred_none": int(pred_none),
        "failed": int(failed),
        "skipped": int(skipped),
        "transcript_hits": int(transcript_hits),
    }

    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()

    if args.physical_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)

    set_run_seed(args.seed)

    q_path = Path(args.questions)
    gt_path = Path(args.gt)
    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)

    if not q_path.exists():
        raise FileNotFoundError(f"Questions file not found: {q_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    prepare_out_dir(
        out_dir=out_dir,
        overwrite=args.overwrite,
        allow_existing=args.allow_existing,
    )

    if args.llama_root not in sys.path:
        sys.path.insert(0, args.llama_root)

    all_questions = load_json(str(q_path))
    qid_to_qs, vid_to_qs = build_question_indexes(all_questions)

    gt_map = {
        str(x["question_id"]).strip(): str(x["correct_answer_key"]).upper()
        for x in load_json(str(gt_path))
        if "question_id" in x
    }

    transcripts = load_transcripts(args.transcripts_json)
    if args.transcripts_json:
        log.info("Loaded %d transcripts from %s", len(transcripts), args.transcripts_json)
    else:
        log.info("No transcript file provided.")

    log.info("Loading model...")
    chat, device = load_official_model(args.cfg_path, gpu_id=args.gpu_id)
    log.info("Model ready on %s", device)

    metrics = evaluate(
        video_dir=video_dir,
        out_dir=out_dir,
        chat=chat,
        qid_to_qs=qid_to_qs,
        vid_to_qs=vid_to_qs,
        gt_map=gt_map,
        transcripts=transcripts,
        emotions_only=not args.all_categories,
        temperature=args.temperature,
        seed=args.seed,
    )

    metrics["transcript_json"] = str(args.transcripts_json) if args.transcripts_json else None

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    log.info(
        "Done. %d/%d = %.2f%%",
        metrics["correct"],
        metrics["total"],
        100 * metrics["accuracy"],
    )


if __name__ == "__main__":
    main()


# out = Path("/mnt/data/AUTHORS_eval_temperature.py")
# out.write_text(script, encoding="utf-8")
# py_compile.compile(str(out), doraise=True)
# print(f"Created {out} ({out.stat().st_size} bytes)")
