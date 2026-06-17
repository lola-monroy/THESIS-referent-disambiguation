#!/usr/bin/env python3
"""
BEING USED : 14/06 AND 15/06 
Text-only evaluation for Emotion-LLaMA on MoMentS.
The model NEVER sees video or audio — inference is driven by transcript + question text only.
The --video-dir is used only to discover question IDs (mp4 filenames); clips are not decoded.

NOTE: run from /scratch/monroy/Emotion-LLaMA so that eval_configs/demo.yaml resolves correctly.

SCRIPT PATH:
  /scratch/monroy/Playground/Experiments_Baseline_RERUN/python_executables/EVALUATE_Transcripts.py

TRANSCRIPTS:
  /scratch/monroy/Playground/Experiments_Baseline/transcripts/transcripts_by_videoid.json
  Keys are video IDs (mp4 stem). ~68 entries covering the MoMentS val set.

===========================================================================
COMBINATION 1 — Transcript only, audio question set
  Input: audio-only clips (IDs only) + transcript injected into prompt
  Output: Experiments_Baseline_RERUN/transcript_only
===========================================================================

cd /scratch/monroy/Emotion-LLaMA && \
CUDA_VISIBLE_DEVICES=<GPU> nice -n 15 taskset -c 0-6 \
  python /scratch/monroy/Playground/Experiments_Baseline_RERUN/python_executables/EVALUATE_Transcripts.py \
  --video-dir /scratch/monroy/Playground/Experiments_Baseline/audio_only/dataset_audio_only \
  --out-dir /scratch/monroy/Playground/Experiments_Baseline_RERUN/transcript_only \
  --transcripts /scratch/monroy/Playground/Experiments_Baseline/transcripts/transcripts_by_videoid.json \
  --temperature 0.1 --overwrite

===========================================================================
COMBINATION 2 — Transcript only, video question set
  Input: video-only clips (IDs only) + transcript injected into prompt
  Output: Experiments_Baseline_RERUN/video_only_transcript
===========================================================================

cd /scratch/monroy/Emotion-LLaMA && \
CUDA_VISIBLE_DEVICES=<GPU> nice -n 15 taskset -c 0-6 \
  python /scratch/monroy/Playground/Experiments_Baseline_RERUN/python_executables/EVALUATE_Transcripts.py \
  --video-dir /scratch/monroy/Playground/Experiments_Baseline/video_only/dataset_video_only \
  --out-dir /scratch/monroy/Playground/Experiments_Baseline_RERUN/video_only_transcript \
  --transcripts /scratch/monroy/Playground/Experiments_Baseline/transcripts/transcripts_by_videoid.json \
  --temperature 0.1 --overwrite

===========================================================================
COMBINATION 3 — LLM only (no transcript), audio question set
  Input: audio-only clips (IDs only), no transcript — model sees question + options only
  Output: Experiments_Baseline_RERUN/llm_only_audio
===========================================================================

cd /scratch/monroy/Emotion-LLaMA && \
CUDA_VISIBLE_DEVICES=<GPU> nice -n 15 taskset -c 0-6 \
  python /scratch/monroy/Playground/Experiments_Baseline_RERUN/python_executables/EVALUATE_Transcripts.py \
  --video-dir /scratch/monroy/Playground/Experiments_Baseline/audio_only/dataset_audio_only \
  --out-dir /scratch/monroy/Playground/Experiments_Baseline_RERUN/llm_only_audio \
  --no-transcript \
  --temperature 0.1

===========================================================================
COMBINATION 4 — LLM only (no transcript), video question set
  Input: video-only clips (IDs only), no transcript — model sees question + options only
  Output: Experiments_Baseline_RERUN/llm_only_video
===========================================================================

cd /scratch/monroy/Emotion-LLaMA && \
CUDA_VISIBLE_DEVICES=<GPU> nice -n 15 taskset -c 0-6 \
  python /scratch/monroy/Playground/Experiments_Baseline_RERUN/python_executables/EVALUATE_Transcripts.py \
  --video-dir /scratch/monroy/Playground/Experiments_Baseline/video_only/dataset_video_only \
  --out-dir /scratch/monroy/Playground/Experiments_Baseline_RERUN/llm_only_video \
  --no-transcript \
  --temperature 0.1


# -- only transcript 
CUDA_VISIBLE_DEVICES=4 nice -n 15 taskset -c 0-6 \
python /scratch/monroy/Playground/EVALUATE_Transcripts.py \
  --video-dir /scratch/monroy/Playground/Experiments_Baseline/audio_only/dataset_audio_only \
  --out-dir /scratch/monroy/Playground/Experiments_AUTHORS/transcript_only \
  --transcripts /scratch/monroy/Playground/Experiments_Baseline/transcripts/transcripts_by_videoid.json \
  --temperature 0 \
  --seed 42 \
  --overwrite

# all 
CUDA_VISIBLE_DEVICES=4 nice -n 15 taskset -c 0-6 \
python /scratch/monroy/Playground/EVALUATE_Transcripts.py \
  --video-dir /scratch/monroy/Playground/datasets/MoMentS_val_videos_emo \
  --out-dir /scratch/monroy/Playground/Experiments_AUTHORS/frames_audio_trans \
  --transcripts /scratch/monroy/Playground/Experiments_Baseline/transcripts/transcripts_by_videoid.json \
  --temperature 0.1 \
  --seed 42 \
  --overwrite
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


DEFAULT_VIDEO_ROOT = "/scratch/monroy/Playground/yolo_bbox_style"
DEFAULT_OUT_ROOT = "/scratch/monroy/Playground/Experiments_YOLO_changeBB/temperature_03"
DEFAULT_QUESTIONS = "/scratch/monroy/Playground/datasets/MoMentS/data/moments_questions_updated.json"
DEFAULT_GT = "/scratch/monroy/Playground/datasets/MoMentS/data/validation/moments_validation_keys.json"
DEFAULT_LLAMA_ROOT = "/scratch/monroy/Emotion-LLaMA"
DEFAULT_CFG_PATH = "eval_configs/demo.yaml"
DEFAULT_TRANSCRIPTS = "/scratch/monroy/Playground/Experiments_Baseline/transcripts/transcripts_by_videoid.json"

EMOTIONS_ONLY = True
MIN_CLIP_SIZE_BYTES = 1024


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Emotion-LLaMA on MoMentS with transcript injection."
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Evaluate one video directory instead of batch-discovering *_audio folders.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path(DEFAULT_VIDEO_ROOT),
        help="Root used to discover batch folders.",
    )
    parser.add_argument(
        "--pattern",
        default="*_audio",
        help="Folder name pattern used with --video-root in batch mode.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help=(
            "Folder names or stems to skip in batch mode, e.g. "
            "blue_t2_audio or blue_t2."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(DEFAULT_OUT_ROOT),
        help="Root directory where result folders are created.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for --video-dir. Not allowed in batch mode.",
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
        "--transcripts",
        default=DEFAULT_TRANSCRIPTS,
        help="Path to transcripts_by_videoid.json.",
    )
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help="Disable transcript injection (run as plain EVALUATE_MOMENTS3.py).",
    )
    parser.add_argument(
        "--use-video",
        action="store_true",
        help="Decode and feed each clip to the model (video + transcript). "
             "Default is text-only inference.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Frames sampled evenly per clip and mean-pooled into the visual slot "
             "when --use-video is set. Default: 16.",
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
        "--gpu-id",
        type=int,
        default=0,
        help="Visible CUDA GPU id to use. With CUDA_VISIBLE_DEVICES=5, this should usually stay 0.",
    )
    parser.add_argument(
        "--physical-gpu",
        help="Set CUDA_VISIBLE_DEVICES inside this script before torch is imported.",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Evaluate all questions instead of only assigned_categories containing Emotions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Use stable output names instead of timestamped names.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used for Python, NumPy, and Torch sampling. Default: 42.",
    )
    return parser.parse_args()


def set_run_seed(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_official_model(cfg_path: str = DEFAULT_CFG_PATH, gpu_id: int = 0):
    """Load Emotion-LLaMA model exactly like cli_inference."""
    import torch

    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.conversation.conversation import Chat

    log.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    log.info(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    log.info(f"torch.cuda.device_count()={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot see CUDA in this process. If nvidia-smi works in your shell, "
            "run with the same environment, for example: "
            "CUDA_VISIBLE_DEVICES=<GPU_ID> conda run -n llama-lola python EVALUATE_MOMENTS3_T.py"
        )

    if gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"--gpu-id {gpu_id} is out of range for the visible CUDA devices "
            f"(count={torch.cuda.device_count()}). If you set CUDA_VISIBLE_DEVICES to one "
            "physical GPU, use --gpu-id 0."
        )

    device = f"cuda:{gpu_id}"

    class _Args:
        def __init__(self, cfg_path: str):
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


def run_inference(chat, video_path: str, question: str, temperature: float) -> str:
    """Run clip inference mirrors cli_inference.py."""
    import torch

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

    if temperature <= 0:
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
        output_text = chat.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
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


def run_inference_video(chat, video_path: str, question: str, temperature: float, num_frames: int) -> str:
    """Whole-clip inference: mean-pool num_frames sampled frames into the visual slot.

    Mirrors run_inference but swaps chat.encode_img (single first frame) for
    chat.encode_video_meanpool (num_frames evenly sampled, mean-pooled).
    """
    import torch

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
        chat.encode_video_meanpool(img_list, num_frames=num_frames)

    if temperature <= 0:
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
        output_text = chat.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
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


def run_inference_text_only(chat, question: str, device: str, temperature: float = 0.2, max_new_tokens: int = 500) -> str:
    """Run inference using only text (no video/audio). Drives the LLaMA backbone directly."""
    import torch
    from transformers import StoppingCriteriaList
    from minigpt4.conversation.conversation import StoppingCriteriaSub

    prompt = f"<s>[INST] {question} [/INST]"
    input_ids = chat.model.llama_tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    input_embeds = chat.model.embed_tokens(input_ids)

    stop_ids = [torch.tensor([2]).to(device)]
    stopping = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_ids)])

    with chat.model.maybe_autocast():
        output = chat.model.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=float(temperature) if temperature > 0 else 1.0,
            top_p=0.9,
            repetition_penalty=1.05,
            stopping_criteria=stopping,
        )

    text = chat.model.llama_tokenizer.decode(output[0], skip_special_tokens=True)
    text = text.split("###")[0].split("Assistant:")[-1].strip()
    return text


def extract_choice_letter(text: str) -> Optional[str]:
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


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_transcripts(path: str) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        log.warning(f"Transcripts file not found: {p}. Proceeding without transcripts.")
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(k).strip(): str(v).strip() for k, v in data.items()}


def write_jsonl(path: Path, obj: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_mcq_prompt(q: Dict, transcript: Optional[str] = None) -> str:
    question = (q.get("question") or "").strip()
    opts = q.get("options") or {}
    a, b, c, d = [opts.get(k, "").strip() for k in "ABCD"]

    transcript_block = ""
    if transcript and transcript.strip():
        transcript_block = f"Transcript (may be noisy):\n{transcript}\n\n"

    return (
        f"{transcript_block}"
        f"{question}\n\n"
        f"Options:\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
        "Task: Analyze the video and choose the single best answer (A, B, C, or D).\n"
        "Instructions:\n"
        "1. IMPORTANT: First, provide a very brief one-sentence reason for EACH option (A, B, C, and D).\n"
        "2. Finally, output a new line exactly in this format: FINAL_ANSWER: [LETTER]\n"
    )


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


def discover_video_dirs(args: argparse.Namespace) -> List[Path]:
    if args.video_dir:
        return [args.video_dir]

    if args.out_dir:
        raise ValueError("--out-dir can only be used together with --video-dir")

    if not args.video_root.exists():
        raise FileNotFoundError(f"Video root not found: {args.video_root}")

    excluded = set(args.exclude)
    excluded.update(name[:-6] for name in args.exclude if name.endswith("_audio"))
    excluded.update(f"{name}_audio" for name in args.exclude if not name.endswith("_audio"))
    return sorted(
        p
        for p in args.video_root.glob(args.pattern)
        if p.is_dir() and p.name not in excluded
    )


def make_out_dir(video_dir: Path, args: argparse.Namespace, stamp: str) -> Path:
    if args.out_dir:
        out_dir = args.out_dir
    else:
        name = video_dir.name[:-6] if video_dir.name.endswith("_audio") else video_dir.name
        suffix = "moments1_eval_t"
        if args.no_timestamp:
            out_dir = args.out_root / f"{name}_{suffix}"
        else:
            out_dir = args.out_root / f"{name}_{suffix}_{stamp}"

    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def clear_previous_outputs(out_dir: Path):
    for name in ["predictions.jsonl", "failed.jsonl", "metrics.json"]:
        path = out_dir / name
        if path.exists():
            path.unlink()


def evaluate_video_dir(
    video_dir: Path,
    out_dir: Path,
    chat,
    qid_to_qs: Dict[str, List[Dict]],
    vid_to_qs: Dict[str, List[Dict]],
    gt_map: Dict[str, str],
    transcript_map: Dict[str, str],
    use_transcript: bool,
    use_video: bool,
    num_frames: int,
    emotions_only: bool,
    temperature: float,
    seed: int,
) -> Dict:
    set_run_seed(seed)

    pred_jsonl = out_dir / "predictions.jsonl"
    fail_jsonl = out_dir / "failed.jsonl"

    total = correct = pred_none = failed = skipped = transcript_hits = 0
    video_files = sorted(video_dir.glob("*.mp4"))
    log.info(f"Found {len(video_files)} videos in {video_dir}")

    for clip_path in video_files:
        stem = clip_path.stem.strip()
        qrecs = qid_to_qs.get(stem) or vid_to_qs.get(stem)

        if not qrecs:
            skipped += 1
            log.warning(f"ID '{stem}' not found in questions JSON. Skipping.")
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
                log.warning(f"No GT found for question_id: {qid}. Skipping.")
                continue

            if not clip_path.exists() or clip_path.stat().st_size < MIN_CLIP_SIZE_BYTES:
                failed += 1
                write_jsonl(
                    fail_jsonl,
                    {"question_id": qid, "video_id": stem, "error": "missing_or_too_small"},
                )
                continue

            # transcript keys are video IDs (stem of the mp4 filename)
            transcript = transcript_map.get(stem, "") if use_transcript else ""
            transcript_found = bool(transcript.strip())
            if transcript_found:
                transcript_hits += 1

            prompt = build_mcq_prompt(qrec, transcript=transcript if use_transcript else None)
            log.info(
                f"Video {stem} | Question {qid} | transcript_found={transcript_found}: Running inference"
            )

            try:
                if use_video:
                    raw = run_inference_video(
                        chat, str(clip_path), prompt,
                        temperature=temperature, num_frames=num_frames,
                    )
                else:
                    raw = run_inference_text_only(chat, prompt, chat.device)
                pred = extract_choice_letter(raw)
                rec = {
                    "question_id": qid,
                    "video_id": stem,
                    "gt": gt,
                    "pred": pred,
                    "correct": pred == gt,
                    "temperature": temperature,
                    "seed": seed,
                    "transcript_used": transcript_found,
                    "transcript": transcript,
                    "raw_response": raw,
                }
                write_jsonl(pred_jsonl, rec)

                total += 1
                correct += int(pred == gt)
                pred_none += int(pred is None)
            except Exception as e:
                failed += 1
                log.exception(f"Error processing question {qid} (video {stem})")
                write_jsonl(fail_jsonl, {"question_id": qid, "video_id": stem, "error": str(e)})

    metrics = {
        "video_dir": str(video_dir),
        "out_dir": str(out_dir),
        "use_video": use_video,
        "num_frames": num_frames if use_video else None,
        "temperature": temperature,
        "seed": seed,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "pred_none": pred_none,
        "failed": failed,
        "skipped": skipped,
        "transcript_hits": transcript_hits,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Evaluation complete for {video_dir.name}. Accuracy: {metrics['accuracy']:.2%}")
    return metrics


def main():
    args = parse_args()

    if args.physical_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.physical_gpu

    set_run_seed(args.seed)

    if args.llama_root not in sys.path:
        sys.path.insert(0, args.llama_root)

    q_path = Path(args.questions)
    gt_path = Path(args.gt)
    if not q_path.exists():
        raise FileNotFoundError(f"Questions file not found: {q_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    use_transcript = not args.no_transcript
    transcript_map = load_transcripts(args.transcripts) if use_transcript else {}
    log.info(f"Transcript injection: {use_transcript} | entries loaded: {len(transcript_map)}")

    video_dirs = discover_video_dirs(args)
    if not video_dirs:
        raise RuntimeError(f"No video directories found using pattern '{args.pattern}'")

    all_questions = load_json(str(q_path))
    qid_to_qs, vid_to_qs = build_question_indexes(all_questions)
    gt_map = {
        str(x["question_id"]).strip(): str(x["correct_answer_key"]).upper()
        for x in load_json(str(gt_path))
        if "question_id" in x
    }

    log.info("Loading model...")
    chat, device = load_official_model(cfg_path=args.cfg_path, gpu_id=args.gpu_id)
    log.info(f"Model ready on {device}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_metrics = []
    for video_dir in video_dirs:
        out_dir = make_out_dir(video_dir, args, stamp)
        if args.overwrite:
            clear_previous_outputs(out_dir)
        log.info(f"Evaluating {video_dir} -> {out_dir}")
        metrics = evaluate_video_dir(
            video_dir=video_dir,
            out_dir=out_dir,
            chat=chat,
            qid_to_qs=qid_to_qs,
            vid_to_qs=vid_to_qs,
            gt_map=gt_map,
            transcript_map=transcript_map,
            use_transcript=use_transcript,
            use_video=args.use_video,
            num_frames=args.num_frames,
            emotions_only=not args.all_categories,
            temperature=args.temperature,
            seed=args.seed,
        )
        all_metrics.append(metrics)

    if len(all_metrics) > 1:
        summary_path = args.out_root / f"moments1_batch_summary_t_{stamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        log.info(f"Wrote batch summary: {summary_path}")


if __name__ == "__main__":
    main()
