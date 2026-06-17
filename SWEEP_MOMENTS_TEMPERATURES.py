#!/usr/bin/env python3
"""
Temperature sweep for MoMentS / Emotion-LLaMA evaluations.

This is intentionally a separate script from EVALUATE_MOMENTS*.py so existing
experiment outputs stay comparable. It loads the model once, evaluates a grid of
temperatures, repeats sampled runs, and writes aggregate accuracy statistics.

CUDA_VISIBLE_DEVICES=3 5 python Playground/SWEEP_MOMENTS_TEMPERATURES.py \
  --video-dir Playground/yolo_bbox_style/blue_t2_audio \
  --temperatures 0 0.05 0.1 0.2 0.3 \
  --repeats 3 \
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
from statistics import mean, pstdev
from typing import Dict, List, Optional

import numpy as np


DEFAULT_VIDEO_ROOT = "/scratch/monroy/Playground/yolo_bbox_style"
DEFAULT_OUT_ROOT = "/scratch/monroy/Playground/Experiments_YOLO_changeBB/temperature_sweep"
DEFAULT_QUESTIONS = "/scratch/monroy/Playground/datasets/MoMentS/data/moments_questions_updated.json"
DEFAULT_GT = "/scratch/monroy/Playground/datasets/MoMentS/data/validation/moments_validation_keys.json"
DEFAULT_LLAMA_ROOT = "/scratch/monroy/Emotion-LLaMA"
DEFAULT_CFG_PATH = "eval_configs/demo.yaml"

EMOTIONS_ONLY = True
MIN_CLIP_SIZE_BYTES = 1024


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep decoding temperature for Emotion-LLaMA MoMentS evals."
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Evaluate one video directory. If omitted, batch-discovers folders.",
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
        "--out-root",
        type=Path,
        default=Path(DEFAULT_OUT_ROOT),
        help="Root directory for sweep outputs.",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help=(
            "Resume an existing sweep directory. Completed runs with metrics.json "
            "are skipped; incomplete runs are restarted."
        ),
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.2, 0.3],
        help="Temperature grid to evaluate.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats per temperature. Temperature 0 is run once by default.",
    )
    parser.add_argument(
        "--repeat-greedy",
        action="store_true",
        help="Also repeat temperature 0 greedy runs.",
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
        help="Emotion-LLaMA eval config path.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Visible CUDA GPU id. With CUDA_VISIBLE_DEVICES=5, usually keep this 0.",
    )
    parser.add_argument(
        "--physical-gpu",
        help="Set CUDA_VISIBLE_DEVICES before torch is imported.",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Evaluate all questions instead of only Emotions-tagged questions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing sweep directory.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed used before each run.",
    )
    return parser.parse_args()


def load_official_model(cfg_path: str = DEFAULT_CFG_PATH, gpu_id: int = 0):
    import torch

    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.conversation.conversation import Chat

    log.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    log.info(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    log.info(f"torch.cuda.device_count()={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see CUDA in this process.")

    if gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"--gpu-id {gpu_id} is out of range for visible CUDA devices "
            f"(count={torch.cuda.device_count()})."
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


def set_run_seed(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_inference(chat, video_path: str, question: str, temperature: float) -> str:
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


def extract_choice_letter(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()

    m = re.search(r"FINAL_ANSWER\s*:?\s*[\[\(]?\s*([A-D])\s*[\]\)]?", t, re.IGNORECASE)
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


def write_jsonl(path: Path, obj: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_mcq_prompt(q: Dict) -> str:
    question = (q.get("question") or "").strip()
    opts = q.get("options") or {}
    a, b, c, d = [opts.get(k, "").strip() for k in "ABCD"]
    return (
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

    if not args.video_root.exists():
        raise FileNotFoundError(f"Video root not found: {args.video_root}")

    return sorted(p for p in args.video_root.glob(args.pattern) if p.is_dir())


def validate_video_dirs(video_dirs: List[Path]):
    for video_dir in video_dirs:
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        if not video_dir.is_dir():
            raise NotADirectoryError(f"Video path is not a directory: {video_dir}")
        if not any(video_dir.glob("*.mp4")):
            raise RuntimeError(
                f"No .mp4 files found in {video_dir}. If you are running from "
                "/scratch/monroy/Playground, use --video-dir yolo_bbox_style/blue_t2_audio "
                "or pass the absolute path."
            )


def temp_label(temperature: float) -> str:
    return f"{temperature:.3g}".replace("-", "neg").replace(".", "p")


def clear_previous_outputs(out_dir: Path):
    for name in ["predictions.jsonl", "failed.jsonl", "metrics.json"]:
        path = out_dir / name
        if path.exists():
            path.unlink()


def apply_resume_config(args: argparse.Namespace):
    if not args.resume_dir:
        return None

    config_path = args.resume_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Resume config not found: {config_path}")

    config = load_json(str(config_path))
    args.video_dir = None
    args.video_dirs_from_config = [resolve_resume_video_dir(Path(path)) for path in config["video_dirs"]]
    args.temperatures = config["temperatures"]
    args.repeats = config["repeats"]
    args.repeat_greedy = config["repeat_greedy"]
    args.base_seed = config["base_seed"]
    args.all_categories = not config.get("emotions_only", True)
    args.questions = config["questions"]
    args.gt = config["gt"]
    return config


def resolve_resume_video_dir(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path

    playground_relative = Path(DEFAULT_VIDEO_ROOT).parent / path
    if playground_relative.exists():
        return playground_relative

    return path


def evaluate_video_dir(
    video_dir: Path,
    out_dir: Path,
    chat,
    qid_to_qs: Dict[str, List[Dict]],
    vid_to_qs: Dict[str, List[Dict]],
    gt_map: Dict[str, str],
    emotions_only: bool,
    temperature: float,
    seed: int,
) -> Dict:
    set_run_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_previous_outputs(out_dir)

    pred_jsonl = out_dir / "predictions.jsonl"
    fail_jsonl = out_dir / "failed.jsonl"

    total = correct = pred_none = failed = skipped = 0
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

            prompt = build_mcq_prompt(qrec)
            log.info(
                f"temp={temperature:g} seed={seed} video={stem} question={qid}: inference"
            )

            try:
                raw = run_inference(chat, str(clip_path), prompt, temperature=temperature)
                pred = extract_choice_letter(raw)
                rec = {
                    "question_id": qid,
                    "video_id": stem,
                    "gt": gt,
                    "pred": pred,
                    "correct": pred == gt,
                    "temperature": temperature,
                    "seed": seed,
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
        "temperature": temperature,
        "seed": seed,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "pred_none": pred_none,
        "failed": failed,
        "skipped": skipped,
    }
    write_json(out_dir / "metrics.json", metrics)
    log.info(
        f"Completed {video_dir.name} temp={temperature:g} seed={seed}: "
        f"{metrics['accuracy']:.2%}"
    )
    return metrics


def summarize_runs(runs: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = {}
    for run in runs:
        grouped.setdefault((run["video_dir"], run["temperature"]), []).append(run)

    summary = []
    for (video_dir, temperature), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        accuracies = [row["accuracy"] for row in rows]
        corrects = [row["correct"] for row in rows]
        summary.append(
            {
                "video_dir": video_dir,
                "temperature": temperature,
                "runs": len(rows),
                "mean_accuracy": mean(accuracies),
                "std_accuracy": pstdev(accuracies) if len(accuracies) > 1 else 0.0,
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),
                "mean_correct": mean(corrects),
                "total": rows[0]["total"] if rows else 0,
                "pred_none_total": sum(row["pred_none"] for row in rows),
                "failed_total": sum(row["failed"] for row in rows),
                "out_dirs": [row["out_dir"] for row in rows],
            }
        )

    return summary


def main():
    args = parse_args()
    apply_resume_config(args)

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    if args.physical_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.physical_gpu

    if args.llama_root not in sys.path:
        sys.path.insert(0, args.llama_root)

    q_path = Path(args.questions)
    gt_path = Path(args.gt)
    if not q_path.exists():
        raise FileNotFoundError(f"Questions file not found: {q_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    video_dirs = getattr(args, "video_dirs_from_config", None) or discover_video_dirs(args)
    if not video_dirs:
        raise RuntimeError(f"No video directories found using pattern '{args.pattern}'")
    validate_video_dirs(video_dirs)

    all_questions = load_json(str(q_path))
    qid_to_qs, vid_to_qs = build_question_indexes(all_questions)
    gt_map = {
        str(x["question_id"]).strip(): str(x["correct_answer_key"]).upper()
        for x in load_json(str(gt_path))
        if "question_id" in x
    }

    if args.resume_dir:
        sweep_dir = args.resume_dir
        if not sweep_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {sweep_dir}")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = args.out_root / f"sweep_{stamp}"
        if sweep_dir.exists() and not args.overwrite:
            raise FileExistsError(f"Sweep directory already exists: {sweep_dir}")
        sweep_dir.mkdir(parents=True, exist_ok=True)

        write_json(
            sweep_dir / "config.json",
            {
                "video_dirs": [str(path) for path in video_dirs],
                "temperatures": args.temperatures,
                "repeats": args.repeats,
                "repeat_greedy": args.repeat_greedy,
                "base_seed": args.base_seed,
                "emotions_only": not args.all_categories,
                "questions": str(q_path),
                "gt": str(gt_path),
            },
        )

    log.info("Loading model...")
    chat, device = load_official_model(cfg_path=args.cfg_path, gpu_id=args.gpu_id)
    log.info(f"Model ready on {device}")

    all_runs = []
    for video_dir in video_dirs:
        base_name = video_dir.name[:-6] if video_dir.name.endswith("_audio") else video_dir.name
        for temperature in args.temperatures:
            run_count = args.repeats if args.repeat_greedy or temperature > 0 else 1
            for repeat_idx in range(run_count):
                seed = args.base_seed + repeat_idx
                out_dir = (
                    sweep_dir
                    / base_name
                    / f"temp_{temp_label(temperature)}"
                    / f"run_{repeat_idx:02d}_seed_{seed}"
                )
                metrics_path = out_dir / "metrics.json"
                if args.resume_dir and metrics_path.exists():
                    log.info(f"Skipping completed run: {out_dir}")
                    metrics = load_json(str(metrics_path))
                else:
                    if args.resume_dir and out_dir.exists():
                        log.info(f"Restarting incomplete run: {out_dir}")
                    metrics = evaluate_video_dir(
                        video_dir=video_dir,
                        out_dir=out_dir,
                        chat=chat,
                        qid_to_qs=qid_to_qs,
                        vid_to_qs=vid_to_qs,
                        gt_map=gt_map,
                        emotions_only=not args.all_categories,
                        temperature=temperature,
                        seed=seed,
                    )
                all_runs.append(metrics)
                write_json(sweep_dir / "runs.json", all_runs)
                write_json(sweep_dir / "temperature_summary.json", summarize_runs(all_runs))

    summary = summarize_runs(all_runs)
    write_json(sweep_dir / "runs.json", all_runs)
    write_json(sweep_dir / "temperature_summary.json", summary)
    log.info(f"Wrote sweep outputs under {sweep_dir}")


if __name__ == "__main__":
    main()
