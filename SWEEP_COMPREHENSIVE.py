#!/usr/bin/env python3
"""
Comprehensive MoMentS sweep: all 7 YOLO bbox styles + no-box baseline.

Default schedule
  t = 0.0  →  3 greedy repeats  (determinism check; all runs should be identical)
  t = 0.3  →  5 stochastic repeats  (reliable mean ± 95 % CI)

Usage
-----
  # Full run — all 8 conditions × 8 runs (56 eval passes × 68 questions each):
  CUDA_VISIBLE_DEVICES=5 conda run -n llama-lola \\
      python Experiments_YOLO_changeBB/SWEEP_COMPREHENSIVE.py

  # No-box baseline only (quick sanity run first):
  CUDA_VISIBLE_DEVICES=5 conda run -n llama-lola \\
      python Experiments_YOLO_changeBB/SWEEP_COMPREHENSIVE.py --no-box-only

  # All 7 styled conditions, skip no-box:
  CUDA_VISIBLE_DEVICES=5 conda run -n llama-lola \\
      python Experiments_YOLO_changeBB/SWEEP_COMPREHENSIVE.py --styles-only

  # Custom temperatures / repeats:
  CUDA_VISIBLE_DEVICES=5 conda run -n llama-lola \\
      python Experiments_YOLO_changeBB/SWEEP_COMPREHENSIVE.py \\
      --temperatures 0 0.3 --greedy-repeats 3 --stochastic-repeats 5
"""

import argparse
import json
import logging
import math
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_NO_BOX_DIR  = "/scratch/monroy/Playground/datasets/MoMentS_val_videos_emo"
# DEFAULT_STYLES_ROOT = "/scratch/monroy/Playground/yolo_bbox_style"
DEFAULT_STYLES_ROOT = "/scratch/monroy/Playground/datasets/YOLO_datasets/yolo_experiments_datasets"
DEFAULT_OUT_ROOT    = "/scratch/monroy/Playground/Experiments_YOLO_sweep"
DEFAULT_QUESTIONS   = "/scratch/monroy/Playground/datasets/MoMentS/data/moments_questions_updated.json"
DEFAULT_GT          = "/scratch/monroy/Playground/datasets/MoMentS/data/validation/moments_validation_keys.json"
DEFAULT_LLAMA_ROOT  = "/scratch/monroy/Emotion-LLaMA"
DEFAULT_CFG_PATH    = "eval_configs/demo.yaml"

EMOTIONS_ONLY       = True
MIN_CLIP_SIZE_BYTES = 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Comprehensive temperature sweep: all styles + no-box baseline."
    )
    p.add_argument("--no-box-dir", default=DEFAULT_NO_BOX_DIR,
                   help="Directory of raw MoMentS videos (no YOLO overlay).")
    p.add_argument("--styles-root", default=DEFAULT_STYLES_ROOT,
                   help="Root containing the *_audio style folders.")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                   help="Parent directory; a sweep_<stamp> subdir is created here.")
    p.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.3],
                   help="Temperature grid. Default: 0.0 and 0.3.")
    p.add_argument("--greedy-repeats", type=int, default=3,
                   help="Repeats for t=0 (should all be identical — determinism check).")
    p.add_argument("--stochastic-repeats", type=int, default=5,
                   help="Repeats for t>0.")
    p.add_argument("--base-seed", type=int, default=42,
                   help="Seeds are base-seed, base-seed+1, … for each repeat.")
    p.add_argument("--no-box-only", action="store_true",
                   help="Only run the no-box baseline, skip styled conditions.")
    p.add_argument("--styles-only", action="store_true",
                   help="Only run styled conditions, skip no-box baseline.")
    p.add_argument("--questions", default=DEFAULT_QUESTIONS)
    p.add_argument("--gt", default=DEFAULT_GT)
    p.add_argument("--llama-root", default=DEFAULT_LLAMA_ROOT)
    p.add_argument("--cfg-path", default=DEFAULT_CFG_PATH)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--physical-gpu",
                   help="Set CUDA_VISIBLE_DEVICES before torch is imported.")
    p.add_argument("--all-categories", action="store_true",
                   help="Evaluate all question categories, not just Emotions.")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow re-using an existing sweep directory.")
    return p.parse_args()


# ── model loading & inference
def load_official_model(cfg_path: str = DEFAULT_CFG_PATH, gpu_id: int = 0):
    import torch
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.conversation.conversation import Chat

    log.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    log.info(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    log.info(f"torch.cuda.device_count()={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see CUDA. Run with CUDA_VISIBLE_DEVICES set.")
    if gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"--gpu-id {gpu_id} out of range for visible CUDA devices "
            f"(count={torch.cuda.device_count()})."
        )
    device = f"cuda:{gpu_id}"

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

def normalize_video_id(stem: str) -> str:
    suffixes = [
        "_faces_filtered_crop",
        "_faces_filtered",
        "_filtered_crop",
        "_filtered",
        "_crop",
    ]
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem

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


# ── helpers ───────────────────────────────────────────────────────────────────
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


def load_json(path):
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


def temp_label(temperature: float) -> str:
    return f"{temperature:.3g}".replace("-", "neg").replace(".", "p")


# ── condition discovery ───────────────────────────────────────────────────────
def collect_conditions(args) -> List[Tuple[str, Path]]:
    """Returns [(label, video_dir), ...] in evaluation order."""
    conditions: List[Tuple[str, Path]] = []

    if not args.styles_only:
        no_box = Path(args.no_box_dir)
        if not no_box.exists():
            log.warning(f"No-box directory not found, skipping: {no_box}")
        elif not any(no_box.glob("*.mp4")):
            log.warning(f"No .mp4 files in no-box directory, skipping: {no_box}")
        else:
            conditions.append(("no_box", no_box))

    if not args.no_box_only:
        styles_root = Path(args.styles_root)
        if not styles_root.exists():
            log.warning(f"Styles root not found, skipping: {styles_root}")
        else:
            # for d in sorted(styles_root.glob("*_audio")):
            #     if d.is_dir() and any(d.glob("*.mp4")):
            #         label = d.name[:-6]  # strip _audio suffix
            #         conditions.append((label, d))
            for d in sorted(styles_root.iterdir()):
                if d.is_dir() and any(d.glob("*.mp4")):
                    label = d.name[:-6] if d.name.endswith("_audio") else d.name
                    conditions.append((label, d))

    return conditions


# ── single-run evaluation ─────────────────────────────────────────────────────
def evaluate_one_run(
    label: str,
    video_dir: Path,
    out_dir: Path,
    chat,
    qid_to_qs: Dict,
    vid_to_qs: Dict,
    gt_map: Dict,
    emotions_only: bool,
    temperature: float,
    seed: int,
) -> Dict:
    set_run_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_jsonl = out_dir / "predictions.jsonl"
    fail_jsonl = out_dir / "failed.jsonl"
    for f in [pred_jsonl, fail_jsonl, out_dir / "metrics.json"]:
        if f.exists():
            f.unlink()

    total = correct = pred_none = failed = skipped = 0
    video_files = sorted(video_dir.glob("*.mp4"))
    log.info(f"[{label}] Found {len(video_files)} videos in {video_dir}")

    for clip_path in video_files:
        raw_stem = clip_path.stem.strip()
        stem = normalize_video_id(raw_stem)
        qrecs = qid_to_qs.get(stem) or vid_to_qs.get(stem)
        if not qrecs:
            skipped += 1
            log.warning(f"[{label}] ID '{stem}' not in questions JSON — skipping.")
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
                log.warning(f"[{label}] No GT for question_id {qid} — skipping.")
                continue

            if clip_path.stat().st_size < MIN_CLIP_SIZE_BYTES:
                failed += 1
                write_jsonl(fail_jsonl, {"question_id": qid, "video_id": stem,
                                          "error": "too_small"})
                continue

            prompt = build_mcq_prompt(qrec)
            log.info(
                f"[{label}] temp={temperature:g} seed={seed} "
                f"video={stem} qid={qid}: inference"
            )

            try:
                raw = run_inference(chat, str(clip_path), prompt, temperature=temperature)
                pred = extract_choice_letter(raw)
                rec = {
                    "question_id": qid,
                    "video_id": stem,
                    "file_stem": raw_stem,
                    "gt": gt,
                    "pred": pred,
                    "correct": pred == gt,
                    "condition": label,
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
                log.exception(f"[{label}] Error on question {qid} (video {stem})")
                write_jsonl(fail_jsonl, {"question_id": qid, "video_id": stem,
                                          "error": str(e)})

    metrics = {
        "condition": label,
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
        f"[{label}] temp={temperature:g} seed={seed} → "
        f"{correct}/{total} = {metrics['accuracy']:.2%}"
    )
    return metrics


# ── aggregation ───────────────────────────────────────────────────────────────
def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95 % CI for a proportion (used for single-run estimates)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# t critical values for small n (two-tailed 95 %)
_T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
           6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _t_ci(accuracies: List[float]) -> Tuple[float, float, float]:
    """Returns (mean, lower, upper) 95 % t-CI for multi-run accuracy."""
    n = len(accuracies)
    m = mean(accuracies)
    if n == 1:
        lo, hi = _wilson_ci(round(m * 68), 68)  # fall back to Wilson
        return m, lo, hi
    s = pstdev(accuracies) * math.sqrt(n / (n - 1))  # sample std
    t = _T_CRIT.get(n - 1, 1.96)
    half = t * s / math.sqrt(n)
    return m, max(0.0, m - half), min(1.0, m + half)


def summarize_runs(runs: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, float], List[Dict]] = {}
    for r in runs:
        key = (r["condition"], r["temperature"])
        grouped.setdefault(key, []).append(r)

    summary = []
    for (condition, temperature), rows in sorted(grouped.items()):
        accs = [r["accuracy"] for r in rows]
        corrs = [r["correct"] for r in rows]
        m, lo, hi = _t_ci(accs)
        summary.append({
            "condition": condition,
            "temperature": temperature,
            "runs": len(rows),
            "mean_accuracy": m,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "std_accuracy": pstdev(accs) if len(accs) > 1 else 0.0,
            "min_accuracy": min(accs),
            "max_accuracy": max(accs),
            "mean_correct": mean(corrs),
            "total": rows[0]["total"] if rows else 0,
            "pred_none_total": sum(r["pred_none"] for r in rows),
            "failed_total": sum(r["failed"] for r in rows),
            "out_dirs": [r["out_dir"] for r in rows],
        })
    return summary


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.no_box_only and args.styles_only:
        raise ValueError("--no-box-only and --styles-only are mutually exclusive.")

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

    conditions = collect_conditions(args)
    if not conditions:
        raise RuntimeError("No valid conditions found. Check --no-box-dir and --styles-root.")

    log.info(f"Conditions to evaluate ({len(conditions)}): {[c[0] for c in conditions]}")
    log.info(f"Temperatures: {args.temperatures}")
    log.info(f"Greedy repeats (t=0): {args.greedy_repeats}  |  "
             f"Stochastic repeats (t>0): {args.stochastic_repeats}")

    all_questions = load_json(str(q_path))
    qid_to_qs, vid_to_qs = build_question_indexes(all_questions)
    gt_map = {
        str(x["question_id"]).strip(): str(x["correct_answer_key"]).upper()
        for x in load_json(str(gt_path))
        if "question_id" in x
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root)
    sweep_dir = out_root / f"sweep_{stamp}"
    if sweep_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Sweep directory already exists: {sweep_dir}")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    total_runs = sum(
        args.greedy_repeats if t == 0 else args.stochastic_repeats
        for t in args.temperatures
    ) * len(conditions)
    log.info(f"Total eval passes planned: {total_runs}")

    write_json(sweep_dir / "config.json", {
        "conditions": [(label, str(path)) for label, path in conditions],
        "temperatures": args.temperatures,
        "greedy_repeats": args.greedy_repeats,
        "stochastic_repeats": args.stochastic_repeats,
        "base_seed": args.base_seed,
        "emotions_only": not args.all_categories,
        "questions": str(q_path),
        "gt": str(gt_path),
        "stamp": stamp,
    })

    log.info("Loading model …")
    import torch
    torch.manual_seed(args.base_seed)
    chat, device = load_official_model(cfg_path=args.cfg_path, gpu_id=args.gpu_id)
    log.info(f"Model ready on {device}")

    all_runs: List[Dict] = []
    for label, video_dir in conditions:
        for temperature in args.temperatures:
            n_repeats = (
                args.greedy_repeats if temperature == 0.0
                else args.stochastic_repeats
            )
            for i in range(n_repeats):
                seed = args.base_seed + i
                out_dir = (
                    sweep_dir
                    / label
                    / f"temp_{temp_label(temperature)}"
                    / f"run_{i:02d}_seed_{seed}"
                )
                metrics = evaluate_one_run(
                    label=label,
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
                # Incremental saves so a crash doesn't lose progress
                write_json(sweep_dir / "runs.json", all_runs)
                write_json(sweep_dir / "temperature_summary.json",
                           summarize_runs(all_runs))

    summary = summarize_runs(all_runs)
    write_json(sweep_dir / "runs.json", all_runs)
    write_json(sweep_dir / "temperature_summary.json", summary)
    log.info(f"Sweep complete. Results in {sweep_dir}")

    # Print a quick summary table to stdout
    print("\n" + "=" * 72)
    print(f"{'CONDITION':<12} {'TEMP':>6} {'RUNS':>5} {'MEAN%':>7} {'CI95':>18}")
    print("=" * 72)
    for row in summary:
        print(
            f"{row['condition']:<12} {row['temperature']:>6.2f} "
            f"{row['runs']:>5} {row['mean_accuracy']:>7.1%} "
            f"  [{row['ci95_lo']:.1%} – {row['ci95_hi']:.1%}]"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
