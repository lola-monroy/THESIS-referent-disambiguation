# Copy form the MoMentS dataset
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


def _project_root() -> Path:
    """Assumes this file lives in the project root."""
    return Path(__file__).resolve().parent


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.realpath(path)
    return str((_project_root() / path).resolve())


def norm_video_path(video_path: str) -> str:
    """
    Normalize dataset video paths relative to <project_root>/videos.
    """
    p = (video_path or "").strip()
    if p.startswith("file://"):
        p = p[7:]
    p = os.path.expanduser(p)
    if os.path.isabs(p):
        return os.path.realpath(p)
    if p.startswith("videos/"):
        p = p[7:]
    return str((_project_root() / "videos" / p).resolve())


def load_gt(
    paths: Optional[str | List[str]], key: str = "correct_answer_key"
) -> Dict[str, str]:
    """
    Load one or more ground-truth files into {question_id: answer_letter}.
    Supports either:
    - list of answer entries
    - {"answers": [...]} wrapper
    """
    if not paths:
        return {}

    path_list = [paths] if isinstance(paths, str) else paths
    combined_gt: Dict[str, str] = {}

    for path in path_list:
        if not path:
            continue
        abs_path = resolve_path(path)
        obj = json.loads(Path(abs_path).read_text(encoding="utf-8"))

        gt: Dict[str, str] = {}
        if isinstance(obj, list):
            for item in obj:
                if "question_id" in item and key in item:
                    gt[str(item["question_id"])] = str(item[key]).upper()
        elif isinstance(obj, dict) and "answers" in obj:
            for item in obj["answers"]:
                if "question_id" in item and key in item:
                    gt[str(item["question_id"])] = str(item[key]).upper()

        combined_gt.update(gt)

    return combined_gt


def load_items(
    path: str,
    categories: Optional[List[str]] = None,
    multimodal_cues: Optional[List[str]] = None,
    max_items: Optional[int] = None,
    filter_available: bool = True,
    ground_truth: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load VQA items and apply optional filtering.

    Required keys per item:
    - question, options, video_path, t_i, t_j
    """
    abs_path = resolve_path(path)
    data = json.loads(Path(abs_path).read_text(encoding="utf-8"))

    required = {"question", "options", "video_path", "t_i", "t_j"}
    for it in data:
        missing = required - set(it)
        if missing:
            raise ValueError(f"Missing keys in item: {missing}")
        if set("ABCD") - set(it["options"].keys()):
            raise ValueError("Options must include A, B, C, and D.")

    if filter_available:
        data = [it for it in data if it.get("video_status_local") == "available"]

    if categories:
        selected = set(categories)
        data = [
            it
            for it in data
            if selected.intersection(set(it.get("assigned_categories", [])))
        ]

    if multimodal_cues:
        selected = set(multimodal_cues)
        data = [
            it
            for it in data
            if selected.intersection(set(it.get("multimodal_cues") or []))
        ]

    if ground_truth:
        data = [it for it in data if str(it.get("question_id", "")) in ground_truth]

    if max_items is not None:
        data = data[:max_items]

    return data


def create_video_segment(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: Optional[str] = None,
) -> str:
    if output_path is None:
        temp_fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="video_segment_")
        os.close(temp_fd)

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-ss",
            str(start_time),
            "-t",
            str(max(0.0, end_time - start_time)),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-avoid_negative_ts",
            "make_zero",
            output_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, check=False
        )
        if result.returncode == 0:
            return output_path

        fallback_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-ss",
            str(start_time),
            "-t",
            str(max(0.0, end_time - start_time)),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            output_path,
        ]
        fallback = subprocess.run(
            fallback_cmd, capture_output=True, text=True, timeout=300, check=False
        )
        return output_path if fallback.returncode == 0 else video_path
    except Exception:
        return video_path


def sample_frames_uniform_with_times(
    video_path: str, n: int = 1
) -> Tuple[List[np.ndarray], List[float], float]:
    p = norm_video_path(video_path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Video not found: {p}")

    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        return [], [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = (total / max(1.0, fps)) if total > 0 else 0.0

    if n <= 0:
        n = 1

    idxs = np.linspace(0, duration * fps, num=n, endpoint=False, dtype=int)
    frames: List[np.ndarray] = []
    times: List[float] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
            times.append(int(idx) / max(1.0, fps))
    cap.release()

    return frames, times, duration


def get_cached_video_segment(
    video_path: str,
    t_i: float,
    t_j: float,
    frame_budget: int = 64,
    full_context: bool = False,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    return_frames: bool = False,
) -> Tuple[str, Dict[str, Any], Optional[List[np.ndarray]], Optional[List[float]]]:
    """
    Return a cached segment entry. If not cached, create segment + sample frames.
    """
    video_path = norm_video_path(video_path)

    if cache_dir is None:
        cache_dir = str((_project_root() / "videos" / "cache").resolve())
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    key = (
        f"{Path(video_path).stem}_t{t_i:.3f}_{t_j:.3f}_frames{frame_budget}_"
        f"{'fullcontext' if full_context else 'segment'}"
    )
    cache_path = Path(cache_dir) / f"{key}.pt"

    metadata = {
        "video_path": video_path,
        "t_i": t_i,
        "t_j": t_j,
        "frame_budget": frame_budget,
        "full_context": full_context,
        "cache_path": str(cache_path),
        "created_at": datetime.now().isoformat(),
        "version": "1.0",
    }

    cache_key_fields = ["t_i", "t_j", "frame_budget", "full_context", "version"]
    if use_cache and cache_path.exists():
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            cached_metadata = cached.get("metadata", {})
            cached_frames = cached.get("frames", [])
            if cached_frames:
                matches = all(
                    cached_metadata.get(field) == metadata.get(field)
                    for field in cache_key_fields
                )
                if matches:
                    metadata["created_at"] = cached_metadata.get(
                        "created_at", metadata["created_at"]
                    )
                    return (
                        str(cache_path),
                        metadata,
                        cached_frames if return_frames else None,
                        cached.get("frame_times") if return_frames else None,
                    )
        except Exception:
            pass

    video_segment_path = video_path
    temp_segment_created = False
    if t_j > t_i:
        video_segment_path = create_video_segment(video_path, t_i, t_j)
        temp_segment_created = video_segment_path != video_path

    try:
        frames, frame_times, _ = sample_frames_uniform_with_times(
            video_segment_path, frame_budget
        )
    finally:
        if (
            temp_segment_created
            and video_segment_path != video_path
            and os.path.exists(video_segment_path)
        ):
            try:
                os.unlink(video_segment_path)
            except Exception as exc:
                print(
                    f"Warning: failed to cleanup temp segment {video_segment_path}: {exc}",
                    file=sys.stderr,
                )

    if not frames:
        raise RuntimeError(
            f"No frames extracted from {video_path} (t_i={t_i}, t_j={t_j})."
        )

    if use_cache:
        try:
            torch.save(
                {"frames": frames, "frame_times": frame_times, "metadata": metadata},
                cache_path,
            )
        except Exception as exc:
            print(f"Warning: failed to save cache {cache_path}: {exc}", file=sys.stderr)

    return (
        str(cache_path),
        metadata,
        frames if return_frames else None,
        frame_times if return_frames else None,
    )


def load_dataset_bundle(
    *,
    items_path: str,
    gt_paths: Optional[str | List[str]] = None,
    tokens_gt_path: Optional[str] = None,
    categories: Optional[List[str]] = None,
    multimodal_cues: Optional[List[str]] = None,
    max_items: Optional[int] = None,
    filter_available: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Convenience loader:
    1) load ground truth
    2) load/filter items
    """
    gt = load_gt(gt_paths)
    items = load_items(
        path=items_path,
        categories=categories,
        multimodal_cues=multimodal_cues,
        max_items=max_items,
        filter_available=filter_available,
        ground_truth=gt if gt else None,
    )

    return items, gt
