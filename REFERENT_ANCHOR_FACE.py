#!/usr/bin/env python3
"""
------------
1) Detects all people in all frames using YOLO.
2) Chooses some candidate frames to find a "good" anchor frame.
3) In each candidate frame, shows the frame to Qwen and asks which of the numbered people is the correct one.
4) When Qwen chooses one, the script checks whether that person has a visible face.
5) If the face exists, it uses that face's facial embedding as an anchor.
6) Then it goes through the entire video and, in each frame, looks for which detection has a face similar to the anchor face, with a small bonus for temporal continuity.
7) Draws a single green box labeled REFERENT.

In two main phases
------------------
1) SEMANTIC GROUNDING:
   "Which person in which frame best matches the question target?"

YOLO cannot do that alone, because YOLO only knows "this is a person".
It does not know who "the little girl" or "the old man" is.
That is why Qwen comes in: it can observe the scene and reason about
who looks like the little girl, the old man, etc.

2) IDENTITY TRACKING:
   "Which boxes in which frames correspond to the same person across time?"

Here we use the face as identity.

Output
------
It saves:
a) an annotated video with the referent box
b) a result.json file with metadata:
c) anchor frame
d) anchor bbox
e) target mention
f) Qwen's reason
g) Qwen's raw response
h) all selected boxes frame by frame

Example how to run
------------------
export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

nice -n 5 python referent_anchor_face.py \
  --video /scratch/monroy/Playground/datasets/MoMentS_val_videos_emo/BrX9b.mp4 \
  --question "How does the little girl feel about the old man when he pretends to fall?" \
  --output-dir /scratch/monroy/Playground/referent_anchor_face_BrX9b_strict \
  --qwen-model Qwen/Qwen2.5-VL-3B-Instruct \
  --yolo-model yolo11x.pt \
  --device cuda \
  --conf 0.35
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from PIL import Image

import torch

# Evita errores de import o de compatibilidad.
class _DummyCompiler:
    @staticmethod
    def is_compiling():
        return False
    iscompiling = is_compiling

    @staticmethod
    def disable(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator

    @staticmethod
    def compile(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator

if not hasattr(torch, "compiler"):
    torch.compiler = _DummyCompiler

from ultralytics import YOLO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    raise ImportError(
        "insightface is required. Install with: pip install insightface onnxruntime"
    ) from e


def safe_mkdir(path: Path) -> None:
    # safe to call even if it already exists (folder)
    path.mkdir(parents=True, exist_ok=True)


def clamp_box(box, w: int, h: int) -> Tuple[int, int, int, int]:
    # Ensure box is within image bounds and has positive area
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def parse_json_from_text(text: str) -> Dict[str, Any]:
    # we ask qwen for a very specific output, however, sometimes it adds extra text before/after, or forgets to format as JSON, so we try to be robust in parsing the JSON object out of the text.
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output:\n{text[:500]}")
    return json.loads(text[start:end + 1])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # cosine similarity between embeddings 
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def read_video_frames(video_path: Path) -> Tuple[List[np.ndarray], float]:
    # Carga todo el video en memoria 
    # returns list of frames in BGR format and the FPS of the video (or 25.0 if it can't be determined)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames found in video: {video_path}")
    return frames, fps if fps > 0 else 25.0


def write_video(frames_bgr: List[np.ndarray], out_path: Path, fps: float) -> None:
    # Escribe todos los frames anotados a un mp4.
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()


def crop_with_context_bgr(
    # cuts a box out of the frame, but with some contect (padding)
    frame_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    pad_frac: float = 0.12,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * pad_frac)
    pad_y = int(bh * pad_frac)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)
    return frame_bgr[y1:y2, x1:x2]


def crop_with_context_pil(
    # same as before but returns a PIL so it can be passed to qwen    
    frame_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    pad_frac: float = 0.12,
) -> Image.Image:
    crop = crop_with_context_bgr(frame_bgr, box, pad_frac=pad_frac)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def sample_frame_indices(n_frames: int, n_samples: int) -> List[int]:
    # chooses frames distributed across video
    # if we say n_anchor_frames=30 it chooses 30 distributed across the video
    if n_frames <= n_samples:
        return list(range(n_frames))
    idxs = np.linspace(0, n_frames - 1, n_samples).round().astype(int).tolist()
    out = []
    seen = set()
    for i in idxs:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


class FaceModels:
    # Incluye toda la parte de InsightFace
    def __init__(self, device: str):
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        try:
            self.face_app = FaceAnalysis(name="buffalo_l", providers=providers)
            self.face_app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(320, 320))
        except Exception:
            self.face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self.face_app.prepare(ctx_id=-1, det_size=(320, 320))

    def detect_faces(self, image_bgr: np.ndarray) -> List[Any]:
        # wrapper para detectar caras.
        return self.face_app.get(image_bgr)

    def best_face_embedding(
        # DADA UNA IMAGEN: Ordena caras por área y elige la más grande.
        # saca embedding
        # la normaliza
        # devuelve embedding + bbox facial local
        # MIGHT CAUSE ERRORS
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
        faces = self.detect_faces(image_bgr)
        if not faces:
            return None, None

        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        face = faces[0]
        emb = face.embedding.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb, face.bbox.tolist()


_GLOBAL_YOLO_MODEL = None
# Carga YOLO una sola vez y lo reutiliza.

def get_yolo(yolo_model_path: str):
    global _GLOBAL_YOLO_MODEL
    if _GLOBAL_YOLO_MODEL is None:
        _GLOBAL_YOLO_MODEL = YOLO(yolo_model_path)
    return _GLOBAL_YOLO_MODEL

def detect_people_all_frames(
    # Para cada frame: llama a YOLO con classes=[0] → solo personas
    # obtiene boxes
    # para cada box calcula:
    # bbox limpia
    # confianza
    # área
    # centro
    # guarda la lista de detecciones de ese frame
    frames_bgr: List[np.ndarray],
    yolo_model_path: str,
    conf: float,
    device: str,
) -> List[List[Dict[str, Any]]]:
    model = get_yolo(yolo_model_path)
    all_dets: List[List[Dict[str, Any]]] = []

    for frame in frames_bgr:
        results = model.predict(
            source=frame,
            classes=[0],
            conf=conf,
            verbose=False,
            device=device,
        )
        dets = []
        if results:
            r = results[0]
            if r.boxes is not None and r.boxes.xyxy is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = (
                    r.boxes.conf.cpu().numpy()
                    if r.boxes.conf is not None
                    else np.ones((len(xyxy),), dtype=np.float32)
                )
                h, w = frame.shape[:2]
                for box, score in zip(xyxy, confs):
                    x1, y1, x2, y2 = clamp_box(box, w, h)
                    area = (x2 - x1) * (y2 - y1)
                    dets.append(
                        {
                            "bbox": (x1, y1, x2, y2),
                            "conf": float(score),
                            "area": float(area),
                            "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                        }
                    )
        all_dets.append(dets)

    return all_dets


def rank_detections_for_anchor(dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Para enseñar candidatos a Qwen, prioriza personas grandes y confiables.
    return sorted(dets, key=lambda d: (d["area"], d["conf"]), reverse=True)


def build_candidate_sheet_for_frame(
        # Construye la imagen que verá Qwen para elegir el referent.
        # toma las top detecciones del frame
        # asigna ids 1, 2, 3, ...
        # copia el frame original
        # dibuja cajas de colores
        # dibuja etiquetas [1], [2], etc.
        # guarda esa imagen
    frame_bgr: np.ndarray,
    dets: List[Dict[str, Any]],
    out_path: Path,
    max_candidates: int = 4,
    tile_size: Tuple[int, int] = (224, 224),
) -> Tuple[Path, Dict[int, Dict[str, Any]]]:
    ranked = rank_detections_for_anchor(dets)[:max_candidates]
    candidate_map = {i + 1: det for i, det in enumerate(ranked)}

    # Instead of cropped collages that lose scale, context, and pose, 
    # we draw labeled bounding boxes on the full original frame (Set-of-Mark style).
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]
    if max(h, w) > 768:
        scale = 768.0 / max(h, w)
        canvas = cv2.resize(canvas, (int(w * scale), int(h * scale)))
        
    colors = [
        (0, 255, 0),    # green
        (0, 0, 255),    # red
        (255, 0, 0),    # blue
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
    ]
    
    for cid, det in candidate_map.items():
        x1, y1, x2, y2 = det["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = colors[(cid - 1) % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 4)
        
        # Draw large label [1], [2], etc.
        label = f"[{cid}]"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        y_text = max(0, y1 - th - 10)
        cv2.rectangle(canvas, (x1, y_text), (x1 + tw + 10, y_text + th + baseline + 10), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 5, y_text + th + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), canvas)
    return out_path, candidate_map


_GLOBAL_QWEN_MODEL = None
_GLOBAL_QWEN_PROCESSOR = None

def load_qwen(model_name: str):
    global _GLOBAL_QWEN_MODEL, _GLOBAL_QWEN_PROCESSOR
    if _GLOBAL_QWEN_MODEL is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _GLOBAL_QWEN_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="eager",
        )
        _GLOBAL_QWEN_PROCESSOR = AutoProcessor.from_pretrained(model_name)
        _GLOBAL_QWEN_MODEL.eval()
    return _GLOBAL_QWEN_MODEL, _GLOBAL_QWEN_PROCESSOR


def build_anchor_prompt(question: str, candidate_ids: List[int]) -> str:
    return f"""You are given a full video frame where candidate people are highlighted with bounding boxes and numbered tags (e.g., [1], [2]).
Below the frame, you are also given close-up cropped images of each candidate to help you see their faces and details clearly.

Task:
Choose the SINGLE candidate number that best matches the semantic target of the question.

Rules:
- Give preference to the person whose feelings or actions are being asked about.
- Look at the WHOLE scene to understand context, actions, and relative sizes (e.g., "little girl" vs "old man").
- Ignore other people mentioned only as context.
- If the TARGET person (e.g. the one the question is about) is NOT clearly visible among the numbered candidates in this specific frame, you MUST set candidate_id to 0.
- candidate_id must be one of: 0, {", ".join(map(str, candidate_ids))}
- Return ONLY valid JSON:
{{"target_mention": "<exact text from question about the target>", "candidate_id": <integer>, "reason": "<reasoning based on the scene context>"}}

Question:
{question}
"""


def choose_anchor_candidate_with_qwen(
    question: str,
    candidate_sheet_path: Path,
    frame_bgr: np.ndarray,
    candidate_map: Dict[int, Dict[str, Any]],
    model: Any,
    processor: Any,
) -> Dict[str, Any]:

    content = [{"type": "image", "image": str(candidate_sheet_path)}]
    
    for cid in sorted(candidate_map.keys()):
        det = candidate_map[cid]
        crop_path = candidate_sheet_path.parent / f"{candidate_sheet_path.stem}_crop_{cid}.jpg"
        crop_pil = crop_with_context_pil(frame_bgr, det["bbox"], pad_frac=0.15)
        crop_pil.thumbnail((256, 256))
        crop_pil.save(crop_path)
        content.append({"type": "text", "text": f"\nCandidate [{cid}]:\n"})
        content.append({"type": "image", "image": str(crop_path)})
        
    content.append({"type": "text", "text": build_anchor_prompt(question, sorted(candidate_map.keys()))})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    try:
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    finally:
        del inputs
        del image_inputs
        del video_inputs
        if 'generated_ids' in locals():
            del generated_ids
        torch.cuda.empty_cache()

    parsed = parse_json_from_text(output_text)
    raw_candidate_id = parsed.get("candidate_id", None)
    if raw_candidate_id is None:
        raise ValueError(f"Missing candidate_id in Qwen output: {output_text}")

    if isinstance(raw_candidate_id, int):
        candidate_id = raw_candidate_id
    else:
        m = re.search(r"-?\d+", str(raw_candidate_id))
        if not m:
            raise ValueError(f"Could not parse candidate_id from Qwen output: {output_text}")
        candidate_id = int(m.group(0))

    if candidate_id == 0:
        raise ValueError("Qwen indicated the target person is not in this frame (candidate_id=0).")

    if candidate_id not in candidate_map:
        raise ValueError(f"Invalid candidate_id={candidate_id}. Raw output:\n{output_text}")

    return {
        "candidate_id": candidate_id,
        "target_mention": str(parsed.get("target_mention", "")).strip(),
        "reason": str(parsed.get("reason", "")).strip(),
        "raw_model_output": output_text,
    }


def choose_anchor_frame_and_box(
    frames_bgr: List[np.ndarray],
    all_dets: List[List[Dict[str, Any]]],
    question: str,
    qwen_model_name: str,
    tmp_dir: Path,
    models: FaceModels,
    n_anchor_frames: int = 6,
    max_candidates_per_frame: int = 4,
) -> Dict[str, Any]:
    qwen_model, qwen_processor = load_qwen(qwen_model_name)
    frame_indices = sample_frame_indices(len(frames_bgr), n_anchor_frames)
    scored = []
    for fi in frame_indices:
        dets = rank_detections_for_anchor(all_dets[fi])[:max_candidates_per_frame]
        total_area = sum(d["area"] for d in dets)
        scored.append((total_area, fi))
    scored.sort(reverse=True)

    last_error = None
    for _, fi in scored:
        dets = rank_detections_for_anchor(all_dets[fi])[:max_candidates_per_frame]
        if not dets:
            continue

        sheet_path, candidate_map = build_candidate_sheet_for_frame(
            frames_bgr[fi],
            dets,
            tmp_dir / f"anchor_frame_{fi}.png",
            max_candidates=max_candidates_per_frame,
            tile_size=(224, 224),
        )

        try:
            qwen_result = choose_anchor_candidate_with_qwen(
                question=question,
                candidate_sheet_path=sheet_path,
                frame_bgr=frames_bgr[fi],
                candidate_map=candidate_map,
                model=qwen_model,
                processor=qwen_processor,
            )
            chosen_det = candidate_map[qwen_result["candidate_id"]]

            crop = crop_with_context_bgr(frames_bgr[fi], chosen_det["bbox"])
            face_emb, local_face_bbox = models.best_face_embedding(crop)
            if face_emb is None or local_face_bbox is None:
                continue

            return {
                "anchor_frame_idx": fi,
                "anchor_bbox": chosen_det["bbox"],
                "anchor_det": chosen_det,
                "anchor_face_embedding": face_emb,
                **qwen_result,
                "candidate_sheet_path": str(sheet_path),
            }
        except Exception as e:
            print(f"       -> Skipping anchor frame {fi} because: {e}")
            last_error = str(e)
            continue

    raise RuntimeError(
        f"Failed to choose anchor frame/candidate with visible face. Last error: {last_error}"
    )


def propagate_anchor_identity_face_strict(
# Una vez ya tienes:
# frame anchor
# bbox anchor
# embedding facial anchor
# la pregunta pasa a ser:“En cada frame, cuál de las detecciones parece la misma persona?”
    frames_bgr: List[np.ndarray],
    all_dets: List[List[Dict[str, Any]]],
    anchor_frame_idx: int,
    anchor_bbox: Tuple[int, int, int, int],
    anchor_face_emb: np.ndarray,
    models: FaceModels,
    face_threshold: float = 0.45,
    temporal_weight: float = 0.10,
    allow_hold_frames: int = 2,
) -> List[Optional[Tuple[int, int, int, int]]]:
    chosen_boxes: List[Optional[Tuple[int, int, int, int]]] = [None] * len(frames_bgr)
    chosen_boxes[anchor_frame_idx] = anchor_bbox

    anchor_center = (
        (anchor_bbox[0] + anchor_bbox[2]) / 2.0,
        (anchor_bbox[1] + anchor_bbox[3]) / 2.0,
    )

    def best_face_match(fi: int, prev_center: Tuple[float, float]):
        dets = all_dets[fi]
        best_score = -1e9
        best_box = None
        best_center = None

        for det in dets:
            crop = crop_with_context_bgr(frames_bgr[fi], det["bbox"])
            if crop.size == 0:
                continue

            det_face_emb, _ = models.best_face_embedding(crop)
            if det_face_emb is None:
                continue

            face_sim = cosine_sim(anchor_face_emb, det_face_emb)
            if face_sim < face_threshold:
                continue

            cx, cy = det["center"]
            dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
            score = face_sim - temporal_weight * (dist / 100.0)

            if score > best_score:
                best_score = score
                best_box = det["bbox"]
                best_center = (cx, cy)

        return best_box, best_center

    prev_center = anchor_center
    prev_box = anchor_bbox
    hold_count = 0
    for fi in range(anchor_frame_idx + 1, len(frames_bgr)):
        best_box, best_center = best_face_match(fi, prev_center)

        if best_box is not None:
            chosen_boxes[fi] = best_box
            prev_box = best_box
            prev_center = best_center
            hold_count = 0
        elif prev_box is not None and hold_count < allow_hold_frames:
            chosen_boxes[fi] = prev_box
            hold_count += 1
        else:
            chosen_boxes[fi] = None

    prev_center = anchor_center
    prev_box = anchor_bbox
    hold_count = 0
    for fi in range(anchor_frame_idx - 1, -1, -1):
        best_box, best_center = best_face_match(fi, prev_center)

        if best_box is not None:
            chosen_boxes[fi] = best_box
            prev_box = best_box
            prev_center = best_center
            hold_count = 0
        elif prev_box is not None and hold_count < allow_hold_frames:
            chosen_boxes[fi] = prev_box
            hold_count += 1
        else:
            chosen_boxes[fi] = None

    return chosen_boxes


def draw_box_and_label(
    frame_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    color=(0, 255, 0),
    thickness=3,
) -> np.ndarray:
    out = frame_bgr.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    y_text = max(0, y1 - th - 8)
    cv2.rectangle(out, (x1, y_text), (x1 + tw + 10, y_text + th + baseline + 8), color, -1)
    cv2.putText(
        out,
        label,
        (x1 + 5, y_text + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def annotate_video(
    # Recorre todos los frames. Si hay caja elegida en ese frame, dibuja "REFERENT".
    frames_bgr: List[np.ndarray],
    chosen_boxes: List[Optional[Tuple[int, int, int, int]]],
) -> List[np.ndarray]:
    annotated = []
    for fi, frame in enumerate(frames_bgr):
        out = frame.copy()
        if chosen_boxes[fi] is not None:
            out = draw_box_and_label(out, chosen_boxes[fi], "REFERENT")
        annotated.append(out)
    return annotated


def parse_args():
    ap = argparse.ArgumentParser()

    # single-sample mode
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--question", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)

    # dataset mode
    ap.add_argument("--dataset-json", type=str, default=None)
    ap.add_argument("--video-root", type=str, default=None)
    ap.add_argument("--dataset-output-root", type=str, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit", type=int, default=None)

    # existing model/settings args
    ap.add_argument("--qwen-model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--yolo-model", type=str, default="yolo11x.pt")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--n-anchor-frames", type=int, default=30)
    ap.add_argument("--max-candidates-per-frame", type=int, default=4)
    ap.add_argument("--allow-hold-frames", type=int, default=2)
    ap.add_argument("--face-threshold", type=float, default=0.45)
    ap.add_argument("--temporal-weight", type=float, default=0.10)
    return ap.parse_args()

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_samples(data):
    """
    Supports:
    - list[dict]
    - dict[question_id] -> dict
    - {"data": [...]}, {"samples": [...]}, {"questions": [...]}
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if all(isinstance(v, dict) for v in data.values()):
            out = []
            for k, v in data.items():
                item = dict(v)
                item.setdefault("question_id", k)
                out.append(item)
            return out

        for key in ["data", "samples", "items", "questions"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("Unsupported dataset JSON structure")


def resolve_sample_id(sample: Dict[str, Any], idx: int) -> str:
    for key in ["question_id", "id", "uid"]:
        if key in sample and sample[key] is not None:
            return str(sample[key])
    return f"sample_{idx:06d}"


def resolve_question(sample: Dict[str, Any]) -> str:
    for key in ["question", "question_text", "query"]:
        if key in sample and sample[key]:
            return str(sample[key])
    raise ValueError(f"Could not find question text in sample: {sample}")


def resolve_video_path(sample: Dict[str, Any], video_root: Path) -> Path:
    sample_id = sample.get("question_id", None)
    if sample_id is None:
        raise ValueError(f"Could not resolve video path for sample: {sample}")

    return video_root / f"{sample_id}.mp4"


_GLOBAL_FACE_MODELS = None

def get_face_models(device: str):
    global _GLOBAL_FACE_MODELS
    if _GLOBAL_FACE_MODELS is None:
        _GLOBAL_FACE_MODELS = FaceModels(device)
    return _GLOBAL_FACE_MODELS

def run_one_sample(
    video_path: Path,
    question: str,
    out_dir: Path,
    args,
):
    tmp_dir = out_dir / "anchor_candidates"
    safe_mkdir(out_dir)
    safe_mkdir(tmp_dir)

    print(f"[1/6] Reading video: {video_path}")
    frames_bgr, fps = read_video_frames(video_path)
    h, w = frames_bgr[0].shape[:2]
    print(f"       frames={len(frames_bgr)} fps={fps:.3f} size={w}x{h}")

    print("[2/6] Detecting people on ALL frames")
    all_dets = detect_people_all_frames(
        frames_bgr=frames_bgr,
        yolo_model_path=args.yolo_model,
        conf=args.conf,
        device=args.device,
    )

    print("[3/6] Loading face model and choosing anchor frame + candidate with Qwen")
    models = get_face_models(args.device)
    anchor_result = choose_anchor_frame_and_box(
        frames_bgr=frames_bgr,
        all_dets=all_dets,
        question=question,
        qwen_model_name=args.qwen_model,
        tmp_dir=tmp_dir,
        models=models,
        n_anchor_frames=args.n_anchor_frames,
        max_candidates_per_frame=args.max_candidates_per_frame,
    )
    print(f"       anchor_frame={anchor_result['anchor_frame_idx']}")
    print(f"       target_mention={anchor_result.get('target_mention', '')}")
    print(f"       reason={anchor_result.get('reason', '')}")

    print("[4/6] Strict face-only propagation across ALL frames")
    chosen_boxes = propagate_anchor_identity_face_strict(
        frames_bgr=frames_bgr,
        all_dets=all_dets,
        anchor_frame_idx=anchor_result["anchor_frame_idx"],
        anchor_bbox=anchor_result["anchor_bbox"],
        anchor_face_emb=anchor_result["anchor_face_embedding"],
        models=models,
        face_threshold=args.face_threshold,
        temporal_weight=args.temporal_weight,
        allow_hold_frames=args.allow_hold_frames,
    )

    print("[5/6] Rendering output video")
    annotated = annotate_video(frames_bgr, chosen_boxes)
    out_video = out_dir / f"{video_path.stem}_referent.mp4"
    write_video(annotated, out_video, fps=fps)

    print("[6/6] Saving metadata")
    serial_boxes = []
    for box in chosen_boxes:
        if box is None:
            serial_boxes.append(None)
        else:
            serial_boxes.append([int(v) for v in box])

    metadata = {
        "video_path": str(video_path),
        "question": question,
        "fps": fps,
        "n_frames": len(frames_bgr),
        "qwen_model": args.qwen_model,
        "yolo_model": args.yolo_model,
        "anchor_frame_idx": int(anchor_result["anchor_frame_idx"]),
        "anchor_bbox": [int(v) for v in anchor_result["anchor_bbox"]],
        "target_mention": anchor_result.get("target_mention", ""),
        "qwen_reason": anchor_result.get("reason", ""),
        "qwen_raw_output": anchor_result.get("raw_model_output", ""),
        "candidate_sheet_path": anchor_result.get("candidate_sheet_path", ""),
        "chosen_boxes": serial_boxes,
        "annotated_video": str(out_video),
    }

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Done. Video: {out_video}")
    print(f"      JSON:  {out_dir / 'result.json'}")
    
def main():
    args = parse_args()

    # DATASET MODE
    if args.dataset_json is not None:
        if args.video_root is None or args.dataset_output_root is None:
            raise ValueError(
                "In dataset mode you must provide --dataset-json, --video-root, and --dataset-output-root"
            )

        dataset_json = Path(args.dataset_json)
        video_root = Path(args.video_root)
        dataset_output_root = Path(args.dataset_output_root)

        safe_mkdir(dataset_output_root)
        manifests_dir = dataset_output_root / "manifests"
        samples_dir = dataset_output_root / "samples"
        safe_mkdir(manifests_dir)
        safe_mkdir(samples_dir)

        data = load_json(dataset_json)
        samples = normalize_samples(data)
        if args.limit is not None:
            samples = samples[:args.limit]

        success_path = manifests_dir / "success.jsonl"
        failed_path = manifests_dir / "failed.jsonl"

        n_ok = 0
        n_fail = 0

        with open(success_path, "a", encoding="utf-8") as f_ok, open(failed_path, "a", encoding="utf-8") as f_fail:
            for idx, sample in enumerate(samples):
                try:
                    sample_id = resolve_sample_id(sample, idx)
                    question = resolve_question(sample)
                    video_path = resolve_video_path(sample, video_root)

                    out_dir = samples_dir / sample_id
                    result_json = out_dir / "result.json"

                    if args.skip_existing and result_json.exists():
                        print(f"[SKIP] {sample_id} already processed")
                        continue

                    if not video_path.exists():
                        print(f"[SKIP] {sample_id} not present in {video_root}")
                        continue

                    print(f"\n===== [{idx+1}/{len(samples)}] {sample_id} =====")
                    run_one_sample(
                        video_path=video_path,
                        question=question,
                        out_dir=out_dir,
                        args=args,
                    )

                    f_ok.write(json.dumps({
                        "sample_id": sample_id,
                        "video_path": str(video_path),
                        "question": question,
                        "output_dir": str(out_dir),
                        "result_json": str(result_json),
                    }, ensure_ascii=False) + "\n")
                    f_ok.flush()
                    n_ok += 1

                except Exception as e:
                    print(f"[FAIL] idx={idx}: {e}")
                    f_fail.write(json.dumps({
                        "index": idx,
                        "sample": sample,
                        "error": str(e),
                    }, ensure_ascii=False) + "\n")
                    f_fail.flush()
                    n_fail += 1

        summary = {
            "dataset_json": str(dataset_json),
            "video_root": str(video_root),
            "dataset_output_root": str(dataset_output_root),
            "processed": len(samples),
            "success": n_ok,
            "failed": n_fail,
        }

        with open(manifests_dir / "processing_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\nFinished dataset mode.")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # SINGLE VIDEO MODE
    if args.video is None or args.question is None or args.output_dir is None:
        raise ValueError(
            "For single-video mode you must provide --video, --question, and --output-dir"
        )

    run_one_sample(
        video_path=Path(args.video),
        question=args.question,
        out_dir=Path(args.output_dir),
        args=args,
    )

if __name__ == "__main__":
    main()