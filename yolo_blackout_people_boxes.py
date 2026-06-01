import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--model", default="yolo11x.pt")
parser.add_argument("--conf", type=float, default=0.25)
args = parser.parse_args()

input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

model = YOLO(args.model)

video_paths = sorted(input_dir.glob("*.mp4"))

for video_path in video_paths:
    out_path = output_dir / video_path.name

    if out_path.exists():
        print("SKIP:", out_path)
        continue

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    print("Processing:", video_path.name)

    for _ in tqdm(range(total)):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False)[0]

        keep = np.zeros_like(frame)

        for box in results.boxes:
            cls = int(box.cls[0])

            # COCO class 0 = person
            if cls != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            keep[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        writer.write(keep)

    cap.release()
    writer.release()

print("DONE")
