#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time License Plate Recognition using YOLOv8 + EasyOCR

Installation (CPU only):
    pip install ultralytics easyocr opencv-python

Installation (GPU, if supported by your system):
    pip install ultralytics easyocr opencv-python torch torchvision --index-url https://download.pytorch.org/whl/cu121

Usage examples:
    # Webcam
    python license_plate_recognition.py --source 0 --weights best.pt

    # RTSP stream
    python license_plate_recognition.py --source rtsp://user:pass@camera-ip:554/stream --weights best.pt
"""

import argparse
import csv
import os
import re
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except Exception:
    _EASYOCR_AVAILABLE = False


# -----------------------------
# Utility functions
# -----------------------------
def parse_source(source_str: str):
    if source_str.isdigit():
        return int(source_str)
    return source_str


def current_timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_csv_has_header(csv_path: str):
    if not csv_path:
        return
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "track_id", "plate_text"])


def write_csv_row(csv_path: str, timestamp: str, track_id: int, plate_text: str):
    if not csv_path:
        return
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, track_id, plate_text])


def clean_plate_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def preprocess_plate_crop(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    return thresh


# -----------------------------
# Centroid Tracker
# -----------------------------
class CentroidTracker:
    def __init__(self, max_missing_frames=30, distance_threshold=80.0):
        self.next_track_id = 1
        self.tracks = {}
        self.max_missing_frames = max_missing_frames
        self.distance_threshold = distance_threshold

    @staticmethod
    def _centroid_from_bbox(xyxy):
        x1, y1, x2, y2 = xyxy
        return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)

    @staticmethod
    def _euclidean_distance(c1, c2):
        return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))

    def update(self, detections):
        for track in self.tracks.values():
            track["missing_frames"] += 1

        assigned_tracks = set()
        track_assignments = {}

        for det_bbox in detections:
            det_centroid = self._centroid_from_bbox(det_bbox)
            best_track_id = None
            best_dist = float("inf")

            for track_id, track in self.tracks.items():
                if track_id in assigned_tracks:
                    continue
                dist = self._euclidean_distance(det_centroid, track["centroid"])
                if dist < best_dist:
                    best_dist = dist
                    best_track_id = track_id

            if best_track_id is not None and best_dist <= self.distance_threshold:
                self.tracks[best_track_id].update(
                    {"bbox": det_bbox, "centroid": det_centroid, "missing_frames": 0}
                )
                assigned_tracks.add(best_track_id)
                track_assignments[best_track_id] = det_bbox
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracks[tid] = {
                    "bbox": det_bbox,
                    "centroid": det_centroid,
                    "missing_frames": 0,
                    "last_ocr_text": "",
                    "last_ocr_frame_idx": -999999,
                    "ocr_history": deque(maxlen=5),
                }
                assigned_tracks.add(tid)
                track_assignments[tid] = det_bbox

        to_delete = [
            tid for tid, t in self.tracks.items() if t["missing_frames"] > self.max_missing_frames
        ]
        for tid in to_delete:
            del self.tracks[tid]

        return track_assignments


# -----------------------------
# Model loading
# -----------------------------
def load_yolo_model(weights_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(weights_path)
    model.to(device)
    return model


# -----------------------------
# OCR initialization
# -----------------------------
def init_easyocr_reader(languages, use_gpu=None):
    if not _EASYOCR_AVAILABLE:
        raise RuntimeError("easyocr not installed. Please install via: pip install easyocr")
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    return easyocr.Reader(languages, gpu=use_gpu)


# -----------------------------
# Main function
# -----------------------------
def run(
    source,
    weights,
    plate_class_id=0,
    plate_class_name=None,
    conf_threshold=0.25,
    iou_threshold=0.45,
    show_window=True,
    csv_path="detections.csv",
    use_tracking=True,
    ocr_frame_cooldown=10,
    resize_width=None,
    languages=None,
):
    if languages is None:
        languages = ["en"]

    cap = cv2.VideoCapture(parse_source(source), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model = load_yolo_model(weights, device)
    model.overrides["conf"] = conf_threshold
    model.overrides["iou"] = iou_threshold

    reader = init_easyocr_reader(languages, use_gpu=(device == "cuda"))

    class_name_to_id = None
    selected_class_id = plate_class_id
    if model.names:
        inv = {str(v).lower(): k for k, v in model.names.items()}
        if plate_class_name and plate_class_name.strip().lower() in inv:
            selected_class_id = inv[plate_class_name.strip().lower()]
        else:
            print(f"[WARN] Plate class name not found, using id={plate_class_id}")

    tracker = CentroidTracker() if use_tracking else None
    frame_idx = 0

    ensure_csv_has_header(csv_path) if csv_path else None

    print("[INFO] Starting stream. Press 'q' to quit.")
    while True:
        grabbed, frame = cap.read()
        if not grabbed or frame is None:
            time.sleep(0.01)
            continue

        original_h, original_w = frame.shape[:2]

        ### >>> RESIZE  <<<
        if resize_width is not None and resize_width > 0:
            h, w = frame.shape[:2]
            if w > resize_width:
                scale = resize_width / float(w)
                new_h = int(h * scale)

                max_height = 720
                if new_h > max_height:
                    scale = max_height / float(h)
                    resize_width = int(w * scale)
                    new_h = max_height

                frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

                if frame_idx % 30 == 0:
                    print(f"[INFO] Resized frame to: {resize_width}x{new_h}")
        ### >>> RESIZE  <<<

        display_frame = frame.copy()

        results = model.predict(source=frame, verbose=False, device=device)
        det_bboxes, det_scores = [], []
        if results and len(results) > 0:
            r0 = results[0]
            if r0.boxes is not None:
                for b in r0.boxes:
                    cls_id = int(b.cls[0].item())
                    if cls_id != selected_class_id:
                        continue
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    det_bboxes.append((x1, y1, x2, y2))
                    det_scores.append(float(b.conf[0].item()))

        track_to_bbox = tracker.update(det_bboxes) if tracker else {
            i + 1: bbox for i, bbox in enumerate(det_bboxes)
        }

        for track_id, bbox in track_to_bbox.items():
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            run_ocr = True
            last_text = ""
            if tracker:
                t = tracker.tracks.get(track_id)
                last_text = t["last_ocr_text"] if t else ""
                last_idx = t["last_ocr_frame_idx"] if t else -999999
                if (frame_idx - last_idx) < ocr_frame_cooldown and last_text:
                    run_ocr = False

            recognized_text = last_text
            if run_ocr:
                pre = preprocess_plate_crop(crop)
                ocr_res = reader.readtext(pre)
                if ocr_res:
                    ocr_res.sort(key=lambda x: x[2], reverse=True)
                    recognized_text = clean_plate_text(ocr_res[0][1])

                if tracker:
                    t = tracker.tracks.get(track_id)
                    if t is not None:
                        if recognized_text:
                            t["ocr_history"].append(recognized_text)
                            if len(t["ocr_history"]) >= 3:
                                vals, counts = np.unique(
                                    np.array(list(t["ocr_history"])), return_counts=True
                                )
                                recognized_text = str(vals[np.argmax(counts)])
                        t["last_ocr_text"] = recognized_text
                        t["last_ocr_frame_idx"] = frame_idx

                if recognized_text and recognized_text != last_text:
                    ts = current_timestamp_str()
                    print(f"{ts} | {recognized_text}")
                    if csv_path:
                        write_csv_row(csv_path, ts, track_id, recognized_text)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = recognized_text if recognized_text else "plate"
            label = f"ID {track_id}: {label}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (x1, y1 - th - base - 6), (x1 + tw + 6, y1), (0, 200, 0), -1)
            cv2.putText(display_frame, label, (x1 + 3, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        if show_window:
            cv2.imshow("License Plate Recognition", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if show_window:
        cv2.destroyAllWindows()


def build_argparser():
    p = argparse.ArgumentParser(description="Real-time LPR using YOLOv8 + EasyOCR")
    p.add_argument("--source", type=str, default="rtsp://admin:NCAI%401024@192.168.1.64:554/Streaming/Channels/101")
    p.add_argument("--weights", type=str, default="license_plate_detector.pt")
    p.add_argument("--plate-class-id", type=int, default=0)
    p.add_argument("--plate-class-name", type=str, default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--csv", type=str, default="detections.csv")
    p.add_argument("--no-tracking", action="store_true")
    p.add_argument("--ocr-cooldown", type=int, default=10)
    p.add_argument("--resize-width", type=int, default=960)
    p.add_argument("--lang", type=str, default="en")
    return p


def main():
    args = build_argparser().parse_args()
    langs = [x.strip() for x in args.lang.split(",") if x.strip()]
    resize_width = args.resize_width if args.resize_width > 0 else None
    csv_path = args.csv if args.csv else None

    run(
        source=args.source,
        weights=args.weights,
        plate_class_id=args.plate_class_id,
        plate_class_name=args.plate_class_name,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        show_window=(not args.no_window),
        csv_path=csv_path,
        use_tracking=(not args.no_tracking),
        ocr_frame_cooldown=max(0, int(args.ocr_cooldown)),
        resize_width=resize_width,
        languages=langs,
    )


if __name__ == "__main__":
    main()
