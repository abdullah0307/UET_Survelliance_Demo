import os
os.environ["ULTRALYTICS_NO_IMSHOW"] = "1"

import cv2
import torch
import easyocr
from ultralytics import YOLO
import numpy as np

# ---------------------------------------------------------
# SIMPLE CENTROID TRACKER
# ---------------------------------------------------------
class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_id = 1
        self.objects = {}  # id -> (x1, y1, x2, y2)

    def update(self, detections):
        """
        detections = list of (x1, y1, x2, y2)
        """

        new_objects = {}
        used_detections = set()

        # --------------------------------------
        # 1. MATCH PREVIOUS OBJECTS TO NEW FRAME
        # --------------------------------------
        for obj_id, old_box in self.objects.items():

            ox1, oy1, ox2, oy2 = old_box
            cx = (ox1 + ox2) // 2
            cy = (oy1 + oy2) // 2

            match_found = False

            for i, (x1, y1, x2, y2) in enumerate(detections):

                # Check if old centroid is INSIDE this new bounding box
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    new_objects[obj_id] = (x1, y1, x2, y2)
                    used_detections.add(i)
                    match_found = True
                    break

            # If old centroid didn't match → object disappeared (do not keep)
            if not match_found:
                pass  # simply do not add it

        # --------------------------------------
        # 2. ADD NEW OBJECTS FOR UNMATCHED DETECTIONS
        # --------------------------------------
        for i, det in enumerate(detections):
            if i not in used_detections:
                new_objects[self.next_id] = det
                self.next_id += 1

        self.objects = new_objects
        return self.objects



# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
USE_CAMERA = False
VIDEO_PATH = "CleanedVideo/192.168.1.64_01_20251114132542233_1.mp4"

SAVE_OUTPUT = True
OUTPUT_PATH = "processed_output.mp4"

CONF_THRESH = 0.35
IOU_THRESH = 0.2

# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------
reader = easyocr.Reader(["en"])
vehicle_detector = YOLO("yolov8n.pt")
plate_detector = YOLO("license_plate_detector.pt")

allowed_vehicles = ["car", "truck", "bus", "motorbike", "motorcycle"]

tracker = CentroidTracker(max_distance=50)

# ---------------------------------------------------------
# VIDEO
# ---------------------------------------------------------
cap = cv2.VideoCapture(0 if USE_CAMERA else VIDEO_PATH)

if not cap.isOpened():
    print("❌ Cannot open video source")
    exit()

if SAVE_OUTPUT:
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    print(f"💾 Saving output to {OUTPUT_PATH}")
else:
    out = None

print("▶ Processing started...")

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⏹ End of video")
        break

    results = vehicle_detector(frame, conf=CONF_THRESH, imgsz=960, iou=IOU_THRESH)[0]

    detections = []
    boxes_xyxy = []

    for box in results.boxes:
        cls = vehicle_detector.names[int(box.cls)]
        if cls not in allowed_vehicles:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((x1, y1, x2, y2))
        boxes_xyxy.append((x1, y1, x2, y2))

    tracked_objects = tracker.update(detections)

    # ---------------------------------------------------------
    # RENDERING VEHICLES + TRACK ID (Top-Right Corner)
    # ---------------------------------------------------------
    for obj_id, (x1, y1, x2, y2) in tracked_objects.items():

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # TOP-RIGHT tracking ID
        cv2.putText(
            frame,
            f"ID {obj_id}",
            (x1 + (x2 - x1)//2, y1 + (y2 - y1)//2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Crop vehicle
        crop_vehicle = frame[y1:y2, x1:x2]

        # Detect license plate inside vehicle crop
        plate_results = plate_detector(crop_vehicle)[0]

        for p_box in plate_results.boxes:
            px1, py1, px2, py2 = map(int, p_box.xyxy[0])
            plate_crop = crop_vehicle[py1:py2, px1:px2]

            # OCR
            ocr = reader.readtext(plate_crop)
            plate_text = "".join([t[1] for t in ocr])

            # Draw plate box on full frame
            cv2.rectangle(
                frame,
                (x1 + px1, y1 + py1),
                (x1 + px2, y1 + py2),
                (255, 0, 0),
                3
            )

            cv2.putText(
                frame,
                f"Plate: {plate_text}",
                (x1, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # Show window
    cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)
    cv2.imshow("Processed Video", frame)

    if SAVE_OUTPUT:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()

print("✅ Processing complete")
