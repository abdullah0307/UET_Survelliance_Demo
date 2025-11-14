import cv2
from ultralytics import YOLO
import os

# -------------------------
# SETTINGS
# -------------------------
VIDEO_SOURCE = "Videos/192.168.1.64_01_20251114132542233_1.mp4"    # Input video
OUTPUT_VIDEO = "CleanedVideo/192.168.1.64_01_20251114132542233_1.mp4"  # Single output file

VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike", "motorcycle"]
CONF_THRESH = 0.4
IOU_THRESH = 0.3

# -------------------------
# LOAD YOLO MODEL
# -------------------------
model = YOLO("yolov8n.pt")

# -------------------------
# OPEN VIDEO INPUT
# -------------------------
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print("Saving only detected frames to:", OUTPUT_VIDEO)

# -------------------------
# PROCESS VIDEO
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESH, iou=IOU_THRESH)[0]

    vehicle_detected = False

    # Check detection
    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        if cls_name in VEHICLE_CLASSES:
            vehicle_detected = True
            break

    # Save only detected frames
    if vehicle_detected:
        out.write(frame)

cap.release()
out.release()

print("DONE! Output saved as:", OUTPUT_VIDEO)
