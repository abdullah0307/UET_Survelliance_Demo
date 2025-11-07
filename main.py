import math
import time

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

import torch
print("Cuda available: ", torch.cuda.is_available())
print("PyTorch available: ", torch.__version__)

# VIDEO_SOURCE = "rtsp://admin:NCAI%401024@192.168.1.64:554/Streaming/Channels/101"
VIDEO_SOURCE = "Videos/Entry Night.mp4"
CONF_THRESH = 0.35
IOU_THRESH = 0.2
# FORCED_WIDTH = 1920 * 2
# FORCED_HEIGHT = 1080 * 2


# ----------------------------
# SIMPLE DEEP SORT-LIKE TRACKER
# ----------------------------
class Track:
    def __init__(self, track_id, bbox):
        self.id = track_id
        self.bbox = bbox  # x1, y1, x2, y2
        self.kf = self.create_kf(bbox)
        self.time_since_update = 0
        self.hits = 1

    @staticmethod
    def create_kf(bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([[1,0,0,0,1,0,0],
                         [0,1,0,0,0,1,0],
                         [0,0,1,0,0,0,1],
                         [0,0,0,1,0,0,0],
                         [0,0,0,0,1,0,0],
                         [0,0,0,0,0,1,0],
                         [0,0,0,0,0,0,1]])

        kf.H = np.array([[1,0,0,0,0,0,0],
                         [0,1,0,0,0,0,0],
                         [0,0,1,0,0,0,0],
                         [0,0,0,1,0,0,0]])

        kf.R *= 10
        kf.P *= 10

        kf.x[:4] = np.array([cx, cy, w, h]).reshape((4,1))
        return kf

    def predict(self):
        self.kf.predict()
        px, py, pw, ph = self.kf.x[:4]
        x1 = int(px - pw/2)
        y1 = int(py - ph/2)
        x2 = int(px + pw/2)
        y2 = int(py + ph/2)
        self.bbox = (x1,y1,x2,y2)
        return self.bbox

    def update(self, bbox):
        x1,y1,x2,y2 = bbox
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        w = x2-x1
        h = y2-y1

        self.kf.update(np.array([[cx],[cy],[w],[h]]))
        self.time_since_update = 0
        self.hits += 1


class DeepSortTracker:
    def __init__(self, max_age=10, iou_threshold=0.1):
        self.next_id = 1
        self.tracks = []
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def iou(self, bb_test, bb_gt):
        xx1 = max(bb_test[0], bb_gt[0])
        yy1 = max(bb_test[1], bb_gt[1])
        xx2 = min(bb_test[2], bb_gt[2])
        yy2 = min(bb_test[3], bb_gt[3])
        w = max(0., xx2 - xx1)
        h = max(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
                  + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
        return o

    def update(self, detections):
        # predict
        predicted = []
        for t in self.tracks:
            predicted.append(t.predict())

        # cost matrix: 1 - IOU
        if len(predicted) > 0 and len(detections) > 0:
            cost = np.zeros((len(predicted), len(detections)))
            for i, p in enumerate(predicted):
                for j, d in enumerate(detections):
                    cost[i,j] = 1 - self.iou(p, d)

            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            row_ind, col_ind = [], []

        assigned_tracks = set()
        assigned_detections = set()

        for r,c in zip(row_ind, col_ind):
            if (1 - cost[r,c]) < self.iou_threshold:
                continue
            self.tracks[r].update(detections[c])
            assigned_tracks.add(r)
            assigned_detections.add(c)

        # new detections
        for i, d in enumerate(detections):
            if i not in assigned_detections:
                self.tracks.append(Track(self.next_id, d))
                self.next_id += 1

        # age unassigned tracks
        alive_tracks = []
        for idx, t in enumerate(self.tracks):
            if idx not in assigned_tracks:
                t.time_since_update += 1
            if t.time_since_update < self.max_age:
                alive_tracks.append(t)
        self.tracks = alive_tracks

        return self.tracks


# --------------------------
# YOLO + DEEPSORT PIPELINE
# --------------------------
model = YOLO("yolov8n.pt")
tracker = DeepSortTracker()

cap = cv2.VideoCapture(VIDEO_SOURCE)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.resize(frame, (FORCED_WIDTH, FORCED_HEIGHT))

    results = model(frame, conf=CONF_THRESH, iou=IOU_THRESH)[0]

    detections = []
    for box in results.boxes:
        cls = model.names[int(box.cls)]
        if cls in ["car", "truck", "bus", "motorbike", "motorcycle"]:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            detections.append((x1,y1,x2,y2))

    # update tracker
    tracks = tracker.update(detections)

    # draw boxes
    for t in tracks:
        x1,y1,x2,y2 = t.bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {t.id}", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.namedWindow("Vehicle Tracking (YOLO + DeepSORT)", cv2.WINDOW_NORMAL)
    cv2.imshow("Vehicle Tracking (YOLO + DeepSORT)", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
