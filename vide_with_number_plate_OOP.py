import os
os.environ["ULTRALYTICS_NO_IMSHOW"] = "1"

import cv2
import torch
import easyocr
import matplotlib.pyplot as plt
from ultralytics import YOLO


class VehiclePlateProcessor:
    def __init__(
        self,
        input_source,
        vehicle_model="yolov8n.pt",
        plate_model="license_plate_detector.pt",
        save_output=False,
        output_path="output_video.mp4",
        conf_thresh=0.35,
        iou_thresh=0.2,
    ):
        # Flags
        self.input_source = input_source
        self.save_output = save_output
        self.output_path = output_path

        # Thresholds
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # OCR
        self.reader = easyocr.Reader(["en"])

        # YOLO Models
        self.vehicle_detector = YOLO(vehicle_model)
        self.plate_detector = YOLO(plate_model)

        # Allowed vehicle classes
        self.allowed_vehicles = [
            "car",
            "truck",
            "bus",
            "motorbike",
            "motorcycle",
        ]

        # Initialize video input
        self.cap = cv2.VideoCapture(input_source)
        if not self.cap.isOpened():
            raise ValueError("❌ Error: Cannot open video source.")

        # Output video writer
        self.out = None
        if self.save_output:
            self._init_video_writer()

        print("✅ System Ready")

    def _init_video_writer(self):
        """Initialize the output video writer if saving is enabled."""
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        print(f"💾 Saving output video to: {self.output_path}")

    def process_frame(self, frame):
        """Process a single frame: detect vehicles, plates, OCR."""
        results = self.vehicle_detector(frame, conf=self.conf_thresh, iou=self.iou_thresh)[0]

        for box in results.boxes:
            cls = self.vehicle_detector.names[int(box.cls)]

            if cls in self.allowed_vehicles:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw vehicle box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, cls, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Crop vehicle
                extracted_vehicle = frame[y1:y2, x1:x2]
                plate_results = self.plate_detector(extracted_vehicle)

                # Detect plates inside vehicle
                for p_box in plate_results[0].boxes:
                    px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                    plate_crop = extracted_vehicle[py1:py2, px1:px2]

                    # OCR reading
                    ocr_result = self.reader.readtext(plate_crop)
                    plate_text = "".join([t[1] for t in ocr_result])

                    # Draw text
                    cv2.putText(frame, f"Plate: {plate_text}",
                                (x1 + 5, y1 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)
        return frame

    def display_frame(self, frame):
        """Display using matplotlib (avoids cv2.imshow issues)."""
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.pause(0.001)
        plt.clf()

    def run(self):
        """Main processing loop."""
        print("▶ Processing Started")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⏹ End of video stream")
                break

            processed_frame = self.process_frame(frame)

            # Save processed frame
            if self.save_output:
                self.out.write(processed_frame)

            # # Display live results
            # self.display_frame(processed_frame)

        self.cap.release()
        if self.out:
            self.out.release()
        plt.close()

        print("✅ Processing Complete")


# -----------------------------------------------------
# RUN THE SYSTEM
# -----------------------------------------------------
if __name__ == "__main__":
    USE_CAMERA = False
    VIDEO_PATH = "CleanedVideo/192.168.1.64_01_20251114132542233_1.mp4"

    processor = VehiclePlateProcessor(
        input_source=0 if USE_CAMERA else VIDEO_PATH,
        save_output=True,                      # 🔥 Set False to disable saving
        output_path="Processed_output.mp4",    # Output file name
    )

    processor.run()
