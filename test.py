import os

from matplotlib import pyplot as plt

os.environ["ULTRALYTICS_NO_IMSHOW"] = "1"

import cv2
import torch
import easyocr
from ultralytics import YOLO


reader = easyocr.Reader(['en'])
print(cv2.__version__)

print("Cuda available: ", torch.cuda.is_available())
print("PyTorch available: ", torch.__version__)

model = YOLO("yolov8n.pt")
np_plate = YOLO("license_plate_detector.pt")

CONF_THRESH = 0.35
IOU_THRESH = 0.2

# Load an image (replace with any image path you have)
img = cv2.imread('Image/test_image.png')

results = model(img, conf=CONF_THRESH, iou=IOU_THRESH)[0]

for box in results.boxes:
    cls = model.names[int(box.cls)]
    if cls in ["car", "truck", "bus", "motorbike", "motorcycle"]:
        x1,y1,x2,y2 = map(int, box.xyxy[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{cls}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        extracted_vehicle = img[y1:y2, x1:x2]
        np_plate_results = np_plate(extracted_vehicle)

        for np_box in np_plate_results[0].boxes:
            np_x1, np_y1, np_x2, np_y2 = map(int, np_box.xyxy[0])
            # cv2.rectangle(extracted_vehicle, (np_x1, np_y1), (np_x2, np_y2), (0, 255, 0), 2)
            # cv2.imshow(f"number plate extracted ares", extracted_vehicle)
            # cv2.imshow(f"number plate extracted plate", extracted_vehicle[np_y1:np_y2, np_x1:np_x2])
            # cv2.imwrite(f'test_number_plate_image.png',  extracted_vehicle[np_y1:np_y2, np_x1:np_x2])
            text = reader.readtext(extracted_vehicle[np_y1:np_y2, np_x1:np_x2])
            refind_text = "".join([i[1] for i in text])
            cv2.putText(img, f"number plate:{refind_text}",(x1 + 2, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)

# Display the image
plt.imshow(img)
plt.show()

# # Wait until a key is pressed
# cv2.waitKey(0)
#
# # Close all windows
# cv2.destroyAllWindows()