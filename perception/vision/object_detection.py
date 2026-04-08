import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
from ultralytics import YOLO
import threading
from controls.headset_controls import start_headset_listener
from understanding import interpret_image

MODEL = YOLO("yolov8n.pt", verbose=False)
TARGET_FPS = 20
DELAY = int(1000 / TARGET_FPS)

def run_viewer(frame):
    model = MODEL
    results = model(frame, conf=0.375, verbose=False)
    r = results[0]
    annotated = r.plot()

    h, w = annotated.shape[:2]
    cx, cy = w // 2, h // 2

    img = annotated.copy()

    # find closest box to center
    closest_box = None
    closest_dist = float('inf')

    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        box_cx = (x1 + x2) // 2
        box_cy = (y1 + y2) // 2
        dist = ((box_cx - cx) ** 2 + (box_cy - cy) ** 2) ** 0.5
        if dist < closest_dist:
            closest_dist = dist
            closest_box = box

    if closest_box is not None:
        x1, y1, x2, y2 = map(int, closest_box.xyxy[0].tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = r.names[int(closest_box.cls)]
        conf = float(closest_box.conf)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.imshow("detections", img)

    return closest_box, r

def trigger_interpretation(r, closest_box):
    if closest_box is None:
        return
    x1, y1, x2, y2 = map(int, closest_box.xyxy[0].tolist())
    label = r.names[int(closest_box.cls)]
    conf = float(closest_box.conf)
    print(f"Selected: {label} ({conf:.2f}) at ({x1}, {y1}, {x2}, {y2})")
    crop = r.orig_img[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (224, 224))
    cv2.imshow("crop", crop_resized)

    def interpret():
        response = interpret_image(crop_resized, label)
        print(f"Description: {response}")

    threading.Thread(target=interpret, daemon=True).start()

trigger_flag = {"fired": False}

def on_button_press():
    trigger_flag["fired"] = True

start_headset_listener(on_button_press)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    closest_box = None
    r = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        closest_box, r = run_viewer(frame)

        if trigger_flag["fired"]:
            trigger_flag["fired"] = False
            trigger_interpretation(r, closest_box)

        key = cv2.waitKey(DELAY) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):  # trigger key
            trigger_interpretation(r, closest_box)

    cap.release()
    cv2.destroyAllWindows()