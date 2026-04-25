import os
import threading

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
from ultralytics import YOLO

from typing import Generator
from Shared.Classes.Message import Message
from Shared.Classes.InputObject import InputObject

from controls.headset_controls import start_headset_listener

MODEL = YOLO("/Users/jscheltema/Documents/Personal/PAL/Shared/Resources/vision_models/yolov8n.pt", verbose=False)
TARGET_FPS = 5
DELAY = int(1000 / TARGET_FPS)


def _find_closest_box(r, cx, cy):
    closest_box = None
    closest_dist = float('inf')
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        dist = (((x1 + x2) // 2 - cx) ** 2 + ((y1 + y2) // 2 - cy) ** 2) ** 0.5
        if dist < closest_dist:
            closest_dist = dist
            closest_box = box
    return closest_box

def _annotate_frame(r, closest_box):
    annotated = r.plot()
    h, w = annotated.shape[:2]
    cx, cy = w // 2, h // 2
    img = annotated.copy()

    if closest_box is not None:
        x1, y1, x2, y2 = map(int, closest_box.xyxy[0].tolist())
        label = r.names[int(closest_box.cls)]
        conf = float(closest_box.conf)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.imshow("detections", img)
    return img

def _process_frame(frame, VISUAL_MODE=False):
    results = MODEL(frame, conf=0.375, verbose=False)
    r = results[0]
    h, w = r.orig_img.shape[:2]
    closest_box = _find_closest_box(r, w // 2, h // 2)
    if VISUAL_MODE:
        _annotate_frame(r, closest_box)
    return closest_box, r

def extract_crop(r, closest_box) -> tuple[np.ndarray, str, float]:
    x1, y1, x2, y2 = map(int, closest_box.xyxy[0].tolist())
    label = r.names[int(closest_box.cls)]
    conf = float(closest_box.conf)
    crop = cv2.resize(r.orig_img[y1:y2, x1:x2], (224, 224))
    return (crop, label, conf)



def run(trigger: threading.Event) -> Generator[Message, None, None]:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        closest_box, r = _process_frame(frame)
        if trigger.is_set():
            trigger.clear()
            if closest_box is None:
                print("[Vision] Trigger fired but no object detected")
            else:
                label = r.names[int(closest_box.cls)]
                conf = float(closest_box.conf)
                print(f"[Vision] Trigger fired, capturing: {label} ({conf:.2f})")
                yield extract_crop(r, closest_box)
        cv2.waitKey(DELAY)