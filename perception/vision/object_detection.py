import cv2
import signal
import sys
from ultralytics import YOLO
import threading

from understanding import interpret_image

MODEL = YOLO("yolov8n.pt")

def run_viewer(image_path: str):
    model = MODEL
    results = model(image_path)
    r = results[0]
    annotated = r.plot()

    def signal_handler(sig, frame):
        cv2.destroyWindow("detections")
        cv2.destroyWindow("crop")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    state = {"crop": None}
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            img = annotated.copy()
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if x1 < x < x2 and y1 < y < y2:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = r.names[int(box.cls)]
                    conf = float(box.conf)
                    cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("detections", img)

        elif event == cv2.EVENT_LBUTTONDOWN:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if x1 < x < x2 and y1 < y < y2:
                    label = r.names[int(box.cls)]
                    conf = float(box.conf)
                    print(f"Selected: {label} ({conf:.2f}) at ({x1}, {y1}, {x2}, {y2})")
                    crop = r.orig_img[y1:y2, x1:x2]
                    cv2.imshow("crop", crop)
                    state["crop"] = crop
                    
                    # run interpretation in background so window stays responsive
                    def interpret():
                        response = interpret_image(crop)
                        print(f"Description: {response}")

                    threading.Thread(target=interpret, daemon=True).start()

                    break

    cv2.imshow("detections", annotated)
    cv2.setMouseCallback("detections", mouse_callback)

    while True:
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            cv2.destroyWindow("detections")
            cv2.destroyWindow("crop")
            break
    
    return state["crop"]