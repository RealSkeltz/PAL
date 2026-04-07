import cv2
from object_detection import run_viewer
import time

TARGET_FPS = 3
DELAY = int(1000 / TARGET_FPS)  # milliseconds

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    run_viewer(frame)
    
    if cv2.waitKey(DELAY) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()