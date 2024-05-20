from ultralytics import YOLO
from collections import defaultdict
from helpers.helper import tracking
import torch
import cv2
import os
import time

# Initialize global variables for selection functionality
selected_id = None
red_timers = {}

if not os.path.exists('models/'):
    os.makedirs('models/')

def selectBbox(event, x, y, flags, param):
    global selected_id, red_timers
    if event == cv2.EVENT_LBUTTONDOWN:
        for track_id, (x1, y1, x2, y2) in bbox_coords.items():
            if x1 < x < x2 and y1 < y < y2:
                if selected_id != track_id:
                    selected_id = track_id
                    red_timers[track_id] = time.time()
                return

if __name__ == "__main__":   
    link = 'assets/Main Out 2.mp4' #IP CAM streaming link
    model = YOLO('models/yolov8l.pt')
    cap = cv2.VideoCapture(link)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bbox_colors = {}
    bbox_coords = {}
    track_history = defaultdict(lambda: [])

    cv2.namedWindow("YOLOv8 Inference Window", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("YOLOv8 Inference Window", selectBbox)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(source=frame, persist=True, classes=[0], device=device, verbose=False)
        frame = tracking(results, frame, bbox_coords, bbox_colors, track_history, selected_id, red_timers)
        
        cv2.imshow("YOLOv8 Inference Window", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
