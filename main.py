from ultralytics import YOLO
from collections import defaultdict
import random 
import torch
import cv2
import numpy as np
import time

def generate_random_color(existing_colors):
    while True:
        color = tuple(random.randint(0, 255) for _ in range(3))
        # Avoid red-ish colors and ensure uniqueness
        if color not in existing_colors and not (color[2] > 150 and color[1] < 100 and color[0] < 100):
            return color

# Initialize global variables for selection functionality
selected_id = None
red_timers = {}

def select_bbox(event, x, y, flags, param):
    global selected_id, red_timers
    if event == cv2.EVENT_LBUTTONDOWN:
        for track_id, (x1, y1, x2, y2) in bbox_coords.items():
            if x1 < x < x2 and y1 < y < y2:
                if selected_id != track_id:
                    selected_id = track_id
                    red_timers[track_id] = time.time()
                return

if __name__ == "__main__":   
    link = 'assets/Corridor Enterence gate.mp4'
    model = YOLO('models/yolov8l.pt')
    cap = cv2.VideoCapture(link)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bbox_colors = {}
    bbox_coords = {}
    track_history = defaultdict(lambda: [])

    cv2.namedWindow("YOLOv8 Inference Window", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("YOLOv8 Inference Window", select_bbox)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(source=frame, persist=True, classes=[0], device=device, verbose=False)
        boxes_xyxy = results[0].boxes.xyxy.cpu()
        current_ids = []

        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for track_id, box_xyxy in zip(track_ids, boxes_xyxy):
                x1, y1, x2, y2 = map(int, box_xyxy)
                track = track_history[track_id]
                track.append((float(x1), float(y1)))

                if len(track) > 50:
                    track.pop(0)

                current_ids.append(track_id)
                bbox_coords[track_id] = (x1, y1, x2, y2)
                
                if track_id not in bbox_colors:
                    bbox_colors[track_id] = generate_random_color(bbox_colors.values())
                
                color = bbox_colors[track_id]
                
                if selected_id == track_id:
                    color = (0, 0, 255)  # Red color
                    elapsed_time = int(time.time() - red_timers[track_id])
                    cv2.putText(frame, f'Timer: {elapsed_time}s', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
                cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        except:
            continue

        # Clean up old trackers
        for track_id in list(bbox_colors):
            if track_id not in current_ids:
                del bbox_colors[track_id]
                if track_id in red_timers:
                    del red_timers[track_id]
                if track_id in bbox_coords:
                    del bbox_coords[track_id]

        cv2.imshow("YOLOv8 Inference Window", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
