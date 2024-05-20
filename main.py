import cv2
import numpy as np
from ultralytics import YOLO
import random 
import torch

def generate_random_color(existing_colors):
    while True:
        color = tuple(random.randint(0, 255) for _ in range(3))
        # Avoid red-ish colors and ensure uniqueness
        if color not in existing_colors and not (color[2] > 150 and color[1] < 100 and color[0] < 100):
            return color

link = 'assets/Corridor Enterence gate.mp4'
model = YOLO('models/yolov8s.pt')
cap = cv2.VideoCapture(link)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bbox_colors = {}

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(source=frame, persist=True, classes = [0], device=device)
    boxes_xyxy = results[0].boxes.xyxy.cpu()
    
    current_ids = []

    # try:
    track_ids = results[0].boxes.id.int().cpu().tolist()
    for track_id, box_xyxy in zip(track_ids, boxes_xyxy):
        x1, y1, x2, y2 = box_xyxy
        
        current_ids.append(track_id)
        if track_id not in bbox_colors:
            bbox_colors[track_id] = generate_random_color(bbox_colors.values())
        
        color = bbox_colors[track_id]
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=2)
        # cv2.putText(frame, f"ID: {str(track_id)}", (int(x1), int(y1)), color=(255,255,255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.35, thickness=1)
    # except:
    #     continue

    cv2.namedWindow("YOLOv8 Inference Window", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLOv8 Inference Window", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()