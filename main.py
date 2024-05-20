import cv2
import numpy as np
from ultralytics import YOLO
import random 
import torch

link = 'assets/Corridor Enterence gate.mp4'
model = YOLO('models/yolov8s.pt')
cap = cv2.VideoCapture(link)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(source=frame, persist=True, classes = [0], device=device)

    boxes_xyxy = results[0].boxes.xyxy.cpu()
    try:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        for track_id, box_xyxy in zip(track_ids, boxes_xyxy):
            x1, y1, x2, y2 = box_xyxy

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,255,0), thickness=2)
            # cv2.putText(frame, f"ID: {str(track_id)}", (int(x1), int(y1)), color=(255,255,255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.35, thickness=1)
    except:
        continue

    cv2.namedWindow("YOLOv8 Inference Window", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLOv8 Inference Window", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()