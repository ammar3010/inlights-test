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

    results = model.track(source=frame, persist=True, classes = [0], device=device, show=True)