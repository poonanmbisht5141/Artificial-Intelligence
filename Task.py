# Task_Crowd_Detection.py - Crowd Detection using YOLOv5

import cv2
import torch
import numpy as np
import pandas as pd
from scipy.spatial import distance

# Parameters
crowd_distance_threshold = 100  # Distance threshold for grouping
crowd_frame_threshold = 10  # Minimum frames to confirm a crowd

# Load YOLOv5 model from local
model = torch.hub.load('./yolov5', 'yolov5s', source='local', pretrained=True)
model.classes = [0]  # Class 0 is 'person'

# Video Input and Output
video_path = 'dataset_video.mp4'
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_crowd_detection.mp4', fourcc, fps, (width, height))

# Tracking variables
frame_count = 0
crowd_history = []  # Stores (frame_number, person_count) when crowd is detected
group_track = {}  # Tracks groups across frames

# Process Video Frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    # Run detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    
    # Extract person detections
    persons = []
    for det in detections:
        xmin, ymin, xmax, ymax, conf, cls = det
        if conf > 0.5:  # Confidence threshold
            cx, cy = int((xmin + xmax) / 2), int((ymin + ymax) / 2)
            persons.append((cx, cy))

    # Calculate pairwise distances and group close individuals
    groups = []
    if len(persons) > 1:
        dists = distance.cdist(persons, persons, 'euclidean')
        visited = set()
        for i in range(len(persons)):
            if i in visited:
                continue
            group = [i]
            visited.add(i)
            for j in range(len(persons)):
                if j != i and j not in visited and dists[i][j] < crowd_distance_threshold:
                    group.append(j)
                    visited.add(j)
            if len(group) >= 3:  # Minimum 3 people to be considered a crowd
                groups.append(group)

    # Track groups across frames
    for group in groups:
        group_key = tuple(sorted(group))
        if group_key in group_track:
            group_track[group_key] += 1
        else:
            group_track[group_key] = 1

    # Check for crowds
    for key in list(group_track.keys()):
        if group_track[key] >= crowd_frame_threshold:
            crowd_history.append((frame_count, len(key)))
            del group_track[key]  # Reset after logging

    # Draw detections and groups
    for (cx, cy) in persons:
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    for group in groups:
        color = (0, 0, 255)
        for idx in group:
            cv2.circle(frame, persons[idx], 5, color, -1)

    # Write frame to output video
    out.write(frame)

cap.release()
out.release()

# Save crowd events to CSV
df = pd.DataFrame(crowd_history, columns=['Frame Number', 'Person Count'])
df.to_csv('crowd_events.csv', index=False)

print("Crowd Detection Completed. Output video and CSV saved.")
