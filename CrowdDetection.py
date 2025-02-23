# Crowd_Detection.py
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Video Input and Output Setup
input_file = 'dataset_video.mp4'
output_file = 'crowd_detection_output.mp4'
cap = cv2.VideoCapture(input_file)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Parameters
DISTANCE_THRESHOLD = 50  # Max distance to be considered "close"
FRAME_THRESHOLD = 10    # Number of consecutive frames for crowd detection

# Data Storage
frame_history = []
crowd_history = []
crowd_events = []
frame_number = 0

# Helper Functions
def calculate_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def detect_crowd(frame_history):
    global crowd_history
    current_crowd = []

    # Analyze the latest frame
    latest_boxes = frame_history[-1]
    centroids = [calculate_centroid(box) for box in latest_boxes]

    # Check distance between each pair of centroids
    for i in range(len(centroids)):
        group = [i]
        for j in range(i+1, len(centroids)):
            if euclidean_distance(centroids[i], centroids[j]) < DISTANCE_THRESHOLD:
                group.append(j)

        # If a group of 3 or more persons is close together, log it
        if len(group) >= 3:
            current_crowd.append(group)

    # Update crowd history
    crowd_history.append(current_crowd)
    if len(crowd_history) > FRAME_THRESHOLD:
        crowd_history.pop(0)

    # Check if the crowd persists for FRAME_THRESHOLD frames
    persistent_crowds = [group for group in current_crowd if all(group in frame for frame in crowd_history)]
    return persistent_crowds

# Main Detection Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    # Perform person detection
    results = model(frame)
    boxes = []
    for r in results:
        for bbox in r.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            cls = int(bbox.cls[0])
            if cls == 0:  # Person class in COCO dataset
                boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Store bounding boxes for crowd detection logic
    frame_history.append(boxes)
    if len(frame_history) > FRAME_THRESHOLD:
        frame_history.pop(0)

    # Detect crowd
    crowds = detect_crowd(frame_history)
    if crowds:
        for group in crowds:
            person_count = len(group)
            cv2.putText(frame, f"Crowd Detected: {person_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Log the event with frame number and count of persons
            crowd_events.append({"Frame Number": frame_number, "Person Count in Crowd": person_count})

    # Write the frame to the output video
    out.write(frame)

    # Display frame (optional, can be commented out for faster processing)
    cv2.imshow('Crowd Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Save crowd event log to CSV
event_df = pd.DataFrame(crowd_events)
event_df.to_csv('crowd_detection_events.csv', index=False)

print("Video saved as crowd_detection_output.mp4")
print("Event log saved as crowd_detection_events.csv")
