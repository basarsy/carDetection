#!/usr/bin/env python3
from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from collections import deque

# Initialize YOLO and Deep SORT
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=15)

video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

track_start_times = {}
track_y_history = {}

CONGESTION_TIME_SEC = 3
FPS = cap.get(cv2.CAP_PROP_FPS)
Y_HISTORY_LENGTH = 5

current_frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    current_frame_count += 1

    results = model(frame, verbose=False)

    detections = []
    matched_ids = set()

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf < 0.3:
                continue

            if cls in [2, 3, 5, 7]:  # vehicles
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    tracks = tracker.update_tracks(detections, frame=frame)

    congested_vehicles = 0
    total_tracked = 0

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        center_y = int((y1 + y2) / 2)

        matched_ids.add(track_id)

        if track_id not in track_y_history:
            track_y_history[track_id] = deque(maxlen=Y_HISTORY_LENGTH)
        track_y_history[track_id].append(center_y)

        y_positions = track_y_history[track_id]
        if len(y_positions) < Y_HISTORY_LENGTH:
            continue

        delta_y = y_positions[-1] - y_positions[0]
        if delta_y <= 0:
            continue

        if track_id not in track_start_times:
            track_start_times[track_id] = current_time

        duration = current_time - track_start_times[track_id]
        total_tracked += 1

        color = (0, 255, 0)
        label = f'ID: {track_id}'

        if duration >= CONGESTION_TIME_SEC:
            congested_vehicles += 1
            color = (0, 0, 255)
            label += " (slow)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if total_tracked > 0:
        density = (congested_vehicles / total_tracked) * 100
    else:
        density = 0

    density = round(density, 2)

    cv2.putText(frame, f"Congestion: {density}%", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Traffic Congestion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
