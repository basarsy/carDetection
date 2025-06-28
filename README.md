# Traffic Congestion Detection using YOLOv8 and Deep SORT

This project implements a real-time traffic congestion detection system using advanced object detection and multi-object tracking techniques. It detects vehicles in video footage and monitors their movement to estimate traffic congestion levels on roads.

---

## Project Overview

Traffic congestion is a critical problem in urban areas, leading to wasted time, increased pollution, and economic losses. Automated systems that can detect congestion levels from video streams can help in traffic management and urban planning.

This project combines the powerful YOLOv8 object detection model with the Deep SORT tracking algorithm to:

- Detect vehicles frame-by-frame in a video stream.
- Track each vehicle across frames with a unique ID.
- Analyze vehicle movement patterns to identify slow or stopped vehicles.
- Calculate congestion as the percentage of vehicles that have been slow for a specific time window.
- Visually display tracked vehicles and congestion information on the video.

---

## How It Works: Step-by-Step Code Explanation

### 1. Initialization

- **YOLOv8 Model:** The script loads a pretrained YOLOv8n (nano) model (`yolov8n.pt`) from Ultralytics to detect objects in each video frame.
- **Deep SORT Tracker:** Deep SORT is initialized to track detected vehicles consistently across frames by associating detections with existing tracked objects.
- The input video file (`input.mp4`) is opened with OpenCV for frame-by-frame processing.
- Several variables and dictionaries are initialized to store tracking information:
  - `track_start_times`: Records the timestamp when a vehicle is first detected as potentially congested.
  - `track_y_history`: Stores a short history of vertical positions (y-coordinates) for each vehicle to analyze its movement.
  - `CONGESTION_TIME_SEC`: Threshold time (in seconds) a vehicle must remain slow to be considered congested.
  - `Y_HISTORY_LENGTH`: Number of frames used to analyze vehicle movement.

---

### 2. Processing Each Video Frame

- The video is read frame-by-frame in a loop.
- Each frame is passed through the YOLOv8 model to detect objects.
- The script filters detections to keep only vehicle classes (car, motorcycle, bus, truck) with confidence above 0.3.
- Bounding boxes and associated info are prepared as detections for Deep SORT.

---

### 3. Tracking Vehicles

- Deep SORT updates tracks with the new detections, maintaining consistent IDs for each vehicle.
- For each confirmed track, the script calculates the vertical center (`center_y`) of the bounding box.
- The vertical center positions are saved in a history buffer (`track_y_history`) for movement analysis.

---

### 4. Congestion Detection Logic

- The script checks the vertical movement of each vehicle over the last few frames (`Y_HISTORY_LENGTH`).
- If the vehicle has moved downward (positive `delta_y`) or hasn't moved forward significantly, it is skipped.
- If the vehicle is detected as not moving forward (no upward movement), the script checks how long it has stayed like that.
- If the duration exceeds `CONGESTION_TIME_SEC`, the vehicle is marked as congested (slow).
- Congested vehicles are highlighted with red bounding boxes and labeled `(slow)`.

---

### 5. Congestion Density Calculation

- The congestion density is calculated as the ratio of congested vehicles to total tracked vehicles, expressed as a percentage.
- This value is displayed on the video frame in real-time.

---

### 6. Visualization and Output

- The script draws bounding boxes and labels for each tracked vehicle on the video frames.
- Green boxes represent normal movement, red boxes indicate congestion.
- Congestion percentage is shown on the top-left corner.
- Pressing `q` exits the video window.

---

## Parameters You Can Adjust

- **`CONGESTION_TIME_SEC`**: Increase or decrease the time a vehicle must stay slow before being considered congested.
- **`conf` threshold**: Set higher to reduce false detections.
- **`Y_HISTORY_LENGTH`**: Number of frames used to analyze movement; larger values smooth out sudden position changes.
- **Vehicle Classes**: Modify tracked classes to suit your use case or camera angle.

---

## Installation

### Requirements

- Python 3.7+
- OpenCV
- Ultralytics YOLOv8
- deep_sort_realtime
- numpy

Install dependencies via pip:

```bash
pip install ultralytics opencv-python numpy deep_sort_realtime

