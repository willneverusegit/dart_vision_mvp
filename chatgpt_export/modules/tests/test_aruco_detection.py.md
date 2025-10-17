# Module: `tests\test_aruco_detection.py`
Hash: `95b9f3502709` · LOC: 1 · Main guard: false

## Imports
- `cv2`\n- `numpy`

## From-Imports
—

## Classes
—

## Functions
—

## Intra-module calls (heuristic)
ArucoDetector, DetectorParameters, VideoCapture, destroyAllWindows, detectMarkers, drawDetectedMarkers, getPredefinedDictionary, imshow, len, ord, print, putText, read, release, waitKey

## Code
```python
"""
Quick test to verify ArUco detection
"""

import cv2
import numpy as np

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# Open webcam
cap = cv2.VideoCapture(0)

print("ArUco Detection Test")
print("Press 'q' to quit")
print("")

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(frame)

    # Draw detected markers
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Print detected IDs
        detected_ids = [id[0] for id in ids]
        print(f"Detected markers: {detected_ids} (Total: {len(detected_ids)})")

        # Draw status
        cv2.putText(frame, f"Markers: {len(detected_ids)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No markers detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("No markers detected")

    # Show frame
    cv2.imshow('ArUco Detection Test', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
