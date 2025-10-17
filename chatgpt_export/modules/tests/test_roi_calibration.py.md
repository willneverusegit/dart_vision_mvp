# Module: `tests\test_roi_calibration.py`
Hash: `a6b4f1f9ec11` · LOC: 1 · Main guard: true

## Imports
- `cv2`\n- `numpy`\n- `sys`

## From-Imports
- `from pathlib import Path`\n- `from src.calibration import ROIProcessor, CalibrationManager, ROIConfig`

## Classes
—

## Functions
- `test_manual_calibration()` (L16): Test manual 4-point calibration with test image\n- `mouse_callback()` (L36)

## Intra-module calls (heuristic)
CalibrationManager, Path, ROIConfig, ROIProcessor, VideoCapture, append, circle, clear, copy, destroyAllWindows, enumerate, get_homography, imshow, insert, len, line, manual_calibration, namedWindow, ord, print, putText, range, read, release, setMouseCallback, set_homography_from_matrix, str, test_manual_calibration, waitKey, warp_roi

## Code
```python
"""
Test ROI Processor and Calibration Manager
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from src.calibration import ROIProcessor, CalibrationManager, ROIConfig


def test_manual_calibration():
    """Test manual 4-point calibration with test image"""

    # Load test video first frame
    video_path = 0
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to load test video")
        return

    print("🎯 Manual Calibration Test")
    print("Click 4 corners of dartboard: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
    print("Press 'r' to reset, 'c' to calibrate, 'q' to quit")

    # Point collection
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")

    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', mouse_callback)

    while True:
        display = frame.copy()

        # Draw collected points
        for i, pt in enumerate(points):
            cv2.circle(display, pt, 5, (0, 255, 0), -1)
            cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw lines between points
        if len(points) > 1:
            for i in range(len(points)):
                cv2.line(display, points[i], points[(i + 1) % len(points)],
                         (0, 255, 0), 2)

        # Instructions
        cv2.putText(display, f"Points: {len(points)}/4", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Calibration', display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            points.clear()
            print("Points reset")
        elif key == ord('c') and len(points) == 4:
            # Perform calibration
            calib_manager = CalibrationManager()
            success = calib_manager.manual_calibration(frame, points)

            if success:
                print("✅ Calibration successful!")

                # Test ROI warping
                roi_processor = ROIProcessor(ROIConfig(roi_size=(400, 400)))
                homography = calib_manager.get_homography()
                roi_processor.set_homography_from_matrix(homography)

                warped = roi_processor.warp_roi(frame)

                cv2.imshow('Warped ROI', warped)
                print("Press any key to continue...")
                cv2.waitKey(0)
            else:
                print("❌ Calibration failed")

            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_manual_calibration()
```
