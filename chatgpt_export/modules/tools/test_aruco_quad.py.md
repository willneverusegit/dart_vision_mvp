# Module: `tools\test_aruco_quad.py`
Hash: `37346e566794` · LOC: 1 · Main guard: false

## Imports
- `argparse`\n- `cv2`\n- `numpy`

## From-Imports
- `from src.calibration.aruco_quad_calibrator import ArucoQuadCalibrator`

## Classes
—

## Functions
—

## Intra-module calls (heuristic)
ArgumentParser, ArucoQuadCalibrator, VideoCapture, _centers_from_corners, add_argument, calibrate_from_frame, destroyAllWindows, detect, draw_debug, imshow, len, ord, parse_args, putText, read, release, set, waitKey

## Code
```python
# tools/test_aruco_quad.py
import cv2, argparse
import numpy as np
from src.calibration.aruco_quad_calibrator import ArucoQuadCalibrator

parser = argparse.ArgumentParser()
parser.add_argument("--webcam", "-w", type=int, default=0)
parser.add_argument("--dict", type=str, default="4X4_50")   # e.g. 4X4_50, 6X6_250
parser.add_argument("--roi", type=int, default=400)
parser.add_argument("--ids", type=int, nargs="*", default=None, help="expected 4 ids, e.g., --ids 0 1 2 3")
parser.add_argument("--width-mm", type=float, default=None)
parser.add_argument("--height-mm", type=float, default=None)
args = parser.parse_args()

DICT_MAP = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "6X6_250": cv2.aruco.DICT_6X6_250,
}

cal = ArucoQuadCalibrator(dict_name=DICT_MAP[args.dict], roi_size=args.roi, expected_ids=args.ids, debug=True)
cap = cv2.VideoCapture(args.webcam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    corners, ids = cal.detect(frame)
    centers = cal._centers_from_corners(corners) if ids is not None else None

    okH, H, mmpp, info = cal.calibrate_from_frame(
        frame, rect_width_mm=args.width_mm, rect_height_mm=args.height_mm
    )
    disp = cal.draw_debug(frame, corners, ids, centers if centers is not None else None, H if okH else None)

    txt = f"markers={0 if ids is None else len(ids)}"
    if okH: txt += f" | H ok | mm/px={mmpp:.4f}" if mmpp is not None else " | H ok"
    cv2.putText(disp, txt, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0) if okH else (0,0,255), 2)
    cv2.imshow("ArucoQuad", disp)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()

```
