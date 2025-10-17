# Module: `src\board\hough_circle.py`
Hash: `b8dbe99a68ea` · LOC: 1 · Main guard: false

## Imports
- `cv2`\n- `numpy`

## From-Imports
- `from __future__ import annotations`\n- `from typing import Optional, Tuple`

## Classes
—

## Functions
- `find_board_circle()` (L7): Return (cx, cy, r) for the strongest circle or None.

## Intra-module calls (heuristic)
HoughCircles, ValueError, around, int, medianBlur, uint16

## Code
```python
# Hough circle detection for board center/radius estimation
from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple

def find_board_circle(gray: np.ndarray,
                      dp: float = 1.2,
                      min_dist: float = 100.0,
                      param1: float = 150.0,
                      param2: float = 60.0,
                      min_radius: int = 100,
                      max_radius: int = 0) -> Optional[Tuple[int,int,int]]:
    """Return (cx, cy, r) for the strongest circle or None.
    gray: uint8 image (pre-blur recommended).
    """
    if gray.ndim != 2:
        raise ValueError("gray image expected")
    # Mild blur to stabilize edges
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))
    # take first/strongest
    x, y, r = circles[0][0]
    return int(x), int(y), int(r)

```
