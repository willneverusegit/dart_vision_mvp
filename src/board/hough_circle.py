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
