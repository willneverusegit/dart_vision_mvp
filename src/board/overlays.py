# Visual debug overlays for sectors and rings
from __future__ import annotations
import cv2
import numpy as np
import math
from typing import Tuple

from .board_mapping import BoardMapper

def draw_sector_labels(img: np.ndarray, mapper: BoardMapper, step_deg: float = 18.0) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    center = (int(mapper.calib.cx), int(mapper.calib.cy))
    r = int(mapper.calib.r_outer_double_px)
    # Draw sector centerlines and labels
    for i in range(20):
        theta_center = (i * step_deg)
        # Re-apply inverse of mapping to get screen angle
        # We construct a point on the circle at that relative angle
        theta = theta_center
        if mapper.cfg.angles.clockwise:
            theta = (360.0 - theta) % 360.0
        theta = (theta + mapper.cfg.angles.theta0_deg + mapper.calib.rotation_deg) % 360.0
        rad = math.radians(theta)
        x = int(center[0] + r * math.cos(rad))
        y = int(center[1] - r * math.sin(rad))
        cv2.line(out, center, (x, y), (0,255,0), 1)
        # Put label near outer edge
        label = str(mapper.cfg.sectors.order[i])
        lx = int(center[0] + int(0.85*r) * math.cos(rad))
        ly = int(center[1] - int(0.85*r) * math.sin(rad))
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out

def draw_ring_circles(img: np.ndarray, mapper: BoardMapper) -> np.ndarray:
    out = img.copy()
    center = (int(mapper.calib.cx), int(mapper.calib.cy))
    r_px = mapper.calib.r_outer_double_px
    rs = mapper.cfg.radii
    radii_norm = [rs.r_bull_inner, rs.r_bull_outer, rs.r_triple_inner, rs.r_triple_outer, rs.r_double_inner, rs.r_double_outer]
    for rn in radii_norm:
        cv2.circle(out, center, int(rn * r_px), (255,0,0), 1)
    return out

def annotate_hit(img: np.ndarray, mapper: BoardMapper, xy: Tuple[int,int]) -> np.ndarray:
    out = img.copy()
    ring, sec, label = mapper.score_from_hit(*xy)
    cv2.circle(out, (int(xy[0]), int(xy[1])), 5, (0,0,255), -1)
    cv2.putText(out, label, (int(xy[0])+8, int(xy[1])-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
    return out
