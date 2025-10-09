"""
Simple ArUco-based calibration (no ChArUco needed)
Uses 4 ArUco markers at dartboard corners
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class ArucoCalibrator:
    """Simple ArUco marker calibration (4 markers at corners)"""

    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

    def detect_markers(self, frame: np.ndarray) -> Tuple[bool, List]:
        """Detect ArUco markers in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is None or len(ids) < 4:
            logger.warning(f"Need 4 markers, found {len(ids) if ids is not None else 0}")
            return False, []

        # Sort markers by ID
        marker_data = [(ids[i][0], corners[i][0]) for i in range(len(ids))]
        marker_data.sort(key=lambda x: x[0])

        logger.info(f"Detected {len(marker_data)} markers: {[m[0] for m in marker_data]}")
        return True, marker_data

    def draw_markers(self, frame: np.ndarray, markers: List) -> np.ndarray:
        """Draw detected markers"""
        display = frame.copy()

        for marker_id, corners in markers:
            # Draw marker outline
            corners_int = corners.astype(int)
            cv2.polylines(display, [corners_int], True, (0, 255, 0), 2)

            # Draw marker ID
            center = corners.mean(axis=0).astype(int)
            cv2.putText(display, f"ID:{marker_id}", tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return display