# -*- coding: utf-8 -*-
"""
ArucoQuadCalibrator
Detect 4 separate ArUco markers (e.g., DICT_4X4_50) placed at the four corners of a rectangle.
Compute a homography (image -> normalized ROI) from their centers, and optionally a mm/px scale
if the real-world rectangle size (width_mm, height_mm) is known.

OpenCV 4.12-ready (ArucoDetector) with legacy fallback.

Usage:
    cal = ArucoQuadCalibrator(dict_name=cv2.aruco.DICT_4X4_50, roi_size=400, expected_ids=[0,1,2,3])
    ok, H, mm_per_px, info = cal.calibrate_from_frame(frame, width_mm=600, height_mm=600)
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict

class ArucoQuadCalibrator:
    def __init__(
        self,
        dict_name: int = cv2.aruco.DICT_4X4_50,
        roi_size: int = 400,
        expected_ids: Optional[List[int]] = None,  # e.g. [0, 1, 2, 3] if you want to enforce IDs
        debug: bool = False
    ):
        self.roi_size = int(roi_size)
        self.debug = debug

        # Dictionary & detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)
        self.det_params = cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.det_params)
        else:
            self.detector = None  # legacy path will use cv2.aruco.detectMarkers

        # Optional list of expected IDs
        self.expected_ids = expected_ids

    # ----------------- Detection -----------------
    def detect(self, frame: np.ndarray) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Return marker corners and ids using new API if available."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.det_params)
        return corners, ids

    @staticmethod
    def _centers_from_corners(corners: List[np.ndarray]) -> np.ndarray:
        """Compute centers (x,y) from list of 4x1x2 corner arrays."""
        ctrs = []
        for c in corners:
            ctr = c[0].mean(axis=0)  # average of the 4 corners
            ctrs.append(ctr)
        return np.asarray(ctrs, dtype=np.float32)  # shape (N,2)

    # ----------------- Geometry helpers -----------------
    @staticmethod
    def _order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
        """
        Order 4 points as TL, TR, BR, BL (standard CV convention).
        pts: (4,2) float32
        """
        pts = pts.astype(np.float32)
        s = pts.sum(axis=1)             # x + y
        d = np.diff(pts, axis=1).ravel()  # x - y

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def estimate_homography_from_centers(self, centers: np.ndarray) -> np.ndarray:
        """
        centers: (4,2) unsorted marker centers.
        Returns 3x3 homography mapping image -> ROI, where ROI corners are:
        (0,0)=TL, (W-1,0)=TR, (W-1,W-1)=BR, (0,W-1)=BL  (square ROI of roi_size)
        """
        assert centers.shape == (4, 2), "need exactly 4 centers"
        src = self._order_points_tl_tr_br_bl(centers)
        W = self.roi_size
        dst = np.array([[0, 0], [W-1, 0], [W-1, W-1], [0, W-1]], dtype=np.float32)
        H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        return H

    @staticmethod
    def _axis_aligned_box(pts: np.ndarray) -> Tuple[float, float]:
        """Return width_px, height_px of the axis-aligned box enclosing points TL,TR,BR,BL."""
        # pts must be ordered TL,TR,BR,BL
        tl, tr, br, bl = pts
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        width_px = 0.5 * (width_top + width_bottom)
        height_px = 0.5 * (height_left + height_right)
        return width_px, height_px

    def estimate_scale_mm_per_px(
        self, centers: np.ndarray, width_mm: float, height_mm: float
    ) -> float:
        """
        Compute mm/px using the average of horizontal and vertical scales.
        Assumes the 4 marker centers form the corners of a physical rectangle of given size.
        """
        ordered = self._order_points_tl_tr_br_bl(centers)
        width_px, height_px = self._axis_aligned_box(ordered)
        sx = width_mm / max(width_px, 1e-6)
        sy = height_mm / max(height_px, 1e-6)
        return float(0.5 * (sx + sy))

    # ----------------- End-to-end -----------------
    def calibrate_from_frame(
        self,
        frame: np.ndarray,
        rect_width_mm: Optional[float] = None,
        rect_height_mm: Optional[float] = None,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[float], Dict]:
        """
        Detect markers, compute homography from their centers, and (optionally) mm/px.
        Returns: (ok, H, mm_per_px, info)
        """
        info: Dict = {"reason": None, "markers": 0, "ids": None}

        corners, ids = self.detect(frame)
        if ids is None or len(ids) < 4:
            info["reason"] = f"Need >=4 markers, got {0 if ids is None else len(ids)}"
            info["markers"] = 0 if ids is None else int(len(ids))
            return False, None, None, info

        # Optionally filter/validate expected IDs
        if self.expected_ids:
            id_list = ids.flatten().tolist()
            missing = [eid for eid in self.expected_ids if eid not in id_list]
            if missing:
                info["reason"] = f"Missing expected IDs: {missing}"
                info["markers"] = len(id_list)
                info["ids"] = id_list
                # Continue anyway if >=4 present (we can still compute H)
            # Optionally, we could filter to only the expected IDs:
            # keep = [i for i, idv in enumerate(id_list) if idv in self.expected_ids]
            # corners = [corners[i] for i in keep]; ids = ids[keep]

        # Take the 4 strongest markers (by area) if more than 4 are visible
        areas = [cv2.contourArea(c.astype(np.float32)) for c in [cn[0] for cn in corners]]
        idx_sorted = np.argsort(areas)[::-1].tolist()
        use_idx = idx_sorted[:4] if len(idx_sorted) >= 4 else idx_sorted

        sel_corners = [corners[i] for i in use_idx]
        sel_ids = ids[use_idx].flatten()
        centers = self._centers_from_corners(sel_corners)

        if centers.shape[0] != 4:
            info["reason"] = f"Could not select 4 centers (got {centers.shape[0]})"
            info["markers"] = len(ids)
            info["ids"] = sel_ids.tolist()
            return False, None, None, info

        try:
            H = self.estimate_homography_from_centers(centers)
        except Exception as e:
            info["reason"] = f"Homography failed: {e}"
            info["markers"] = len(ids)
            info["ids"] = sel_ids.tolist()
            return False, None, None, info

        mm_per_px = None
        if rect_width_mm is not None and rect_height_mm is not None:
            try:
                mm_per_px = self.estimate_scale_mm_per_px(centers, rect_width_mm, rect_height_mm)
            except Exception as e:
                info["reason"] = f"Scale estimation failed: {e}"

        info.update({
            "markers": int(len(ids)),
            "ids": sel_ids.tolist(),
            "centers": centers.tolist()
        })
        return True, H, mm_per_px, info

    # ----------------- Debug drawing -----------------
    def draw_debug(
        self, frame: np.ndarray, corners: List[np.ndarray], ids: Optional[np.ndarray],
        centers: Optional[np.ndarray] = None, H: Optional[np.ndarray] = None
    ) -> np.ndarray:
        out = frame.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(out, corners, ids)
        if centers is not None and len(centers) == 4:
            for i, pt in enumerate(centers):
                cv2.circle(out, tuple(pt.astype(int)), 6, (0, 255, 255), -1)
                cv2.putText(out, f"C{i}", tuple((pt+np.array([8, -8])).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        if H is not None:
            # draw projected ROI square for sanity
            W = self.roi_size
            roi_pts = np.array([[0,0],[W-1,0],[W-1,W-1],[0,W-1]], dtype=np.float32).reshape(-1,1,2)
            invH = np.linalg.inv(H)
            img_pts = cv2.perspectiveTransform(roi_pts, invH)  # map ROI->image to draw
            img_pts = img_pts.reshape(-1,2).astype(int)
            for i in range(4):
                cv2.line(out, tuple(img_pts[i]), tuple(img_pts[(i+1)%4]), (0,255,0), 2)
        return out

    def to_yaml_dict(self, H=None, mm_per_px=None, rect_size_mm=None, used_ids=None) -> dict:
        return {
            "type": "aruco_quad",
            "aruco": {
                "dictionary": int(self.aruco_dict.bytesList.shape[0]),  # informational
                "expected_ids": used_ids if used_ids is not None else self.expected_ids,
            },
            "roi": {"size_px": int(self.roi_size)},
            "homography": {"H": H} if H is not None else None,
            "scale": {
                "mm_per_px": float(mm_per_px) if mm_per_px is not None else None,
                "rect_width_mm": rect_size_mm[0] if rect_size_mm else None,
                "rect_height_mm": rect_size_mm[1] if rect_size_mm else None,
            }
        }
