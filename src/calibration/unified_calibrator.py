# unified_calibrator.py
# Python 3.10+ | pip install opencv-contrib-python pyyaml
"""
Unified calibration with prioritized ChArUco (real multi-frame calibration),
then ArUco, then Manual fallback. OpenCV 4.5–4.12+ compatible.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -------------------- Data structures --------------------
class CalibrationMethod(Enum):
    CHARUCO = "charuco"
    ARUCO = "aruco"
    MANUAL = "manual"
    UNKNOWN = "unknown"


@dataclass
class CalibrationResult:
    success: bool
    method: CalibrationMethod
    homography: Optional[np.ndarray]
    camera_matrix: Optional[np.ndarray]
    dist_coeffs: Optional[np.ndarray]
    center_px: Tuple[int, int]
    roi_board_radius: float
    mm_per_px: float
    message: str
    rms: float = 0.0  # RMS reprojection error if available
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None


# -------------------- Unified calibrator --------------------
class UnifiedCalibrator:
    """
    Unified calibration with automatic fallback chain:
    1) ChArUco (multi-frame, highest accuracy, pose + intrinsics)
    2) ArUco (single-frame homography)
    3) Manual 4-point homography
    """

    def __init__(
        self,
        squares_x: int = 5,
        squares_y: int = 7,
        square_length_m: float = 0.04,  # 40 mm
        marker_length_m: float = 0.03,  # 30 mm
        dict_type: int = cv2.aruco.DICT_6X6_250,
        min_charuco_samples: int = 8,
        board_diameter_mm: float = 340.0
    ):
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length_m
        self.marker_length = marker_length_m
        self.min_charuco_samples = min_charuco_samples
        self.board_diameter_mm = board_diameter_mm

        # --- Dictionary ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)

        # --- Charuco board (compatibility bridge) ---
        self.board = self._make_charuco_board(
            self.squares_x, self.squares_y, self.square_length, self.marker_length, self.aruco_dict
        )

        # --- Detector parameters (tuned for webcams, indoor light) ---
        self.det_params = self._create_detector_params()

        # --- Detectors (new API preferred, legacy fallback) ---
        self._has_new = hasattr(cv2.aruco, "ArucoDetector")
        self._aruco = cv2.aruco.ArucoDetector(self.aruco_dict, self.det_params) if self._has_new else None
        self._charuco = cv2.aruco.CharucoDetector(self.board) if hasattr(cv2.aruco, "CharucoDetector") else None

        # --- Calibration buffers/state for ChArUco ---
        self._samples_corners: List[np.ndarray] = []
        self._samples_ids: List[np.ndarray] = []
        self._image_size: Optional[Tuple[int, int]] = None
        self.K: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None
        self._rms: float = 0.0

        logger.info(
            f"UnifiedCalibrator ready | Board {squares_x}x{squares_y}, sq={square_length_m*1000:.0f}mm, "
            f"mk={marker_length_m*1000:.0f}mm | API new={self._has_new}"
        )

    # ---------- API/Board helpers ----------
    @staticmethod
    def _make_charuco_board(sx, sy, sq_len, mk_len, dictionary):
        # Prefer factory; fall back to constructor depending on build
        if hasattr(cv2.aruco, "CharucoBoard_create"):
            return cv2.aruco.CharucoBoard_create(
                squaresX=sx, squaresY=sy,
                squareLength=sq_len, markerLength=mk_len,
                dictionary=dictionary
            )
        else:
            return cv2.aruco.CharucoBoard(
                (sx, sy), sq_len, mk_len, dictionary
            )

    @staticmethod
    def _create_detector_params() -> "cv2.aruco.DetectorParameters":
        p = cv2.aruco.DetectorParameters()
        # Adaptive thresholding
        p.adaptiveThreshConstant = 7
        p.adaptiveThreshWinSizeMin = 5
        p.adaptiveThreshWinSizeMax = 35
        p.adaptiveThreshWinSizeStep = 10
        # Geometry/size
        p.minMarkerPerimeterRate = 0.02
        p.maxMarkerPerimeterRate = 4.0
        p.polygonalApproxAccuracyRate = 0.03
        p.minCornerDistanceRate = 0.02
        # Refinement
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 5
        p.cornerRefinementMaxIterations = 50
        return p

    def _detect_markers(self, gray):
        """
        Thin wrapper to unify ArUco marker detection across API variants.
        Returns: (corners, ids, rejected)
        """
        if hasattr(cv2.aruco, "ArucoDetector") and self._aruco is not None:
            return self._aruco.detectMarkers(gray)
        # Legacy fallback
        return cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.det_params)

    def tune_params_for_resolution(self, width: int, height: int) -> "cv2.aruco.DetectorParameters":
        """
        Final, field-tested tuning for OpenCV 4.12 ArUco/ChArUco detection.
        Optimized for bright scenes (white squares, strong lighting) and 1080p+,
        but degrades gracefully to 720p.

        Args:
            width, height: actual capture resolution of the first valid frame.

        Returns:
            cv2.aruco.DetectorParameters with tuned values.
        """
        p = cv2.aruco.DetectorParameters()
        short_side = min(width, height)

        # --- Adaptive thresholding (robust under bright/overexposed white squares) ---
        p.adaptiveThreshWinSizeMin = 5
        p.adaptiveThreshWinSizeMax = 45
        p.adaptiveThreshWinSizeStep = 10
        # Lower constant → more sensitive to faint marker borders on bright paper
        p.adaptiveThreshConstant = 5

        # --- Marker size model (relative perimeter to image size) ---
        if short_side < 900:  # ~720p
            p.minMarkerPerimeterRate = 0.012  # allow reasonably small markers
            p.maxMarkerPerimeterRate = 5.0
        else:  # 1080p+
            p.minMarkerPerimeterRate = 0.008  # allow even smaller markers
            p.maxMarkerPerimeterRate = 6.0

        # --- Geometry / spacing ---
        p.polygonalApproxAccuracyRate = 0.03
        p.minCornerDistanceRate = 0.05
        p.minDistanceToBorder = 3  # avoid border artefacts on tight frames
        # Keep default markerBorderBits=1 (explicit for clarity)
        p.markerBorderBits = 1

        # --- Subpixel corner refinement (sharper corners on bright prints) ---
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 5
        p.cornerRefinementMaxIterations = 60
        p.cornerRefinementMinAccuracy = 0.1

        # --- Robustness knobs ---
        p.minOtsuStdDev = 2.0  # tolerate flatter histograms
        p.detectInvertedMarker = True  # safety if contrast flips locally

        return p

    def set_detector_params(self, params: "cv2.aruco.DetectorParameters"):
        """Apply externally tuned params and rebuild ArUco detector."""
        self.det_params = params
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._aruco = cv2.aruco.ArucoDetector(self.aruco_dict, self.det_params)
        # CharucoDetector bleibt wie ist

    def _interpolate_charuco(self, gray, corners, ids):
        if ids is None or len(ids) == 0:
            return None, None

        if hasattr(cv2.aruco, "CharucoDetector") and self._charuco is not None:
            try:
                result = self._charuco.detectBoard(
                    gray,
                    markerCorners=corners,
                    markerIds=ids
                )
                if isinstance(result, tuple):
                    if len(result) == 3:
                        _, ch_corners, ch_ids = result
                    elif len(result) == 2:
                        ch_corners, ch_ids = result
                    else:
                        ch_corners, ch_ids = None, None
                else:
                    ch_corners, ch_ids = None, None

                if ch_corners is None or ch_ids is None or len(ch_ids) < 4:
                    return None, None
                return ch_corners, ch_ids
            except Exception as e:
                print(f"[WARN] CharucoDetector.detectBoard failed (new API path): {e}")

        # Legacy fallback
        try:
            ok, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners, markerIds=ids, image=gray, board=self.board
            )
            if not ok or ch_corners is None or ch_ids is None or len(ch_ids) < 4:
                return None, None
            return ch_corners, ch_ids
        except Exception as e:
            print(f"[WARN] interpolateCornersCharuco fallback failed: {e}")
            return None, None

    def detect_charuco(self, frame):
        """Return (marker_corners, marker_ids, charuco_corners, charuco_ids) or (None,..) if not found."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        mk_c, mk_ids, _ = self._detect_markers(gray)
        ch_c = ch_ids = None
        if mk_ids is not None and len(mk_ids) > 0:
            ch_c, ch_ids = self._interpolate_charuco(gray, mk_c, mk_ids)
        return mk_c, mk_ids, ch_c, ch_ids

    # ---------- ChArUco sampling/calibration ----------
    def add_charuco_sample(self, frame) -> bool:
        """Collect one valid ChArUco sample for later calibration."""
        mk_c, mk_ids, ch_c, ch_ids = self.detect_charuco(frame)
        if ch_ids is None or len(ch_ids) < 4:
            return False
        if self._image_size is None:
            self._image_size = (frame.shape[1], frame.shape[0])
        self._samples_corners.append(ch_c)
        self._samples_ids.append(ch_ids)
        logger.info(f"[Charuco] Sample #{len(self._samples_corners)} collected ({len(ch_ids)} corners)")
        return True

    def calibrate_charuco(self) -> float:
        """Run full ChArUco calibration from collected samples. Returns RMS."""
        if len(self._samples_corners) < self.min_charuco_samples:
            raise RuntimeError(f"Need >={self.min_charuco_samples} ChArUco samples, have {len(self._samples_corners)}.")
        flags = (cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_CB_FAST_CHECK)
        rms, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=self._samples_corners,
            charucoIds=self._samples_ids,
            board=self.board,
            imageSize=self._image_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=flags
        )
        self.K, self.D, self._rms = K, D, float(rms)
        logger.info(f"[Charuco] Calibration done | RMS={self._rms:.4f}")
        return self._rms

    def estimate_pose_charuco(self, frame):
        """Return (ok, rvec, tvec) using current charuco calibration."""
        if self.K is None or self.D is None:
            return False, None, None
        _, _, ch_c, ch_ids = self.detect_charuco(frame)
        if ch_ids is None or len(ch_ids) < 4:
            return False, None, None
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(ch_c, ch_ids, self.board, self.K, self.D)
        return bool(ok), rvec, tvec

    def reset_charuco_samples(self):
        self._samples_corners.clear()
        self._samples_ids.clear()
        self._image_size = None
        self._rms = 0.0

    # ---------- ArUco detection for fallback ----------
    def _detect_aruco_markers(self, frame) -> Tuple[bool, Optional[List]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self._detect_markers(gray)
        if ids is None or len(ids) < 4:
            return False, None
        marker_data = [(int(ids[i][0]), corners[i][0]) for i in range(len(ids))]
        marker_data.sort(key=lambda x: x[0])
        return True, marker_data

    # ---------- Homography helpers ----------
    @staticmethod
    def _homography_and_metrics(src_points: np.ndarray, roi_size: int, board_diameter_mm: float):
        dst_points = np.float32([[0, 0], [roi_size, 0], [roi_size, roi_size], [0, roi_size]])
        H = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points)

        center_x = float(np.mean(src_points[:, 0]))
        center_y = float(np.mean(src_points[:, 1]))
        board_width_px = float(np.linalg.norm(src_points[1] - src_points[0]))
        mm_per_px = board_diameter_mm / max(board_width_px, 1e-6)
        roi_board_radius = roi_size * 0.4
        return H, (int(center_x), int(center_y)), roi_board_radius, mm_per_px

    # ---------- Automatic chain ----------
    def calibrate_auto(
        self,
        frame: np.ndarray,
        dartboard_points: Optional[List[Tuple[int, int]]] = None,
        roi_size: int = 400
    ) -> CalibrationResult:
        """
        1) If ChArUco calibration exists (K,D), return pose + (optional) homography.
        2) Else try ArUco homography.
        3) Else try Manual homography (needs 4 points).
        """
        # 1) ChArUco (preferred)
        if self.K is not None and self.D is not None:
            ok, rvec, tvec = self.estimate_pose_charuco(frame)
            msg = "[Charuco] Pose OK" if ok else "[Charuco] No pose (not enough corners)"
            H = None
            if dartboard_points and len(dartboard_points) == 4:
                H, center, roi_r, mmpp = self._homography_and_metrics(
                    np.float32(dartboard_points), roi_size, self.board_diameter_mm
                )
            else:
                # center/mmpp fallback from pose distance only
                center = (frame.shape[1] // 2, frame.shape[0] // 2)
                roi_r = roi_size * 0.4
                mmpp = 1.0
            return CalibrationResult(
                success=True,
                method=CalibrationMethod.CHARUCO,
                homography=H,
                camera_matrix=self.K,
                dist_coeffs=self.D,
                center_px=center,
                roi_board_radius=roi_r,
                mm_per_px=mmpp,
                message=f"{msg} | RMS={self._rms:.4f}",
                rms=self._rms,
                rvec=rvec, tvec=tvec
            )

        # 2) ArUco (homography)
        ok_mk, markers = self._detect_aruco_markers(frame)
        if ok_mk:
            if dartboard_points and len(dartboard_points) == 4:
                board_corners = np.float32(dartboard_points)
            else:
                # Use first 4 marker centers
                board_corners = np.float32(
                    [markers[i][1].mean(axis=0) for i in range(min(4, len(markers)))]
                )
            H, center, roi_r, mmpp = self._homography_and_metrics(board_corners, roi_size, self.board_diameter_mm)
            return CalibrationResult(
                success=True,
                method=CalibrationMethod.ARUCO,
                homography=H,
                camera_matrix=None,
                dist_coeffs=None,
                center_px=center,
                roi_board_radius=roi_r,
                mm_per_px=mmpp,
                message=f"[Aruco] {len(markers)} markers → homography"
            )

        # 3) Manual (needs explicit 4 points)
        if dartboard_points and len(dartboard_points) == 4:
            H, center, roi_r, mmpp = self._homography_and_metrics(
                np.float32(dartboard_points), roi_size, self.board_diameter_mm
            )
            return CalibrationResult(
                success=True,
                method=CalibrationMethod.MANUAL,
                homography=H,
                camera_matrix=None,
                dist_coeffs=None,
                center_px=center,
                roi_board_radius=roi_r,
                mm_per_px=mmpp,
                message="[Manual] 4 points → homography"
            )

        # All failed
        return CalibrationResult(
            success=False,
            method=CalibrationMethod.UNKNOWN,
            homography=None,
            camera_matrix=None,
            dist_coeffs=None,
            center_px=(0, 0),
            roi_board_radius=160.0,
            mm_per_px=1.0,
            message="No calibration possible (collect ChArUco samples or provide 4 points)"
        )

    # ---------- Visualization ----------
    @staticmethod
    def draw_overlays(frame, marker_corners=None, marker_ids=None,
                      charuco_corners=None, charuco_ids=None,
                      K=None, D=None, rvec=None, tvec=None, axis_len=0.08):
        img = frame.copy()
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
        if charuco_ids is not None and len(charuco_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids, (0, 255, 0))
        if K is not None and D is not None and rvec is not None and tvec is not None:
            cv2.drawFrameAxes(img, K, D, rvec, tvec, axis_len)
        return img

    def to_yaml_dict(self) -> dict:
        """
        Produce a YAML-friendly dict for a ChArUco calibration result.
        Only includes intrinsics if available.
        """
        out = {
            "type": "charuco",
            "board": {
                "squares_x": self.squares_x,
                "squares_y": self.squares_y,
                "square_length_m": float(self.square_length),
                "marker_length_m": float(self.marker_length),
                "dictionary": int(self.aruco_dict),
            },
        }
        if getattr(self, "camera_matrix", None) is not None:
            out["camera"] = {
                "matrix": self.K,
                "dist_coeffs": self.D if getattr(self, "dist_coeffs", None) is not None else None,
                "rms_px": float(getattr(self, "last_rms", -1.0)),
                "image_size": list(getattr(self, "last_image_size", (0, 0))),
            }
        # Optional homography if you computed one (manual 4-corner)
        if getattr(self, "H", None) is not None:
            out["homography"] = {"H": self.H}
        return out
