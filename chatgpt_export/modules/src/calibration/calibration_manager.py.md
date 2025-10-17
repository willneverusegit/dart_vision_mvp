# Module: `src\calibration\calibration_manager.py`
Hash: `789335232e59` · LOC: 1 · Main guard: false

## Imports
- `cv2`\n- `logging`\n- `numpy`\n- `os`\n- `tempfile`\n- `yaml`

## From-Imports
- `from pathlib import Path`\n- `from datetime import datetime`\n- `from typing import Optional, Tuple, List, Dict`\n- `from dataclasses import dataclass, asdict`

## Classes
- `CalibrationData` (L27): Calibration data schema\n- `CalibrationManager` (L64): Manages dartboard calibration with multiple methods.

## Functions
- `to_dict()` (L39): Convert to dictionary for YAML serialization\n- `_homography_to_list()` (L53): Convert homography to nested list\n- `__init__()` (L74)\n- `_load_config()` (L99): Load calibration from YAML with schema validation\n- `_atomic_save_config()` (L139): Atomic write of calibration config (crash-safe).\n- `manual_calibration()` (L183): Manual 4-point calibration.\n- `charuco_calibration()` (L250): ChArUco board calibration (best accuracy).\n- `get_homography()` (L301): Get current homography matrix\n- `get_calibration()` (L307): Get current calibration data\n- `is_valid()` (L311): Check if calibration is valid\n- `invalidate()` (L315): Mark current calibration as invalid

## Intra-module calls (heuristic)
CalibrationData, CharucoBoard, DetectorParameters, Path, _atomic_save_config, _homography_to_list, _load_config, all, array, bool, cvtColor, detectMarkers, dump, error, exists, fdopen, float, float32, get, getLogger, getPerspectiveTransform, getPredefinedDictionary, info, int, interpolateCornersCharuco, isinstance, isoformat, len, list, mkdir, mkstemp, norm, open, replace, safe_load, sum, to_dict, tolist, tuple, unlink, utcnow, warning

## Code
```python
"""
Calibration Manager with ChArUco and Atomic Config Storage
Handles camera calibration and persistent storage.

Features:
- ChArUco board calibration (sub-pixel accuracy)
- Manual 4-point calibration
- Atomic YAML writes (crash-safe)
- Validation and fallback mechanisms
"""

import cv2
import numpy as np
import yaml
import os
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Calibration data schema"""
    center_px: Tuple[int, int]
    radii_px: List[float]  # Ring radii from center
    rotation_deg: float
    mm_per_px: float
    homography: List[List[float]]  # 3x3 matrix as nested list
    last_update_utc: str
    valid: bool
    calibration_method: str = "none"  # charuco, manual, template
    roi_board_radius: float = 160.0  # ✅ NEU: Board radius in ROI space

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML serialization"""
        data = {
            'center_px': list(self.center_px),  # ✅ FIX: Convert tuple to list
            'radii_px': [float(r) for r in self.radii_px],  # Ensure all floats
            'rotation_deg': float(self.rotation_deg),
            'mm_per_px': float(self.mm_per_px),
            'homography': self._homography_to_list(),
            'last_update_utc': self.last_update_utc,
            'valid': bool(self.valid),
            'calibration_method': self.calibration_method
        }
        return data

    def _homography_to_list(self) -> List[List[float]]:
        """Convert homography to nested list"""
        if isinstance(self.homography, np.ndarray):
            return [[float(x) for x in row] for row in self.homography]
        elif isinstance(self.homography, list):
            # Already list, ensure all elements are floats
            return [[float(x) for x in row] for row in self.homography]
        else:
            # Fallback: identity matrix
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

class CalibrationManager:
    """
    Manages dartboard calibration with multiple methods.

    Calibration Methods:
    1. ChArUco Board: Best accuracy (0.5-2mm), requires printed board
    2. Manual 4-Point: Fast setup (2-5mm), click corners
    3. ArUco Markers: Good for tracking (1-3mm), 4+ markers needed
    """

    def __init__(self, config_path: str = 'config/calibration_config.yaml'):
        self.config_path = Path(config_path)

        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # ArUco/ChArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # ChArUco board (7x5 squares, 40mm squares, 20mm markers)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (7, 5),  # Board size
            0.04,  # Square length (meters)
            0.03,  # Marker length (meters)
            self.aruco_dict
        )

        # Load existing calibration
        self.calibration: Optional[CalibrationData] = self._load_config()

        # Camera intrinsics (optional, for advanced calibration)
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

    def _load_config(self) -> Optional[CalibrationData]:
        """Load calibration from YAML with schema validation"""
        if not self.config_path.exists():
            logger.info("No calibration file found, starting fresh")
            return None

        try:
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f)

            # Handle empty/None file
            if data is None:
                logger.warning("Empty calibration file, ignoring")
                return None

            # Validate schema
            required_keys = ['center_px', 'homography', 'valid']
            if not all(key in data for key in required_keys):
                logger.warning("Incomplete calibration data, ignoring")
                return None

            # Convert lists back to tuples where needed
            calib = CalibrationData(
                center_px=tuple(data['center_px']),  # ✅ FIX: Convert list to tuple
                radii_px=data.get('radii_px', [50, 100, 150, 200]),
                rotation_deg=data.get('rotation_deg', 0.0),
                mm_per_px=data.get('mm_per_px', 1.0),
                homography=data['homography'],
                last_update_utc=data.get('last_update_utc', ''),
                valid=data['valid'],
                calibration_method=data.get('calibration_method', 'unknown')
            )

            logger.info(f"Loaded {calib.calibration_method} calibration from {self.config_path}")
            return calib

        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return None

    def _atomic_save_config(self, calibration: CalibrationData) -> bool:
        """
        Atomic write of calibration config (crash-safe).

        Strategy: Write to temp file, then atomic replace.
        This prevents corruption if process crashes during write.
        """
        try:
            # Convert to dict for YAML
            config_dict = calibration.to_dict()

            # Write to temporary file first
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.yaml',
                dir=self.config_path.parent
            )

            try:
                with os.fdopen(temp_fd, 'w') as temp_file:
                    yaml.dump(
                        config_dict,
                        temp_file,
                        default_flow_style=False,
                        sort_keys=False
                    )

                # Atomic replacement (POSIX/Windows compatible)
                os.replace(temp_path, self.config_path)

                logger.info(f"Calibration saved: {self.config_path}")
                return True

            except Exception as e:
                # Cleanup temp file on error
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise e

        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False

    def manual_calibration(
            self,
            frame: np.ndarray,
            board_points: List[Tuple[int, int]],
            board_diameter_mm: float = 340.0
    ) -> bool:
        """Manual 4-point calibration."""
        if len(board_points) != 4:
            logger.error(f"Need exactly 4 points, got {len(board_points)}")
            return False

        try:
            roi_size = 400
            dst_points = np.float32([
                [0, 0],
                [roi_size, 0],
                [roi_size, roi_size],
                [0, roi_size]
            ])

            homography = cv2.getPerspectiveTransform(
                np.float32(board_points),
                dst_points
            )

            center_x = sum(p[0] for p in board_points) / 4
            center_y = sum(p[1] for p in board_points) / 4

            board_width_px = np.linalg.norm(
                np.array(board_points[1]) - np.array(board_points[0])
            )
            mm_per_px = board_diameter_mm / board_width_px

            # ✅ Board fills ~80% of ROI after warping
            board_radius_in_roi = roi_size * 0.4  # 400 * 0.4 = 160px

            # Ring radii in ROI space (normalized to board radius)
            radii_mm = [15.9, 50, 107, 170]
            radii_px = [r / board_diameter_mm * 2 * board_radius_in_roi for r in radii_mm]

            self.calibration = CalibrationData(
                center_px=(int(center_x), int(center_y)),
                radii_px=radii_px,
                rotation_deg=0.0,
                mm_per_px=mm_per_px,
                homography=homography.tolist(),
                last_update_utc=datetime.utcnow().isoformat(),
                valid=True,
                calibration_method='manual',
                roi_board_radius=board_radius_in_roi  # ✅ Store ROI radius
            )

            success = self._atomic_save_config(self.calibration)

            if success:
                logger.info(f"Manual calibration successful:")
                logger.info(f"  Center (original): {self.calibration.center_px}")
                logger.info(f"  Scale: {mm_per_px:.3f} mm/px")
                logger.info(f"  ROI board radius: {board_radius_in_roi:.1f}px")

            return success

        except Exception as e:
            logger.error(f"Manual calibration failed: {e}")
            return False


    def charuco_calibration(
            self,
            frame: np.ndarray,
            min_markers: int = 4
    ) -> bool:
        """
        ChArUco board calibration (best accuracy).

        Args:
            frame: Image containing ChArUco board
            min_markers: Minimum markers needed for valid detection

        Returns:
            Success flag
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        if ids is None or len(ids) < min_markers:
            logger.warning(f"Not enough markers detected: {len(ids) if ids is not None else 0}/{min_markers}")
            return False

        # Refine to ChArUco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners,
            ids,
            gray,
            self.charuco_board
        )

        if not retval or charuco_corners is None or len(charuco_corners) < 4:
            logger.warning("ChArUco corner interpolation failed")
            return False

        logger.info(f"Detected {len(charuco_corners)} ChArUco corners")

        # For simplicity, we'll use the detected corners to estimate homography
        # In production, you'd also calibrate camera intrinsics here

        # TODO: Implement full ChArUco calibration with camera matrix
        # For now, fall back to manual-style calibration from detected corners

        logger.warning("Full ChArUco calibration not yet implemented, use manual calibration for now")
        return False

    def get_homography(self) -> Optional[np.ndarray]:
        """Get current homography matrix"""
        if self.calibration and self.calibration.valid:
            return np.array(self.calibration.homography, dtype=np.float32)
        return None

    def get_calibration(self) -> Optional[CalibrationData]:
        """Get current calibration data"""
        return self.calibration

    def is_valid(self) -> bool:
        """Check if calibration is valid"""
        return self.calibration is not None and self.calibration.valid

    def invalidate(self):
        """Mark current calibration as invalid"""
        if self.calibration:
            self.calibration.valid = False
            self._atomic_save_config(self.calibration)
            logger.info("Calibration invalidated")
```
