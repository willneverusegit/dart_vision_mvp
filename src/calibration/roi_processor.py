"""
ROI Processor with Homography and Polar Unwrap
Reduces processing load by isolating dartboard region.

Performance Impact: ~50% CPU reduction when combined with motion gating
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ROIConfig:
    """ROI processing configuration"""
    roi_size: Tuple[int, int] = (400, 400)  # Standard square dartboard view
    polar_enabled: bool = False
    polar_radius: int = 200
    interpolation: int = cv2.INTER_LINEAR  # High quality for display
    interpolation_fast: int = cv2.INTER_NEAREST  # Fast mode for motion detection (~2-3x faster)


class ROIProcessor:
    """
    Region of Interest processor for dartboard isolation.

    Key Features:
    - Perspective correction via homography
    - Optional polar coordinate unwrapping
    - CPU-efficient early data reduction
    - Identity transform fallback for robustness
    """

    def __init__(self, config: Optional[ROIConfig] = None):
        self.config = config or ROIConfig()

        # Transformation matrices
        self.homography: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None

        # Polar unwrap parameters
        self.polar_center: Optional[Tuple[int, int]] = None

        # Statistics
        self.transforms_applied = 0
        self.fallback_count = 0

    def set_homography(
            self,
            src_points: np.ndarray,
            dst_points: Optional[np.ndarray] = None
    ) -> bool:
        """
        Set homography transformation from source points.

        Args:
            src_points: 4 corner points in original image (clockwise from top-left)
            dst_points: 4 corner points in normalized view (None = use roi_size)

        Returns:
            Success flag
        """
        if src_points.shape[0] != 4:
            logger.error(f"Need exactly 4 points, got {src_points.shape[0]}")
            return False

        # Default destination: centered square
        if dst_points is None:
            w, h = self.config.roi_size
            dst_points = np.float32([
                [0, 0],  # Top-left
                [w, 0],  # Top-right
                [w, h],  # Bottom-right
                [0, h]  # Bottom-left
            ])

        try:
            # Calculate homography
            self.homography = cv2.getPerspectiveTransform(
                np.float32(src_points),
                np.float32(dst_points)
            )

            # Calculate inverse for back-projection
            self.inverse_homography = cv2.getPerspectiveTransform(
                np.float32(dst_points),
                np.float32(src_points)
            )

            logger.info("Homography set successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to calculate homography: {e}")
            return False

    def set_homography_from_matrix(self, matrix: np.ndarray) -> bool:
        """
        Set homography directly from 3x3 matrix.

        Args:
            matrix: 3x3 homography matrix

        Returns:
            Success flag
        """
        if matrix.shape != (3, 3):
            logger.error(f"Homography must be 3x3, got {matrix.shape}")
            return False

        try:
            self.homography = matrix.copy()
            self.inverse_homography = np.linalg.inv(matrix)
            logger.info("Homography matrix loaded")
            return True
        except Exception as e:
            logger.error(f"Invalid homography matrix: {e}")
            return False

    def warp_roi(self, frame: np.ndarray, fast_mode: bool = False) -> np.ndarray:
        """
        Apply perspective transform to extract ROI.

        Args:
            frame: Input image
            fast_mode: Use INTER_NEAREST for speed (~2-3x faster, ideal for motion detection).
                       Default False uses INTER_LINEAR for quality (display)

        Returns:
            Warped ROI (or original frame if no homography set)

        Performance:
            - fast_mode=True: ~10-15% FPS improvement for motion detection pipeline
            - fast_mode=False: Better quality for final display/visualization
        """
        if self.homography is None:
            logger.warning("No homography set, returning identity transform")
            self.fallback_count += 1
            return self._identity_fallback(frame)

        # PERFORMANCE OPTIMIZATION: Choose interpolation based on use case
        interpolation = self.config.interpolation_fast if fast_mode else self.config.interpolation

        try:
            warped = cv2.warpPerspective(
                frame,
                self.homography,
                self.config.roi_size,
                flags=interpolation
            )

            self.transforms_applied += 1
            return warped

        except Exception as e:
            logger.error(f"Warp failed: {e}, using fallback")
            self.fallback_count += 1
            return self._identity_fallback(frame)

    def _identity_fallback(self, frame: np.ndarray) -> np.ndarray:
        """
        Fallback: Center-crop to ROI size without transformation.
        """
        h, w = frame.shape[:2]
        roi_w, roi_h = self.config.roi_size

        # Calculate center crop
        start_x = max(0, (w - roi_w) // 2)
        start_y = max(0, (h - roi_h) // 2)

        # Ensure we don't exceed bounds
        end_x = min(w, start_x + roi_w)
        end_y = min(h, start_y + roi_h)

        cropped = frame[start_y:end_y, start_x:end_x]

        # Resize if needed
        if cropped.shape[:2] != (roi_h, roi_w):
            cropped = cv2.resize(cropped, self.config.roi_size)

        return cropped

    def polar_unwrap(
            self,
            roi_frame: np.ndarray,
            center: Optional[Tuple[int, int]] = None,
            radius: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert circular dartboard to linear polar coordinates.

        Useful for ring/sector analysis. Converts concentric circles
        to horizontal lines.

        Args:
            roi_frame: ROI image (should be square)
            center: Circle center (None = image center)
            radius: Circle radius (None = use config)

        Returns:
            Polar-unwrapped image
        """
        if not self.config.polar_enabled:
            return roi_frame

        # Auto-detect center if not provided
        if center is None:
            if self.polar_center is None:
                h, w = roi_frame.shape[:2]
                self.polar_center = (w // 2, h // 2)
            center = self.polar_center

        # Use config radius if not provided
        if radius is None:
            radius = self.config.polar_radius

        try:
            # warpPolar: converts circular to rectangular
            # Output: width = circumference, height = 360 degrees
            polar = cv2.warpPolar(
                roi_frame,
                (2 * radius, 360),  # Output size
                center,
                radius,
                cv2.WARP_POLAR_LINEAR
            )

            return polar

        except Exception as e:
            logger.error(f"Polar unwrap failed: {e}")
            return roi_frame

    def project_point_to_roi(
            self,
            point: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """
        Project point from original frame to ROI coordinates.

        Args:
            point: (x, y) in original frame

        Returns:
            (x, y) in ROI frame, or None if projection fails
        """
        if self.homography is None:
            return None

        try:
            # Convert point to homogeneous coordinates
            pt = np.array([[point[0], point[1]]], dtype=np.float32)
            pt_transformed = cv2.perspectiveTransform(
                pt.reshape(1, 1, 2),
                self.homography
            )

            x, y = pt_transformed[0][0]

            # Check if point is within ROI bounds
            roi_w, roi_h = self.config.roi_size
            if 0 <= x < roi_w and 0 <= y < roi_h:
                return (int(x), int(y))

            return None

        except Exception as e:
            logger.error(f"Point projection failed: {e}")
            return None

    def project_point_to_original(
            self,
            point: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """
        Project point from ROI back to original frame coordinates.

        Args:
            point: (x, y) in ROI frame

        Returns:
            (x, y) in original frame, or None if projection fails
        """
        if self.inverse_homography is None:
            return None

        try:
            pt = np.array([[point[0], point[1]]], dtype=np.float32)
            pt_transformed = cv2.perspectiveTransform(
                pt.reshape(1, 1, 2),
                self.inverse_homography
            )

            x, y = pt_transformed[0][0]
            return (int(x), int(y))

        except Exception as e:
            logger.error(f"Inverse projection failed: {e}")
            return None

    def get_stats(self) -> dict:
        """Get ROI processing statistics"""
        return {
            'transforms_applied': self.transforms_applied,
            'fallback_count': self.fallback_count,
            'fallback_rate': self.fallback_count / max(self.transforms_applied, 1),
            'homography_set': self.homography is not None
        }