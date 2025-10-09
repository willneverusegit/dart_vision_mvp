"""
ChArUco Board Calibration
High-accuracy camera and dartboard calibration using ChArUco patterns.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


class CharucoCalibrator:
    """
    ChArUco-based calibration for camera intrinsics and dartboard pose.

    Achieves sub-pixel accuracy (0.5-2mm) and handles lens distortion.
    """

    def __init__(
            self,
            squares_x: int = 7,
            squares_y: int = 5,
            square_length: float = 0.04,  # 40mm
            marker_length: float = 0.02,  # 20mm
            dict_type: int = cv2.aruco.DICT_6X6_250
    ):
        """
        Args:
            squares_x: Number of squares horizontally
            squares_y: Number of squares vertically
            square_length: Square size in meters
            marker_length: Marker size in meters
            dict_type: ArUco dictionary type
        """
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length

        # Create ArUco dictionary and board
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            self.aruco_dict
        )

        # Detector parameters
        self.detector_params = cv2.aruco.DetectorParameters()

        # Calibration results
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.calibrated = False

    def detect_board(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect ChArUco board in frame.

        Args:
            frame: Input image (BGR or grayscale)

        Returns:
            (success, charuco_corners, charuco_ids)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.detector_params
        )

        if ids is None or len(ids) < 4:
            logger.warning(f"Not enough markers detected: {len(ids) if ids is not None else 0}")
            return False, None, None

        # Interpolate ChArUco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners,
            ids,
            gray,
            self.charuco_board
        )

        if not retval or charuco_corners is None or len(charuco_corners) < 4:
            logger.warning("ChArUco corner interpolation failed")
            return False, None, None

        logger.info(f"Detected {len(charuco_corners)} ChArUco corners from {len(ids)} markers")
        return True, charuco_corners, charuco_ids

    def calibrate_camera_multi_frame(
            self,
            frames: List[np.ndarray],
            image_size: Tuple[int, int]
    ) -> bool:
        """
        Calibrate camera using multiple frames of ChArUco board.

        Args:
            frames: List of images showing ChArUco board from different angles
            image_size: (width, height) of images

        Returns:
            Success flag
        """
        all_corners = []
        all_ids = []

        for i, frame in enumerate(frames):
            success, corners, ids = self.detect_board(frame)

            if success:
                all_corners.append(corners)
                all_ids.append(ids)
                logger.info(f"Frame {i + 1}/{len(frames)}: {len(corners)} corners detected")
            else:
                logger.warning(f"Frame {i + 1}/{len(frames)}: Board not detected")

        if len(all_corners) < 3:
            logger.error(f"Not enough valid frames: {len(all_corners)}/3 minimum")
            return False

        # Calibrate camera
        try:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                all_corners,
                all_ids,
                self.charuco_board,
                image_size,
                None,
                None
            )

            if not ret:
                logger.error("Camera calibration failed")
                return False

            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.calibrated = True

            logger.info("Camera calibration successful!")
            logger.info(f"Camera matrix:\n{camera_matrix}")
            logger.info(f"Distortion coefficients: {dist_coeffs.ravel()}")

            return True

        except Exception as e:
            logger.error(f"Calibration error: {e}")
            return False

    def calibrate_camera_single_frame(
            self,
            frame: np.ndarray
    ) -> bool:
        """
        Quick calibration from single frame (less accurate).

        Args:
            frame: Image showing ChArUco board frontally

        Returns:
            Success flag
        """
        success, corners, ids = self.detect_board(frame)

        if not success:
            return False

        # Use simplified calibration (assumes minimal distortion)
        h, w = frame.shape[:2]

        # Estimate camera matrix
        self.camera_matrix = np.array([
            [w, 0, w / 2],
            [0, w, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        # Assume no distortion for quick calibration
        self.dist_coeffs = np.zeros(5, dtype=np.float32)
        self.calibrated = True

        logger.info("Quick single-frame calibration complete (assumes minimal distortion)")
        return True

    def get_dartboard_homography(
            self,
            frame: np.ndarray,
            dartboard_corners_3d: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Calculate homography from ChArUco board to dartboard.

        Args:
            frame: Image with visible ChArUco board
            dartboard_corners_3d: 4 corners of dartboard in board coordinate system

        Returns:
            Homography matrix or None
        """
        success, corners, ids = self.detect_board(frame)

        if not success or not self.calibrated:
            return None

        # Estimate board pose
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            corners,
            ids,
            self.charuco_board,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            None
        )

        if not retval:
            logger.error("Board pose estimation failed")
            return None

        # Project dartboard corners to image
        dartboard_corners_2d, _ = cv2.projectPoints(
            dartboard_corners_3d,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )

        dartboard_corners_2d = dartboard_corners_2d.reshape(-1, 2)

        # Calculate homography
        roi_size = 400
        dst_points = np.float32([
            [0, 0],
            [roi_size, 0],
            [roi_size, roi_size],
            [0, roi_size]
        ])

        homography = cv2.getPerspectiveTransform(
            dartboard_corners_2d.astype(np.float32),
            dst_points
        )

        return homography

    def undistort_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Remove lens distortion from image.

        Args:
            frame: Distorted input image

        Returns:
            Undistorted image
        """
        if not self.calibrated:
            logger.warning("Camera not calibrated, returning original image")
            return frame

        h, w = frame.shape[:2]

        # Get optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (w, h),
            1,
            (w, h)
        )

        # Undistort
        undistorted = cv2.undistort(
            frame,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            new_camera_matrix
        )

        # Crop to ROI
        x, y, w, h = roi
        undistorted = undistorted[y:y + h, x:x + w]

        return undistorted

    def draw_detected_board(
            self,
            frame: np.ndarray,
            corners: np.ndarray,
            ids: np.ndarray
    ) -> np.ndarray:
        """
        Draw detected ChArUco board on frame for visualization.

        Args:
            frame: Input image
            corners: ChArUco corners
            ids: ChArUco IDs

        Returns:
            Annotated image
        """
        display = frame.copy()

        # Draw detected corners
        cv2.aruco.drawDetectedCornersCharuco(
            display,
            corners,
            ids,
            (0, 255, 0)
        )

        return display

    def save_calibration(self, filepath: str) -> bool:
        """Save calibration to file"""
        if not self.calibrated:
            logger.error("No calibration to save")
            return False

        try:
            np.savez(
                filepath,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                squares_x=self.squares_x,
                squares_y=self.squares_y,
                square_length=self.square_length,
                marker_length=self.marker_length
            )
            logger.info(f"Calibration saved: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False

    def load_calibration(self, filepath: str) -> bool:
        """Load calibration from file"""
        try:
            data = np.load(filepath)

            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.squares_x = int(data['squares_x'])
            self.squares_y = int(data['squares_y'])
            self.square_length = float(data['square_length'])
            self.marker_length = float(data['marker_length'])
            self.calibrated = True

            logger.info(f"Calibration loaded: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False