"""
Dart Vision MVP - Main Application
CPU-optimized dart detection and scoring system.

Usage:
    python main.py --video test_videos/dart_throw_1.mp4
    python main.py --webcam 0
    python main.py --calibrate
"""

import cv2
import argparse
import logging
import time
import sys
import numpy as np
from pathlib import Path
from typing import Optional

from src.utils.performance_profiler import PerformanceProfiler
from src.capture import ThreadedCamera, CameraConfig, FPSCounter
from src.calibration import ROIProcessor, CalibrationManager, ROIConfig
from src.calibration.charuco_calibrator import CharucoCalibrator
from src.vision import (
    MotionDetector, MotionConfig,
    DartImpactDetector, DartDetectorConfig,
    FieldMapper, FieldMapperConfig
)


def setup_logging():
    """Setup logging with UTF-8 encoding (no emoji issues on Windows)"""
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler('dart_vision.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Console handler (safe for Windows console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s - %(message)s'
    ))

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


class DartVisionApp:
    """
    Main Dart Vision Application.

    Orchestrates all components for real-time dart detection.
    """

    def __init__(self, args):
        self.args = args

        # Components (initialized in setup())
        self.camera: Optional[ThreadedCamera] = None
        self.calib_manager: Optional[CalibrationManager] = None
        self.roi_processor: Optional[ROIProcessor] = None
        self.motion_detector: Optional[MotionDetector] = None
        self.dart_detector: Optional[DartImpactDetector] = None
        self.field_mapper: Optional[FieldMapper] = None
        self.fps_counter: Optional[FPSCounter] = None

        # State
        self.running = False
        self.paused = False
        self.show_debug = True
        self.show_motion = False
        self.frame_count = 0

        # Statistics
        self.session_start = time.time()
        self.total_darts_detected = 0

        # Cleanup guard
        self._cleaned_up = False

        # ✅ NEU: Performance profiler
        self.profiler = PerformanceProfiler()

    def setup(self) -> bool:
        """Initialize all components"""
        logger.info("=" * 60)
        logger.info("DART VISION MVP - Initializing")
        logger.info("=" * 60)

        # 1. Calibration Manager
        logger.info("Loading calibration...")
        self.calib_manager = CalibrationManager()

        if not self.calib_manager.is_valid():
            logger.warning("WARNING: No valid calibration found!")

            if self.args.calibrate:
                logger.info("Starting calibration mode...")

                # Run calibration first
                if not self._run_calibration_mode():
                    logger.error("Calibration failed")
                    return False

                # Reload calibration after successful calibration
                self.calib_manager = CalibrationManager()

                if not self.calib_manager.is_valid():
                    logger.error("Calibration data not loaded after calibration")
                    return False
            else:
                logger.warning("Running with identity transform (reduced accuracy)")
                logger.info("Tip: Run with --calibrate flag to calibrate first")
        else:
            calib_data = self.calib_manager.get_calibration()
            logger.info(f"Calibration loaded: {calib_data.calibration_method}")
            logger.info(f"   Center: {calib_data.center_px}")
            logger.info(f"   Scale: {calib_data.mm_per_px:.3f} mm/px")
            logger.info(f"   ROI Board Radius: {calib_data.roi_board_radius:.1f}px")

        # 2. ROI Processor
        logger.info("Setting up ROI processor...")
        self.roi_processor = ROIProcessor(ROIConfig(
            roi_size=(400, 400),
            polar_enabled=False
        ))

        if self.calib_manager.is_valid():
            homography = self.calib_manager.get_homography()
            self.roi_processor.set_homography_from_matrix(homography)

        # 3. Motion Detector
        logger.info("Initializing motion detector...")
        self.motion_detector = MotionDetector(MotionConfig(
            var_threshold=self.args.motion_threshold,
            motion_pixel_threshold=self.args.motion_pixels,
            detect_shadows=True
        ))

        # 4. Dart Detector
        logger.info("Initializing dart detector...")
        self.dart_detector = DartImpactDetector(DartDetectorConfig(
            confirmation_frames=self.args.confirmation_frames,
            position_tolerance_px=10,  # Reduced from 20 to minimize false positives
            min_area=10,
            max_area=1000
        ))

        # 5. Field Mapper (for scoring)
        logger.info("Setting up field mapper...")
        self.field_mapper = FieldMapper(FieldMapperConfig())

        # 6. FPS Counter
        self.fps_counter = FPSCounter(window_size=30)

        # 7. Camera
        logger.info("Opening camera...")

        if self.args.video:
            camera_src = self.args.video
            logger.info(f"Video source: {camera_src}")
        else:
            camera_src = self.args.webcam
            logger.info(f"Webcam index: {camera_src}")

        camera_config = CameraConfig(
            src=camera_src,
            max_queue_size=5,
            buffer_size=1,
            width=self.args.width,
            height=self.args.height
        )

        self.camera = ThreadedCamera(camera_config)

        if not self.camera.start():
            logger.error("Failed to start camera")
            return False

        logger.info("All components initialized")
        logger.info("")
        logger.info("Controls:")
        logger.info("   'q' - Quit")
        logger.info("   'p' - Pause/Resume")
        logger.info("   'd' - Toggle debug overlay")
        logger.info("   'm' - Toggle motion visualization")
        logger.info("   'r' - Reset dart detections")
        logger.info("   'c' - Recalibrate")
        logger.info("   's' - Save screenshot")
        logger.info("")

        return True

    def _run_calibration_mode(self) -> bool:
        """Interactive calibration mode"""
        logger.info("")
        logger.info("CALIBRATION MODE")
        logger.info("=" * 60)
        logger.info("Choose calibration method:")
        logger.info("")
        logger.info("1. Manual 4-Point Calibration")
        logger.info("   - Quick setup (30 seconds)")
        logger.info("   - Accuracy: 2-5mm")
        logger.info("   - No special equipment needed")
        logger.info("")
        logger.info("2. ChArUco Board Calibration")
        logger.info("   - Accurate setup (2 minutes)")
        logger.info("   - Accuracy: 0.5-2mm (sub-pixel!)")
        logger.info("   - Corrects lens distortion")
        logger.info("   - Requires printed ChArUco board")
        logger.info("")

        # Get user choice
        print("Enter choice (1 or 2): ", end='', flush=True)

        try:
            choice = input().strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("Calibration cancelled")
            return False

        if choice == '2':
            return self._run_charuco_calibration()
        else:
            return self._run_manual_calibration()

    def _run_manual_calibration(self) -> bool:
        """Manual 4-point calibration (original method)"""
        logger.info("")
        logger.info("MANUAL CALIBRATION")
        logger.info("=" * 60)
        logger.info("Instructions:")
        logger.info("1. Position dartboard clearly in view")
        logger.info("2. Click 4 corners in order:")
        logger.info("   - Top-Left")
        logger.info("   - Top-Right")
        logger.info("   - Bottom-Right")
        logger.info("   - Bottom-Left")
        logger.info("3. Press 'c' to confirm calibration")
        logger.info("4. Press 'r' to reset points")
        logger.info("5. Press 'q' to cancel")
        logger.info("")

        # ... (rest of original _run_calibration_mode code)
        # ... (copy from previous main.py, lines ~200-290)

    def _run_charuco_calibration(self) -> bool:
        """ChArUco board calibration mode"""
        logger.info("")
        logger.info("CHARUCO CALIBRATION")
        logger.info("=" * 60)
        logger.info("Prerequisites:")
        logger.info("1. ChArUco board printed and mounted next to dartboard")
        logger.info("   (Run: python generate_charuco_board.py)")
        logger.info("2. Board must be on SAME PLANE as dartboard")
        logger.info("3. Good lighting (no shadows on markers)")
        logger.info("")
        logger.info("Instructions:")
        logger.info("1. Position camera to see BOTH board and dartboard")
        logger.info("2. Press SPACE to capture frame")
        logger.info("3. Click 4 dartboard corners in captured frame")
        logger.info("4. Press 'c' to complete calibration")
        logger.info("5. Press 'q' to cancel")
        logger.info("")

        # Determine camera source
        if self.args.video:
            camera_src = self.args.video
            logger.info(f"Using video file: {camera_src}")
        else:
            camera_src = self.args.webcam
            logger.info(f"Using webcam: {camera_src}")

        # Initialize ChArUco calibrator
        calibrator = CharucoCalibrator(
            squares_x=7,
            squares_y=5,
            square_length=0.04,  # 40mm
            marker_length=0.02  # 20mm
        )

        # Check if existing calibration file exists
        calib_file = Path('config/charuco_camera_calib.npz')
        if calib_file.exists():
            logger.info(f"Found existing ChArUco calibration: {calib_file}")
            print("Load existing calibration? (y/n): ", end='', flush=True)

            try:
                load_choice = input().strip().lower()
                if load_choice == 'y':
                    if calibrator.load_calibration(str(calib_file)):
                        logger.info("ChArUco calibration loaded successfully")
                    else:
                        logger.warning("Failed to load calibration, will create new one")
            except (EOFError, KeyboardInterrupt):
                logger.info("Skipping load")

        # Temporary camera for calibration
        temp_camera = ThreadedCamera(CameraConfig(
            src=camera_src,
            max_queue_size=2
        ))

        if not temp_camera.start():
            logger.error("Failed to start camera for calibration")
            return False

        # Wait for camera to stabilize
        time.sleep(0.5)

        # ✅ DEBUG: Log camera info
        logger.info(f"Camera started, reading frames...")

        # Capture loop
        logger.info("Press SPACE to capture frame...")
        captured_frame = None
        frame_count = 0  # ✅ DEBUG

        while True:
            ret, frame = temp_camera.read()

            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1  # ✅ DEBUG

            # ✅ DEBUG: Log every 30 frames
            if frame_count % 30 == 0:
                logger.info(f"Reading frames... (frame {frame_count})")

            # Detect ChArUco board in live view
            success, corners, ids = calibrator.detect_board(frame)

            display = frame.copy()

            if success:
                # ✅ DEBUG: Log detection
                if frame_count % 30 == 0:
                    logger.info(f"Board detected! {len(corners)} corners")

                # Draw detected board
                display = calibrator.draw_detected_board(display, corners, ids)

                # Status text
                cv2.putText(display, f"ChArUco: {len(corners)} corners detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, "Press SPACE to capture",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # ✅ DEBUG: Log no detection
                if frame_count % 30 == 0:
                    logger.warning("Board NOT detected in this frame")

                # Warning
                cv2.putText(display, "ChArUco board NOT detected!",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display, "Make sure board is visible",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ✅ DEBUG: Add frame counter to display
            cv2.putText(display, f"Frame: {frame_count}",
                        (10, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('ChArUco Calibration', display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                logger.info("Calibration cancelled")
                temp_camera.stop()
                cv2.destroyAllWindows()
                return False

            elif key == ord(' ') and success:  # Space bar
                captured_frame = frame.copy()
                captured_corners = corners
                captured_ids = ids
                logger.info(f"Frame captured! ({len(captured_corners)} corners)")
                break

            elif key == ord(' ') and not success:  # ✅ DEBUG
                logger.warning("Cannot capture: Board not detected!")

        temp_camera.stop()
        cv2.destroyAllWindows()

        if captured_frame is None:
            logger.error("No frame captured")
            return False

        # ✅ DEBUG: Verify captured data
        logger.info(f"Captured frame shape: {captured_frame.shape}")
        logger.info(f"Captured corners: {len(captured_corners) if captured_corners is not None else 0}")

        # Perform camera calibration (single frame, quick method)
        if not calibrator.calibrated:
            logger.info("Performing camera calibration...")
            if not calibrator.calibrate_camera_single_frame(captured_frame):
                logger.error("Camera calibration failed")
                return False

            # Save camera calibration
            calib_file.parent.mkdir(parents=True, exist_ok=True)
            calibrator.save_calibration(str(calib_file))

        # Quick calibration (estimate camera matrix)
        logger.info("Estimating camera parameters...")
        calibrator.calibrate_camera_single_frame(captured_frame)

        # Now get dartboard corners manually
        logger.info("")
        logger.info("Now click 4 DARTBOARD corners:")
        logger.info("  1. Top-Left")
        logger.info("  2. Top-Right")
        logger.info("  3. Bottom-Right")
        logger.info("  4. Bottom-Left")

        dartboard_points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(dartboard_points) < 4:
                dartboard_points.append((x, y))
                logger.info(f"Corner {len(dartboard_points)}/4: ({x}, {y})")

        cv2.namedWindow('Click Dartboard Corners')
        cv2.setMouseCallback('Click Dartboard Corners', mouse_callback)

        while True:
            display = captured_frame.copy()

            # Draw clicked points
            for i, pt in enumerate(dartboard_points):
                cv2.circle(display, pt, 8, (0, 255, 255), -1)
                cv2.putText(display, str(i + 1), (pt[0] + 15, pt[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # Draw lines
            if len(dartboard_points) > 1:
                for i in range(len(dartboard_points)):
                    cv2.line(display, dartboard_points[i],
                             dartboard_points[(i + 1) % len(dartboard_points)],
                             (0, 255, 255), 2)

            # Status
            status = f"Dartboard corners: {len(dartboard_points)}/4"
            if len(dartboard_points) < 4:
                status += " - Click next corner"
            else:
                status += " - Press 'c' to confirm, 'r' to reset"

            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('Click Dartboard Corners', display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                logger.info("Cancelled")
                cv2.destroyAllWindows()
                return False

            elif key == ord('r'):
                dartboard_points.clear()
                logger.info("Points reset")

            elif key == ord('c') and len(dartboard_points) == 4:
                break

        cv2.destroyAllWindows()

        # Calculate homography (undistorted coordinates)
        logger.info("Calculating homography...")

        dartboard_points_array = np.array(dartboard_points, dtype=np.float32).reshape(-1, 1, 2)

        # Undistort points (minimal distortion assumed)
        dartboard_points_undistorted = cv2.undistortPoints(
            dartboard_points_array,
            calibrator.camera_matrix,
            calibrator.dist_coeffs,
            P=calibrator.camera_matrix
        )
        dartboard_points_undistorted = dartboard_points_undistorted.reshape(-1, 2)

        # ROI destination points
        roi_size = 400
        dst_points = np.float32([
            [0, 0],
            [roi_size, 0],
            [roi_size, roi_size],
            [0, roi_size]
        ])

        homography = cv2.getPerspectiveTransform(
            dartboard_points_undistorted.astype(np.float32),
            dst_points
        )

        # Calculate calibration parameters
        center_x = sum(p[0] for p in dartboard_points) / 4
        center_y = sum(p[1] for p in dartboard_points) / 4

        board_width_px = np.linalg.norm(
            np.array(dartboard_points[1]) - np.array(dartboard_points[0])
        )

        board_diameter_mm = 340.0
        mm_per_px = board_diameter_mm / board_width_px

        board_radius_in_roi = roi_size * 0.4
        radii_mm = [15.9, 50, 107, 170]
        radii_px = [r / board_diameter_mm * 2 * board_radius_in_roi for r in radii_mm]

        # Create calibration data
        from src.calibration.calibration_manager import CalibrationData
        from datetime import datetime

        calibration_data = CalibrationData(
            center_px=(int(center_x), int(center_y)),
            radii_px=radii_px,
            rotation_deg=0.0,
            mm_per_px=mm_per_px,
            homography=homography.tolist(),
            last_update_utc=datetime.utcnow().isoformat(),
            valid=True,
            calibration_method='charuco_simplified',
            roi_board_radius=board_radius_in_roi
        )

        # Save calibration
        self.calib_manager.calibration = calibration_data
        if self.calib_manager._atomic_save_config(calibration_data):
            logger.info("ChArUco calibration successful!")
            logger.info(f"  Method: Simplified (ChArUco verified)")
            logger.info(f"  Center: {calibration_data.center_px}")
            logger.info(f"  Scale: {mm_per_px:.3f} mm/px")
            logger.info(f"  ROI Board Radius: {board_radius_in_roi:.1f}px")

            # Show result
            roi_processor = ROIProcessor(ROIConfig(roi_size=(400, 400)))
            roi_processor.set_homography_from_matrix(homography)

            undistorted_frame = calibrator.undistort_image(captured_frame)
            warped = roi_processor.warp_roi(undistorted_frame)

            # Draw rings
            roi_center = (200, 200)
            roi_radius = int(board_radius_in_roi)

            cv2.circle(warped, roi_center, roi_radius, (0, 255, 0), 2)
            cv2.circle(warped, roi_center, int(roi_radius * 0.05), (255, 255, 0), 1)
            cv2.circle(warped, roi_center, int(roi_radius * 0.095), (255, 255, 0), 1)
            cv2.circle(warped, roi_center, int(roi_radius * 0.53), (0, 255, 0), 1)
            cv2.circle(warped, roi_center, int(roi_radius * 0.58), (0, 255, 0), 1)
            cv2.circle(warped, roi_center, int(roi_radius * 0.94), (0, 0, 255), 1)

            cv2.imshow('ChArUco Calibration Result', warped)
            logger.info("Showing calibrated ROI - Press any key...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return True
        else:
            logger.error("Failed to save calibration")
            return False


    def process_frame(self, frame):
        """Process single frame through pipeline"""

        self.frame_count += 1
        timestamp = time.time() - self.session_start

        # === PIPELINE STAGES ===

        # Stage 1: ROI Extraction (CPU reduction: ~92%)
        roi_frame = self._stage_roi(frame)  # ✅ Use profiled method

        # Stage 2: Motion Detection (gating trigger)
        motion_detected, motion_event, fg_mask = self._stage_motion(
            roi_frame, self.frame_count, timestamp
        )  # ✅ Use profiled method

        # Stage 3: Dart Detection (only if motion detected)
        dart_impact = None
        if motion_detected:
            dart_impact = self._stage_dart(
                roi_frame, fg_mask, self.frame_count, timestamp
            )  # ✅ Use profiled method

            if dart_impact:
                self.total_darts_detected += 1

                # Calculate score with calibrated ROI parameters
                if self.calib_manager.is_valid():
                    calib = self.calib_manager.get_calibration()

                    # ROI scoring coordinates
                    roi_center = (200, 200)
                    roi_radius = calib.roi_board_radius

                    score, multiplier, segment = self.field_mapper.point_to_score(
                        dart_impact.position,
                        roi_center,
                        roi_radius
                    )

                    total_score = score * multiplier

                    logger.info(f"DART #{self.total_darts_detected}")
                    logger.info(f"   Score: {total_score} ({multiplier}x{score})")
                    logger.info(f"   Segment: {segment}")
                    logger.info(f"   Position (ROI): {dart_impact.position}")
                    logger.info(f"   Confidence: {dart_impact.confidence:.2f}")

        return roi_frame, motion_detected, fg_mask, dart_impact

    # ✅ NEU: Profiled stage methods
    def _stage_roi(self, frame):
        """ROI extraction stage (profiled)"""
        start = time.perf_counter()
        result = self.roi_processor.warp_roi(frame)
        elapsed = (time.perf_counter() - start) * 1000
        self.profiler.timings["ROI Extraction"].append(elapsed)
        return result

    def _stage_motion(self, roi_frame, frame_index, timestamp):
        """Motion detection stage (profiled)"""
        start = time.perf_counter()
        result = self.motion_detector.detect_motion(roi_frame, frame_index, timestamp)
        elapsed = (time.perf_counter() - start) * 1000
        self.profiler.timings["Motion Detection"].append(elapsed)
        return result

    def _stage_dart(self, roi_frame, fg_mask, frame_index, timestamp):
        """Dart detection stage (profiled)"""
        start = time.perf_counter()
        result = self.dart_detector.detect_dart(roi_frame, fg_mask, frame_index, timestamp)
        elapsed = (time.perf_counter() - start) * 1000
        self.profiler.timings["Dart Detection"].append(elapsed)

    def create_visualization(self, frame, roi_frame, motion_detected, fg_mask, dart_impact):
        """Create visualization overlay"""

        # Resize frames for display
        display_main = cv2.resize(frame, (800, 600))
        display_roi = cv2.resize(roi_frame, (400, 400))

        # Motion overlay (if enabled)
        if self.show_motion and motion_detected:
            fg_mask_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            fg_mask_color = cv2.resize(fg_mask_color, (400, 400))
            display_roi = cv2.addWeighted(display_roi, 0.7, fg_mask_color, 0.3, 0)

        # Draw dartboard rings for reference
        if self.calib_manager.is_valid() and self.show_debug:
            calib = self.calib_manager.get_calibration()
            roi_center = (200, 200)
            roi_radius = int(calib.roi_board_radius)

            # Draw board outline
            cv2.circle(display_roi, roi_center, roi_radius, (0, 255, 0), 2)

            # Draw ring boundaries
            cv2.circle(display_roi, roi_center, int(roi_radius * 0.05), (255, 255, 0), 1)  # Bull
            cv2.circle(display_roi, roi_center, int(roi_radius * 0.095), (255, 255, 0), 1)  # Outer bull
            cv2.circle(display_roi, roi_center, int(roi_radius * 0.53), (0, 255, 0), 1)  # Triple inner
            cv2.circle(display_roi, roi_center, int(roi_radius * 0.58), (0, 255, 0), 1)  # Triple outer
            cv2.circle(display_roi, roi_center, int(roi_radius * 0.94), (0, 0, 255), 1)  # Double inner
        if self.calib_manager.is_valid() and len(self.dart_detector.get_confirmed_impacts()) > 0:
            # Create heatmap overlay
            heatmap = np.zeros((400, 400, 3), dtype=np.uint8)

            for impact in self.dart_detector.get_confirmed_impacts():
                # Draw semi-transparent circle
                overlay = heatmap.copy()
                cv2.circle(overlay, impact.position, 30, (0, 0, 255), -1)
                cv2.addWeighted(heatmap, 0.7, overlay, 0.3, 0, heatmap)

            # Blend heatmap with ROI
            display_roi = cv2.addWeighted(display_roi, 0.8, heatmap, 0.2, 0)


        # Draw confirmed dart impacts
        for impact in self.dart_detector.get_confirmed_impacts():
            cv2.circle(display_roi, impact.position, 12, (0, 255, 255), 2)
            cv2.circle(display_roi, impact.position, 3, (0, 255, 255), -1)

        # Debug overlay
        if self.show_debug:
            fps_stats = self.fps_counter.get_stats()
            motion_stats = self.motion_detector.get_stats()

            # FPS
            cv2.putText(display_roi, f"FPS: {fps_stats.fps_median:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Frame time
            cv2.putText(display_roi, f"Time: {fps_stats.frame_time_ms:.1f}ms",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Motion status
            status_color = (0, 255, 0) if motion_detected else (128, 128, 128)
            cv2.putText(display_roi, f"Motion: {'YES' if motion_detected else 'NO'}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # Dart count
            cv2.putText(display_roi, f"Darts: {self.total_darts_detected}",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Gate efficiency
            cv2.putText(display_roi, f"Gate: {motion_stats['gate_efficiency']:.0%}",
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # ✅ NEW: Score history sidebar
            if self.show_debug and self.dart_detector.get_confirmed_impacts():
                impacts = self.dart_detector.get_confirmed_impacts()

                # Calculate scores for last 10 darts
                score_history = []
                for impact in impacts[-10:]:
                    if self.calib_manager.is_valid():
                        calib = self.calib_manager.get_calibration()
                        roi_center = (200, 200)
                        roi_radius = calib.roi_board_radius

                        score, multiplier, segment = self.field_mapper.point_to_score(
                            impact.position,
                            roi_center,
                            roi_radius
                        )

                        total = score * multiplier
                        score_history.append(total)

                # Draw score history
                y_pos = 180
                cv2.putText(display_roi, "Recent Scores:", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                y_pos += 25
                for i, score in enumerate(score_history[-5:], 1):
                    cv2.putText(display_roi, f"{i}. {score}", (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_pos += 20
        # Combine displays
        # Create black canvas
        canvas = np.zeros((600, 1200, 3), dtype=np.uint8)

        # Place main view (left)
        canvas[0:600, 0:800] = display_main

        # Place ROI view (right, centered vertically)
        y_offset = (600 - 400) // 2
        canvas[y_offset:y_offset+400, 800:1200] = display_roi

        return canvas

    def run(self):
        """Main application loop"""

        if not self.setup():
            logger.error("Setup failed, exiting")
            return

        self.running = True
        logger.info("Starting main loop...")

        try:
            while self.running:
                # Read frame
                ret, frame = self.camera.read(timeout=0.1)

                if not ret:
                    continue

                # Update FPS
                self.fps_counter.update()

                # Process frame (unless paused)
                if not self.paused:
                    roi_frame, motion_detected, fg_mask, dart_impact = self.process_frame(frame)
                else:
                    # Just pass through when paused
                    roi_frame = self.roi_processor.warp_roi(frame)
                    motion_detected = False
                    fg_mask = np.zeros((400, 400), dtype=np.uint8)
                    dart_impact = None

                # Create visualization
                display = self.create_visualization(
                    frame, roi_frame, motion_detected, fg_mask, dart_impact
                )

                # Pause indicator
                if self.paused:
                    cv2.putText(display, "PAUSED", (500, 50),
                               cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 165, 255), 3)

                # Show
                cv2.imshow('Dart Vision MVP', display)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    logger.info("Quit requested")
                    self.running = False

                elif key == ord('p'):
                    self.paused = not self.paused
                    logger.info(f"{'Paused' if self.paused else 'Resumed'}")

                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    logger.info(f"Debug overlay: {'ON' if self.show_debug else 'OFF'}")

                elif key == ord('m'):
                    self.show_motion = not self.show_motion
                    logger.info(f"Motion overlay: {'ON' if self.show_motion else 'OFF'}")

                elif key == ord('r'):
                    self.dart_detector.clear_impacts()
                    self.total_darts_detected = 0
                    logger.info("Dart detections reset")

                elif key == ord('s'):
                    filename = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, display)
                    logger.info(f"Screenshot saved: {filename}")

                elif key == ord('c'):
                    logger.info("Recalibration requested...")
                    self.running = False
                    cv2.destroyAllWindows()
                    self.cleanup()

                    # Restart in calibration mode
                    self.args.calibrate = True
                    self.__init__(self.args)
                    self.run()
                    return

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        # Guard against multiple calls
        if self._cleaned_up:
            return

        self._cleaned_up = True
        logger.info("Cleaning up...")

        if self.camera:
            self.camera.stop()

        cv2.destroyAllWindows()

        # ✅ NEU: Print profiling report
        if self.profiler and len(self.profiler.timings) > 0:
            logger.info(self.profiler.get_report())

        # Print final statistics
        self._print_final_stats()

        logger.info("Shutdown complete")

    def _print_final_stats(self):
        """Print session statistics"""

        session_duration = time.time() - self.session_start

        logger.info("")
        logger.info("=" * 60)
        logger.info("SESSION STATISTICS")
        logger.info("=" * 60)

        # Performance
        if self.fps_counter:
            fps_stats = self.fps_counter.get_stats()
            logger.info(f"\nPerformance:")
            logger.info(f"   Median FPS:     {fps_stats.fps_median:.1f}")
            logger.info(f"   P95 FPS:        {fps_stats.fps_p95:.1f}")
            logger.info(f"   Mean FPS:       {fps_stats.fps_mean:.1f}")
            logger.info(f"   Frame Time:     {fps_stats.frame_time_ms:.1f} ms")
            logger.info(f"   P95 Frame Time: {fps_stats.frame_time_p95_ms:.1f} ms")

        # Motion
        if self.motion_detector:
            motion_stats = self.motion_detector.get_stats()
            logger.info(f"\nMotion Detection:")
            logger.info(f"   Frames Processed: {motion_stats['frames_processed']}")
            logger.info(f"   Motion Frames:    {motion_stats['motion_frames']}")
            logger.info(f"   Motion Rate:      {motion_stats['motion_rate']:.1%}")
            logger.info(f"   Gated Ops:        {motion_stats['gated_operations']}")
            logger.info(f"   Gate Efficiency:  {motion_stats['gate_efficiency']:.1%}")

        # Darts
        logger.info(f"\nDart Detection:")
        logger.info(f"   Total Darts:   {self.total_darts_detected}")
        logger.info(f"   Darts/Minute:  {self.total_darts_detected / (session_duration / 60):.1f}")

        # ROI
        if self.roi_processor:
            roi_stats = self.roi_processor.get_stats()
            logger.info(f"\nROI Processing:")
            logger.info(f"   Transforms:    {roi_stats['transforms_applied']}")
            logger.info(f"   Fallbacks:     {roi_stats['fallback_count']}")
            logger.info(f"   Fallback Rate: {roi_stats['fallback_rate']:.1%}")

        # Camera
        if self.camera:
            cam_stats = self.camera.get_stats()
            logger.info(f"\nCamera:")
            logger.info(f"   Frames Captured: {cam_stats['frames_captured']}")
            logger.info(f"   Frames Dropped:  {cam_stats['frames_dropped']}")
            logger.info(f"   Drop Rate:       {cam_stats['drop_rate']:.2%}")

        # Session
        logger.info(f"\nSession:")
        logger.info(f"   Duration:     {session_duration:.1f} seconds")
        logger.info(f"   Total Frames: {self.frame_count}")

        logger.info("")


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description='Dart Vision MVP - CPU-optimized dart detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with video file
  python main.py --video test_videos/dart_throw_1.mp4
  
  # Run with webcam
  python main.py --webcam 0
  
  # Calibrate first, then run
  python main.py --calibrate --webcam 0
  
  # Custom parameters
  python main.py --video test.mp4 --motion-threshold 75 --confirmation-frames 5
        """
    )

    # Input source
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--video', '-v', type=str,
                            help='Path to video file')
    input_group.add_argument('--webcam', '-w', type=int, default=0,
                            help='Webcam index (default: 0)')

    # Calibration
    parser.add_argument('--calibrate', '-c', action='store_true',
                       help='Run calibration mode')

    # Camera settings
    parser.add_argument('--width', type=int, default=None,
                       help='Camera width (default: auto)')
    parser.add_argument('--height', type=int, default=None,
                       help='Camera height (default: auto)')

    # Detection parameters
    parser.add_argument('--motion-threshold', type=int, default=50,
                       help='MOG2 variance threshold (default: 50)')
    parser.add_argument('--motion-pixels', type=int, default=500,
                       help='Minimum motion pixels (default: 500)')
    parser.add_argument('--confirmation-frames', type=int, default=3,
                       help='Frames needed for dart confirmation (default: 3)')

    args = parser.parse_args()

    # Create and run app
    app = DartVisionApp(args)
    app.run()


if __name__ == "__main__":
    main()