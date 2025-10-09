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
from pathlib import Path
from typing import Optional

from src.capture import ThreadedCamera, CameraConfig, FPSCounter
from src.calibration import ROIProcessor, CalibrationManager, ROIConfig
from src.vision import (
    MotionDetector, MotionConfig,
    DartImpactDetector, DartDetectorConfig,
    FieldMapper, FieldMapperConfig
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dart_vision.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

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

    def setup(self) -> bool:
        """Initialize all components"""
        logger.info("=" * 60)
        logger.info("🎯 DART VISION MVP - Initializing")
        logger.info("=" * 60)

        # 1. Calibration Manager
        logger.info("Loading calibration...")
        self.calib_manager = CalibrationManager()

        if not self.calib_manager.is_valid():
            logger.warning("⚠️  No valid calibration found!")

            if self.args.calibrate:
                logger.info("Starting calibration mode...")
                return self._run_calibration_mode()
            else:
                logger.warning("Running with identity transform (reduced accuracy)")
                logger.info("Tip: Run with --calibrate flag to calibrate first")
        else:
            calib_data = self.calib_manager.get_calibration()
            logger.info(f"✅ Calibration loaded: {calib_data.calibration_method}")
            logger.info(f"   Center: {calib_data.center_px}")
            logger.info(f"   Scale: {calib_data.mm_per_px:.3f} mm/px")

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
            position_tolerance_px=20,
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
            logger.error("❌ Failed to start camera")
            return False

        logger.info("✅ All components initialized")
        logger.info("")
        logger.info("🎮 Controls:")
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
        logger.info("📐 CALIBRATION MODE")
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

        # Temporary camera for calibration
        temp_camera = ThreadedCamera(CameraConfig(
            src=self.args.webcam if not self.args.video else self.args.video,
            max_queue_size=2
        ))

        if not temp_camera.start():
            logger.error("Failed to start camera for calibration")
            return False

        # Wait for camera to stabilize
        time.sleep(0.5)

        # Get calibration frame
        ret, calib_frame = temp_camera.read()
        temp_camera.stop()

        if not ret:
            logger.error("Failed to capture calibration frame")
            return False

        # Run interactive calibration
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                logger.info(f"Point {len(points)}/4: ({x}, {y})")

        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', mouse_callback)

        while True:
            display = calib_frame.copy()

            # Draw points
            for i, pt in enumerate(points):
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
                cv2.putText(display, str(i+1), (pt[0]+10, pt[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw lines
            if len(points) > 1:
                for i in range(len(points)):
                    cv2.line(display, points[i], points[(i+1) % len(points)],
                            (0, 255, 0), 2)

            # Status text
            status = f"Points: {len(points)}/4 - "
            if len(points) < 4:
                status += "Click next corner"
            else:
                status += "Press 'c' to confirm"

            cv2.putText(display, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Calibration', display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                logger.info("Calibration cancelled")
                cv2.destroyAllWindows()
                return False

            elif key == ord('r'):
                points.clear()
                logger.info("Points reset")

            elif key == ord('c') and len(points) == 4:
                # Perform calibration
                success = self.calib_manager.manual_calibration(
                    calib_frame,
                    points,
                    board_diameter_mm=340.0
                )

                cv2.destroyAllWindows()

                if success:
                    logger.info("✅ Calibration successful!")
                    return True
                else:
                    logger.error("❌ Calibration failed")
                    return False

        cv2.destroyAllWindows()
        return False

    def process_frame(self, frame):
        """Process single frame through pipeline"""

        self.frame_count += 1
        timestamp = time.time() - self.session_start

        # === PIPELINE STAGES ===

        # Stage 1: ROI Extraction (CPU reduction: ~92%)
        roi_frame = self.roi_processor.warp_roi(frame)

        # Stage 2: Motion Detection (gating trigger)
        motion_detected, motion_event, fg_mask = self.motion_detector.detect_motion(
            roi_frame,
            self.frame_count,
            timestamp
        )

        # Stage 3: Dart Detection (only if motion detected)
        dart_impact = None
        if motion_detected:
            dart_impact = self.dart_detector.detect_dart(
                roi_frame,
                fg_mask,
                self.frame_count,
                timestamp
            )

            if dart_impact:
                self.total_darts_detected += 1

                # Calculate score with calibrated ROI parameters
                if self.calib_manager.is_valid():
                    calib = self.calib_manager.get_calibration()

                    # Use calibrated ROI center and radius
                    roi_center = (200, 200)  # Always center of 400x400 ROI
                    roi_radius = calib.roi_board_radius  # ✅ From calibration

                    score, multiplier, segment = self.field_mapper.point_to_score(
                        dart_impact.position,
                        roi_center,
                        roi_radius
                    )

                    total_score = score * multiplier

                    logger.info(f"🎯 DART #{self.total_darts_detected}")
                    logger.info(f"   Score: {total_score} ({multiplier}x{score})")
                    logger.info(f"   Segment: {segment}")
                    logger.info(f"   Position (ROI): {dart_impact.position}")
                    logger.info(f"   Confidence: {dart_impact.confidence:.2f}")

            return roi_frame, motion_detected, fg_mask, dart_impact

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
        logger.info("🚀 Starting main loop...")

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
                    self.run()
                    return

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up...")

        if self.camera:
            self.camera.stop()

        cv2.destroyAllWindows()

        # Print final statistics
        self._print_final_stats()

        logger.info("✅ Shutdown complete")

    def _print_final_stats(self):
        """Print session statistics"""

        session_duration = time.time() - self.session_start

        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 SESSION STATISTICS")
        logger.info("=" * 60)

        # Performance
        if self.fps_counter:
            fps_stats = self.fps_counter.get_stats()
            logger.info(f"\n⚡ Performance:")
            logger.info(f"   Median FPS:     {fps_stats.fps_median:.1f}")
            logger.info(f"   P95 FPS:        {fps_stats.fps_p95:.1f}")
            logger.info(f"   Mean FPS:       {fps_stats.fps_mean:.1f}")
            logger.info(f"   Frame Time:     {fps_stats.frame_time_ms:.1f} ms")
            logger.info(f"   P95 Frame Time: {fps_stats.frame_time_p95_ms:.1f} ms")

        # Motion
        if self.motion_detector:
            motion_stats = self.motion_detector.get_stats()
            logger.info(f"\n🔍 Motion Detection:")
            logger.info(f"   Frames Processed: {motion_stats['frames_processed']}")
            logger.info(f"   Motion Frames:    {motion_stats['motion_frames']}")
            logger.info(f"   Motion Rate:      {motion_stats['motion_rate']:.1%}")
            logger.info(f"   Gated Ops:        {motion_stats['gated_operations']}")
            logger.info(f"   Gate Efficiency:  {motion_stats['gate_efficiency']:.1%}")

        # Darts
        logger.info(f"\n🎯 Dart Detection:")
        logger.info(f"   Total Darts:   {self.total_darts_detected}")
        logger.info(f"   Darts/Minute:  {self.total_darts_detected / (session_duration / 60):.1f}")

        # ROI
        if self.roi_processor:
            roi_stats = self.roi_processor.get_stats()
            logger.info(f"\n📐 ROI Processing:")
            logger.info(f"   Transforms:    {roi_stats['transforms_applied']}")
            logger.info(f"   Fallbacks:     {roi_stats['fallback_count']}")
            logger.info(f"   Fallback Rate: {roi_stats['fallback_rate']:.1%}")

        # Camera
        if self.camera:
            cam_stats = self.camera.get_stats()
            logger.info(f"\n📹 Camera:")
            logger.info(f"   Frames Captured: {cam_stats['frames_captured']}")
            logger.info(f"   Frames Dropped:  {cam_stats['frames_dropped']}")
            logger.info(f"   Drop Rate:       {cam_stats['drop_rate']:.2%}")

        # Session
        logger.info(f"\n⏱️  Session:")
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
    import numpy as np  # Add missing import
    main()