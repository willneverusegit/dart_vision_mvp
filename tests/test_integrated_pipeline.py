"""
Test integrated pipeline: ROI + Motion + Dart Detection
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import time
from src.calibration import ROIProcessor, CalibrationManager, ROIConfig
from src.vision import MotionDetector, MotionConfig, DartImpactDetector, DartDetectorConfig
from src.capture import FPSCounter


def test_integrated_pipeline():
    """Test complete pipeline with video"""

    print("🎯 Integrated Pipeline Test")
    print("=" * 50)

    # Load calibration (if exists)
    calib_manager = CalibrationManager()

    if not calib_manager.is_valid():
        print("⚠️  No calibration found, using identity transform")
        print("   Run test_roi_calibration.py first for best results!")

    # Setup components
    roi_processor = ROIProcessor(ROIConfig(roi_size=(400, 400)))

    if calib_manager.is_valid():
        homography = calib_manager.get_homography()
        roi_processor.set_homography_from_matrix(homography)
        print("✅ Calibration loaded")

    motion_detector = MotionDetector(MotionConfig(
        var_threshold=50,
        motion_pixel_threshold=500
    ))

    dart_detector = DartImpactDetector(DartDetectorConfig(
        confirmation_frames=3,
        position_tolerance_px=20
    ))

    fps_counter = FPSCounter()

    # Open video
    video_path = "test_videos/39_sec.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Failed to open {video_path}")
        return

    frame_index = 0
    dart_impacts = []

    print("\n▶️  Processing video...")
    print("Press 'q' to quit, 'p' to pause")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            fps_counter.update()
            timestamp = frame_index / 30.0  # Assume 30 FPS

            # === PIPELINE STAGES ===

            # 1. ROI Extraction (CPU reduction)
            roi_frame = roi_processor.warp_roi(frame)

            # 2. Motion Detection (gating trigger)
            motion_detected, motion_event, fg_mask = motion_detector.detect_motion(
                roi_frame,
                frame_index,
                timestamp
            )

            # 3. Dart Detection (only if motion)
            dart_impact = None
            if motion_detected:
                dart_impact = dart_detector.detect_dart(
                    roi_frame,
                    fg_mask,
                    frame_index,
                    timestamp
                )

                if dart_impact:
                    dart_impacts.append(dart_impact)
                    print(f"\n🎯 DART DETECTED!")
                    print(f"   Position: {dart_impact.position}")
                    print(f"   Confidence: {dart_impact.confidence:.2f}")
                    print(f"   Frame: {dart_impact.confirmed_frame}")

            # === VISUALIZATION ===

            # Resize for display
            display_original = cv2.resize(frame, (640, 480))
            display_roi = cv2.resize(roi_frame, (400, 400))

            # Add motion mask overlay
            if motion_detected:
                fg_mask_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                fg_mask_color = cv2.resize(fg_mask_color, (400, 400))
                display_roi = cv2.addWeighted(display_roi, 0.7, fg_mask_color, 0.3, 0)

            # Draw dart impacts on ROI
            for impact in dart_detector.get_confirmed_impacts():
                cv2.circle(display_roi, impact.position, 10, (0, 255, 255), 2)
                cv2.circle(display_roi, impact.position, 3, (0, 255, 255), -1)

            # FPS overlay
            fps_stats = fps_counter.get_stats()
            cv2.putText(
                display_roi,
                f"FPS: {fps_stats.fps_median:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Motion status
            status_color = (0, 255, 0) if motion_detected else (128, 128, 128)
            cv2.putText(
                display_roi,
                f"Motion: {'YES' if motion_detected else 'NO'}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2
            )

            # Dart count
            cv2.putText(
                display_roi,
                f"Darts: {len(dart_impacts)}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # Show frames
            cv2.imshow('Original', display_original)
            cv2.imshow('ROI + Detection', display_roi)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)

            frame_index += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # === RESULTS ===

    print("\n" + "=" * 50)
    print("📊 PIPELINE RESULTS")
    print("=" * 50)

    # FPS stats
    fps_stats = fps_counter.get_stats()
    print(f"\n⚡ Performance:")
    print(f"   Median FPS:  {fps_stats.fps_median:.1f}")
    print(f"   P95 FPS:     {fps_stats.fps_p95:.1f}")
    print(f"   Frame Time:  {fps_stats.frame_time_ms:.1f} ms")

    # Motion stats
    motion_stats = motion_detector.get_stats()
    print(f"\n🔍 Motion Detection:")
    print(f"   Total Frames:    {motion_stats['frames_processed']}")
    print(f"   Motion Frames:   {motion_stats['motion_frames']}")
    print(f"   Motion Rate:     {motion_stats['motion_rate']:.1%}")
    print(f"   Gate Efficiency: {motion_stats['gate_efficiency']:.1%}")

    # Dart detections
    print(f"\n🎯 Dart Detections:")
    print(f"   Total Impacts:   {len(dart_impacts)}")

    for i, impact in enumerate(dart_impacts, 1):
        print(f"   #{i}: Frame {impact.confirmed_frame}, "
              f"Pos {impact.position}, Conf {impact.confidence:.2f}")

    # ROI stats
    roi_stats = roi_processor.get_stats()
    print(f"\n📐 ROI Processing:")
    print(f"   Transforms:   {roi_stats['transforms_applied']}")
    print(f"   Fallbacks:    {roi_stats['fallback_count']}")
    print(f"   Fallback Rate: {roi_stats['fallback_rate']:.1%}")

    print("\n✅ Test complete!\n")


if __name__ == "__main__":
    test_integrated_pipeline()