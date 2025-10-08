"""
Example: Video analysis with different MOG2 parameters
"""

import cv2
from src.utils.video_analyzer import VideoAnalyzer


def create_motion_detector(var_threshold: int):
    """Factory function for different MOG2 configs"""
    return cv2.createBackgroundSubtractorMOG2(
        detectShadows=True,
        varThreshold=var_threshold
    )


def test_mog2_parameters():
    """Test different MOG2 variance thresholds"""

    analyzer = VideoAnalyzer(output_dir="results/mog2_tuning")

    # Your test videos
    video_paths = [
        "test_videos/darts_3Pfeile.mp4",
        "test_videos/VID-20251008-WA0001.mp4",
        "test_videos/VID-20251008-WA0003.mp4"
    ]

    # Define configs to test
    configs = {}

    for var_threshold in [25, 50, 75, 99]:
        config_name = f"MOG2_var{var_threshold}"

        # Create closure to capture var_threshold
        def make_callback(vt):
            bg_sub = create_motion_detector(vt)
            motion_threshold = 500

            def callback(frame, frame_idx):
                # Apply background subtraction
                fg_mask = bg_sub.apply(frame)

                # Clean noise
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

                # Count motion pixels
                motion_pixels = cv2.countNonZero(fg_mask)

                if motion_pixels > motion_threshold:
                    # Find contours
                    contours, _ = cv2.findContours(
                        fg_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours:
                        # Return largest contour as detection
                        largest = max(contours, key=cv2.contourArea)
                        M = cv2.moments(largest)

                        if M['m00'] > 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])

                            return {
                                'position': (cx, cy),
                                'confidence': min(motion_pixels / 1000, 1.0),
                                'motion_pixels': motion_pixels
                            }

                return None

            return callback

        configs[config_name] = make_callback(var_threshold)

    # Run batch analysis
    analyzer.batch_analyze(
        video_paths=video_paths,
        configs=configs,
        max_frames=300  # Limit to 10s per video for quick test
    )


if __name__ == "__main__":
    test_mog2_parameters()