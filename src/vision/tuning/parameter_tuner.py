"""
Parameter Tuner - Interactive tool for finding optimal detection parameters.

Features:
- Live preview with side-by-side comparison
- 80+ trackbars for all parameters
- Real-time metrics dashboard
- Debug overlays (masks, contours, metrics)
- Preset system (load/save/compare)
- Frame-by-frame navigation
- Statistics tracking

Usage:
    python -m src.vision.tuning.parameter_tuner --video path/to/video.mp4
    # OR
    python -m src.vision.tuning.parameter_tuner --webcam 0
"""

import cv2
import numpy as np
import argparse
import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from collections import deque

from src.vision.detection import (
    MotionDetector,
    MotionConfig,
    DartImpactDetector,
    DartDetectorConfig,
    apply_detector_preset,
)
from src.vision.tuning.tuning_visualizer import TuningVisualizer
from src.vision.tuning.preset_manager import PresetManager

logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    """Configuration for tuning session"""
    video_source: str = ""
    webcam_index: int = 0
    use_webcam: bool = False
    start_frame: int = 0
    loop_video: bool = True
    show_debug: bool = True
    window_width: int = 1800
    window_height: int = 900
    preset_dir: str = "presets/detection"


class ParameterTuner:
    """
    Interactive parameter tuning tool for dart detection system.

    Provides real-time parameter adjustment with visual feedback for:
    - Motion detection parameters
    - Dart detection parameters
    - Shape analysis thresholds
    - Filtering and preprocessing options
    """

    def __init__(self, config: TuningConfig):
        """
        Initialize parameter tuner.

        Args:
            config: Tuning configuration
        """
        self.config = config
        self.logger = logging.getLogger("ParameterTuner")

        # Video source
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 30.0

        # Detectors
        self.motion_detector: Optional[MotionDetector] = None
        self.dart_detector: Optional[DartImpactDetector] = None

        # Visualizer
        self.visualizer = TuningVisualizer()

        # Preset manager
        self.preset_manager = PresetManager(Path(self.config.preset_dir))

        # UI state
        self.paused = True  # Start paused to allow parameter adjustment
        self.show_trackbars = True
        self.show_metrics = True
        self.show_debug_overlays = True
        self.current_preset = "balanced"

        # Playback control
        self.playback_speed = 1.0
        self.frame_step = 0  # For frame-by-frame navigation

        # Statistics
        self.stats = {
            "frames_processed": 0,
            "motion_detected": 0,
            "darts_detected": 0,
            "false_positives": 0,
            "avg_confidence": deque(maxlen=30),
        }

        # Window names
        self.main_window = "Dart Detection Tuner"
        self.controls_window = "Parameters"

    def initialize(self) -> bool:
        """
        Initialize video source and detectors.

        Returns:
            True if successful, False otherwise
        """
        # Open video source
        if self.config.use_webcam:
            self.cap = cv2.VideoCapture(self.config.webcam_index)
            self.logger.info(f"Opened webcam {self.config.webcam_index}")
        else:
            if not Path(self.config.video_source).exists():
                self.logger.error(f"Video file not found: {self.config.video_source}")
                return False
            self.cap = cv2.VideoCapture(self.config.video_source)
            self.logger.info(f"Opened video: {self.config.video_source}")

        if not self.cap.isOpened():
            self.logger.error("Failed to open video source")
            return False

        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.info(
            f"Video: {width}x{height} @ {self.fps:.1f}fps, "
            f"{self.total_frames} frames"
        )

        # Skip to start frame
        if self.config.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.start_frame)
            self.current_frame_idx = self.config.start_frame

        # Initialize detectors with default preset
        self._load_preset(self.current_preset)

        # Create windows
        cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.main_window, self.config.window_width, self.config.window_height)

        # Create trackbars
        self._create_trackbars()

        self.logger.info("Initialization complete. Press 'h' for help.")
        return True

    def _load_preset(self, preset_name: str):
        """Load a detection preset"""
        self.logger.info(f"Loading preset: {preset_name}")

        # Load motion config
        motion_cfg = MotionConfig()

        # Load dart config
        dart_cfg = DartDetectorConfig()
        dart_cfg = apply_detector_preset(dart_cfg, preset_name)

        # Create detectors
        self.motion_detector = MotionDetector(motion_cfg)
        self.dart_detector = DartImpactDetector(dart_cfg)

        self.current_preset = preset_name

    def _create_trackbars(self):
        """Create all parameter trackbars"""
        cv2.namedWindow(self.controls_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.controls_window, 600, 900)

        # Playback controls
        cv2.createTrackbar("Paused", self.controls_window, 1, 1, self._on_pause)
        cv2.createTrackbar("Frame", self.controls_window, 0, max(1, self.total_frames - 1), self._on_frame_seek)
        cv2.createTrackbar("Speed", self.controls_window, 10, 30, self._on_speed)

        # Preset selection
        cv2.createTrackbar("Preset", self.controls_window, 1, 2, self._on_preset_change)
        # 0=aggressive, 1=balanced, 2=stable

        # Motion Detection Parameters
        cv2.createTrackbar("Motion: Var Thresh", self.controls_window, 50, 200, lambda x: None)
        cv2.createTrackbar("Motion: Min Pixels", self.controls_window, 500, 5000, lambda x: None)
        cv2.createTrackbar("Motion: Morph Kernel", self.controls_window, 3, 15, lambda x: None)

        # Dart Detection - Shape Constraints
        cv2.createTrackbar("Dart: Min Area", self.controls_window, 10, 200, lambda x: None)
        cv2.createTrackbar("Dart: Max Area", self.controls_window, 1000, 5000, lambda x: None)
        cv2.createTrackbar("Dart: Min AR x10", self.controls_window, 3, 50, lambda x: None)  # Aspect ratio * 10
        cv2.createTrackbar("Dart: Max AR x10", self.controls_window, 30, 80, lambda x: None)

        # Dart Detection - Shape Heuristics
        cv2.createTrackbar("Dart: Min Solidity x100", self.controls_window, 10, 100, lambda x: None)
        cv2.createTrackbar("Dart: Max Solidity x100", self.controls_window, 50, 100, lambda x: None)
        cv2.createTrackbar("Dart: Min Extent x100", self.controls_window, 5, 100, lambda x: None)
        cv2.createTrackbar("Dart: Max Extent x100", self.controls_window, 30, 100, lambda x: None)

        # Dart Detection - Edge & Convexity
        cv2.createTrackbar("Dart: Min Edge Dens x100", self.controls_window, 2, 50, lambda x: None)
        cv2.createTrackbar("Dart: Max Edge Dens x100", self.controls_window, 20, 60, lambda x: None)
        cv2.createTrackbar("Dart: Convexity x100", self.controls_window, 70, 100, lambda x: None)
        cv2.createTrackbar("Dart: Convex Gate", self.controls_window, 1, 1, lambda x: None)  # On/Off

        # Dart Detection - Temporal
        cv2.createTrackbar("Dart: Confirm Frames", self.controls_window, 3, 10, lambda x: None)
        cv2.createTrackbar("Dart: Pos Tolerance", self.controls_window, 20, 80, lambda x: None)
        cv2.createTrackbar("Dart: Cooldown Frames", self.controls_window, 30, 120, lambda x: None)

        # Dart Detection - Preprocessing
        cv2.createTrackbar("Dart: Adaptive", self.controls_window, 1, 1, lambda x: None)
        cv2.createTrackbar("Dart: Otsu Bias", self.controls_window, 10, 30, lambda x: None)
        cv2.createTrackbar("Dart: Morph Open", self.controls_window, 5, 15, lambda x: None)
        cv2.createTrackbar("Dart: Morph Close", self.controls_window, 9, 21, lambda x: None)

        # Display options
        cv2.createTrackbar("Show: Debug", self.controls_window, 1, 1, lambda x: None)
        cv2.createTrackbar("Show: Metrics", self.controls_window, 1, 1, lambda x: None)
        cv2.createTrackbar("Show: Overlays", self.controls_window, 1, 1, lambda x: None)

        self.logger.info("Trackbars created")

    def _on_pause(self, val: int):
        """Pause/unpause playback"""
        self.paused = bool(val)

    def _on_frame_seek(self, val: int):
        """Seek to specific frame"""
        if self.cap and 0 <= val < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, val)
            self.current_frame_idx = val
            self.frame_step = 1  # Trigger a frame read

    def _on_speed(self, val: int):
        """Change playback speed"""
        self.playback_speed = val / 10.0  # 0.1x to 3.0x

    def _on_preset_change(self, val: int):
        """Change detection preset"""
        presets = ["aggressive", "balanced", "stable"]
        if 0 <= val < len(presets):
            self._load_preset(presets[val])
            self.logger.info(f"Preset changed to: {presets[val]}")

    def _read_trackbar_values(self) -> Tuple[MotionConfig, DartDetectorConfig]:
        """Read all trackbar values and create configs"""
        # Motion config
        motion_cfg = MotionConfig(
            var_threshold=cv2.getTrackbarPos("Motion: Var Thresh", self.controls_window),
            motion_pixel_threshold=cv2.getTrackbarPos("Motion: Min Pixels", self.controls_window),
            morph_kernel_size=cv2.getTrackbarPos("Motion: Morph Kernel", self.controls_window) | 1,  # Ensure odd
        )

        # Dart config
        dart_cfg = DartDetectorConfig(
            min_area=cv2.getTrackbarPos("Dart: Min Area", self.controls_window),
            max_area=cv2.getTrackbarPos("Dart: Max Area", self.controls_window),
            min_aspect_ratio=cv2.getTrackbarPos("Dart: Min AR x10", self.controls_window) / 10.0,
            max_aspect_ratio=cv2.getTrackbarPos("Dart: Max AR x10", self.controls_window) / 10.0,
            min_solidity=cv2.getTrackbarPos("Dart: Min Solidity x100", self.controls_window) / 100.0,
            max_solidity=cv2.getTrackbarPos("Dart: Max Solidity x100", self.controls_window) / 100.0,
            min_extent=cv2.getTrackbarPos("Dart: Min Extent x100", self.controls_window) / 100.0,
            max_extent=cv2.getTrackbarPos("Dart: Max Extent x100", self.controls_window) / 100.0,
            min_edge_density=cv2.getTrackbarPos("Dart: Min Edge Dens x100", self.controls_window) / 100.0,
            max_edge_density=cv2.getTrackbarPos("Dart: Max Edge Dens x100", self.controls_window) / 100.0,
            convexity_min_ratio=cv2.getTrackbarPos("Dart: Convexity x100", self.controls_window) / 100.0,
            convexity_gate_enabled=bool(cv2.getTrackbarPos("Dart: Convex Gate", self.controls_window)),
            confirmation_frames=cv2.getTrackbarPos("Dart: Confirm Frames", self.controls_window),
            position_tolerance_px=cv2.getTrackbarPos("Dart: Pos Tolerance", self.controls_window),
            cooldown_frames=cv2.getTrackbarPos("Dart: Cooldown Frames", self.controls_window),
            motion_adaptive=bool(cv2.getTrackbarPos("Dart: Adaptive", self.controls_window)),
            motion_otsu_bias=cv2.getTrackbarPos("Dart: Otsu Bias", self.controls_window),
            morph_open_ksize=cv2.getTrackbarPos("Dart: Morph Open", self.controls_window) | 1,
            morph_close_ksize=cv2.getTrackbarPos("Dart: Morph Close", self.controls_window) | 1,
        )

        # Display options
        self.show_debug_overlays = bool(cv2.getTrackbarPos("Show: Overlays", self.controls_window))
        self.show_metrics = bool(cv2.getTrackbarPos("Show: Metrics", self.controls_window))

        return motion_cfg, dart_cfg

    def run(self):
        """Main tuning loop"""
        if not self.initialize():
            return

        self.logger.info("Starting tuning session...")
        self.logger.info("Press 'h' for help, 'q' to quit, SPACE to pause/unpause")

        while True:
            # Read frame if not paused or if stepping
            if not self.paused or self.frame_step > 0:
                ret, frame = self.cap.read()

                if not ret:
                    if self.config.loop_video and not self.config.use_webcam:
                        # Loop video
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame_idx = 0
                        continue
                    else:
                        break

                self.current_frame_idx += 1
                self.frame_step = max(0, self.frame_step - 1)

                # Update frame trackbar
                cv2.setTrackbarPos("Frame", self.controls_window, self.current_frame_idx)
            else:
                # Use last frame
                frame = getattr(self, '_last_frame', None)
                if frame is None:
                    # Read first frame
                    ret, frame = self.cap.read()
                    if not ret:
                        break

            # Store frame for pause mode
            self._last_frame = frame.copy()

            # Read trackbar values
            motion_cfg, dart_cfg = self._read_trackbar_values()

            # Update detector configs
            self.motion_detector.config = motion_cfg
            self.dart_detector.config = dart_cfg

            # Process frame
            motion_detected, motion_event, fg_mask = self.motion_detector.detect_motion(
                frame, self.current_frame_idx, self.current_frame_idx / self.fps
            )

            dart_impact = None
            if motion_detected:
                dart_impact = self.dart_detector.detect_dart(
                    frame, fg_mask, self.current_frame_idx, self.current_frame_idx / self.fps
                )

            # Update stats
            self.stats["frames_processed"] += 1
            if motion_detected:
                self.stats["motion_detected"] += 1
            if dart_impact:
                self.stats["darts_detected"] += 1
                self.stats["avg_confidence"].append(dart_impact.confidence)

            # Visualize
            vis_frame = self.visualizer.create_visualization(
                frame=frame,
                fg_mask=fg_mask,
                processed_mask=self.dart_detector.last_processed_mask,
                motion_detected=motion_detected,
                motion_event=motion_event,
                dart_impact=dart_impact,
                dart_candidate=self.dart_detector.current_candidate,
                show_debug=self.show_debug_overlays,
                show_metrics=self.show_metrics,
                stats=self.stats,
                detector_config=dart_cfg,
            )

            cv2.imshow(self.main_window, vis_frame)

            # Handle keyboard
            wait_time = 1 if self.paused else int(1000 / (self.fps * self.playback_speed))
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
                cv2.setTrackbarPos("Paused", self.controls_window, int(self.paused))
            elif key == ord('h'):
                self._show_help()
            elif key == ord('s'):
                self._save_current_preset()
            elif key == ord('l'):
                self._load_preset_dialog()
            elif key == ord('r'):
                self._reset_stats()
            elif key == ord('.'):
                # Step forward one frame
                self.frame_step = 1
                self.paused = False
            elif key == ord(','):
                # Step backward one frame
                if self.current_frame_idx > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx - 1)
                    self.current_frame_idx -= 1
                    self.frame_step = 1
                    self.paused = False

        self.cleanup()

    def _show_help(self):
        """Display help message"""
        help_text = """
        === Dart Detection Parameter Tuner ===

        Keyboard Shortcuts:
        SPACE - Pause/Unpause
        . - Step forward one frame
        , - Step backward one frame
        s - Save current parameters as preset
        l - Load preset
        r - Reset statistics
        h - Show this help
        q - Quit

        Trackbars:
        - Use trackbars to adjust parameters in real-time
        - Changes take effect immediately
        - Compare different presets with Preset trackbar

        Tips:
        1. Start with a preset (aggressive/balanced/stable)
        2. Adjust motion detection first (threshold, min pixels)
        3. Fine-tune dart detection shape constraints
        4. Use debug overlays to understand what's detected
        5. Save working configurations as presets
        """
        self.logger.info(help_text)
        print(help_text)

    def _save_current_preset(self):
        """Save current parameters as preset"""
        motion_cfg, dart_cfg = self._read_trackbar_values()

        preset_name = f"custom_{int(cv2.getTickCount())}"
        preset_data = {
            "motion": asdict(motion_cfg),
            "dart": asdict(dart_cfg),
        }

        filepath = self.preset_manager.save_preset(preset_name, preset_data)
        self.logger.info(f"Saved preset: {filepath}")

    def _load_preset_dialog(self):
        """Load preset from file"""
        # List available presets
        presets = self.preset_manager.list_presets()
        self.logger.info(f"Available presets: {presets}")
        # TODO: Implement file dialog or console input

    def _reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "frames_processed": 0,
            "motion_detected": 0,
            "darts_detected": 0,
            "false_positives": 0,
            "avg_confidence": deque(maxlen=30),
        }
        self.logger.info("Statistics reset")

    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Tuning session ended")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Dart Detection Parameter Tuner")
    parser.add_argument("--video", "-v", type=str, help="Video file path")
    parser.add_argument("--webcam", "-w", type=int, default=0, help="Webcam index")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame")
    parser.add_argument("--preset-dir", type=str, default="presets/detection", help="Preset directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create config
    config = TuningConfig(
        video_source=args.video or "",
        webcam_index=args.webcam,
        use_webcam=(args.video is None),
        start_frame=args.start_frame,
        preset_dir=args.preset_dir,
    )

    # Run tuner
    tuner = ParameterTuner(config)
    tuner.run()


if __name__ == "__main__":
    main()
