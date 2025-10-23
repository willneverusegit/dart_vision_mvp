"""
Enhanced Motion Detector (Refactored) - Modular motion detection.

Orchestrates background subtraction, adaptive thresholding, filtering,
and temporal gating for robust motion detection.

NEW: Modular architecture with clean separation of concerns.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from collections import deque

from .background_subtractor import BackgroundSubtractor
from .adaptive_threshold import AdaptiveThreshold
from .motion_filter import MotionFilter
from .temporal_gate import TemporalGate


logger = logging.getLogger(__name__)


@dataclass
class MotionEvent:
    """Motion detection event"""
    timestamp: float
    center: Tuple[int, int]
    area: float
    intensity: float  # 0.0 - 1.0
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    frame_index: int
    brightness: Optional[float] = None
    threshold_used: Optional[int] = None


@dataclass
class MotionConfig:
    """Motion detection configuration"""
    # Background subtraction
    var_threshold: int = 50
    detect_shadows: bool = True
    history: int = 500

    # Motion thresholds
    motion_pixel_threshold: int = 500
    min_contour_area: int = 100
    max_contour_area: int = 5000

    # Morphological operations
    morph_kernel_size: int = 3

    # Event history
    event_history_size: int = 10

    # Adaptive Otsu-Bias
    adaptive_otsu_enabled: bool = True
    brightness_dark_threshold: float = 60.0
    brightness_bright_threshold: float = 150.0
    otsu_bias_dark: int = -15
    otsu_bias_normal: int = 0
    otsu_bias_bright: int = 10

    # Temporal-Gate (Search Mode)
    search_mode_enabled: bool = True
    search_mode_trigger_frames: int = 90
    search_mode_threshold_drop: int = 150
    search_mode_duration_frames: int = 30


class MotionDetector:
    """
    Refactored motion detector with modular components.

    Components:
    - BackgroundSubtractor: MOG2 background subtraction
    - AdaptiveThreshold: Brightness-based threshold adaptation
    - MotionFilter: Morphological cleanup
    - TemporalGate: Search mode after stillness
    """

    def __init__(self, config: Optional[MotionConfig] = None):
        """
        Initialize motion detector.

        Args:
            config: Motion detection configuration
        """
        self.config = config or MotionConfig()

        # Initialize components
        self.bg_subtractor = BackgroundSubtractor(
            var_threshold=self.config.var_threshold,
            detect_shadows=self.config.detect_shadows,
            history=self.config.history,
        )

        self.adaptive_threshold = AdaptiveThreshold(
            enabled=self.config.adaptive_otsu_enabled,
            dark_threshold=self.config.brightness_dark_threshold,
            bright_threshold=self.config.brightness_bright_threshold,
            dark_bias=self.config.otsu_bias_dark,
            normal_bias=self.config.otsu_bias_normal,
            bright_bias=self.config.otsu_bias_bright,
        )

        self.motion_filter = MotionFilter(
            morph_kernel_size=self.config.morph_kernel_size,
            min_blob_area=self.config.min_contour_area,
        )

        self.temporal_gate = TemporalGate(
            enabled=self.config.search_mode_enabled,
            trigger_frames=self.config.search_mode_trigger_frames,
            duration_frames=self.config.search_mode_duration_frames,
            threshold_drop=self.config.search_mode_threshold_drop,
        )

        # Event tracking
        self.motion_events: deque = deque(maxlen=self.config.event_history_size)
        self.last_motion_frame: Optional[int] = None

        # Statistics
        self.frames_processed = 0
        self.motion_frames = 0

    def detect_motion(
        self, frame: np.ndarray, frame_index: int, timestamp: float
    ) -> Tuple[bool, Optional[MotionEvent], np.ndarray]:
        """
        Detect motion in frame.

        Args:
            frame: Input BGR frame
            frame_index: Current frame index
            timestamp: Current timestamp

        Returns:
            Tuple of (motion_detected, motion_event, foreground_mask)
        """
        self.frames_processed += 1

        # 1. Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # 2. Compute brightness for adaptive threshold
        brightness = self.adaptive_threshold.compute_brightness(frame)

        # 3. Apply morphological filtering
        cleaned_mask = self.motion_filter.apply(fg_mask)

        # 4. Count motion pixels
        motion_pixels = cv2.countNonZero(cleaned_mask)

        # 5. Get adaptive threshold
        threshold = self.config.motion_pixel_threshold
        if self.config.adaptive_otsu_enabled:
            threshold_bias = self.adaptive_threshold.get_threshold_bias(brightness)
            threshold = max(0, threshold + threshold_bias)

        # 6. Apply temporal gate adjustment
        if self.config.search_mode_enabled:
            gate_adjustment = self.temporal_gate.get_threshold_adjustment()
            threshold = max(0, threshold + gate_adjustment)

        # 7. Check if motion detected
        motion_detected = motion_pixels >= threshold

        # 8. Update temporal gate
        self.temporal_gate.update(motion_detected, frame_index)

        # 9. Create motion event if detected
        motion_event = None
        if motion_detected:
            self.motion_frames += 1
            self.last_motion_frame = frame_index

            # Find largest contour
            contours, _ = cv2.findContours(
                cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                # Filter by area
                if self.config.min_contour_area <= area <= self.config.max_contour_area:
                    # Calculate bounding box and center
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    center = (x + w // 2, y + h // 2)

                    # Calculate intensity (normalized area)
                    max_area = frame.shape[0] * frame.shape[1]
                    intensity = min(1.0, area / max_area)

                    # Create event
                    motion_event = MotionEvent(
                        timestamp=timestamp,
                        center=center,
                        area=area,
                        intensity=intensity,
                        bounding_box=(x, y, w, h),
                        frame_index=frame_index,
                        brightness=brightness,
                        threshold_used=threshold,
                    )

                    self.motion_events.append(motion_event)

        return motion_detected, motion_event, cleaned_mask

    def get_statistics(self) -> dict:
        """
        Get combined statistics from all components.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "frames_processed": self.frames_processed,
            "motion_frames": self.motion_frames,
            "motion_rate": (
                self.motion_frames / max(1, self.frames_processed)
            ) * 100,
            "avg_brightness": self.adaptive_threshold.get_average_brightness(),
            "search_mode_active": self.temporal_gate.is_search_mode_active(),
        }

        # Add component stats
        stats["adaptive_threshold"] = self.adaptive_threshold.get_stats()
        stats["motion_filter"] = self.motion_filter.get_stats()
        stats["temporal_gate"] = self.temporal_gate.get_stats()

        return stats

    def reset_statistics(self):
        """Reset all statistics"""
        self.frames_processed = 0
        self.motion_frames = 0
        self.adaptive_threshold.reset_stats()
        self.motion_filter.reset_stats()
        self.temporal_gate.reset_stats()

    def update_config(self, new_config: MotionConfig):
        """
        Update configuration and propagate to components.

        Args:
            new_config: New motion configuration
        """
        # Update background subtractor if needed
        if (
            new_config.var_threshold != self.config.var_threshold
            or new_config.detect_shadows != self.config.detect_shadows
        ):
            self.bg_subtractor.update_parameters(
                var_threshold=new_config.var_threshold,
                detect_shadows=new_config.detect_shadows,
            )

        # Update motion filter if needed
        if new_config.morph_kernel_size != self.config.morph_kernel_size:
            self.motion_filter.update_kernel_size(new_config.morph_kernel_size)

        # Update adaptive threshold (recreate if needed)
        if (
            new_config.adaptive_otsu_enabled != self.config.adaptive_otsu_enabled
            or new_config.brightness_dark_threshold != self.config.brightness_dark_threshold
            or new_config.brightness_bright_threshold != self.config.brightness_bright_threshold
        ):
            self.adaptive_threshold = AdaptiveThreshold(
                enabled=new_config.adaptive_otsu_enabled,
                dark_threshold=new_config.brightness_dark_threshold,
                bright_threshold=new_config.brightness_bright_threshold,
                dark_bias=new_config.otsu_bias_dark,
                normal_bias=new_config.otsu_bias_normal,
                bright_bias=new_config.otsu_bias_bright,
            )

        # Update temporal gate (recreate if needed)
        if (
            new_config.search_mode_enabled != self.config.search_mode_enabled
            or new_config.search_mode_trigger_frames != self.config.search_mode_trigger_frames
        ):
            self.temporal_gate = TemporalGate(
                enabled=new_config.search_mode_enabled,
                trigger_frames=new_config.search_mode_trigger_frames,
                duration_frames=new_config.search_mode_duration_frames,
                threshold_drop=new_config.search_mode_threshold_drop,
            )

        # Store new config
        self.config = new_config
