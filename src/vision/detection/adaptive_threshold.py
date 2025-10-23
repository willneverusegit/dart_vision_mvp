"""
Adaptive Threshold - Brightness-based threshold adaptation.

Automatically adjusts detection thresholds based on frame brightness
to maintain consistent detection across varying lighting conditions.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional
from collections import deque


logger = logging.getLogger(__name__)


class AdaptiveThreshold:
    """
    Adaptive threshold calculator based on frame brightness.

    Adjusts detection sensitivity dynamically:
    - Dark frames (< 60): More sensitive (lower threshold)
    - Normal frames (60-150): Baseline sensitivity
    - Bright frames (> 150): Less sensitive (higher threshold)
    """

    def __init__(
        self,
        enabled: bool = True,
        dark_threshold: float = 60.0,
        bright_threshold: float = 150.0,
        dark_bias: int = -15,
        normal_bias: int = 0,
        bright_bias: int = 10,
        history_size: int = 30,
    ):
        """
        Initialize adaptive threshold calculator.

        Args:
            enabled: Enable adaptive thresholding
            dark_threshold: Brightness below this is considered dark
            bright_threshold: Brightness above this is considered bright
            dark_bias: Bias for dark frames (negative = more sensitive)
            normal_bias: Bias for normal frames
            bright_bias: Bias for bright frames (positive = less sensitive)
            history_size: Number of frames for brightness history
        """
        self.enabled = enabled
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.dark_bias = dark_bias
        self.normal_bias = normal_bias
        self.bright_bias = bright_bias

        # Brightness tracking
        self.brightness_history: deque = deque(maxlen=history_size)

        # Statistics
        self.stats = {
            "dark_frames": 0,
            "normal_frames": 0,
            "bright_frames": 0,
            "adjustments": 0,
        }

    def compute_brightness(self, frame: np.ndarray) -> float:
        """
        Compute average frame brightness (0-255).

        Args:
            frame: BGR or grayscale frame

        Returns:
            Average brightness value
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        brightness = float(np.mean(gray))
        self.brightness_history.append(brightness)

        return brightness

    def get_threshold_bias(self, brightness: float) -> int:
        """
        Get threshold bias based on frame brightness.

        Args:
            brightness: Current frame brightness (0-255)

        Returns:
            Threshold bias value
        """
        if not self.enabled:
            return 0

        # Categorize frame
        if brightness < self.dark_threshold:
            self.stats["dark_frames"] += 1
            bias = self.dark_bias
        elif brightness > self.bright_threshold:
            self.stats["bright_frames"] += 1
            bias = self.bright_bias
        else:
            self.stats["normal_frames"] += 1
            bias = self.normal_bias

        if bias != 0:
            self.stats["adjustments"] += 1

        return bias

    def get_adaptive_threshold(
        self, base_threshold: int, brightness: float
    ) -> int:
        """
        Calculate adaptive threshold from base threshold and brightness.

        Args:
            base_threshold: Base threshold value
            brightness: Current frame brightness

        Returns:
            Adjusted threshold value
        """
        bias = self.get_threshold_bias(brightness)
        adjusted = int(np.clip(base_threshold + bias, 0, 255))

        return adjusted

    def get_average_brightness(self) -> Optional[float]:
        """
        Get average brightness over history.

        Returns:
            Average brightness or None if no history
        """
        if not self.brightness_history:
            return None
        return float(np.mean(list(self.brightness_history)))

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "dark_frames": 0,
            "normal_frames": 0,
            "bright_frames": 0,
            "adjustments": 0,
        }

    def get_stats(self) -> dict:
        """Get current statistics"""
        return self.stats.copy()
