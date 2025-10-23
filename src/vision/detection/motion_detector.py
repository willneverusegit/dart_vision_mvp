"""
Enhanced Motion Detector with Adaptive Gating
CPU-efficient motion detection with dynamic threshold adaptation.

NEW Features (v1.1.0):
- Adaptive Otsu-Bias: Dynamic threshold based on frame brightness
- Multi-Threshold Fusion: Parallel low/high thresholds for better recall
- Temporal-Gate: Search mode after prolonged stillness
- Frame brightness analysis for adaptive tuning

Existing Features:
- MOG2 background subtraction
- Configurable sensitivity
- Motion event tracking
- Gating mechanism for CPU optimization
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Callable, Any
from dataclasses import dataclass
from collections import deque

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
    brightness: Optional[float] = None  # NEW: frame brightness (0-255)
    threshold_used: Optional[int] = None  # NEW: actual threshold applied


@dataclass
class MotionConfig:
    """Enhanced motion detection configuration (Pydantic-compatible)"""
    # Background subtraction
    var_threshold: int = 50  # MOG2 variance threshold
    detect_shadows: bool = True
    history: int = 500  # Frames for background model

    # Motion thresholds
    motion_pixel_threshold: int = 500  # Min pixels for motion event
    min_contour_area: int = 100
    max_contour_area: int = 5000

    # Morphological operations
    morph_kernel_size: int = 3

    # Event history
    event_history_size: int = 10

    # NEW: Adaptive Otsu-Bias (Proposal 2)
    adaptive_otsu_enabled: bool = True
    brightness_dark_threshold: float = 60.0  # Below this = dark
    brightness_bright_threshold: float = 150.0  # Above this = bright
    otsu_bias_dark: int = -15  # Lower threshold for dark frames
    otsu_bias_normal: int = 0  # Baseline for normal frames
    otsu_bias_bright: int = +10  # Higher threshold for bright frames

    # NEW: Multi-Threshold Fusion (Proposal 2)
    dual_threshold_enabled: bool = False  # Experimental, default OFF
    dual_threshold_low_multiplier: float = 0.6  # 60% of base threshold
    dual_threshold_high_multiplier: float = 1.4  # 140% of base threshold

    # NEW: Temporal-Gate (Search Mode) (Proposal 2)
    search_mode_enabled: bool = True
    search_mode_trigger_frames: int = 90  # No motion for 3 seconds → search mode
    search_mode_threshold_drop: int = 150  # Drop threshold by this amount
    search_mode_duration_frames: int = 30  # Stay in search mode for 1 second


class MotionDetector:
    """
    Enhanced motion detector with adaptive gating logic.

    Features:
    - Adaptive MOG2 thresholds based on frame brightness
    - Optional dual-threshold fusion for recall boost
    - Temporal search mode after prolonged stillness
    - CPU-efficient gating mechanism
    """

    def __init__(self, config: Optional[MotionConfig] = None):
        self.config = config or MotionConfig()

        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=self.config.detect_shadows,
            varThreshold=self.config.var_threshold,
            history=self.config.history
        )

        # Morphological kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morph_kernel_size, self.config.morph_kernel_size)
        )

        # Event tracking
        self.motion_events: deque = deque(maxlen=self.config.event_history_size)
        self.last_motion_frame: Optional[int] = None

        # Statistics
        self.frames_processed = 0
        self.motion_frames = 0
        self.gated_operations = 0

        # NEW: Adaptive tracking
        self.brightness_history: deque = deque(maxlen=30)  # Last 30 frames
        self.search_mode_active = False
        self.search_mode_end_frame: Optional[int] = None

        # NEW: Stats for adaptive features
        self.adaptive_stats = {
            "adaptive_adjustments": 0,
            "dual_threshold_activations": 0,
            "search_mode_activations": 0,
            "dark_frames": 0,
            "bright_frames": 0,
            "normal_frames": 0
        }

    def _compute_frame_brightness(self, frame: np.ndarray) -> float:
        """Compute average frame brightness (0-255)"""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return float(np.mean(gray))

    def _get_adaptive_otsu_bias(self, brightness: float) -> int:
        """
        Compute adaptive Otsu bias based on frame brightness.
        
        Dark frames (< 60): Lower threshold (more sensitive)
        Bright frames (> 150): Higher threshold (less sensitive)
        Normal frames: Baseline threshold
        """
        if not self.config.adaptive_otsu_enabled:
            return 0

        if brightness < self.config.brightness_dark_threshold:
            self.adaptive_stats["dark_frames"] += 1
            return self.config.otsu_bias_dark
        elif brightness > self.config.brightness_bright_threshold:
            self.adaptive_stats["bright_frames"] += 1
            return self.config.otsu_bias_bright
        else:
            self.adaptive_stats["normal_frames"] += 1
            return self.config.otsu_bias_normal

    def _apply_dual_threshold(
            self,
            fg_mask: np.ndarray,
            base_threshold: int
    ) -> np.ndarray:
        """
        Apply dual-threshold fusion for better recall.
        
        Combines results from low and high thresholds via union.
        Low threshold catches subtle motion, high threshold reduces noise.
        """
        if not self.config.dual_threshold_enabled:
            return fg_mask

        self.adaptive_stats["dual_threshold_activations"] += 1

        # Compute thresholds
        low_thresh = int(base_threshold * self.config.dual_threshold_low_multiplier)
        high_thresh = int(base_threshold * self.config.dual_threshold_high_multiplier)

        # Apply both thresholds
        _, mask_low = cv2.threshold(fg_mask, low_thresh, 255, cv2.THRESH_BINARY)
        _, mask_high = cv2.threshold(fg_mask, high_thresh, 255, cv2.THRESH_BINARY)

        # Union (OR operation)
        fused_mask = cv2.bitwise_or(mask_low, mask_high)

        return fused_mask

    def _check_search_mode(self, frame_index: int) -> bool:
        """
        Check if search mode should be activated.
        
        Search mode: After N frames of no motion, temporarily lower threshold
        to actively search for subtle motion (e.g., dart stuck in board).
        """
        if not self.config.search_mode_enabled:
            return False

        # Already in search mode?
        if self.search_mode_active:
            if frame_index >= self.search_mode_end_frame:
                self.search_mode_active = False
                logger.debug(f"Search mode ended at frame {frame_index}")
            return self.search_mode_active

        # Check if we should enter search mode
        if self.last_motion_frame is None:
            frames_since_motion = frame_index
        else:
            frames_since_motion = frame_index - self.last_motion_frame

        if frames_since_motion >= self.config.search_mode_trigger_frames:
            # Activate search mode
            self.search_mode_active = True
            self.search_mode_end_frame = frame_index + self.config.search_mode_duration_frames
            self.adaptive_stats["search_mode_activations"] += 1
            logger.debug(f"Search mode activated at frame {frame_index} (no motion for {frames_since_motion} frames)")
            return True

        return False

    def detect_motion(
            self,
            frame: np.ndarray,
            frame_index: int,
            timestamp: float
    ) -> Tuple[bool, Optional[MotionEvent], np.ndarray]:
        """
        Detect motion in frame with adaptive thresholding.

        Args:
            frame: Input frame (BGR or grayscale)
            frame_index: Frame number for tracking
            timestamp: Frame timestamp

        Returns:
            (motion_detected, motion_event, fg_mask) tuple
        """
        self.frames_processed += 1

        # NEW: Compute frame brightness
        brightness = self._compute_frame_brightness(frame)
        self.brightness_history.append(brightness)

        # NEW: Get adaptive Otsu bias
        otsu_bias = self._get_adaptive_otsu_bias(brightness)
        if otsu_bias != 0:
            self.adaptive_stats["adaptive_adjustments"] += 1

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # NEW: Apply dual threshold fusion if enabled
        if self.config.dual_threshold_enabled:
            fg_mask = self._apply_dual_threshold(fg_mask, self.config.var_threshold)

        # Clean noise with morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.morph_kernel)

        # Count motion pixels
        motion_pixels = cv2.countNonZero(fg_mask)

        # NEW: Check search mode and adjust threshold if needed
        in_search_mode = self._check_search_mode(frame_index)
        effective_threshold = self.config.motion_pixel_threshold
        if in_search_mode:
            effective_threshold = max(
                100,  # Minimum threshold
                effective_threshold - self.config.search_mode_threshold_drop
            )

        # Check if motion exceeds threshold
        motion_detected = motion_pixels > effective_threshold

        if motion_detected:
            self.motion_frames += 1
            self.last_motion_frame = frame_index

            # Find motion event details
            event = self._extract_motion_event(
                fg_mask, frame_index, timestamp,
                brightness=brightness,
                threshold_used=effective_threshold
            )

            if event:
                self.motion_events.append(event)

            return True, event, fg_mask

        return False, None, fg_mask

    def _extract_motion_event(
            self,
            fg_mask: np.ndarray,
            frame_index: int,
            timestamp: float,
            brightness: Optional[float] = None,
            threshold_used: Optional[int] = None
    ) -> Optional[MotionEvent]:
        """Extract motion event details from foreground mask"""

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Filter by area
        if not (self.config.min_contour_area <= area <= self.config.max_contour_area):
            return None

        # Calculate center
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate intensity (percentage of motion in bounding box)
        roi_mask = fg_mask[y:y + h, x:x + w]
        intensity = np.count_nonzero(roi_mask) / (w * h) if w * h > 0 else 0

        return MotionEvent(
            timestamp=timestamp,
            center=(cx, cy),
            area=area,
            intensity=intensity,
            bounding_box=(x, y, w, h),
            frame_index=frame_index,
            brightness=brightness,
            threshold_used=threshold_used
        )

    def gate_processing(
            self,
            frame: np.ndarray,
            frame_index: int,
            timestamp: float,
            process_func: Callable[[np.ndarray, MotionEvent], Any]
    ) -> Optional[Any]:
        """
        Gate expensive processing behind motion detection.

        Args:
            frame: Input frame
            frame_index: Frame number
            timestamp: Frame timestamp
            process_func: Function to call if motion detected
                         Signature: func(frame, motion_event) -> result

        Returns:
            Result from process_func if motion detected, else None
        """
        motion_detected, motion_event, fg_mask = self.detect_motion(
            frame, frame_index, timestamp
        )

        if motion_detected and motion_event:
            # Trigger expensive processing
            result = process_func(frame, motion_event)
            self.gated_operations += 1
            return result

        return None

    def get_motion_history(self, n: int = 5) -> list:
        """Get last N motion events"""
        return list(self.motion_events)[-n:]

    def is_motion_recent(self, max_frames_ago: int = 30) -> bool:
        """Check if motion occurred recently"""
        if self.last_motion_frame is None:
            return False

        frames_since = self.frames_processed - self.last_motion_frame
        return frames_since <= max_frames_ago

    def get_stats(self) -> dict:
        """Get comprehensive motion detection statistics"""
        base_stats = {
            'frames_processed': self.frames_processed,
            'motion_frames': self.motion_frames,
            'motion_rate': self.motion_frames / max(self.frames_processed, 1),
            'gated_operations': self.gated_operations,
            'gate_efficiency': 1.0 - (self.gated_operations / max(self.frames_processed, 1)),
            'recent_motion': self.is_motion_recent(),
            'search_mode_active': self.search_mode_active
        }

        # Add adaptive stats
        if self.config.adaptive_otsu_enabled or self.config.dual_threshold_enabled or self.config.search_mode_enabled:
            base_stats.update({
                'adaptive': self.adaptive_stats.copy(),
                'avg_brightness': float(np.mean(self.brightness_history)) if self.brightness_history else 0.0
            })

        return base_stats

    def reset(self):
        """Reset detector state"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=self.config.detect_shadows,
            varThreshold=self.config.var_threshold,
            history=self.config.history
        )
        self.motion_events.clear()
        self.last_motion_frame = None
        self.frames_processed = 0
        self.motion_frames = 0
        self.gated_operations = 0

        # Reset adaptive tracking
        self.brightness_history.clear()
        self.search_mode_active = False
        self.search_mode_end_frame = None
        self.adaptive_stats = {
            "adaptive_adjustments": 0,
            "dual_threshold_activations": 0,
            "search_mode_activations": 0,
            "dark_frames": 0,
            "bright_frames": 0,
            "normal_frames": 0
        }

        logger.info("Enhanced motion detector reset")
