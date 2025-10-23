# Module: `src\vision\motion_detector.py`
Hash: `5c5514f3b7db` · LOC: 1 · Main guard: false

## Imports
- `cv2`\n- `logging`\n- `numpy`

## From-Imports
- `from typing import Optional, Tuple, Callable, Any`\n- `from dataclasses import dataclass`\n- `from collections import deque`

## Classes
- `MotionEvent` (L19): Motion detection event\n- `MotionConfig` (L30): Motion detection configuration\n- `MotionDetector` (L49): Motion detector with gating logic for CPU optimization.

## Functions
- `__init__()` (L60)\n- `detect_motion()` (L85): Detect motion in frame.\n- `_extract_motion_event()` (L131): Extract motion event details from foreground mask\n- `gate_processing()` (L181): Gate expensive processing behind motion detection.\n- `get_motion_history()` (L213): Get last N motion events\n- `is_motion_recent()` (L217): Check if motion occurred recently\n- `get_stats()` (L225): Get motion detection statistics\n- `reset()` (L236): Reset detector state

## Intra-module calls (heuristic)
MotionConfig, MotionEvent, _extract_motion_event, append, apply, boundingRect, clear, contourArea, countNonZero, count_nonzero, createBackgroundSubtractorMOG2, deque, detect_motion, findContours, getLogger, getStructuringElement, info, int, is_motion_recent, list, max, moments, morphologyEx, process_func

## Code
```python
"""
Enhanced Motion Detector with Gating Logic
CPU-efficient motion detection optimized for dart throws.

Performance: Only triggers expensive operations when motion detected
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


@dataclass
class MotionConfig:
    """Motion detection configuration"""
    # Background subtraction
    var_threshold: int = 50  # MOG2 variance threshold (tune with video_analyzer!)
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


class MotionDetector:
    """
    Motion detector with gating logic for CPU optimization.

    Features:
    - MOG2 background subtraction
    - Configurable sensitivity
    - Motion event tracking
    - Gating mechanism for expensive operations
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

    def detect_motion(
            self,
            frame: np.ndarray,
            frame_index: int,
            timestamp: float
    ) -> Tuple[bool, Optional[MotionEvent], np.ndarray]:
        """
        Detect motion in frame.

        Args:
            frame: Input frame (BGR or grayscale)
            frame_index: Frame number for tracking
            timestamp: Frame timestamp

        Returns:
            (motion_detected, motion_event, fg_mask) tuple
        """
        self.frames_processed += 1

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Clean noise with morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.morph_kernel)

        # Count motion pixels
        motion_pixels = cv2.countNonZero(fg_mask)

        # Check if motion exceeds threshold
        motion_detected = motion_pixels > self.config.motion_pixel_threshold

        if motion_detected:
            self.motion_frames += 1
            self.last_motion_frame = frame_index

            # Find motion event details
            event = self._extract_motion_event(fg_mask, frame_index, timestamp)

            if event:
                self.motion_events.append(event)

            return True, event, fg_mask

        return False, None, fg_mask

    def _extract_motion_event(
            self,
            fg_mask: np.ndarray,
            frame_index: int,
            timestamp: float
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
            frame_index=frame_index
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
        """Get motion detection statistics"""
        return {
            'frames_processed': self.frames_processed,
            'motion_frames': self.motion_frames,
            'motion_rate': self.motion_frames / max(self.frames_processed, 1),
            'gated_operations': self.gated_operations,
            'gate_efficiency': 1.0 - (self.gated_operations / max(self.frames_processed, 1)),
            'recent_motion': self.is_motion_recent()
        }

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
        logger.info("Motion detector reset")
```
