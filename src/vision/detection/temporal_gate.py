"""
Temporal Gate - Search mode after prolonged stillness.

Implements smart gating logic that increases sensitivity after
periods of no motion to catch subtle movements.
"""

import logging
from typing import Optional


logger = logging.getLogger(__name__)


class TemporalGate:
    """
    Temporal gating with search mode activation.

    Features:
    - Track frames since last motion
    - Activate search mode after stillness
    - Temporarily reduce thresholds
    - Auto-deactivation after duration
    """

    def __init__(
        self,
        enabled: bool = True,
        trigger_frames: int = 90,
        duration_frames: int = 30,
        threshold_drop: int = 150,
    ):
        """
        Initialize temporal gate.

        Args:
            enabled: Enable temporal gating
            trigger_frames: Frames without motion to trigger search mode
            duration_frames: How long search mode stays active
            threshold_drop: How much to reduce threshold in search mode
        """
        self.enabled = enabled
        self.trigger_frames = trigger_frames
        self.duration_frames = duration_frames
        self.threshold_drop = threshold_drop

        # State
        self.frames_since_motion = 0
        self.search_mode_active = False
        self.search_mode_end_frame: Optional[int] = None
        self.current_frame = 0

        # Statistics
        self.stats = {
            "search_mode_activations": 0,
            "frames_in_search_mode": 0,
        }

    def update(self, motion_detected: bool, frame_index: int):
        """
        Update gate state based on motion detection.

        Args:
            motion_detected: Whether motion was detected this frame
            frame_index: Current frame index
        """
        self.current_frame = frame_index

        if not self.enabled:
            return

        if motion_detected:
            # Motion detected - reset counter and deactivate search mode
            self.frames_since_motion = 0
            self.search_mode_active = False
            self.search_mode_end_frame = None
        else:
            # No motion - increment counter
            self.frames_since_motion += 1

            # Check if we should activate search mode
            if (
                not self.search_mode_active
                and self.frames_since_motion >= self.trigger_frames
            ):
                self._activate_search_mode()

        # Check if search mode should end
        if self.search_mode_active:
            self.stats["frames_in_search_mode"] += 1
            if (
                self.search_mode_end_frame is not None
                and frame_index >= self.search_mode_end_frame
            ):
                self._deactivate_search_mode()

    def _activate_search_mode(self):
        """Activate search mode"""
        self.search_mode_active = True
        self.search_mode_end_frame = self.current_frame + self.duration_frames
        self.stats["search_mode_activations"] += 1

        logger.debug(
            f"Search mode activated at frame {self.current_frame}, "
            f"will end at {self.search_mode_end_frame}"
        )

    def _deactivate_search_mode(self):
        """Deactivate search mode"""
        self.search_mode_active = False
        self.search_mode_end_frame = None

        logger.debug(f"Search mode deactivated at frame {self.current_frame}")

    def is_search_mode_active(self) -> bool:
        """
        Check if search mode is currently active.

        Returns:
            True if search mode is active
        """
        return self.search_mode_active if self.enabled else False

    def get_threshold_adjustment(self) -> int:
        """
        Get threshold adjustment for current state.

        Returns:
            Threshold reduction amount (negative value)
        """
        if self.is_search_mode_active():
            return -self.threshold_drop
        return 0

    def reset(self):
        """Reset gate state"""
        self.frames_since_motion = 0
        self.search_mode_active = False
        self.search_mode_end_frame = None

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "search_mode_activations": 0,
            "frames_in_search_mode": 0,
        }

    def get_stats(self) -> dict:
        """Get current statistics"""
        return self.stats.copy()
