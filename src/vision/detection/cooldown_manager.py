"""
Cooldown Manager - Prevent re-detection of same dart.

Manages cooldown regions around confirmed impacts to prevent
detecting the same dart multiple times in quick succession.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional


logger = logging.getLogger(__name__)


class CooldownManager:
    """
    Manage cooldown regions for dart impacts.

    Features:
    - Spatial cooldown (radius around impact)
    - Temporal cooldown (frames after impact)
    - Automatic cleanup of expired regions
    - Overlap checking
    """

    def __init__(
        self,
        cooldown_frames: int = 30,
        cooldown_radius_px: int = 50,
    ):
        """
        Initialize cooldown manager.

        Args:
            cooldown_frames: How long cooldown lasts (frames)
            cooldown_radius_px: Radius around impact (pixels)
        """
        self.cooldown_frames = cooldown_frames
        self.cooldown_radius_px = cooldown_radius_px

        # Active cooldown regions: [(position, expiry_frame), ...]
        self.cooldown_regions: List[Tuple[Tuple[int, int], int]] = []

        # Statistics
        self.stats = {
            "cooldowns_added": 0,
            "cooldowns_expired": 0,
            "candidates_blocked": 0,
        }

    def add_cooldown(self, position: Tuple[int, int], current_frame: int):
        """
        Add a cooldown region.

        Args:
            position: Impact position (x, y)
            current_frame: Current frame index
        """
        expiry_frame = current_frame + self.cooldown_frames
        self.cooldown_regions.append((position, expiry_frame))
        self.stats["cooldowns_added"] += 1

        logger.debug(
            f"Cooldown added at {position}, expires at frame {expiry_frame}"
        )

    def is_in_cooldown(self, position: Tuple[int, int], current_frame: int) -> bool:
        """
        Check if position is in cooldown.

        Args:
            position: Position to check (x, y)
            current_frame: Current frame index

        Returns:
            True if position is in active cooldown region
        """
        # Clean up expired regions first
        self._cleanup_expired(current_frame)

        # Check all active regions
        for cooldown_pos, expiry_frame in self.cooldown_regions:
            if expiry_frame > current_frame:
                distance = self._compute_distance(position, cooldown_pos)
                if distance <= self.cooldown_radius_px:
                    self.stats["candidates_blocked"] += 1
                    logger.debug(
                        f"Position {position} blocked by cooldown at {cooldown_pos}, "
                        f"dist={distance:.1f}px"
                    )
                    return True

        return False

    def _cleanup_expired(self, current_frame: int):
        """
        Remove expired cooldown regions.

        Args:
            current_frame: Current frame index
        """
        before_count = len(self.cooldown_regions)

        # Keep only non-expired regions
        self.cooldown_regions = [
            (pos, expiry)
            for pos, expiry in self.cooldown_regions
            if expiry > current_frame
        ]

        after_count = len(self.cooldown_regions)
        expired = before_count - after_count

        if expired > 0:
            self.stats["cooldowns_expired"] += expired
            logger.debug(f"Cleaned up {expired} expired cooldown(s)")

    def _compute_distance(
        self, pos1: Tuple[int, int], pos2: Tuple[int, int]
    ) -> float:
        """
        Compute Euclidean distance between positions.

        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)

        Returns:
            Distance in pixels
        """
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return float(np.sqrt(dx**2 + dy**2))

    def get_active_cooldowns(self) -> List[Tuple[Tuple[int, int], int]]:
        """
        Get list of active cooldown regions.

        Returns:
            List of (position, expiry_frame) tuples
        """
        return self.cooldown_regions.copy()

    def reset(self):
        """Clear all cooldown regions"""
        self.cooldown_regions.clear()
        logger.debug("All cooldowns cleared")

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "cooldowns_added": 0,
            "cooldowns_expired": 0,
            "candidates_blocked": 0,
        }

    def get_stats(self) -> dict:
        """Get current statistics"""
        return self.stats.copy()
