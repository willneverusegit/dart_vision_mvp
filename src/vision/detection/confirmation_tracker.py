"""
Confirmation Tracker - Multi-frame dart confirmation (Land-and-Stick).

Tracks dart candidates across frames to confirm stable impacts
and reject transient blobs (hands, shadows, etc.).
"""

import numpy as np
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from collections import deque


logger = logging.getLogger(__name__)


@dataclass
class DartCandidate:
    """Potential dart detection"""
    position: Tuple[int, int]
    area: float
    confidence: float
    frame_index: int
    timestamp: float
    aspect_ratio: float
    extent: float
    solidity: float
    edge_density: float
    convexity: Optional[float] = None


@dataclass
class DartImpact:
    """Confirmed dart impact"""
    position: Tuple[int, int]
    confidence: float
    first_detected_frame: int
    confirmed_frame: int
    confirmation_count: int
    timestamp: float
    refine_score: Optional[float] = None
    raw_position: Optional[Tuple[int, int]] = None
    refined_position: Optional[Tuple[int, int]] = None


class ConfirmationTracker:
    """
    Track and confirm dart impacts across frames.

    Features:
    - Multi-frame temporal confirmation (Land-and-Stick)
    - Position stability checking
    - Candidate history management
    - Automatic reset on position jump
    """

    def __init__(
        self,
        confirmation_frames: int = 3,
        position_tolerance_px: int = 20,
        history_size: int = 20,
    ):
        """
        Initialize confirmation tracker.

        Args:
            confirmation_frames: Frames needed to confirm dart
            position_tolerance_px: Max pixel movement between frames
            history_size: Max candidate history size
        """
        self.confirmation_frames = confirmation_frames
        self.position_tolerance_px = position_tolerance_px
        self.history_size = history_size

        # Current tracking state
        self.current_candidate: Optional[DartCandidate] = None
        self.confirmation_count = 0

        # History
        self.candidate_history: deque = deque(maxlen=history_size)

        # Statistics
        self.stats = {
            "candidates_tracked": 0,
            "confirmations": 0,
            "resets_position_jump": 0,
            "resets_timeout": 0,
        }

    def update(
        self,
        new_candidate: Optional[DartCandidate],
        frame_index: int,
        timestamp: float,
    ) -> Optional[DartImpact]:
        """
        Update tracker with new candidate.

        Args:
            new_candidate: New dart candidate or None
            frame_index: Current frame index
            timestamp: Current timestamp

        Returns:
            DartImpact if confirmed, None otherwise
        """
        # No candidate this frame
        if new_candidate is None:
            # Keep current candidate for now (allows temporary occlusion)
            return None

        # Track candidate
        self.stats["candidates_tracked"] += 1
        self.candidate_history.append(new_candidate)

        # First candidate
        if self.current_candidate is None:
            self.current_candidate = new_candidate
            self.confirmation_count = 1
            logger.debug(
                f"New candidate at {new_candidate.position}, conf={new_candidate.confidence:.2f}"
            )
            return None

        # Check if same candidate (position stability)
        distance = self._compute_distance(
            self.current_candidate.position, new_candidate.position
        )

        if distance <= self.position_tolerance_px:
            # Same candidate - increment confirmation
            self.confirmation_count += 1
            logger.debug(
                f"Candidate confirmed {self.confirmation_count}/{self.confirmation_frames}, "
                f"dist={distance:.1f}px"
            )

            # Check if fully confirmed
            if self.confirmation_count >= self.confirmation_frames:
                impact = self._create_impact(new_candidate, frame_index, timestamp)
                self.stats["confirmations"] += 1
                logger.info(
                    f"Dart CONFIRMED at {impact.position}, conf={impact.confidence:.2f}"
                )

                # Reset tracker
                self.current_candidate = None
                self.confirmation_count = 0

                return impact

            # Update current candidate (use newer one)
            self.current_candidate = new_candidate

        else:
            # Position jumped - new candidate
            logger.debug(
                f"Position jump: {distance:.1f}px > {self.position_tolerance_px}px, "
                f"resetting tracker"
            )
            self.stats["resets_position_jump"] += 1

            self.current_candidate = new_candidate
            self.confirmation_count = 1

        return None

    def _compute_distance(
        self, pos1: Tuple[int, int], pos2: Tuple[int, int]
    ) -> float:
        """
        Compute Euclidean distance between two positions.

        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)

        Returns:
            Distance in pixels
        """
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return float(np.sqrt(dx**2 + dy**2))

    def _create_impact(
        self, candidate: DartCandidate, frame_index: int, timestamp: float
    ) -> DartImpact:
        """
        Create confirmed dart impact from candidate.

        Args:
            candidate: Dart candidate
            frame_index: Current frame index
            timestamp: Current timestamp

        Returns:
            DartImpact object
        """
        # Get first frame from history
        first_frame = candidate.frame_index
        if self.candidate_history:
            first_frame = min(c.frame_index for c in self.candidate_history)

        return DartImpact(
            position=candidate.position,
            confidence=candidate.confidence,
            first_detected_frame=first_frame,
            confirmed_frame=frame_index,
            confirmation_count=self.confirmation_count,
            timestamp=timestamp,
        )

    def reset(self):
        """Reset tracker state"""
        self.current_candidate = None
        self.confirmation_count = 0
        logger.debug("Tracker reset")

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "candidates_tracked": 0,
            "confirmations": 0,
            "resets_position_jump": 0,
            "resets_timeout": 0,
        }

    def get_stats(self) -> dict:
        """Get current statistics"""
        return self.stats.copy()

    def get_current_candidate(self) -> Optional[DartCandidate]:
        """Get current candidate being tracked"""
        return self.current_candidate

    def get_confirmation_progress(self) -> Tuple[int, int]:
        """
        Get confirmation progress.

        Returns:
            Tuple of (current_count, required_count)
        """
        return self.confirmation_count, self.confirmation_frames
