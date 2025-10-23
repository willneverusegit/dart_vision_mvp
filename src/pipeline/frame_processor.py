"""
Frame Processor - Main processing pipeline for dart detection.

Orchestrates the complete frame processing flow:
- ROI warping and preprocessing
- Motion detection
- Dart impact detection
- Board mapping and scoring
- Statistics and heatmap updates
- Auto-alignment (optional)
"""

import logging
import time
import cv2
import numpy as np
from typing import Optional, Tuple

from src.vision import DartImpactDetector, MotionDetector
from src.calibration.roi_processor import ROIProcessor
from src.board import BoardMapper
from src.game.game import DemoGame
from src.analytics.stats_accumulator import StatsAccumulator

logger = logging.getLogger(__name__)


class FrameProcessor:
    """
    Handles the main frame processing pipeline.

    Responsibilities:
    - Frame preprocessing (ROI warping, mask application)
    - Motion detection
    - Dart impact detection and scoring
    - Game state updates
    - Statistics and heatmap tracking
    - Auto-alignment integration
    """

    def __init__(self, app):
        """
        Initialize frame processor.

        Args:
            app: DartVisionApp instance
        """
        self.app = app
        self.logger = logging.getLogger("FrameProcessor")

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, bool, Optional[np.ndarray], Optional[any]]:
        """
        Process a single frame through the complete pipeline.

        Args:
            frame: Input BGR frame

        Returns:
            Tuple of (roi_frame, motion_detected, fg_mask, impact):
            - roi_frame: Warped ROI frame
            - motion_detected: Whether motion was detected
            - fg_mask: Foreground mask (can be None)
            - impact: Dart impact object (can be None)
        """
        self.app.frame_count += 1
        timestamp = time.time() - self.app.session_start

        # Apply ROI adjustments if dirty
        self._apply_effective_H_if_dirty()

        # Warp ROI
        roi_frame = self.app.roi.warp_roi(frame)

        # Motion detection
        motion_detected, motion_event, fg_mask = self.app.motion.detect_motion(
            roi_frame, self.app.frame_count, timestamp
        )

        # Apply annulus mask to ignore net/background
        if self.app._roi_annulus_mask is not None and fg_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, self.app._roi_annulus_mask)

        # Dart detection
        impact = None
        if motion_detected:
            impact = self._process_dart_detection(roi_frame, fg_mask, timestamp)

        # Show processed motion mask for tuning
        self._show_debug_mask()

        # Auto-align (every 15 frames in ALIGN mode)
        self._process_auto_align(roi_frame)

        return roi_frame, motion_detected, fg_mask, impact

    def _process_dart_detection(
        self, roi_frame: np.ndarray, fg_mask: np.ndarray, timestamp: float
    ) -> Optional[any]:
        """
        Process dart detection when motion is detected.

        Args:
            roi_frame: ROI frame
            fg_mask: Foreground mask
            timestamp: Current timestamp

        Returns:
            Dart impact object or None
        """
        impact = self.app.dart.detect_dart(
            roi_frame, fg_mask, self.app.frame_count, timestamp
        )

        if impact:
            self.app.total_darts += 1

            # Score dart if board mapper available
            if self.app.board_mapper is not None:
                self._score_dart(impact)

            # Debug logging
            if self.app.show_debug:
                self.logger.info(
                    f"[DART #{self.app.total_darts}] pos={impact.position} "
                    f"conf={impact.confidence:.2f}"
                )

            # Update heatmaps
            self._update_heatmaps(impact)

        return impact

    def _score_dart(self, impact: any) -> None:
        """
        Score a dart impact using the board mapper.

        Args:
            impact: Dart impact object
        """
        ring, sector, label = self.app.board_mapper.score_from_hit(
            float(impact.position[0]), float(impact.position[1])
        )
        pts = self._points_from_mapping(ring, sector)
        impact.score_label = label  # e.g., "D20", "T5", "25", "50"

        # Game update
        self.app.last_msg = self.app.game.apply_points(pts, label)
        if self.app.show_debug:
            self.logger.info(f"[SCORE] {label} -> {pts} | {self.app.last_msg}")

        # Stats update (ROI coordinates)
        self.app.stats.add(
            ring=ring,
            sector=sector,
            points=pts,
            cx=float(impact.position[0]),
            cy=float(impact.position[1]),
        )

    def _points_from_mapping(self, ring: str, sector: int) -> int:
        """
        Convert board mapper result to points.

        Args:
            ring: Ring name (e.g., "double", "triple", "bull_inner")
            sector: Sector number

        Returns:
            Points scored
        """
        if ring == "bull_inner":
            return 50
        if ring == "bull_outer":
            return 25
        if ring == "double":
            return 2 * int(sector)
        if ring == "triple":
            return 3 * int(sector)
        if ring.startswith("single"):
            return int(sector)
        # Fallback
        try:
            return int(sector)
        except Exception:
            return 0

    def _update_heatmaps(self, impact: any) -> None:
        """
        Update heatmap accumulators with dart impact.

        Args:
            impact: Dart impact object
        """
        try:
            cx, cy = int(impact.position[0]), int(impact.position[1])

            # Cartesian heatmap
            if self.app.hm is not None:
                self.app.hm.add_hit(cx, cy, weight=1.0)

            # Polar heatmap (requires board mapper)
            if self.app.ph is not None and self.app.board_mapper is not None:
                # Ring/sector were computed in _score_dart
                ring, sector, _ = self.app.board_mapper.score_from_hit(
                    float(impact.position[0]), float(impact.position[1])
                )
                self.app.ph.add(ring, sector)
        except Exception as e:
            if self.app.show_debug:
                self.logger.debug(f"[HM] update skipped: {e}")

    def _show_debug_mask(self) -> None:
        """Show processed motion mask for tuning (debug mode)."""
        if (
            self.app.show_mask_debug
            and self.app.dart is not None
            and getattr(self.app.dart, "last_processed_mask", None) is not None
        ):
            ROI_SIZE = (400, 400)  # From app config
            pm = self.app.dart.last_processed_mask
            pm_show = (
                pm
                if (pm.shape[1] == ROI_SIZE[0] and pm.shape[0] == ROI_SIZE[1])
                else cv2.resize(pm, ROI_SIZE, interpolation=cv2.INTER_NEAREST)
            )
            cv2.imshow("MotionMask (processed)", pm_show)

    def _process_auto_align(self, roi_frame: np.ndarray) -> None:
        """
        Process auto-alignment if enabled.

        Args:
            roi_frame: ROI frame to analyze
        """
        if hasattr(self.app, "hough_aligner") and self.app.hough_aligner is not None:
            self.app.hough_aligner.process_auto_align(roi_frame, self.app.frame_count)

    def _apply_effective_H_if_dirty(self) -> None:
        """Apply effective homography if ROI adjustments changed."""
        if not getattr(self.app, "_roi_adjust_dirty", False):
            return
        if self.app.homography is None:
            self.app._roi_adjust_dirty = False
            return
        Heff = self._effective_H()
        self.app.roi.set_homography_from_matrix(Heff)
        self.app._roi_adjust_dirty = False

    def _effective_H(self) -> np.ndarray:
        """Build effective homography from ROI adjustments."""
        if self.app.homography is None:
            return None
        ROI_CENTER = (200.0, 200.0)  # From app config
        A = self.app.build_roi_adjust_matrix(
            ROI_CENTER, self.app.roi_tx, self.app.roi_ty, self.app.roi_scale, self.app.roi_rot_deg
        )
        return A @ self.app.homography
