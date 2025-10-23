"""
Dart Impact Detector (Refactored) - Modular multi-frame dart detection.

Orchestrates specialized modules for dart detection:
- MaskPreprocessor: Adaptive binarization and morphology
- ShapeAnalyzer: Multi-metric shape analysis with convexity gate
- ConfirmationTracker: Multi-frame temporal confirmation (Land-and-Stick)
- CooldownManager: Spatial/temporal cooldown to prevent re-detection

This refactored version maintains the same API as the original DartImpactDetector
but with improved modularity, testability, and maintainability.
"""

import cv2
import numpy as np
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass, replace

from .mask_preprocessor import MaskPreprocessor
from .shape_analyzer import ShapeAnalyzer, ShapeMetrics
from .confirmation_tracker import ConfirmationTracker, DartCandidate, DartImpact
from .cooldown_manager import CooldownManager


logger = logging.getLogger(__name__)


@dataclass
class DartDetectorConfig:
    """Dart detector configuration (Pydantic-compatible)"""
    # Shape constraints
    min_area: int = 10
    max_area: int = 1000
    min_aspect_ratio: float = 0.3
    max_aspect_ratio: float = 3.0

    # Advanced shape heuristics
    min_solidity: float = 0.1
    max_solidity: float = 0.95
    min_extent: float = 0.05
    max_extent: float = 0.75
    min_edge_density: float = 0.02
    max_edge_density: float = 0.35
    preferred_aspect_ratio: float = 0.35
    aspect_ratio_tolerance: float = 1.5

    # Edge detection
    edge_canny_threshold1: int = 40
    edge_canny_threshold2: int = 120

    # Confidence weighting
    circularity_weight: float = 0.35
    solidity_weight: float = 0.2
    extent_weight: float = 0.15
    edge_weight: float = 0.15
    aspect_ratio_weight: float = 0.15

    # Temporal confirmation
    confirmation_frames: int = 3
    position_tolerance_px: int = 20

    # Cooldown
    cooldown_frames: int = 30
    cooldown_radius_px: int = 50

    # History
    candidate_history_size: int = 20

    # Motion mask preprocessing
    motion_mask_smoothing_kernel: int = 7
    motion_adaptive: bool = True
    motion_otsu_bias: int = 8
    motion_min_area_px: int = 24
    morph_open_ksize: int = 3
    morph_close_ksize: int = 5
    motion_min_white_frac: float = 0.015

    # Convexity-Gate
    convexity_gate_enabled: bool = True
    convexity_min_ratio: float = 0.70


# Presets
DETECTOR_PRESETS = {
    "aggressive": dict(
        motion_adaptive=True, motion_otsu_bias=6,
        morph_open_ksize=3, morph_close_ksize=7,
        motion_min_white_frac=0.008,
        min_area=6, max_area=1600,
        min_aspect_ratio=0.25, max_aspect_ratio=3.6,
        min_solidity=0.08, max_solidity=0.98,
        min_extent=0.04, max_extent=0.80,
        min_edge_density=0.012, max_edge_density=0.42,
        preferred_aspect_ratio=0.35, aspect_ratio_tolerance=1.9,
        edge_canny_threshold1=28, edge_canny_threshold2=90,
        circularity_weight=0.28, solidity_weight=0.22,
        extent_weight=0.14, edge_weight=0.22, aspect_ratio_weight=0.14,
        confirmation_frames=2, position_tolerance_px=24,
        cooldown_frames=22, cooldown_radius_px=42,
        candidate_history_size=20, motion_mask_smoothing_kernel=5,
        convexity_gate_enabled=True, convexity_min_ratio=0.65,
    ),
    "balanced": dict(
        motion_adaptive=True, motion_otsu_bias=10,
        morph_open_ksize=5, morph_close_ksize=9,
        motion_min_white_frac=0.012,
        min_area=10, max_area=1100,
        min_aspect_ratio=0.30, max_aspect_ratio=3.0,
        min_solidity=0.10, max_solidity=0.95,
        min_extent=0.05, max_extent=0.75,
        min_edge_density=0.018, max_edge_density=0.36,
        preferred_aspect_ratio=0.35, aspect_ratio_tolerance=1.6,
        edge_canny_threshold1=38, edge_canny_threshold2=120,
        circularity_weight=0.33, solidity_weight=0.20,
        extent_weight=0.15, edge_weight=0.17, aspect_ratio_weight=0.15,
        confirmation_frames=3, position_tolerance_px=20,
        cooldown_frames=28, cooldown_radius_px=48,
        candidate_history_size=22, motion_mask_smoothing_kernel=5,
        convexity_gate_enabled=True, convexity_min_ratio=0.70,
    ),
    "stable": dict(
        motion_adaptive=True, motion_otsu_bias=14,
        morph_open_ksize=7, morph_close_ksize=11,
        motion_min_white_frac=0.015,
        min_area=14, max_area=900,
        min_aspect_ratio=0.34, max_aspect_ratio=2.6,
        min_solidity=0.12, max_solidity=0.92,
        min_extent=0.06, max_extent=0.68,
        min_edge_density=0.022, max_edge_density=0.32,
        preferred_aspect_ratio=0.35, aspect_ratio_tolerance=1.35,
        edge_canny_threshold1=50, edge_canny_threshold2=165,
        circularity_weight=0.35, solidity_weight=0.20,
        extent_weight=0.15, edge_weight=0.12, aspect_ratio_weight=0.18,
        confirmation_frames=4, position_tolerance_px=18,
        cooldown_frames=35, cooldown_radius_px=55,
        candidate_history_size=24, motion_mask_smoothing_kernel=7,
        convexity_gate_enabled=True, convexity_min_ratio=0.75,
    ),
}


def apply_detector_preset(cfg: DartDetectorConfig, name: str) -> DartDetectorConfig:
    """Apply a named preset to config"""
    name = (name or "").lower()
    params = DETECTOR_PRESETS.get(name)
    if not params:
        logger.warning(f"Unknown preset '{name}', using defaults")
        return cfg
    return replace(cfg, **params)


class DartDetectorRefactored:
    """
    Modular dart impact detector.

    Architecture:
    - MaskPreprocessor: Cleans motion masks
    - ShapeAnalyzer: Evaluates blob shapes
    - ConfirmationTracker: Tracks candidates across frames
    - CooldownManager: Prevents re-detection

    Features:
    - Multi-frame temporal confirmation (Land-and-Stick)
    - Multi-metric shape analysis (5 metrics)
    - Convexity gate filtering (filters hands/shadows)
    - Spatial/temporal cooldown regions
    - Hot-reload configuration
    - Per-module statistics
    """

    def __init__(self, config: Optional[DartDetectorConfig] = None):
        """
        Initialize dart detector.

        Args:
            config: Detector configuration
        """
        self.config = config or DartDetectorConfig()

        # Initialize modules
        self.mask_preprocessor = MaskPreprocessor(
            adaptive=self.config.motion_adaptive,
            otsu_bias=self.config.motion_otsu_bias,
            open_kernel_size=self.config.morph_open_ksize,
            close_kernel_size=self.config.morph_close_ksize,
            min_blob_area=self.config.motion_min_area_px,
            min_white_fraction=self.config.motion_min_white_frac,
        )

        self.shape_analyzer = ShapeAnalyzer(
            min_area=self.config.min_area,
            max_area=self.config.max_area,
            min_aspect_ratio=self.config.min_aspect_ratio,
            max_aspect_ratio=self.config.max_aspect_ratio,
            preferred_aspect_ratio=self.config.preferred_aspect_ratio,
            aspect_ratio_tolerance=self.config.aspect_ratio_tolerance,
            min_solidity=self.config.min_solidity,
            max_solidity=self.config.max_solidity,
            min_extent=self.config.min_extent,
            max_extent=self.config.max_extent,
            min_edge_density=self.config.min_edge_density,
            max_edge_density=self.config.max_edge_density,
            edge_canny_threshold1=self.config.edge_canny_threshold1,
            edge_canny_threshold2=self.config.edge_canny_threshold2,
            convexity_gate_enabled=self.config.convexity_gate_enabled,
            convexity_min_ratio=self.config.convexity_min_ratio,
            circularity_weight=self.config.circularity_weight,
            solidity_weight=self.config.solidity_weight,
            extent_weight=self.config.extent_weight,
            edge_weight=self.config.edge_weight,
            aspect_ratio_weight=self.config.aspect_ratio_weight,
        )

        self.confirmation_tracker = ConfirmationTracker(
            confirmation_frames=self.config.confirmation_frames,
            position_tolerance_px=self.config.position_tolerance_px,
            history_size=self.config.candidate_history_size,
        )

        self.cooldown_manager = CooldownManager(
            cooldown_frames=self.config.cooldown_frames,
            cooldown_radius_px=self.config.cooldown_radius_px,
        )

        # State
        self.confirmed_impacts: List[DartImpact] = []
        self.last_processed_mask: Optional[np.ndarray] = None

    def detect_dart(
        self,
        frame: np.ndarray,
        motion_mask: np.ndarray,
        frame_index: int,
        timestamp: float,
    ) -> Optional[DartImpact]:
        """
        Detect dart impact with multi-frame confirmation.

        Args:
            frame: Current BGR frame
            motion_mask: Motion detection mask
            frame_index: Current frame index
            timestamp: Current timestamp

        Returns:
            DartImpact if confirmed, None otherwise
        """
        # Step 1: Preprocess motion mask
        processed_mask = self.mask_preprocessor.preprocess(motion_mask)
        self.last_processed_mask = processed_mask

        if processed_mask is None:
            # Mask failed quality check
            self.confirmation_tracker.update(None, frame_index, timestamp)
            return None

        # Step 2: Find contours
        contours, _ = cv2.findContours(
            processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            self.confirmation_tracker.update(None, frame_index, timestamp)
            return None

        # Step 3: Analyze shapes and find best candidate
        frame_gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )

        best_candidate: Optional[DartCandidate] = None
        best_confidence = 0.0

        for contour in contours:
            result = self.shape_analyzer.analyze_contour(contour, frame_gray)

            if result is None:
                continue

            metrics, confidence = result

            # Check cooldown
            if self.cooldown_manager.is_in_cooldown(metrics.center, frame_index):
                continue

            # Track best candidate
            if confidence > best_confidence:
                best_confidence = confidence
                best_candidate = DartCandidate(
                    position=metrics.center,
                    area=metrics.area,
                    confidence=confidence,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    aspect_ratio=metrics.aspect_ratio,
                    extent=metrics.extent,
                    solidity=metrics.solidity,
                    edge_density=metrics.edge_density,
                    convexity=metrics.convexity,
                )

        # Step 4: Update confirmation tracker
        impact = self.confirmation_tracker.update(
            best_candidate, frame_index, timestamp
        )

        # Step 5: Handle confirmed impact
        if impact is not None:
            # Add cooldown region
            self.cooldown_manager.add_cooldown(impact.position, frame_index)

            # Track confirmed impact
            self.confirmed_impacts.append(impact)

            logger.info(
                f"🎯 Dart confirmed at {impact.position}, "
                f"conf={impact.confidence:.2f}, "
                f"convexity={best_candidate.convexity:.2f}"
            )

        return impact

    def update_config(self, config: DartDetectorConfig):
        """
        Hot-reload configuration.

        Args:
            config: New detector configuration
        """
        self.config = config

        # Update mask preprocessor
        self.mask_preprocessor.update_parameters(
            adaptive=config.motion_adaptive,
            otsu_bias=config.motion_otsu_bias,
            open_kernel_size=config.morph_open_ksize,
            close_kernel_size=config.morph_close_ksize,
        )

        # Update shape analyzer (requires recreation)
        self.shape_analyzer = ShapeAnalyzer(
            min_area=config.min_area,
            max_area=config.max_area,
            min_aspect_ratio=config.min_aspect_ratio,
            max_aspect_ratio=config.max_aspect_ratio,
            preferred_aspect_ratio=config.preferred_aspect_ratio,
            aspect_ratio_tolerance=config.aspect_ratio_tolerance,
            min_solidity=config.min_solidity,
            max_solidity=config.max_solidity,
            min_extent=config.min_extent,
            max_extent=config.max_extent,
            min_edge_density=config.min_edge_density,
            max_edge_density=config.max_edge_density,
            edge_canny_threshold1=config.edge_canny_threshold1,
            edge_canny_threshold2=config.edge_canny_threshold2,
            convexity_gate_enabled=config.convexity_gate_enabled,
            convexity_min_ratio=config.convexity_min_ratio,
            circularity_weight=config.circularity_weight,
            solidity_weight=config.solidity_weight,
            extent_weight=config.extent_weight,
            edge_weight=config.edge_weight,
            aspect_ratio_weight=config.aspect_ratio_weight,
        )

        # Note: ConfirmationTracker and CooldownManager don't support
        # hot-reload as changing their parameters mid-tracking could
        # cause inconsistent state. They would need to be recreated.

        logger.debug("Configuration updated")

    def get_confirmed_impacts(self) -> List[DartImpact]:
        """Get all confirmed dart impacts"""
        return self.confirmed_impacts.copy()

    def clear_impacts(self):
        """Clear confirmed impacts"""
        self.confirmed_impacts.clear()
        logger.info("Dart impacts cleared")

    def reset(self):
        """Reset all detector state"""
        self.confirmation_tracker.reset()
        self.cooldown_manager.reset()
        self.confirmed_impacts.clear()
        self.last_processed_mask = None
        logger.info("Detector reset")

    def get_stats(self) -> dict:
        """
        Get aggregated statistics from all modules.

        Returns:
            Dictionary with statistics from each module
        """
        return {
            "preprocessor": self.mask_preprocessor.get_stats(),
            "shape_analyzer": self.shape_analyzer.get_stats(),
            "confirmation_tracker": self.confirmation_tracker.get_stats(),
            "cooldown_manager": self.cooldown_manager.get_stats(),
            "confirmed_impacts": len(self.confirmed_impacts),
        }

    def reset_stats(self):
        """Reset statistics in all modules"""
        self.mask_preprocessor.reset_stats()
        self.shape_analyzer.reset_stats()
        self.confirmation_tracker.reset_stats()
        self.cooldown_manager.reset_stats()
        logger.debug("Statistics reset")

    def get_current_candidate(self) -> Optional[DartCandidate]:
        """Get current candidate being tracked"""
        return self.confirmation_tracker.get_current_candidate()

    def get_confirmation_progress(self) -> Tuple[int, int]:
        """
        Get confirmation progress.

        Returns:
            Tuple of (current_count, required_count)
        """
        return self.confirmation_tracker.get_confirmation_progress()

    def get_active_cooldowns(self) -> List[Tuple[Tuple[int, int], int]]:
        """
        Get active cooldown regions.

        Returns:
            List of (position, expiry_frame) tuples
        """
        return self.cooldown_manager.get_active_cooldowns()

    def get_last_processed_mask(self) -> Optional[np.ndarray]:
        """Get last preprocessed motion mask"""
        return self.last_processed_mask
