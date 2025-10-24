"""Enhanced dart impact detector with serializable configuration."""

from __future__ import annotations

import cv2
import numpy as np
import logging
import math
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, replace, asdict
from collections import deque

if TYPE_CHECKING:
    from .config_schema import DartDetectorConfigSchema

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
    convexity: Optional[float] = None  # NEW: convex_area / contour_area


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

    # Impact refine
    refine_enabled: bool = True
    refine_threshold: float = 0.45
    refine_roi_size_px: int = 80
    refine_canny_lo: int = 60
    refine_canny_hi: int = 180
    refine_hough_thresh: int = 30
    refine_min_line_len: int = 10
    refine_max_line_gap: int = 4

    # Tip refine
    tip_refine_enabled: bool = True
    tip_roi_px: int = 36
    tip_search_px: int = 14
    tip_max_shift_px: int = 16
    tip_edge_weight: float = 0.6
    tip_dark_weight: float = 0.4
    tip_canny_lo: int = 60
    tip_canny_hi: int = 180

    # NEW: Convexity-Gate (Proposal 1)
    convexity_gate_enabled: bool = True
    convexity_min_ratio: float = 0.70  # Min convex_hull_area / contour_area
    hierarchy_filter_enabled: bool = True  # Prefer top-level contours

    @classmethod
    def from_schema(cls, schema: "DartDetectorConfigSchema | Dict[str, Any]") -> "DartDetectorConfig":
        """Create config from Pydantic schema or raw dict."""
        if hasattr(schema, "model_dump"):
            data = schema.model_dump()
        else:
            data = dict(schema)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Return dict representation for YAML serialization."""
        return asdict(self)


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
        refine_enabled=True, refine_threshold=0.40,
        convexity_gate_enabled=True, convexity_min_ratio=0.65,
        hierarchy_filter_enabled=True
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
        refine_enabled=True, refine_threshold=0.45,
        convexity_gate_enabled=True, convexity_min_ratio=0.70,
        hierarchy_filter_enabled=True
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
        refine_enabled=True, refine_threshold=0.50,
        convexity_gate_enabled=True, convexity_min_ratio=0.75,
        hierarchy_filter_enabled=True
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


class DartImpactDetector:
    """Enhanced dart impact detector with convexity-gate and hierarchy filtering."""

    def __init__(self, config: Optional[DartDetectorConfig] = None):
        self.config = config or DartDetectorConfig()

        # Tracking
        self.current_candidate: Optional[DartCandidate] = None
        self.confirmation_count = 0

        # History
        self.candidate_history: deque = deque(maxlen=self.config.candidate_history_size)
        self.confirmed_impacts: List[DartImpact] = []

        # Cooldown tracking
        self.cooldown_regions: List[Tuple[Tuple[int, int], int]] = []
        self.last_processed_mask = None

        # Stats
        self.stats = {
            "total_candidates": 0,
            "convexity_rejected": 0,
            "hierarchy_rejected": 0,
            "shape_rejected": 0,
            "confirmed_impacts": 0
        }

    def _preprocess_motion_mask(self, motion_mask: np.ndarray) -> np.ndarray:
        """Adaptive binarization + morphology for stable motion contours."""
        mm = motion_mask
        if mm.ndim == 3:
            mm = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)

        if not self.config.motion_adaptive:
            return mm.astype(np.uint8, copy=False)

        # Adaptive Otsu + Bias
        _t, _ = cv2.threshold(mm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = int(np.clip(_t + self.config.motion_otsu_bias, 0, 255))
        _, bw = cv2.threshold(mm, t, 255, cv2.THRESH_BINARY)

        # Morphology (open→close)
        k1 = max(1, int(self.config.morph_open_ksize))
        k1 += (k1 % 2 == 0)
        k2 = max(1, int(self.config.morph_close_ksize))
        k2 += (k2 % 2 == 0)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1)))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)))

        # Remove micro-blobs
        num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < self.config.motion_min_area_px:
                bw[labels == i] = 0

        return bw.astype(np.uint8, copy=False)

    def _find_best_candidate(
            self,
            frame: np.ndarray,
            processed_mask: np.ndarray,
            frame_index: int,
            timestamp: float
    ) -> Optional[DartCandidate]:
        """Find best dart candidate with convexity-gate and hierarchy filter."""

        # Find contours with hierarchy (NEW)
        if self.config.hierarchy_filter_enabled:
            contours, hierarchy = cv2.findContours(
                processed_mask,
                cv2.RETR_TREE,  # Keep parent-child relationships
                cv2.CHAIN_APPROX_SIMPLE
            )
        else:
            contours, hierarchy = cv2.findContours(
                processed_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

        if len(contours) == 0:
            return None

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        candidates = []

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)

            # Basic area filter
            if not (self.config.min_area <= area <= self.config.max_area):
                continue

            self.stats["total_candidates"] += 1

            # NEW: Hierarchy filter (prefer top-level contours)
            if self.config.hierarchy_filter_enabled and hierarchy is not None:
                # hierarchy[0][idx] = [next, prev, first_child, parent]
                parent_idx = hierarchy[0][idx][3]
                if parent_idx != -1:  # Has a parent → skip nested contours
                    self.stats["hierarchy_rejected"] += 1
                    continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0:
                continue

            aspect_ratio = float(w) / float(h)
            if not (self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio):
                self.stats["shape_rejected"] += 1
                continue

            # Solidity (convex hull)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0

            if not (self.config.min_solidity <= solidity <= self.config.max_solidity):
                self.stats["shape_rejected"] += 1
                continue

            # NEW: Convexity-Gate
            convexity = solidity  # Same as solidity for single contours
            if self.config.convexity_gate_enabled:
                if convexity < self.config.convexity_min_ratio:
                    self.stats["convexity_rejected"] += 1
                    continue

            # Extent
            extent = area / float(w * h)
            if not (self.config.min_extent <= extent <= self.config.max_extent):
                self.stats["shape_rejected"] += 1
                continue

            # Edge density
            roi = frame_gray[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            edges = cv2.Canny(roi, self.config.edge_canny_threshold1, self.config.edge_canny_threshold2)
            edge_density = np.count_nonzero(edges) / float(w * h)

            if not (self.config.min_edge_density <= edge_density <= self.config.max_edge_density):
                self.stats["shape_rejected"] += 1
                continue

            # Centroid
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Circularity
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

            # Scoring
            circularity_score = np.clip(circularity, 0.0, 1.0)
            solidity_score = np.clip(
                (solidity - self.config.min_solidity) /
                max(self.config.max_solidity - self.config.min_solidity, 1e-6),
                0.0, 1.0
            )
            extent_score = np.clip(
                (extent - self.config.min_extent) /
                max(self.config.max_extent - self.config.min_extent, 1e-6),
                0.0, 1.0
            )
            edge_score = np.clip(
                (edge_density - self.config.min_edge_density) /
                max(self.config.max_edge_density - self.config.min_edge_density, 1e-6),
                0.0, 1.0
            )

            preferred_ratio = self.config.preferred_aspect_ratio
            tolerance = preferred_ratio * self.config.aspect_ratio_tolerance
            ratio_delta = abs(aspect_ratio - preferred_ratio)
            aspect_ratio_score = np.clip(
                1.0 - (ratio_delta / max(tolerance, 1e-6)),
                0.0, 1.0
            )

            total_weight = (
                    self.config.circularity_weight +
                    self.config.solidity_weight +
                    self.config.extent_weight +
                    self.config.edge_weight +
                    self.config.aspect_ratio_weight
            )

            confidence = (
                    self.config.circularity_weight * circularity_score +
                    self.config.solidity_weight * solidity_score +
                    self.config.extent_weight * extent_score +
                    self.config.edge_weight * edge_score +
                    self.config.aspect_ratio_weight * aspect_ratio_score
            )
            if total_weight > 0:
                confidence = confidence / total_weight

            candidate = DartCandidate(
                position=(cx, cy),
                area=area,
                confidence=confidence,
                frame_index=frame_index,
                timestamp=timestamp,
                aspect_ratio=aspect_ratio,
                solidity=solidity,
                extent=extent,
                edge_density=edge_density,
                convexity=convexity  # NEW
            )

            candidates.append(candidate)

        if not candidates:
            return None

        # Return best candidate by confidence
        return max(candidates, key=lambda c: c.confidence)

    def detect_dart(
            self,
            frame: np.ndarray,
            motion_mask: np.ndarray,
            frame_index: int,
            timestamp: float
    ) -> Optional[DartImpact]:
        """Detect dart impact with temporal confirmation and cooldown."""

        # Update cooldowns
        self.cooldown_regions = [
            (pos, until) for pos, until in self.cooldown_regions
            if until > frame_index
        ]

        # Preprocess motion mask
        processed_mask = self._preprocess_motion_mask(motion_mask)
        self.last_processed_mask = processed_mask.copy()

        # Quick gate: very little white pixels?
        white_frac = np.count_nonzero(processed_mask) / float(processed_mask.size)
        if white_frac < self.config.motion_min_white_frac:
            self._reset_tracking()
            return None

        # Find best candidate
        candidate = self._find_best_candidate(frame, processed_mask, frame_index, timestamp)

        if candidate is None:
            self._reset_tracking()
            return None

        # Check cooldown
        for cooldown_pos, _ in self.cooldown_regions:
            dx = candidate.position[0] - cooldown_pos[0]
            dy = candidate.position[1] - cooldown_pos[1]
            distance = np.sqrt(dx * dx + dy * dy)
            if distance < self.config.cooldown_radius_px:
                self._reset_tracking()
                return None

        # Temporal confirmation
        if self._is_same_position(candidate, self.current_candidate):
            self.confirmation_count += 1
        else:
            self.current_candidate = candidate
            self.confirmation_count = 1

        self.candidate_history.append(candidate)

        # Confirmed?
        if self.confirmation_count >= self.config.confirmation_frames:
            impact = DartImpact(
                position=candidate.position,
                confidence=candidate.confidence,
                first_detected_frame=self.current_candidate.frame_index,
                confirmed_frame=frame_index,
                confirmation_count=self.confirmation_count,
                timestamp=timestamp
            )

            self.confirmed_impacts.append(impact)
            self.stats["confirmed_impacts"] += 1

            # Set cooldown
            self.cooldown_regions.append((candidate.position, frame_index + self.config.cooldown_frames))

            self._reset_tracking()

            logger.info(f"🎯 Dart confirmed at {impact.position} (conf={impact.confidence:.2f}, convexity={candidate.convexity:.2f})")
            return impact

        return None

    def _is_same_position(
            self,
            candidate: Optional[DartCandidate],
            reference: Optional[DartCandidate]
    ) -> bool:
        """Check if two candidates are at same position"""
        if candidate is None or reference is None:
            return False

        dx = candidate.position[0] - reference.position[0]
        dy = candidate.position[1] - reference.position[1]
        distance = np.sqrt(dx * dx + dy * dy)

        return distance < self.config.position_tolerance_px

    def _reset_tracking(self):
        """Reset temporal tracking"""
        self.current_candidate = None
        self.confirmation_count = 0

    def get_confirmed_impacts(self) -> List[DartImpact]:
        """Get all confirmed dart impacts"""
        return self.confirmed_impacts.copy()

    def clear_impacts(self):
        """Clear confirmed impacts"""
        self.confirmed_impacts.clear()
        logger.info("Dart impacts cleared")

    def get_stats(self) -> dict:
        """Get detector statistics"""
        return self.stats.copy()


# FieldMapper kept for compatibility (unchanged)
@dataclass
class FieldMapperConfig:
    """Field mapper configuration"""
    sector_scores: List[int] = None
    bull_inner_radius: float = 0.05
    bull_outer_radius: float = 0.095
    triple_inner_radius: float = 0.53
    triple_outer_radius: float = 0.58
    double_inner_radius: float = 0.94
    double_outer_radius: float = 1.00

    def __post_init__(self):
        if self.sector_scores is None:
            self.sector_scores = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


class FieldMapper:
    """Maps pixel coordinates to dartboard scores."""

    def __init__(self, config: Optional[FieldMapperConfig] = None):
        self.config = config or FieldMapperConfig()
        self.sector_angle_deg = 18
        self.sector_offset_deg = 0

    def point_to_score(
            self,
            point: Tuple[int, int],
            center: Tuple[int, int],
            radius: float
    ) -> Tuple[int, int, int]:
        """Convert pixel coordinates to dartboard score."""
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        distance = np.sqrt(dx * dx + dy * dy)
        norm_distance = distance / radius

        # Determine ring
        if norm_distance <= self.config.bull_inner_radius:
            return (50, 0, 0)  # Bull's eye
        elif norm_distance <= self.config.bull_outer_radius:
            return (25, 0, 0)  # Outer bull
        elif norm_distance > 1.0:
            return (0, 0, 0)  # Miss

        # Determine sector
        angle = np.degrees(np.arctan2(-dy, dx))
        angle = (angle + 360 + self.sector_offset_deg + self.sector_angle_deg / 2) % 360
        sector_index = int(angle / self.sector_angle_deg) % 20
        sector_score = self.config.sector_scores[sector_index]

        # Determine multiplier
        if self.config.triple_inner_radius <= norm_distance <= self.config.triple_outer_radius:
            return (sector_score * 3, 3, sector_score)
        elif self.config.double_inner_radius <= norm_distance <= self.config.double_outer_radius:
            return (sector_score * 2, 2, sector_score)
        else:
            return (sector_score, 1, sector_score)
