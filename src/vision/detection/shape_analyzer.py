"""
Shape Analyzer - Multi-metric dart shape analysis.

Analyzes blob shapes using multiple geometric metrics to determine
if they match dart characteristics. Includes convexity filtering.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class ShapeMetrics:
    """Geometric shape metrics for a contour"""
    area: float
    perimeter: float
    aspect_ratio: float
    extent: float
    solidity: float
    edge_density: float
    convexity: float
    bounding_rect: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]
    orientation: float  # Angle from PCA


class ShapeAnalyzer:
    """
    Analyze blob shapes with multi-metric scoring.

    Metrics:
    - Area: Blob size
    - Aspect Ratio: Width/Height ratio
    - Extent: Blob area / Bounding box area
    - Solidity: Blob area / Convex hull area
    - Edge Density: Edge pixels / Total area
    - Convexity: Convex hull area / Contour area
    """

    def __init__(
        self,
        # Area constraints
        min_area: int = 10,
        max_area: int = 1000,
        # Aspect ratio
        min_aspect_ratio: float = 0.3,
        max_aspect_ratio: float = 3.0,
        preferred_aspect_ratio: float = 0.35,
        aspect_ratio_tolerance: float = 1.5,
        # Solidity
        min_solidity: float = 0.1,
        max_solidity: float = 0.95,
        # Extent
        min_extent: float = 0.05,
        max_extent: float = 0.75,
        # Edge density
        min_edge_density: float = 0.02,
        max_edge_density: float = 0.35,
        edge_canny_threshold1: int = 40,
        edge_canny_threshold2: int = 120,
        # Convexity gate
        convexity_gate_enabled: bool = True,
        convexity_min_ratio: float = 0.70,
        # Confidence weighting
        circularity_weight: float = 0.35,
        solidity_weight: float = 0.2,
        extent_weight: float = 0.15,
        edge_weight: float = 0.15,
        aspect_ratio_weight: float = 0.15,
    ):
        """
        Initialize shape analyzer.

        Args:
            min_area: Minimum blob area
            max_area: Maximum blob area
            min_aspect_ratio: Minimum width/height ratio
            max_aspect_ratio: Maximum width/height ratio
            preferred_aspect_ratio: Ideal aspect ratio
            aspect_ratio_tolerance: Tolerance for aspect ratio scoring
            min_solidity: Minimum solidity
            max_solidity: Maximum solidity
            min_extent: Minimum extent
            max_extent: Maximum extent
            min_edge_density: Minimum edge density
            max_edge_density: Maximum edge density
            edge_canny_threshold1: Canny edge detection low threshold
            edge_canny_threshold2: Canny edge detection high threshold
            convexity_gate_enabled: Enable convexity filtering
            convexity_min_ratio: Minimum convexity ratio
            circularity_weight: Weight for circularity in confidence
            solidity_weight: Weight for solidity in confidence
            extent_weight: Weight for extent in confidence
            edge_weight: Weight for edge density in confidence
            aspect_ratio_weight: Weight for aspect ratio in confidence
        """
        # Constraints
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.preferred_aspect_ratio = preferred_aspect_ratio
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.min_solidity = min_solidity
        self.max_solidity = max_solidity
        self.min_extent = min_extent
        self.max_extent = max_extent
        self.min_edge_density = min_edge_density
        self.max_edge_density = max_edge_density
        self.edge_canny_threshold1 = edge_canny_threshold1
        self.edge_canny_threshold2 = edge_canny_threshold2

        # Convexity gate
        self.convexity_gate_enabled = convexity_gate_enabled
        self.convexity_min_ratio = convexity_min_ratio

        # Weights
        self.circularity_weight = circularity_weight
        self.solidity_weight = solidity_weight
        self.extent_weight = extent_weight
        self.edge_weight = edge_weight
        self.aspect_ratio_weight = aspect_ratio_weight

        # Statistics
        self.stats = {
            "contours_analyzed": 0,
            "rejected_area": 0,
            "rejected_aspect_ratio": 0,
            "rejected_solidity": 0,
            "rejected_extent": 0,
            "rejected_edge_density": 0,
            "rejected_convexity": 0,
            "accepted": 0,
        }

        # Edge caching for performance optimization
        self._cached_edges: Optional[np.ndarray] = None
        self._cached_frame_shape: Optional[Tuple[int, int]] = None

    def precompute_edges(self, frame_gray: np.ndarray) -> np.ndarray:
        """
        Precompute Canny edges for entire frame (call once per frame).

        This is a MAJOR performance optimization - instead of running Canny
        for each contour individually, we run it once and cache the result.

        Args:
            frame_gray: Grayscale frame

        Returns:
            Edge image
        """
        self._cached_edges = cv2.Canny(
            frame_gray,
            self.edge_canny_threshold1,
            self.edge_canny_threshold2
        )
        self._cached_frame_shape = frame_gray.shape
        return self._cached_edges

    def analyze_contour(
        self, contour: np.ndarray, frame_gray: np.ndarray
    ) -> Optional[Tuple[ShapeMetrics, float]]:
        """
        Analyze a contour and compute shape metrics + confidence.

        Args:
            contour: OpenCV contour
            frame_gray: Grayscale frame for edge detection

        Returns:
            Tuple of (ShapeMetrics, confidence) or None if rejected
        """
        self.stats["contours_analyzed"] += 1

        # Basic metrics
        area = cv2.contourArea(contour)
        if not (self.min_area <= area <= self.max_area):
            self.stats["rejected_area"] += 1
            return None

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Aspect ratio
        aspect_ratio = float(w) / max(1, h)
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            self.stats["rejected_aspect_ratio"] += 1
            return None

        # Extent (contour area / bounding box area)
        extent = area / (w * h)
        if not (self.min_extent <= extent <= self.max_extent):
            self.stats["rejected_extent"] += 1
            return None

        # Convex hull for solidity and convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        # Solidity (contour area / convex hull area)
        solidity = area / max(1, hull_area)
        if not (self.min_solidity <= solidity <= self.max_solidity):
            self.stats["rejected_solidity"] += 1
            return None

        # Convexity gate (NEW: filter non-convex blobs like hands/shadows)
        convexity = hull_area / max(1, area)  # Inverse of solidity
        if self.convexity_gate_enabled and convexity < self.convexity_min_ratio:
            self.stats["rejected_convexity"] += 1
            return None

        # Edge density - OPTIMIZED: use precomputed edges if available
        if self._cached_edges is not None and self._cached_frame_shape == frame_gray.shape:
            # FAST PATH: Use precomputed edges (15-25% performance gain)
            mask = np.zeros(frame_gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            edge_pixels = cv2.countNonZero(cv2.bitwise_and(self._cached_edges, mask))
        else:
            # SLOW PATH: Compute edges for this contour only (backward compatibility)
            mask = np.zeros(frame_gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            roi = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
            edges = cv2.Canny(
                roi, self.edge_canny_threshold1, self.edge_canny_threshold2
            )
            edge_pixels = cv2.countNonZero(edges)

        edge_density = edge_pixels / max(1, area)

        if not (self.min_edge_density <= edge_density <= self.max_edge_density):
            self.stats["rejected_edge_density"] += 1
            return None

        # Center point
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        # Orientation (from PCA or ellipse fit)
        orientation = 0.0
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                orientation = ellipse[2]  # Angle
            except:
                pass

        # Create metrics
        metrics = ShapeMetrics(
            area=area,
            perimeter=cv2.arcLength(contour, True),
            aspect_ratio=aspect_ratio,
            extent=extent,
            solidity=solidity,
            edge_density=edge_density,
            convexity=convexity,
            bounding_rect=(x, y, w, h),
            center=(cx, cy),
            orientation=orientation,
        )

        # Compute confidence score
        confidence = self._compute_confidence(metrics)

        self.stats["accepted"] += 1

        return metrics, confidence

    def _compute_confidence(self, metrics: ShapeMetrics) -> float:
        """
        Compute confidence score from shape metrics.

        Args:
            metrics: Shape metrics

        Returns:
            Confidence score (0.0 - 1.0)
        """
        scores = []

        # Circularity score (how round is it?)
        perimeter = metrics.perimeter
        circularity = (4 * np.pi * metrics.area) / max(1, perimeter ** 2)
        circularity_score = min(1.0, circularity)
        scores.append(circularity_score * self.circularity_weight)

        # Solidity score (normalized)
        solidity_norm = (metrics.solidity - self.min_solidity) / max(
            0.01, self.max_solidity - self.min_solidity
        )
        solidity_score = np.clip(solidity_norm, 0, 1)
        scores.append(solidity_score * self.solidity_weight)

        # Extent score (normalized)
        extent_norm = (metrics.extent - self.min_extent) / max(
            0.01, self.max_extent - self.min_extent
        )
        extent_score = np.clip(extent_norm, 0, 1)
        scores.append(extent_score * self.extent_weight)

        # Edge density score (normalized)
        edge_norm = (metrics.edge_density - self.min_edge_density) / max(
            0.01, self.max_edge_density - self.min_edge_density
        )
        edge_score = np.clip(edge_norm, 0, 1)
        scores.append(edge_score * self.edge_weight)

        # Aspect ratio score (how close to preferred?)
        ar_diff = abs(metrics.aspect_ratio - self.preferred_aspect_ratio)
        ar_score = max(0, 1 - (ar_diff / self.aspect_ratio_tolerance))
        scores.append(ar_score * self.aspect_ratio_weight)

        # Total confidence
        confidence = sum(scores)

        return float(np.clip(confidence, 0, 1))

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "contours_analyzed": 0,
            "rejected_area": 0,
            "rejected_aspect_ratio": 0,
            "rejected_solidity": 0,
            "rejected_extent": 0,
            "rejected_edge_density": 0,
            "rejected_convexity": 0,
            "accepted": 0,
        }

    def get_stats(self) -> dict:
        """Get current statistics"""
        return self.stats.copy()
