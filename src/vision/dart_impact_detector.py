"""
Dart Impact Detector with Temporal Confirmation
Detects dart impacts using shape analysis and temporal stability.

Features:
- Multi-frame confirmation (Land-and-Stick logic)
- Shape-based dart detection
- False positive filtering
"""

import cv2
import numpy as np
import logging
import math
from typing import Optional, List, Tuple
from dataclasses import dataclass, replace
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
    # NEU (optional, aber sehr hilfreich):
    raw_position: Optional[Tuple[int, int]] = None  # vor Tip-Refine
    refined_position: Optional[Tuple[int, int]] = None  # nach Tip-Refine


@dataclass
class DartDetectorConfig:
    """Dart detector configuration"""
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
    aspect_ratio_tolerance: float = 1.5  # multiplier on preferred ratio for scoring

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

    # ✅ NEW: Cooldown
    cooldown_frames: int = 30  # Ignore region for N frames after detection
    cooldown_radius_px: int = 50  # Radius around last detection

    # History
    candidate_history_size: int = 20

    # Motion mask pre-processing
    motion_mask_smoothing_kernel: int = 7  # 0 to disable, odd numbers recommended

    # Impact-Refine (A-2)
    refine_enabled: bool = True
    refine_threshold: float = 0.45  # 0.35–0.50 sind gute Startwerte
    refine_roi_size_px: int = 80  # Kantenprüfung in kleinem Fenster
    refine_canny_lo: int = 60
    refine_canny_hi: int = 180
    refine_hough_thresh: int = 30  # Accumulator-Schwelle
    refine_min_line_len: int = 10
    refine_max_line_gap: int = 4
    # Tip refine (A-2b)
    tip_refine_enabled: bool = True
    tip_roi_px: int = 36  # Größe des quadratischen ROI um den Kandidaten
    tip_search_px: int = 14  # maximale Weglänge vom Zentrum (px)
    tip_max_shift_px: int = 16  # harte Kappe für Verschiebung
    tip_edge_weight: float = 0.6  # Gewicht Kante vs. Dunkelheit
    tip_dark_weight: float = 0.4  # (edge_weight + dark_weight = 1.0)
    tip_canny_lo: int = 60
    tip_canny_hi: int = 180

    motion_adaptive: bool = True
    motion_otsu_bias: int = +8  # Otsu + Bias
    motion_min_area_px: int = 24  # nach Morphologie erneut prüfen
    morph_open_ksize: int = 3
    morph_close_ksize: int = 5
    motion_min_white_pct: float = 0.02  # 0.02% des ROI reichen als Aktivität


from dataclasses import replace

DETECTOR_PRESETS = {
        # finds more, toleranter, etwas mehr False Positives möglich
        "aggressive": dict(
            motion_otsu_bias=+4,
            morph_open_ksize=3,
            morph_close_ksize=5,
            min_area=6, max_area=1600,
            min_aspect_ratio=0.25, max_aspect_ratio=3.5,
            min_solidity=0.08, max_solidity=0.98,
            min_extent=0.04, max_extent=0.80,
            min_edge_density=0.015, max_edge_density=0.40,
            preferred_aspect_ratio=0.35, aspect_ratio_tolerance=1.8,
            edge_canny_threshold1=30, edge_canny_threshold2=90,
            circularity_weight=0.30, solidity_weight=0.20,
            extent_weight=0.15, edge_weight=0.20, aspect_ratio_weight=0.15,
            confirmation_frames=2, position_tolerance_px=24,
            cooldown_frames=25, cooldown_radius_px=45,
            candidate_history_size=20, motion_mask_smoothing_kernel=7,
        ),

        # dein bisheriger „Allrounder“
        "balanced": dict(
            min_area=10, max_area=1000,
            min_aspect_ratio=0.3, max_aspect_ratio=3.0,
            min_solidity=0.10, max_solidity=0.95,
            min_extent=0.05, max_extent=0.75,
            min_edge_density=0.02, max_edge_density=0.35,
            preferred_aspect_ratio=0.35, aspect_ratio_tolerance=1.5,
            edge_canny_threshold1=40, edge_canny_threshold2=120,
            circularity_weight=0.35, solidity_weight=0.20,
            extent_weight=0.15, edge_weight=0.15, aspect_ratio_weight=0.15,
            confirmation_frames=3, position_tolerance_px=20,
            cooldown_frames=30, cooldown_radius_px=50,
            candidate_history_size=20, motion_mask_smoothing_kernel=5,
        ),

        # strenger, sehr robuste Treffer, weniger False Positives
        "stable": dict(
            min_area=14, max_area=900,
            min_aspect_ratio=0.35, max_aspect_ratio=2.6,
            min_solidity=0.12, max_solidity=0.92,
            min_extent=0.06, max_extent=0.70,
            min_edge_density=0.025, max_edge_density=0.30,
            preferred_aspect_ratio=0.35, aspect_ratio_tolerance=1.3,
            edge_canny_threshold1=55, edge_canny_threshold2=165,
            circularity_weight=0.35, solidity_weight=0.20,
            extent_weight=0.15, edge_weight=0.10, aspect_ratio_weight=0.20,
            confirmation_frames=4, position_tolerance_px=18,
            cooldown_frames=40, cooldown_radius_px=55,
            candidate_history_size=24, motion_mask_smoothing_kernel=7,
        ),
    }
def _preprocess_motion_mask(self, motion_mask: np.ndarray) -> np.ndarray:
    """Adaptive Binarisierung + Morphologie für stabilere Konturen."""
    mm = motion_mask
    if mm.ndim == 3:
        mm = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)
    if not self.config.motion_adaptive:
        return mm

    # Otsu + Bias
    _t, _ = cv2.threshold(mm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = int(np.clip(_t + self.config.motion_otsu_bias, 0, 255))
    _, bw = cv2.threshold(mm, t, 255, cv2.THRESH_BINARY)

    # Morph Open → Close
    k1 = max(1, int(self.config.morph_open_ksize));  k1 += (k1 % 2 == 0)
    k2 = max(1, int(self.config.morph_close_ksize)); k2 += (k2 % 2 == 0)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)))

    # Entferne sehr kleine Blobs (zusätzlich zu min_area/max_area im späteren Schritt)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    for i in range(1, num):  # 0 ist Hintergrund
        if stats[i, cv2.CC_STAT_AREA] < self.config.motion_min_area_px:
            bw[labels == i] = 0

    return bw.astype(np.uint8, copy=False)

def apply_detector_preset(cfg: DartDetectorConfig, name: str) -> DartDetectorConfig:
        name = (name or "").lower()
        params = DETECTOR_PRESETS.get(name)
        if not params:
            return cfg
        return replace(cfg, **params)

class DartImpactDetector:
    """Dart impact detector with temporal confirmation and cooldown."""

    def __init__(self, config: Optional[DartDetectorConfig] = None):
        self.config = config or DartDetectorConfig()

        # Tracking
        self.current_candidate: Optional[DartCandidate] = None
        self.confirmation_count = 0

        # History
        self.candidate_history: deque = deque(maxlen=self.config.candidate_history_size)
        self.confirmed_impacts: List[DartImpact] = []

        # ✅ NEW: Cooldown tracking
        self.cooldown_regions: List[Tuple[Tuple[int, int], int]] = []  # (position, frame_until)

    def detect_dart(
            self,
            frame: np.ndarray,
            motion_mask: np.ndarray,
            frame_index: int,
            timestamp: float
    ) -> Optional[DartImpact]:
        """Detect dart impact with temporal confirmation and cooldown."""

        # ✅ Update cooldowns (remove expired)
        self.cooldown_regions = [
            (pos, until) for pos, until in self.cooldown_regions
            if until > frame_index
        ]

        # Pre-process motion mask to stabilise contours
        processed_mask = self._preprocess_motion_mask(motion_mask)
        # Quick gate: sehr wenig weiße Pixel? → sparen
        if motion_mask is not None:
            white_pct = 100.0 * (float(np.count_nonzero(motion_mask)) / float(motion_mask.size))
            if white_pct < getattr(self.config, "motion_min_white_pct", 0.02):
                self._reset_tracking()
                return None

        # Find dart-like shapes
        candidates = self._find_dart_shapes(frame, processed_mask, frame_index, timestamp)

        # ✅ Filter candidates in cooldown regions
        candidates = [
            c for c in candidates
            if not self._is_in_cooldown(c.position, frame_index)
        ]

        if not candidates:
            self._reset_tracking()
            return None

        # Get best candidate
        best_candidate = candidates[0]
        self.candidate_history.append(best_candidate)

        # >>> A-2: Refine-Gate (sekundäre Prüfung per Kanten/Linien im kleinen ROI)
        if getattr(self.config, "refine_enabled", True):
            rs = self._refine_impact_in_roi(
                frame,
                # HINWEIS: Hier 'frame' benutzen, wenn best_candidate.position Bildkoordinaten dieses Frames sind.
                (int(best_candidate.position[0]), int(best_candidate.position[1])),
                debug=False  # oder z.B. debug=self.config.debug falls vorhanden
            )
            # Visual Debug Overlay (zeigt Refine-Score am Kandidaten)
            cx, cy = int(best_candidate.position[0]), int(best_candidate.position[1])
            color = (0, 255, 0) if rs >= self.config.refine_threshold else (0, 0, 255)
            cv2.putText(
                frame, f"Ref:{rs:.2f}", (cx, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )
            if rs < float(getattr(self.config, "refine_threshold", 0.35)):
                # Kandidat verwerfen – KEIN Tracking-Update/Kühlzeit
                if __debug__:
                    logger.debug(
                        f"[ImpactRefine] reject pos={best_candidate.position} score={rs:.2f} "
                        f"< {self.config.refine_threshold:.2f}"
                    )
                self._reset_tracking()
                return None
            # (optional) rs weiterreichen/später loggen:
            # best_candidate.refine_score = float(rs)

        # >>> A-2b: Spitzen-Refine – Position vorsichtig nachziehen
        if getattr(self.config, "tip_refine_enabled", True):
            rx, ry = self._refine_tip_position(frame, best_candidate.position, debug=False)
            best_candidate = DartCandidate(
                position=(int(rx), int(ry)),
                area=best_candidate.area,
                confidence=best_candidate.confidence,
                frame_index=best_candidate.frame_index,
                timestamp=best_candidate.timestamp,
                aspect_ratio=best_candidate.aspect_ratio,
                solidity=best_candidate.solidity,
                extent=best_candidate.extent,
                edge_density=best_candidate.edge_density
            )

        # Check temporal stability (nur wenn refine bestanden)
        if self._is_same_position(best_candidate, self.current_candidate):
            self.confirmation_count += 1
        else:
            self.current_candidate = best_candidate
            self.confirmation_count = 1

        # Check if confirmed
        if self.confirmation_count >= self.config.confirmation_frames:
            impact = DartImpact(
                position=(best_candidate.position[0], best_candidate.position[1]),  # refined
                confidence=best_candidate.confidence,
                first_detected_frame=self.current_candidate.frame_index,
                confirmed_frame=frame_index,
                confirmation_count=self.confirmation_count,
                timestamp=timestamp,
                raw_position=(self.current_candidate.position if hasattr(self.current_candidate, "position") else None),
                refined_position=(best_candidate.position[0], best_candidate.position[1]),
                refine_score=float(rs) if 'rs' in locals() else None,
            )

            self.confirmed_impacts.append(impact)

            # ✅ Add cooldown region
            cooldown_until = frame_index + self.config.cooldown_frames
            self.cooldown_regions.append((impact.position, cooldown_until))

            self._reset_tracking()

            logger.info(f"Dart impact confirmed at {impact.position}, confidence={impact.confidence:.2f}")
            return impact

        return None

    def _is_in_cooldown(self, position: Tuple[int, int], frame_index: int) -> bool:
        """Check if position is in cooldown region."""
        for cooldown_pos, cooldown_until in self.cooldown_regions:
            if cooldown_until <= frame_index:
                continue

            # Check distance
            dx = position[0] - cooldown_pos[0]
            dy = position[1] - cooldown_pos[1]
            distance = np.sqrt(dx * dx + dy * dy)

            if distance < self.config.cooldown_radius_px:
                return True

        return False

    def _preprocess_motion_mask(self, motion_mask: np.ndarray) -> np.ndarray:
        """Reduce noise in motion mask before contour extraction."""

        kernel_size = self.config.motion_mask_smoothing_kernel
        if kernel_size and kernel_size > 1:
            if kernel_size % 2 == 0:
                kernel_size += 1

            blurred = cv2.GaussianBlur(motion_mask, (kernel_size, kernel_size), 0)
            _, thresh = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return thresh

        return motion_mask

    def _find_dart_shapes(
            self,
            frame: np.ndarray,
            motion_mask: np.ndarray,
            frame_index: int,
            timestamp: float
    ) -> List[DartCandidate]:
        """Find dart-like objects in motion mask"""

        # Find contours
        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        areas = np.fromiter(
            (cv2.contourArea(contour) for contour in contours),
            dtype=np.float32,
            count=len(contours)
        )
        valid_indices = np.nonzero(
            (areas >= self.config.min_area) & (areas <= self.config.max_area)
        )[0]
        if valid_indices.size == 0:
            return []

        candidates = []
        for idx in valid_indices:
            contour = contours[idx]
            area = float(areas[idx])


            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            if w == 0 or h == 0:
                continue

            # Calculate aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0

            # Filter by aspect ratio (darts are elongated)
            if not (self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio):
                continue

            # Calculate center
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Evaluate solidity (area relative to convex hull)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0

            if not (self.config.min_solidity <= solidity <= self.config.max_solidity):
                continue

            # Extent of filled area within the bounding box
            extent = area / float(w * h) if (w * h) > 0 else 0.0

            if not (self.config.min_extent <= extent <= self.config.max_extent):
                continue

            # Edge density to ensure crisp object edges
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            if roi.ndim == 3:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                roi_gray = roi

            edges = cv2.Canny(
                roi_gray,
                self.config.edge_canny_threshold1,
                self.config.edge_canny_threshold2
            )
            edge_density = (
                    float(np.count_nonzero(edges)) /
                    float(edges.shape[0] * edges.shape[1])
            ) if edges.size else 0.0

            if edge_density < self.config.min_edge_density:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
            circularity_score = np.clip(1.0 - circularity, 0.0, 1.0)

            solidity_score = np.clip(
                    (solidity - self.config.min_solidity) /
                    max(self.config.max_solidity - self.config.min_solidity, 1e-6),
                    0.0,
                    1.0
            )

            extent_score = np.clip(
                    (extent - self.config.min_extent) /
                    max(self.config.max_extent - self.config.min_extent, 1e-6),
                    0.0,
                    1.0
            )

            edge_score = np.clip(
                    (edge_density - self.config.min_edge_density) /
                    max(self.config.max_edge_density - self.config.min_edge_density, 1e-6),
                    0.0,
                    1.0
            )

            preferred_ratio = self.config.preferred_aspect_ratio
            tolerance = preferred_ratio * self.config.aspect_ratio_tolerance
            ratio_delta = abs(aspect_ratio - preferred_ratio)
            aspect_ratio_score = np.clip(
                    1.0 - (ratio_delta / max(tolerance, 1e-6)),
                    0.0,
                    1.0
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
                solidity = solidity,
                extent = extent,
                edge_density = edge_density
            )

            # --- PCA-Orientierung (Dominantrichtung im Bounding-ROI) ---
            roi_full = frame[y:y + h, x:x + w]
            if roi_full.size == 0:
                continue
            roi_gray2 = cv2.cvtColor(roi_full, cv2.COLOR_BGR2GRAY) if roi_full.ndim == 3 else roi_full
            edges2 = cv2.Canny(roi_gray2, 60, 180)
            ys2, xs2 = np.nonzero(edges2)

            orient_ok = True
            have_cal_center = hasattr(self.config, "cal_cx") and hasattr(self.config, "cal_cy")
            if xs2.size >= 12 and have_cal_center:
                pts2 = np.column_stack([xs2.astype(np.float32), ys2.astype(np.float32)])
                mean2 = pts2.mean(axis=0)
                pts2 -= mean2
                C2 = np.dot(pts2.T, pts2) / max(len(pts2), 1)
                _, evecs2 = np.linalg.eigh(C2)  # kleine, große Eigenwerte → letzter Vektor ist Hauptachse
                v = evecs2[:, 1]
                v = v / (np.linalg.norm(v) + 1e-6)

                # radialer Vektor: vom Boardzentrum (ROI) zum Kandidaten
                radial_dx = float(cx) - float(self.config.cal_cx)
                radial_dy = float(cy) - float(self.config.cal_cy)
                nr = math.hypot(radial_dx, radial_dy)
                if nr > 1e-3:  # nur prüfen, wenn sinnvoller Abstand existiert
                    vx, vy = float(v[0]), float(v[1])
                    dot = abs(vx * radial_dx + vy * radial_dy)
                    nv = math.hypot(vx, vy) + 1e-6
                    cosang = np.clip(dot / (nv * nr), 0.0, 1.0)
                    ang = math.degrees(math.acos(cosang))
                    if ang > getattr(self.config, "radial_angle_max_deg", 28.0):
                        orient_ok = False
            # wenn kein cal_cx/cy gesetzt → Gate überspringen (orient_ok bleibt True)
            if not orient_ok:
                continue

            candidates.append(candidate)

        # Sort by confidence (best first)
        candidates.sort(key=lambda c: c.confidence, reverse=True)

        return candidates

    def _refine_impact_in_roi(
            self,
            frame_bgr: np.ndarray,
            center_xy: tuple[int, int],
            debug: bool = False
    ) -> float:
        """
        Sekundäres Gate: prüft im kleinen ROI um 'center_xy', ob dart-ähnliche
        Kanten/Linien vorhanden sind. Rückgabe: Score 0..1 (>= threshold => ok).
        """
        x, y = center_xy
        h, w = frame_bgr.shape[:2]
        R = int(self.config.refine_roi_size_px)
        x0 = max(0, x - R // 2)
        y0 = max(0, y - R // 2)
        x1 = min(w, x + R // 2)
        y1 = min(h, y + R // 2)

        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, self.config.refine_canny_lo, self.config.refine_canny_hi)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            self.config.refine_hough_thresh,
            minLineLength=self.config.refine_min_line_len,
            maxLineGap=self.config.refine_max_line_gap
        )
        if lines is None:
            return 0.0

        # Winkel/Vertikalität bewerten (Pfeile stehen meist ~radial => im ROI oft steil)
        orientations = []
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            dx, dy = x2 - x1, y2 - y1
            L = float(np.hypot(dx, dy))
            if L < 5:
                continue
            a = abs(np.degrees(np.arctan2(dy, dx)))  # 0..180
            # Distanz zur "steilen" Richtung (0 oder 90 ist ok – wir nehmen die bessere)
            orientations.append(1.0 - min(abs(a - 90.0), a) / 90.0)
        if not orientations:
            return 0.0

        vert_score = float(np.mean(orientations))  # 0..1
        density = min(len(orientations) / 10.0, 1.0)  # saturiert bei 10 Linien
        score = 0.6 * vert_score + 0.4 * density  # gewichtete Mischung

        if debug:
            dbg = np.hstack([gray, edges])
            cv2.imshow("Refine-ROI", dbg)

        return float(score)

    def _refine_tip_position(
            self,
            frame_bgr: np.ndarray,
            center_xy: tuple[int, int],
            debug: bool = False
    ) -> tuple[int, int]:
        """
        Verfeinert die Impact-Position auf die Dartspitze.
        Nutzt Kanten im Mini-ROI, schätzt Hauptorientierung (PCA),
        prüft beide Richtungen (±v) und nimmt das Ende mit mehr Kante + dunklerem Patch.
        """
        cx, cy = int(center_xy[0]), int(center_xy[1])
        h, w = frame_bgr.shape[:2]
        R = int(self.config.tip_roi_px)
        x0 = max(0, cx - R // 2);
        y0 = max(0, cy - R // 2)
        x1 = min(w, cx + R // 2);
        y1 = min(h, cy + R // 2)
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return cx, cy

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, self.config.tip_canny_lo, self.config.tip_canny_hi)

        # --- PCA auf Kantenpixeln für Hauptorientierung ---
        ys, xs = np.nonzero(edges)
        if xs.size < 10:
            return cx, cy

        pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        mean = pts.mean(axis=0)  # (mx, my) im ROI
        pts_z = pts - mean
        C = np.dot(pts_z.T, pts_z) / max(len(pts_z), 1)  # 2x2 Kovarianz
        evals, evecs = np.linalg.eigh(C)  # sort unsorted: kleiner->größer
        v = evecs[:, 1]  # Hauptachse (Eigenvektor für größte Eigenzahl)
        v = v / (np.linalg.norm(v) + 1e-6)
        # von ROI in Bildkoordinaten: v_x rechts positiv, v_y nach unten positiv
        # zwei Kandidatsrichtungen: +v und -v
        candidates = []
        L = int(self.config.tip_search_px)
        for sgn in (+1.0, -1.0):
            vx, vy = (float(v[0]) * sgn, float(v[1]) * sgn)
            # Probe entlang der Linie in N Schritten
            N = L
            xs_line = (mean[0] + np.linspace(1, N, N) * vx).astype(np.int32)
            ys_line = (mean[1] + np.linspace(1, N, N) * vy).astype(np.int32)
            mask = (xs_line >= 0) & (ys_line >= 0) & (xs_line < roi.shape[1]) & (ys_line < roi.shape[0])
            xs_line, ys_line = xs_line[mask], ys_line[mask]
            if xs_line.size == 0:
                continue

            # Kantenstärke am Ende (letzte 3 Pixel mitteln)
            tail_idx = max(0, xs_line.size - 3)
            edge_tail = float(edges[ys_line[tail_idx:], xs_line[tail_idx:]].mean()) / 255.0

            # Dunkelheit am Ende (grau klein = dunkel)
            dark_tail = 1.0 - float(gray[ys_line[tail_idx:], xs_line[tail_idx:]].mean()) / 255.0

            score = self.config.tip_edge_weight * edge_tail + self.config.tip_dark_weight * dark_tail

            # Endpunkt in Bildkoordinaten
            end_x = int(x0 + xs_line[-1])
            end_y = int(y0 + ys_line[-1])
            candidates.append((score, end_x, end_y))

        if not candidates:
            return cx, cy

        # Bester Endpunkt
        candidates.sort(key=lambda t: t[0], reverse=True)
        _, ex, ey = candidates[0]

        # Verschiebung kappen
        dx = np.clip(ex - cx, -self.config.tip_max_shift_px, self.config.tip_max_shift_px)
        dy = np.clip(ey - cy, -self.config.tip_max_shift_px, self.config.tip_max_shift_px)
        rx, ry = cx + int(dx), cy + int(dy)

        if debug:
            dbg = roi.copy()
            # Lokales Koord. visual: ROI-Zentrum (cx,cy) → (cx-x0, cy-y0)
            cv2.circle(dbg, (int(mean[0]), int(mean[1])), 3, (0, 255, 255), -1, cv2.LINE_AA)
            # Linie ±v
            p0 = (int(mean[0] - v[0] * L), int(mean[1] - v[1] * L))
            p1 = (int(mean[0] + v[0] * L), int(mean[1] + v[1] * L))
            cv2.line(dbg, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(dbg, (ex - x0, ey - y0), 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.imshow("Tip-Refine ROI", dbg)

        return int(rx), int(ry)

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
        """Clear confirmed impacts (e.g., new game round)"""
        self.confirmed_impacts.clear()
        logger.info("Dart impacts cleared")


@dataclass
class FieldMapperConfig:
    """Field mapper configuration"""
    # Sector configuration (standard dartboard)
    sector_scores: List[int] = None

    # Ring radii (normalized to board radius = 1.0)
    bull_inner_radius: float = 0.05  # 50 points
    bull_outer_radius: float = 0.095  # 25 points
    triple_inner_radius: float = 0.53
    triple_outer_radius: float = 0.58
    double_inner_radius: float = 0.94
    double_outer_radius: float = 1.00

    def __post_init__(self):
        if self.sector_scores is None:
            # Standard dartboard layout (clockwise from top, 20 at 12 o'clock)
            self.sector_scores = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


class FieldMapper:
    """
    Maps pixel coordinates to dartboard scores.

    Converts (x, y) position to sector and multiplier.
    """

    def __init__(self, config: Optional[FieldMapperConfig] = None):
        self.config = config or FieldMapperConfig()

        self.sector_angle_deg = 18  # 360° / 20 sectors
        self.sector_offset_deg = 0  # Offset to center 20 at top

    def point_to_score(
            self,
            point: Tuple[int, int],
            center: Tuple[int, int],
            radius: float
    ) -> Tuple[int, int, int]:
        """
        Convert pixel coordinates to dartboard score.

        Args:
            point: (x, y) position in image
            center: (x, y) dartboard center
            radius: Dartboard radius in pixels

        Returns:
            (score, multiplier, segment) tuple
            - score: Base score (1-20, 25, or 50)
            - multiplier: 1=single, 2=double, 3=triple
            - segment: Sector index (0-19, or -1 for bull)
        """
        # Calculate polar coordinates
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        distance = np.sqrt(dx * dx + dy * dy)
        angle_rad = np.arctan2(dy, dx)

        # Normalize distance to radius
        norm_distance = distance / radius if radius > 0 else 0

        # Check bulls
        if norm_distance <= self.config.bull_inner_radius:
            return (50, 1, -1)  # Inner bull

        if norm_distance <= self.config.bull_outer_radius:
            return (25, 1, -1)  # Outer bull

        # Outside board
        if norm_distance > self.config.double_outer_radius:
            return (0, 0, -1)

        # Determine sector
        angle_deg = np.degrees(angle_rad)
        adjusted_angle = (angle_deg + 90 + self.sector_offset_deg) % 360
        sector_index = int(adjusted_angle / self.sector_angle_deg) % 20
        base_score = self.config.sector_scores[sector_index]

        # Determine multiplier from ring
        if self.config.triple_inner_radius <= norm_distance <= self.config.triple_outer_radius:
            multiplier = 3
        elif self.config.double_inner_radius <= norm_distance <= self.config.double_outer_radius:
            multiplier = 2
        else:
            multiplier = 1

        return (base_score, multiplier, sector_index)
