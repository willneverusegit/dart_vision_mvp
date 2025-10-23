# Hough circle detection for board center/radius estimation
from __future__ import annotations
import cv2
import numpy as np
import math
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.board import BoardConfig


def find_board_circle(gray: np.ndarray,
                      dp: float = 1.2,
                      min_dist: float = 100.0,
                      param1: float = 150.0,
                      param2: float = 60.0,
                      min_radius: int = 100,
                      max_radius: int = 0) -> Optional[Tuple[int,int,int]]:
    """Return (cx, cy, r) for the strongest circle or None.
    gray: uint8 image (pre-blur recommended).
    """
    if gray.ndim != 2:
        raise ValueError("gray image expected")
    # Mild blur to stabilize edges
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))
    # take first/strongest
    x, y, r = circles[0][0]
    return int(x), int(y), int(r)


class HoughCircleAligner:
    """
    Advanced Hough circle detection for dartboard alignment.

    Provides robust ring detection with:
    - Ratio consistency checking
    - Jitter guard
    - EMA stabilization
    - Auto sector rotation from edge analysis
    """

    def __init__(self, board_config: Optional['BoardConfig'] = None):
        """
        Initialize aligner.

        Args:
            board_config: BoardConfig with expected ring ratios
        """
        self.board_config = board_config

    def ratio_consistent(self, circles: np.ndarray, r_out: float) -> bool:
        """
        Check if inner rings appear in expected ratio to r_out.

        Args:
            circles: Detected circles array (N, 3) with (x, y, r)
            r_out: Outer double ring radius

        Returns:
            True if at least 2 expected rings found within tolerance
        """
        if self.board_config is None or r_out <= 0:
            return True

        target = {
            "D_in": self.board_config.radii.r_double_inner,  # ~0.9
            "T_out": self.board_config.radii.r_triple_outer,  # ~0.55
            "T_in": self.board_config.radii.r_triple_inner,  # ~0.45
        }

        rs = np.array([float(c[2]) for c in circles if float(c[2]) < r_out], dtype=float)
        if rs.size == 0:
            return False

        ratios = rs / r_out
        ok = 0
        for r in target.values():
            if np.min(np.abs(ratios - float(r))) < 0.1:  # 10% tolerance
                ok += 1
        return ok >= 2  # at least two "hits"

    def refine_rings(
        self,
        roi_bgr: np.ndarray,
        current_center: Tuple[float, float],
        current_radius: float
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find concentric rings via HoughCircles in ROI panel.

        Returns (cx, cy, r_double_outer_px) or None.
        With jitter guard & EMA stabilization.

        Args:
            roi_bgr: ROI image (BGR)
            current_center: Current center estimate (cx, cy)
            current_radius: Current outer double radius estimate

        Returns:
            Tuple of (cx, cy, r_outer) or None if detection failed
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return None

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)

        # Search window around current value (±15%)
        r0 = float(current_radius)
        rmin = max(10, int(r0 * 0.85))
        rmax = int(r0 * 1.15)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.15, minDist=18,
            param1=170, param2=32, minRadius=rmin, maxRadius=rmax
        )

        if circles is None:
            return None

        c = np.uint16(np.around(circles[0]))  # (N,3): x,y,r
        r_out_guess = float(max(c, key=lambda k: k[2])[2])

        # Ratio check (ensures Triple/Double ratio is correct)
        if not self.ratio_consistent(c, r_out_guess):
            return None

        # Take largest circle and average nearby ones
        close = [k for k in c if abs(float(k[2]) - r_out_guess) < r0 * 0.03]
        cx = float(np.mean([k[0] for k in close]))
        cy = float(np.mean([k[1] for k in close]))
        r_out = float(np.mean([k[2] for k in close]))

        return cx, cy, r_out

    def refine_center_legacy(
        self,
        roi_bgr: np.ndarray,
        current_center: Tuple[float, float],
        current_radius: float,
        gain_center: float = 0.35,
        gain_scale: float = 0.30,
        use_clahe: bool = False
    ) -> Optional[dict]:
        """
        LEGACY/DEBUG: Find outer double ring and adjust overlay center & scale.
        Uses current overlay center/radius as reference, smooth gains, robustness.

        Args:
            roi_bgr: ROI image (BGR)
            current_center: Current center (cx, cy)
            current_radius: Current radius
            gain_center: Gain for center adjustment (0..1)
            gain_scale: Gain for scale adjustment (0..1)
            use_clahe: Whether to apply CLAHE preprocessing

        Returns:
            Dict with adjustment info or None if detection failed
        """
        if roi_bgr is None:
            return None

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        gray = cv2.medianBlur(gray, 5)

        cx_cur, cy_cur = current_center
        r_cur = float(current_radius)

        # Hough radius window around current target radius
        r_min = int(max(10, 0.88 * r_cur))
        r_max = int(min(roi_bgr.shape[1], 1.12 * r_cur))

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=100,
            param1=120, param2=40,
            minRadius=r_min, maxRadius=r_max
        )

        if circles is None or len(circles) == 0:
            return None

        cs = np.round(circles[0, :]).astype(int)

        # Take circle closest to (r_cur) and near expected center
        def score(c):
            cx, cy, r = c
            dr = abs(r - r_cur)
            dc = abs(cx - cx_cur) + abs(cy - cy_cur)
            return (dr, dc)

        cs = sorted(cs, key=score)
        cx, cy, r = [int(v) for v in cs[0]]

        # Deviations
        dx_meas = float(cx) - cx_cur
        dy_meas = float(cy) - cy_cur
        sr_meas = float(r) / max(r_cur, 1e-6)

        # Sanity checks (reject large jumps)
        roi_w, roi_h = roi_bgr.shape[1], roi_bgr.shape[0]
        if (abs(dx_meas) > roi_w * 0.4) or (abs(dy_meas) > roi_h * 0.4):
            return None
        if not (0.85 <= sr_meas <= 1.15):
            return None

        # Calculate adjustments
        dx_adj = gain_center * dx_meas
        dy_adj = gain_center * dy_meas
        scale_adj = 1.0 + gain_scale * (sr_meas - 1.0)

        return {
            'dx': dx_adj,
            'dy': dy_adj,
            'scale': scale_adj,
            'raw_dx': dx_meas,
            'raw_dy': dy_meas,
            'raw_scale': sr_meas,
            'detected_center': (cx, cy),
            'detected_radius': r
        }

    def auto_sector_rotation_from_edges(
        self,
        roi_bgr: np.ndarray,
        calibration: object,
        board_config: 'BoardConfig'
    ) -> Optional[Tuple[float, float]]:
        """
        Estimate rotation offset (degrees) for sector alignment from edge histogram.
        Uses 20-fold periodicity of sectors.

        Args:
            roi_bgr: ROI image (BGR)
            calibration: Calibration object with cx, cy, r_outer_double_px, rotation_deg
            board_config: BoardConfig with sector info

        Returns:
            Tuple of (delta_deg, kappa) or None if signal too weak
            - delta_deg: Suggested rotation adjustment
            - kappa: Quality metric (0..1) from N-fold phase sum
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return None

        # 1) Extract edges in ROI
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 70, 200)

        ys, xs = np.nonzero(edges)
        if xs.size < 200:  # Too few points
            return None

        # 2) Calculate angles relative to current calibration
        # Screen angles: 0° at +x, CCW; y-axis is down → negate dy
        dx = xs.astype(np.float32) - float(calibration.cx)
        dy = ys.astype(np.float32) - float(calibration.cy)
        theta_screen = (np.degrees(np.arctan2(-dy, dx)) + 360.0) % 360.0

        # Effective theta0 (boundary or center) from config
        theta0 = board_config.angles.theta0_effective(board_config.sectors)

        # Rebase to relative angle scale of mapper
        theta_rel = (theta_screen - float(calibration.rotation_deg) - theta0) % 360.0
        if board_config.angles.clockwise:
            theta_rel = (360.0 - theta_rel) % 360.0

        # Optional: Mask ring annulus (singles area usually has strongest sector edges)
        r = np.hypot(dx, dy) / max(float(calibration.r_outer_double_px), 1e-6)
        annulus = (r > 0.30) & (r < 0.92)  # Between single-inner and double-inner
        theta_rel = theta_rel[annulus]
        if theta_rel.size < 200:
            return None

        # 3) Aggregate 20-fold periodicity: Z = mean(exp(i*N*theta))
        N = 20
        theta_rad = np.deg2rad(theta_rel)
        Z = np.exp(1j * N * theta_rad).mean()
        kappa = float(np.abs(Z))  # 0..1 signal strength

        # Weak signal? abort
        if not np.isfinite(kappa) or kappa < 0.03:
            return None

        # Phase position → estimated offset relative to current 0-line
        # If peaks at theta = theta0_est + k*(360/N), then angle(Z) ≈ N * theta0_est
        phi = float(np.angle(Z))  # rad, (-π, π]
        theta0_est_deg = (phi * 180.0 / math.pi) / N  # degrees, typically in (-9, 9]
        delta_deg = -theta0_est_deg  # Correction: we want peak at 0°

        # Clamp to sensible range (prevents "jumps")
        width = float(board_config.sectors.width_deg)  # usually 18
        half = width * 0.5
        while delta_deg <= -half:
            delta_deg += width
        while delta_deg > half:
            delta_deg -= width

        # Limit small updates (e.g., max ±5°)
        delta_deg = float(np.clip(delta_deg, -5.0, 5.0))

        return float(delta_deg), kappa
