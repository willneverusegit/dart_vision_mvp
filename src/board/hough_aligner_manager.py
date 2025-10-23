"""
Hough Aligner Manager - Auto-alignment using Hough circle detection.

Provides intelligent board alignment using:
- Hough circle detection for center/radius refinement
- Edge-based sector rotation estimation
- EMA smoothing for stability
- Jitter guard for robust updates
"""

import logging
import math
import numpy as np
import cv2
from typing import Optional, Tuple
from pathlib import Path

from src.board import BoardConfig, Calibration
from src.board.config_models import UnifiedCalibration
from src.calibration.calib_io import compute_effective_H, save_unified_calibration, mapper_calibration_from_unified

logger = logging.getLogger(__name__)


class HoughAlignerManager:
    """
    Manages automatic board alignment using Hough circle detection.

    Features:
    - Concentric ring detection via HoughCircles
    - Sector rotation estimation via edge histogram analysis
    - EMA smoothing for stable updates
    - Jitter guard to prevent spurious corrections
    - Debug visualization support
    """

    def __init__(self, app, board_cfg: Optional[BoardConfig] = None):
        """
        Initialize Hough aligner manager.

        Args:
            app: DartVisionApp instance
            board_cfg: Board configuration (optional, can be set later)
        """
        self.app = app
        self.board_cfg = board_cfg
        self.logger = logging.getLogger("HoughAlignerManager")

        # State for EMA smoothing
        self._last_hough: Optional[Tuple[float, float, float]] = None

    def refine_rings(self, roi_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Find concentric rings via HoughCircles in ROI panel.

        Returns (cx, cy, r_double_outer_px) or None if detection fails.
        Includes jitter guard & EMA stabilization.

        Args:
            roi_bgr: BGR ROI frame

        Returns:
            Tuple of (cx, cy, r_outer_double_px) or None
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return None

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)

        # Expected radius range (±15%)
        if self.app.board_mapper is not None:
            r0 = float(self.app.board_mapper.calib.r_outer_double_px)
        elif self.app.uc is not None:
            r0 = float(self.app.uc.overlay_adjust.r_outer_double_px)
        else:
            r0 = float(self.app.roi_board_radius) * float(self.app.overlay_scale)

        rmin = max(10, int(r0 * 0.85))
        rmax = int(r0 * 1.15)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.15, minDist=18,
            param1=170, param2=32, minRadius=int(rmin), maxRadius=int(rmax)
        )

        if circles is None:
            return None

        c = np.uint16(np.around(circles[0]))  # (N,3): x,y,r
        r_out_guess = float(max(c, key=lambda k: k[2])[2])

        # Ratio-Check (ensures Triple/Double ratio is correct)
        if not self._ratio_consistent(c, r_out_guess):
            return None

        # Take largest circle and average nearby ones
        close = [k for k in c if abs(float(k[2]) - r_out_guess) < r0 * 0.03]
        cx = float(np.mean([k[0] for k in close]))
        cy = float(np.mean([k[1] for k in close]))
        r_out = float(np.mean([k[2] for k in close]))

        # --- EMA smoothing (gentle stabilization) ---
        alpha = 0.25
        if self._last_hough is not None:
            last_cx, last_cy, last_r = self._last_hough
            cx = alpha * cx + (1 - alpha) * last_cx
            cy = alpha * cy + (1 - alpha) * last_cy
            r_out = alpha * r_out + (1 - alpha) * last_r
        self._last_hough = (cx, cy, r_out)

        # --- Jitter-Guard (catch jumps) ---
        if self.app.board_mapper is not None:
            cal = self.app.board_mapper.calib
            dc = float(np.hypot(cx - cal.cx, cy - cal.cy))
            dr = abs(r_out - cal.r_outer_double_px)
            if dc > 8.0 or dr > 4.0:
                self.logger.debug(f"[HoughRings] jitter guard: skip (Δc={dc:.1f}px, Δr={dr:.1f}px)")
                return None

        # Debug: Display detected circles
        if getattr(self.app, "show_debug", False):
            dbg = roi_bgr.copy()
            cv2.circle(dbg, (int(round(cx)), int(round(cy))), int(round(r_out)), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Hough-Rings (debug)", dbg)

        return cx, cy, r_out

    def auto_sector_rotation_from_edges(
        self, roi_bgr: np.ndarray, apply: bool = True
    ) -> Optional[Tuple[float, float]]:
        """
        Estimate rotation offset (in degrees) for sector alignment.

        Uses angle histogram from edge points, exploiting the 20-fold periodicity
        of dart board segments.

        Args:
            roi_bgr: BGR ROI frame
            apply: Whether to apply the correction immediately

        Returns:
            Tuple of (delta_deg, kappa) or None if signal is weak:
            - delta_deg: Suggested correction to add to calib.rotation_deg
            - kappa: Magnitude of N-fold phase sum (0..1) as quality measure
        """
        if roi_bgr is None or roi_bgr.size == 0 or self.board_cfg is None:
            return None

        # 1) Edges in ROI
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 70, 200)

        ys, xs = np.nonzero(edges)
        if xs.size < 200:  # Too few points → unreliable signal
            return None

        # 2) Angles relative to current calibration (like in mapper)
        #    Screen angle: 0° at +x, CCW; y-axis points down → negate dy
        cal = self._overlay_calib()
        dx = xs.astype(np.float32) - float(cal.cx)
        dy = ys.astype(np.float32) - float(cal.cy)
        theta_screen = (np.degrees(np.arctan2(-dy, dx)) + 360.0) % 360.0

        # Effective theta0 (boundary or center) from config
        theta0 = self.board_cfg.angles.theta0_effective(self.board_cfg.sectors)
        # Rebase to relative angle scale of mapper
        theta_rel = (theta_screen - float(cal.rotation_deg) - theta0) % 360.0
        if self.board_cfg.angles.clockwise:
            theta_rel = (360.0 - theta_rel) % 360.0

        # Optional: Ring annulus mask (singles area usually has strongest sector edges)
        r = np.hypot(dx, dy) / max(float(cal.r_outer_double_px), 1e-6)
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
        # If peaks lie at theta = theta0_est + k*(360/N), then angle(Z) ≈ N * theta0_est
        phi = float(np.angle(Z))  # rad, (-π, π]
        theta0_est_deg = (phi * 180.0 / math.pi) / N  # in degrees, typ. in (-9, 9]
        delta_deg = -theta0_est_deg  # Correction: we want peak at 0°

        # Clamp to reasonable range (prevent "jumps")
        width = float(self.board_cfg.sectors.width_deg)  # usually 18
        half = width * 0.5
        while delta_deg <= -half:
            delta_deg += width
        while delta_deg > half:
            delta_deg -= width

        # Optional: gently limit small updates (e.g. max ±5°)
        delta_deg = float(np.clip(delta_deg, -5.0, 5.0))

        # Apply?
        if apply and self.app.uc is not None:
            self.app.uc.overlay_adjust.rotation_deg = float(
                self.app.uc.overlay_adjust.rotation_deg + delta_deg
            )
            # Refresh + persist
            self.app.homography_eff = compute_effective_H(self.app.uc)
            if hasattr(self.app, "_sync_mapper_from_unified"):
                self.app._sync_mapper_from_unified()
                self.app._roi_annulus_mask = None
                self.app._ensure_roi_annulus_mask()
            save_unified_calibration(self.app.calib_path, self.app.uc)

        # Debug plot (optional)
        if getattr(self.app, "show_debug", False):
            self._plot_angle_histogram(theta_rel, delta_deg, kappa, width, N)

        return float(delta_deg), kappa

    def process_auto_align(self, roi_frame: np.ndarray, frame_count: int) -> None:
        """
        Process auto-alignment if conditions are met.

        Runs every 15 frames in ALIGN mode when align_auto is enabled.

        Args:
            roi_frame: ROI frame to analyze
            frame_count: Current frame count
        """
        from src.ui.overlay_renderer import OVERLAY_ALIGN

        # Only run in ALIGN mode with auto-align enabled, every 15 frames
        if (
            self.app.overlay_mode != OVERLAY_ALIGN
            or not self.app.align_auto
            or frame_count % 15 != 0
        ):
            return

        # 1) Rings/Hough first → stabilize center/radius
        res = self.refine_rings(roi_frame)
        if res is None:
            self.logger.debug("[AutoAlign] Hough: no circle / rejected (ratio/jump)")
        else:
            cx, cy, r_out = res
            if self.app.uc is None:
                self.logger.warning("[AutoAlign] Unified calib missing; skip Hough update")
            else:
                base_cx, base_cy = self.app.uc.metrics.center_px
                self.app.uc.overlay_adjust.center_dx_px = float(cx) - float(base_cx)
                self.app.uc.overlay_adjust.center_dy_px = float(cy) - float(base_cy)
                self.app.uc.overlay_adjust.r_outer_double_px = float(r_out)
                self.app.homography_eff = compute_effective_H(self.app.uc)
                self.app._sync_mapper_from_unified()
                # After successful geometry update:
                self.app._roi_annulus_mask = None
                self.app._ensure_roi_annulus_mask()
                if self.app.dart is not None and self.app.board_mapper is not None:
                    self.app.dart.config.cal_cx = float(self.app.board_mapper.calib.cx)
                    self.app.dart.config.cal_cy = float(self.app.board_mapper.calib.cy)

                # Optional: save every update to persist
                save_unified_calibration(self.app.calib_path, self.app.uc)

        # 2) Then angle fine-tuning (sectors)
        res2 = self.auto_sector_rotation_from_edges(roi_frame, apply=True)
        if res2 is not None:
            dth, kappa = res2
            self.logger.info(f"[AutoRot] Δθ={dth:+.2f}°, κ={kappa:.2f}")

    def _ratio_consistent(self, circles: np.ndarray, r_out: float) -> bool:
        """
        Check if detected circles have consistent triple/double ring ratios.

        Args:
            circles: Array of detected circles (N, 3): x, y, r
            r_out: Outer double radius guess

        Returns:
            True if ratios are consistent with dartboard geometry
        """
        # Expected ratios for dartboard rings
        # r_triple ≈ 0.66 * r_outer_double
        # r_single ≈ 0.50 * r_outer_double
        expected_triple = r_out * 0.66
        expected_single = r_out * 0.50

        # Check if we have rings near expected ratios
        has_triple = any(abs(float(c[2]) - expected_triple) < r_out * 0.05 for c in circles)
        has_single = any(abs(float(c[2]) - expected_single) < r_out * 0.05 for c in circles)

        # At least one ring besides outer double should match
        return has_triple or has_single

    def _overlay_calib(self) -> Calibration:
        """Get current overlay geometry (single source)."""
        if self.app.board_mapper is not None:
            return self.app.board_mapper.calib
        # Fallback if mapper not ready but UC available
        if self.app.uc is not None:
            return mapper_calibration_from_unified(self.app.uc)
        # Last fallback: Legacy (just to prevent crashes)
        ROI_CENTER = (200.0, 200.0)  # Default
        return Calibration(
            cx=float(ROI_CENTER[0] + getattr(self.app, "overlay_center_dx", 0.0)),
            cy=float(ROI_CENTER[1] + getattr(self.app, "overlay_center_dy", 0.0)),
            r_outer_double_px=float(self.app.roi_board_radius) * float(
                getattr(self.app, "overlay_scale", 1.0)
            ),
            rotation_deg=float(getattr(self.app, "overlay_rot_deg", 0.0)),
        )

    def _plot_angle_histogram(
        self, theta_rel: np.ndarray, delta_deg: float, kappa: float, width: float, N: int
    ) -> None:
        """
        Plot angle histogram for debugging sector alignment.

        Args:
            theta_rel: Relative angles array
            delta_deg: Estimated correction angle
            kappa: Signal strength
            width: Sector width in degrees
            N: Number of sectors (periodicity)
        """
        bins = 120
        hist, edges_deg = np.histogram(theta_rel, bins=bins, range=(0.0, 360.0))
        h = np.zeros((120, 360, 3), dtype=np.uint8)
        hh = (hist / max(hist.max(), 1)) * (h.shape[0] - 1)
        for i in range(360):
            b = int((i / 360.0) * bins)
            val = int(hh[b])
            if val > 0:
                cv2.line(h, (i, h.shape[0] - 1), (i, h.shape[0] - 1 - val), (180, 255, 180), 1)
        # Mark expected 20-sector grid after correction
        for k in range(N):
            ang = (k * width) % 360.0
            x = int(round(ang))
            cv2.line(h, (x, 0), (x, h.shape[0] - 1), (100, 200, 255), 1)
        cv2.putText(
            h,
            f"dth:{delta_deg:+.2f} deg  kappa:{kappa:.2f}",
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Angle-Histogram (20x)", h)
