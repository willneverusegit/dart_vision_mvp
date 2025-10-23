"""
Calibration Loader - Unified calibration loading and application.

Supports multiple calibration schemas:
- Unified schema (preferred): homography + metrics + overlay_adjust + roi_adjust
- Typed schema: charuco | aruco_quad | homography_only
- Legacy schema: flat structure with method field

Provides backward/forward compatibility across all calibration formats.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any

from src.board.config_models import UnifiedCalibration, Homography, Metrics, OverlayAdjust, ROIAdjust
from src.calibration.calib_io import compute_effective_H

logger = logging.getLogger(__name__)


class CalibLoader:
    """
    Handles loading and applying calibration data to DartVisionApp.

    Supports three calibration schema types:
    1. Unified (preferred): Single source of truth with all parameters
    2. Typed: Explicit type field (charuco, aruco_quad, homography_only)
    3. Legacy: Flat structure with method field

    All schemas are converted to UnifiedCalibration internally for consistency.
    """

    def __init__(self, app):
        """
        Initialize calibration loader.

        Args:
            app: DartVisionApp instance to apply calibration to
        """
        self.app = app
        self.logger = logging.getLogger("CalibLoader")

    def apply_yaml_data(self, data: dict):
        """
        Apply calibration from YAML data (typed schema).

        Supports:
        - type: "charuco" - ChArUco calibration with intrinsics
        - type: "aruco_quad" - ArUco-Quad homography calibration
        - type: "homography_only" - Simple homography without intrinsics

        Args:
            data: Dictionary loaded from YAML
        """
        t = (data or {}).get("type")

        if t == "charuco":
            self._apply_charuco(data)
        elif t == "aruco_quad":
            self._apply_aruco_quad(data)
        elif t == "homography_only":
            self._apply_homography_only(data)
        else:
            self.logger.warning("[LOAD] Unknown or missing type in YAML.")

    def apply_calibration(self, cfg: dict):
        """
        Apply calibration from any schema format (unified, typed, or legacy).

        Automatically detects schema type and applies appropriate loader.

        Args:
            cfg: Dictionary loaded from calibration file
        """
        if not cfg:
            return

        # --- Unified schema (preferred) ---
        root = cfg.get("calibration", cfg)
        unified_keys = ("homography", "metrics", "overlay_adjust", "roi_adjust")

        if all(k in root for k in unified_keys):
            try:
                self._apply_unified(root)
                return
            except Exception as e:
                self.logger.warning(f"Unified calibration present but failed to parse/apply: {e}")

        # --- Typed schema ---
        if "type" in cfg:
            self.apply_yaml_data(cfg)
            return

        # --- Legacy schema ---
        self._apply_legacy(cfg)

    def _apply_unified(self, root: dict):
        """
        Apply unified calibration schema (single source of truth).

        Args:
            root: Dictionary with unified schema keys
        """
        # Build UnifiedCalibration model
        self.app.uc = UnifiedCalibration(
            homography=Homography(**root["homography"]),
            metrics=Metrics(**root["metrics"]),
            overlay_adjust=OverlayAdjust(**root["overlay_adjust"]),
            roi_adjust=ROIAdjust(**root["roi_adjust"]),
        )

        # Base & effective homography
        self.app.homography = np.asarray(self.app.uc.homography.H, dtype=np.float64)
        self.app.homography_eff = compute_effective_H(self.app.uc)

        # Sync mapper immediately
        self.app._sync_mapper_from_unified()
        self.app._roi_annulus_mask = None
        self.app._ensure_roi_annulus_mask()

        self.logger.info("[CALIB] Applied unified calibration")

    def _apply_charuco(self, data: dict):
        """
        Apply ChArUco calibration with camera intrinsics.

        Args:
            data: Dictionary with ChArUco calibration data
        """
        cam = data.get("camera", {})
        K = cam.get("matrix")
        D = cam.get("dist_coeffs")

        if K is not None:
            self.app.cal.K = np.array(K, dtype=np.float64)
        if D is not None:
            self.app.cal.D = np.array(D, dtype=np.float64).reshape(-1, 1)

        self.app.cal._rms = float(cam.get("rms_px", 0.0))
        self.app.cal.last_image_size = tuple(cam.get("image_size", (0, 0)))

        H = (data.get("homography") or {}).get("H")
        if H is not None:
            self.app.homography = np.array(H, dtype=np.float64)

        self.logger.info("[LOAD] Applied ChArUco intrinsics from YAML.")

        # Apply overlay & ROI adjustments
        self._apply_overlay_adjust(data)
        self._apply_roi_adjust(data)

    def _apply_aruco_quad(self, data: dict):
        """
        Apply ArUco-Quad homography calibration.

        Args:
            data: Dictionary with ArUco-Quad calibration data
        """
        H = (data.get("homography") or {}).get("H")
        if H is not None:
            self.app.homography = np.array(H, dtype=np.float64)

        scale = data.get("scale") or {}
        self.app.mm_per_px = scale.get("mm_per_px")

        self.logger.info("[LOAD] Applied ArUco-Quad homography from YAML.")

        # Apply overlay & ROI adjustments
        self._apply_overlay_adjust(data)
        self._apply_roi_adjust(data)

    def _apply_homography_only(self, data: dict):
        """
        Apply homography-only calibration (no intrinsics).

        Args:
            data: Dictionary with homography calibration data
        """
        H = (data.get("homography") or {}).get("H")
        if H is not None:
            self.app.homography = np.array(H, dtype=np.float64)

        metrics = data.get("metrics") or {}
        self.app.mm_per_px = metrics.get("mm_per_px")

        self.logger.info("[LOAD] Applied Homography-only from YAML.")

        # Apply overlay & ROI adjustments
        self._apply_overlay_adjust(data)
        self._apply_roi_adjust(data)

    def _apply_legacy(self, cfg: dict):
        """
        Apply legacy flat calibration schema.

        Args:
            cfg: Dictionary with legacy calibration data
        """
        # Extract homography (can be list or {"H": list})
        H_node = cfg.get("homography")
        if isinstance(H_node, dict):
            H_list = H_node.get("H")
        else:
            H_list = H_node

        self.app.homography = np.array(H_list, dtype=np.float64) if H_list is not None else None
        self.app.mm_per_px = float(cfg.get("mm_per_px", 1.0))
        self.app.center_px = tuple(cfg.get("center_px", [0, 0]))
        self.app.roi_board_radius = float(cfg.get("roi_board_radius", 160.0))

        # Camera intrinsics (optional)
        if cfg.get("camera_matrix") is not None:
            self.app.cal.K = np.array(cfg["camera_matrix"], dtype=np.float64)
        if cfg.get("dist_coeffs") is not None:
            self.app.cal.D = np.array(cfg["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

        # Apply overlay adjustments
        self._apply_overlay_adjust_legacy(cfg)

        # Build unified structure from legacy data
        self._build_unified_from_legacy()

    def _apply_overlay_adjust(self, data: dict):
        """
        Apply overlay adjustment parameters (typed schema).

        Args:
            data: Dictionary containing overlay_adjust section
        """
        ov = (data or {}).get("overlay_adjust") or {}

        if "rotation_deg" in ov:
            self.app.overlay_rot_deg = float(ov["rotation_deg"])
        if "scale" in ov:
            self.app.overlay_scale = float(ov["scale"])
        if "center_dx_px" in ov:
            self.app.overlay_center_dx = float(ov["center_dx_px"])
        if "center_dy_px" in ov:
            self.app.overlay_center_dy = float(ov["center_dy_px"])

        # Recalculate scale from absolute radius if provided
        abs_r = ov.get("r_outer_double_px")
        if abs_r is not None and self.app.roi_board_radius and self.app.roi_board_radius > 0:
            self.app.overlay_scale = float(abs_r) / float(self.app.roi_board_radius)

    def _apply_roi_adjust(self, data: dict):
        """
        Apply ROI adjustment parameters.

        Args:
            data: Dictionary containing roi_adjust section
        """
        roi_adj = (data or {}).get("roi_adjust") or {}

        self.app.roi_rot_deg = float(roi_adj.get("rot_deg", self.app.roi_rot_deg))
        self.app.roi_scale = float(roi_adj.get("scale", self.app.roi_scale))
        self.app.roi_tx = float(roi_adj.get("tx_px", self.app.roi_tx))
        self.app.roi_ty = float(roi_adj.get("ty_px", self.app.roi_ty))
        self.app._roi_adjust_dirty = True

    def _apply_overlay_adjust_legacy(self, cfg: dict):
        """
        Apply overlay adjustment parameters (legacy schema).

        Args:
            cfg: Dictionary containing overlay_adjust section
        """
        ov = (cfg or {}).get("overlay_adjust") or {}

        if "rotation_deg" in ov:
            self.app.overlay_rot_deg = float(ov["rotation_deg"])
        if "scale" in ov:
            self.app.overlay_scale = float(ov["scale"])
        if "center_dx_px" in ov:
            self.app.overlay_center_dx = float(ov["center_dx_px"])
        if "center_dy_px" in ov:
            self.app.overlay_center_dy = float(ov["center_dy_px"])

        # Recalculate scale from absolute radius
        abs_r = ov.get("r_outer_double_px")
        if abs_r is not None and self.app.roi_board_radius and self.app.roi_board_radius > 0:
            self.app.overlay_scale = float(abs_r) / float(self.app.roi_board_radius)

    def _build_unified_from_legacy(self):
        """
        Build UnifiedCalibration from legacy data for consistency.

        Converts legacy calibration into unified format so the rest of the
        pipeline can work uniformly with self.app.uc and self.app.homography_eff.
        """
        try:
            if self.app.homography is not None:
                H = self.app.homography
            else:
                H = np.eye(3, dtype=np.float64)

            # Baseline center
            cx0 = float(getattr(self.app, "ROI_CENTER", (0.0, 0.0))[0])
            cy0 = float(getattr(self.app, "ROI_CENTER", (0.0, 0.0))[1])
            r_od = float(self.app.roi_board_radius) if hasattr(self.app, "roi_board_radius") else 160.0
            rot = float(getattr(self.app, "overlay_rot_deg", 0.0))
            dx = float(getattr(self.app, "overlay_center_dx", 0.0))
            dy = float(getattr(self.app, "overlay_center_dy", 0.0))

            self.app.uc = UnifiedCalibration(
                homography=Homography(H=H.tolist()),
                metrics=Metrics(center_px=(cx0, cy0), roi_board_radius=r_od),
                overlay_adjust=OverlayAdjust(
                    rotation_deg=rot,
                    r_outer_double_px=float(r_od * float(getattr(self.app, "overlay_scale", 1.0))),
                    center_dx_px=dx,
                    center_dy_px=dy,
                ),
                roi_adjust=ROIAdjust(),  # Neutral (unknown)
            )

            self.app.homography_eff = compute_effective_H(self.app.uc)
            self.app._sync_mapper_from_unified()
            self.app._roi_annulus_mask = None
            self.app._ensure_roi_annulus_mask()

            # Propagate to dart detector if available
            if self.app.dart is not None and self.app.board_mapper is not None:
                self.app.dart.config.cal_cx = float(self.app.board_mapper.calib.cx)
                self.app.dart.config.cal_cy = float(self.app.board_mapper.calib.cy)

            self.logger.info("[CALIB] Built unified structure from legacy data")

        except Exception as e:
            # If this fails, legacy mode continues without unified structure
            self.logger.debug(f"[CALIB] Could not build unified from legacy: {e}")
