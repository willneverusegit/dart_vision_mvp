"""
Calibration UI Manager - Interactive calibration interface.

Provides clean, reusable calibration UI with:
- ChArUco/AruCo detection
- Manual 4-corner homography
- ArUco-Quad one-shot calibration
- Real-time quality metrics (HUD)
- Interactive visual feedback
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, Tuple, List, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class CalibrationUIManager:
    """
    Manages interactive calibration UI with multiple calibration methods.

    Supports:
    - ArUco-Quad calibration (one-shot from 4 markers)
    - Manual 4-corner homography (click-based)
    - ChArUco calibration (collect samples)
    """

    def __init__(
        self,
        calibrator,  # UnifiedCalibrator instance
        camera,  # ThreadedCamera instance
        roi_size: Tuple[int, int] = (400, 400),
        aruco_quad_calibrator=None,  # Optional ArucoQuadCalibrator
        aruco_rect_mm: Optional[Tuple[float, float]] = None,
        use_clahe: bool = False,
        hud_renderer=None  # Optional HUDRenderer for metrics
    ):
        """
        Initialize calibration UI manager.

        Args:
            calibrator: UnifiedCalibrator instance
            camera: ThreadedCamera instance
            roi_size: ROI size (width, height)
            aruco_quad_calibrator: Optional ArucoQuadCalibrator
            aruco_rect_mm: Expected ArUco quad size (width_mm, height_mm)
            use_clahe: Whether to apply CLAHE preprocessing
            hud_renderer: Optional HUDRenderer for quality metrics
        """
        self.calibrator = calibrator
        self.camera = camera
        self.roi_size = roi_size
        self.aruco_quad = aruco_quad_calibrator
        self.aruco_rect_mm = aruco_rect_mm
        self.use_clahe = use_clahe
        self.hud_renderer = hud_renderer

        # State
        self.homography: Optional[np.ndarray] = None
        self.center_px: Optional[Tuple[float, float]] = None
        self.roi_board_radius: Optional[float] = None
        self.mm_per_px: Optional[float] = None

        # Manual calibration state
        self.clicked_pts: List[Tuple[int, int]] = []
        self.captured_for_manual: Optional[np.ndarray] = None

        # Callbacks
        self.on_calibration_complete: Optional[Callable] = None

    def run_interactive_ui(self) -> bool:
        """
        Run interactive calibration UI.

        Returns:
            True if calibration was successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("INTERACTIVE CALIBRATION UI")
        logger.info("=" * 60)
        logger.info("Keys:")
        logger.info("  a = ArUco-Quad calibration (one-shot)")
        logger.info("  m = Manual 4-corner (click TL, TR, BR, BL)")
        logger.info("  s = Save calibration")
        logger.info("  q = Quit")
        logger.info("=" * 60)

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self._on_mouse)

        try:
            while True:
                ok, frame = self.camera.read()
                if not ok:
                    continue

                display = self._render_calibration_frame(frame)
                cv2.imshow("Calibration", display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('a'):
                    self._handle_aruco_quad_calibration(frame)
                elif key == ord('m'):
                    self._handle_manual_calibration_start(frame)
                elif key == ord('s'):
                    if self._handle_save_calibration():
                        return True

        finally:
            cv2.destroyWindow("Calibration")

        return self.homography is not None

    def _render_calibration_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Render calibration frame with overlays.

        Args:
            frame: Input frame

        Returns:
            Frame with overlays
        """
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # Detect markers
        mk_c, mk_ids, ch_c, ch_ids = self.calibrator.detect_charuco(frame)

        # Draw markers
        if mk_ids is not None and len(mk_ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, mk_c, mk_ids)

        # Draw ChArUco corners
        if ch_c is not None and len(ch_c) > 0:
            for pt in ch_c:
                p = tuple(pt.astype(int).ravel())
                cv2.circle(display, p, 3, (0, 255, 0), -1)

        # HUD with metrics
        if self.hud_renderer is not None:
            b_mean, f_var, e_pct = self.hud_renderer.compute_metrics(gray)
            hud_color = self.hud_renderer.get_metrics_color(b_mean, f_var, e_pct)

            y = 30
            lines = [
                f"B:{b_mean:.0f}  F:{int(f_var)}  E:{e_pct:.1f}%   (targets B~135-150, F>800, E~4-12%)",
                "Keys: [a] aruco-quad  [m] manual-4  [s] save  [q] quit"
            ]
            for line in lines:
                cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2, cv2.LINE_AA)
                y += 22

            self.hud_renderer.draw_traffic_light(display, b_mean, f_var, e_pct, org=(12, 105))
        else:
            # Fallback: simple text
            cv2.putText(display, "Keys: [a] aruco  [m] manual  [s] save  [q] quit",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw clicked points (manual mode)
        if self.captured_for_manual is not None:
            for i, pt in enumerate(self.clicked_pts):
                cv2.circle(display, pt, 8, (0, 255, 255), -1)
                cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if len(self.clicked_pts) < 4:
                msg = f"Click corner {len(self.clicked_pts) + 1}/4"
                cv2.putText(display, msg, (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return display

    def _handle_aruco_quad_calibration(self, frame: np.ndarray):
        """Handle ArUco-Quad one-shot calibration."""
        if self.aruco_quad is None:
            logger.warning("[ArucoQuad] Not configured")
            return

        okH, H, mmpp, info = self.aruco_quad.calibrate_from_frame(
            frame,
            rect_width_mm=self.aruco_rect_mm[0] if self.aruco_rect_mm else None,
            rect_height_mm=self.aruco_rect_mm[1] if self.aruco_rect_mm else None
        )

        if not okH:
            logger.warning(
                f"[ArucoQuad] Failed: {info.get('reason', 'unknown')}, markers={info.get('markers', 0)}"
            )
            return

        # Store homography
        self.calibrator.H = H
        self.calibrator.last_image_size = frame.shape[1], frame.shape[0]

        # Calculate center and radius
        h, w = frame.shape[:2]
        self.center_px = (w * 0.5, h * 0.5)

        # Derive radius from mm/px and board diameter
        r_od_px = None
        try:
            if mmpp and hasattr(self.calibrator, "board_diameter_mm"):
                r_od_px = float(self.calibrator.board_diameter_mm) * 0.5 / float(mmpp)
        except Exception:
            pass

        if r_od_px is None:
            r_od_px = 420.0  # Default fallback

        self.homography = H
        self.roi_board_radius = r_od_px
        self.mm_per_px = mmpp

        # Visual feedback
        disp = self.aruco_quad.draw_debug(frame, [], None, None, H)
        cv2.imshow("ArucoQuad Preview", disp)

        logger.info(f"[ArucoQuad] ✓ H computed | ids={info.get('ids')} | mm/px={mmpp:.4f}")

        # Callback
        if self.on_calibration_complete:
            self.on_calibration_complete(H, self.center_px, r_od_px, mmpp)

    def _handle_manual_calibration_start(self, frame: np.ndarray):
        """Start manual 4-corner calibration."""
        self.captured_for_manual = frame.copy()
        self.clicked_pts.clear()
        logger.info("[Manual] Frame captured. Click 4 corners (TL, TR, BR, BL).")

    def _handle_save_calibration(self) -> bool:
        """
        Handle save calibration command.

        Returns:
            True if saved successfully and should exit UI
        """
        # Manual calibration: compute H from 4 clicked points
        if self.homography is None and self.captured_for_manual is not None and len(self.clicked_pts) == 4:
            from src.calibration.unified_calibrator import UnifiedCalibrator

            H, center, roi_r, mmpp = UnifiedCalibrator._homography_and_metrics(
                np.float32(self.clicked_pts),
                roi_size=self.roi_size[0],
                board_diameter_mm=self.calibrator.board_diameter_mm
            )

            self.homography = H
            self.center_px = center
            self.roi_board_radius = roi_r
            self.mm_per_px = mmpp

            logger.info(f"[Manual] ✓ H computed from 4 corners")

            # Callback
            if self.on_calibration_complete:
                self.on_calibration_complete(H, center, roi_r, mmpp)

        # Check if we have a homography to save
        if self.homography is None:
            logger.warning("No homography yet. Use ArUco-Quad (a) or Manual (m) first.")
            return False

        logger.info("[Calibration] ✓ Calibration complete, ready to save")
        return True

    def _on_mouse(self, event, x: int, y: int, flags, param):
        """Mouse callback for manual calibration."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.captured_for_manual is not None and len(self.clicked_pts) < 4:
                self.clicked_pts.append((x, y))
                logger.info(f"[Manual] Corner {len(self.clicked_pts)}/4: ({x}, {y})")

    def get_calibration_result(self) -> Optional[dict]:
        """
        Get calibration result.

        Returns:
            Dict with homography, center, radius, mm_per_px or None
        """
        if self.homography is None:
            return None

        return {
            'homography': self.homography,
            'center_px': self.center_px,
            'roi_board_radius': self.roi_board_radius,
            'mm_per_px': self.mm_per_px
        }
