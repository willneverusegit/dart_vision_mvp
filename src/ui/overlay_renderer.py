"""
Overlay Renderer - Board overlays and visualization for Dart Vision MVP.

Handles rendering of:
- Board rings and sector labels
- Dart impact markers
- Game HUD
- Different overlay modes (MIN, RINGS, FULL, ALIGN)
- Heatmaps and analytics overlays
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from src.board import BoardMapper, BoardConfig, Calibration
    from src.overlay.heatmap import HeatmapAccumulator
    from src.analytics.polar_heatmap import PolarHeatmap
    from src.game.game import DemoGame, GameMode
    from src.vision import DartImpactDetector


# Overlay modes
OVERLAY_MIN = 0    # Only hits & game HUD (presentation mode)
OVERLAY_RINGS = 1  # Rings + ROI circle
OVERLAY_FULL = 2   # Full: Rings + sectors + technical HUDs
OVERLAY_ALIGN = 3  # Alignment mode with thick circles


class OverlayRenderer:
    """Renders board overlays and game visualization."""

    def __init__(
        self,
        roi_size: Tuple[int, int] = (400, 400),
        main_size: Tuple[int, int] = (800, 600),
        canvas_size: Tuple[int, int] = (1200, 600)
    ):
        """
        Initialize overlay renderer.

        Args:
            roi_size: Size of ROI panel (width, height)
            main_size: Size of main camera view (width, height)
            canvas_size: Size of output canvas (width, height)
        """
        self.roi_size = roi_size
        self.main_size = main_size
        self.canvas_size = canvas_size
        self.roi_center = (roi_size[0] // 2, roi_size[1] // 2)
        self._roi_y_offset = (self.canvas_size[1] - self.roi_size[1]) // 2
        self._main_panel_tl = (0, 0)
        self._main_panel_br = (self.main_size[0] - 1, self.main_size[1] - 1)
        self._roi_panel_tl = (self.main_size[0], self._roi_y_offset)
        self._roi_panel_br = (
            self.main_size[0] + self.roi_size[0] - 1,
            self._roi_y_offset + self.roi_size[1] - 1,
        )
        self._main_slice = np.s_[0:self.main_size[1], 0:self.main_size[0]]
        self._roi_slice = np.s_[
            self._roi_y_offset:self._roi_y_offset + self.roi_size[1],
            self.main_size[0]:self.main_size[0] + self.roi_size[0],
        ]
        self._canvas_background = self._build_canvas_background()

    def _build_canvas_background(self) -> np.ndarray:
        """Precompute the gradient canvas with glass panels."""
        background = self._create_gradient_background()
        self._add_glass_panel(background, self._main_panel_tl, self._main_panel_br)
        self._add_glass_panel(background, self._roi_panel_tl, self._roi_panel_br)
        return background

    def _create_gradient_background(self) -> np.ndarray:
        """Create a subtle horizontal/vertical gradient as base background."""
        height, width = self.canvas_size[1], self.canvas_size[0]
        left_color = np.array([36, 32, 58], dtype=np.float32)
        right_color = np.array([92, 32, 110], dtype=np.float32)
        horizontal = np.linspace(0.0, 1.0, width, dtype=np.float32)[:, None]
        row = left_color + (right_color - left_color) * horizontal
        gradient = np.tile(row[np.newaxis, :, :], (height, 1, 1))
        vertical = np.linspace(1.0, 0.82, height, dtype=np.float32)[:, None, None]
        gradient = gradient * vertical
        return np.clip(gradient, 0, 255).astype(np.uint8)

    def _add_glass_panel(
        self,
        base: np.ndarray,
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int],
        opacity: float = 0.32
    ) -> None:
        """Overlay a blurred translucent panel to emulate glassmorphism."""
        overlay = np.zeros_like(base, dtype=np.uint8)
        cv2.rectangle(overlay, top_left, bottom_right, (255, 255, 255), -1, cv2.LINE_AA)
        blurred = cv2.GaussianBlur(overlay, (0, 0), sigmaX=25, sigmaY=25)
        cv2.addWeighted(blurred, opacity, base, 1.0, 0.0, dst=base)

        # Add a faint highlight band near the top of the panel for depth.
        highlight = np.zeros_like(base, dtype=np.uint8)
        y0, y1 = top_left[1], bottom_right[1]
        highlight_height = max(8, int((y1 - y0) * 0.28))
        highlight_bottom = min(y1, y0 + highlight_height)
        cv2.rectangle(
            highlight,
            top_left,
            (bottom_right[0], highlight_bottom),
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )
        highlight = cv2.GaussianBlur(highlight, (0, 0), sigmaX=13, sigmaY=13)
        cv2.addWeighted(highlight, opacity * 0.55, base, 1.0, 0.0, dst=base)

    def _draw_panel_border(
        self,
        canvas: np.ndarray,
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int]
    ) -> None:
        """Draw a crisp border and accent line for glass panels."""
        cv2.rectangle(canvas, top_left, bottom_right, (235, 235, 245), 1, cv2.LINE_AA)
        if bottom_right[0] - top_left[0] > 30:
            accent_start = (top_left[0] + 12, top_left[1] + 8)
            accent_end = (bottom_right[0] - 12, top_left[1] + 8)
            cv2.line(canvas, accent_start, accent_end, (200, 180, 255), 1, cv2.LINE_AA)

    def render_main_panel(
        self,
        frame: np.ndarray,
        overlay_mode: int,
        hud_metrics: Optional[Tuple[float, float, float]] = None,
        draw_traffic_light_fn: Optional[callable] = None
    ) -> np.ndarray:
        """
        Render main camera view with optional HUD.

        Args:
            frame: Raw camera frame
            overlay_mode: Current overlay mode (OVERLAY_MIN, RINGS, FULL, ALIGN)
            hud_metrics: Optional tuple of (brightness, focus, edge_density)
            draw_traffic_light_fn: Optional function to draw traffic light

        Returns:
            Resized and annotated main panel
        """
        disp_main = cv2.resize(frame, self.main_size)

        # HUD in FULL mode only
        if overlay_mode == OVERLAY_FULL and hud_metrics is not None:
            b_mean, f_var, e_pct = hud_metrics

            # Determine HUD color
            ok_b = 120.0 <= b_mean <= 170.0
            ok_f = f_var >= 800.0
            ok_e = 3.5 <= e_pct <= 15.0
            hud_col = (0, 255, 0) if (ok_b and ok_f and ok_e) else (0, 200, 200)

            # Draw metrics text
            cv2.putText(
                disp_main,
                f"B:{b_mean:.0f} F:{int(f_var)} E:{e_pct:.1f}%",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                hud_col,
                2,
                cv2.LINE_AA
            )

            # Optional traffic light
            if draw_traffic_light_fn is not None:
                draw_traffic_light_fn(disp_main, b_mean, f_var, e_pct, org=(10, 50))

        return disp_main

    def render_roi_base(
        self,
        roi_frame: np.ndarray,
        motion_detected: bool = False,
        fg_mask: Optional[np.ndarray] = None,
        annulus_mask: Optional[np.ndarray] = None,
        show_motion: bool = False,
        show_mask: bool = False
    ) -> np.ndarray:
        """
        Render base ROI panel with optional motion/mask overlays.

        Args:
            roi_frame: Warped ROI frame
            motion_detected: Whether motion was detected
            fg_mask: Foreground mask (from motion detector)
            annulus_mask: ROI annulus mask for filtering
            show_motion: Whether to show motion overlay
            show_mask: Whether to show annulus mask

        Returns:
            ROI panel with base overlays
        """
        disp_roi = cv2.resize(roi_frame, self.roi_size)

        # Motion overlay
        if show_motion and motion_detected and fg_mask is not None:
            fg_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            fg_color = cv2.resize(fg_color, self.roi_size)
            disp_roi = cv2.addWeighted(disp_roi, 0.9, fg_color, 0.6, 0.3)

        # Annulus mask overlay
        if show_mask and annulus_mask is not None:
            disp_roi = cv2.addWeighted(
                disp_roi,
                1.0,
                cv2.cvtColor(annulus_mask, cv2.COLOR_GRAY2BGR),
                0.25,
                0.0
            )

        return disp_roi

    def render_overlay_rings(
        self,
        img: np.ndarray,
        calibration: 'Calibration',
        overlay_mode: int,
        board_mapper: Optional['BoardMapper'] = None,
        board_cfg: Optional['BoardConfig'] = None
    ) -> np.ndarray:
        """
        Render board ring overlays based on mode.

        Args:
            img: Image to draw on
            calibration: Board calibration data
            overlay_mode: Current overlay mode
            board_mapper: Optional BoardMapper for full overlay
            board_cfg: Optional BoardConfig for ALIGN mode

        Returns:
            Image with ring overlays
        """
        from src.board import draw_ring_circles, draw_sector_labels

        cx = int(round(calibration.cx))
        cy = int(round(calibration.cy))
        r_base = int(round(calibration.r_outer_double_px))

        # RINGS mode: Simple outer ring
        if overlay_mode >= OVERLAY_RINGS:
            cv2.circle(img, (cx, cy), max(1, r_base), (0, 255, 0), 2, cv2.LINE_AA)

        # FULL mode: Detailed mapping
        if overlay_mode == OVERLAY_FULL and board_mapper is not None:
            img[:] = draw_ring_circles(img, board_mapper)
            img[:] = draw_sector_labels(img, board_mapper)

            # Center coordinates
            cv2.putText(
                img,
                f"cx:{calibration.cx:.1f} cy:{calibration.cy:.1f}",
                (self.roi_size[0] - 180, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 220, 220),
                1,
                cv2.LINE_AA
            )

        # ALIGN mode: Thick alignment circles
        if overlay_mode == OVERLAY_ALIGN and board_cfg is not None:
            radii = board_cfg.radii

            # Calculate all ring radii
            rings = [
                (int(r_base), (0, 255, 0), 1.5),  # Double outer
                (int(r_base * radii.r_double_inner), (0, 255, 0), 1),  # Double inner
                (int(r_base * radii.r_triple_outer), (0, 200, 255), 0.75),  # Triple outer
                (int(r_base * radii.r_triple_inner), (0, 200, 255), 0.75),  # Triple inner
                (int(r_base * radii.r_bull_outer), (255, 200, 0), 0.5),  # Bull outer
                (int(r_base * radii.r_bull_inner), (255, 200, 0), 0.25),  # Bull inner
            ]

            # Draw concentric circles
            for radius, color, thickness in rings:
                cv2.circle(img, (cx, cy), max(radius, 1), color, int(thickness), cv2.LINE_AA)

            # ALIGN HUD
            cv2.putText(
                img,
                "ALIGN",
                (self.roi_size[0] - 180, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                img,
                f"cx:{cx} cy:{cy} rpx:{r_base}",
                (self.roi_size[0] - 180, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                img,
                "Keys: t=Hough once, z=Auto Hough",
                (self.roi_size[0] - 290, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

        return img

    def render_impact_markers(
        self,
        img: np.ndarray,
        dart_detector: 'DartImpactDetector'
    ) -> np.ndarray:
        """
        Render dart impact markers.

        Args:
            img: Image to draw on
            dart_detector: Dart impact detector with confirmed impacts

        Returns:
            Image with impact markers
        """
        for imp in dart_detector.get_confirmed_impacts():
            # Refined position (yellow)
            px, py = int(imp.position[0]), int(imp.position[1])
            cv2.circle(img, (px, py), 2, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (px, py), 1, (0, 255, 255), -1, cv2.LINE_AA)

            # Optional: raw position (white) for debugging
            if hasattr(imp, "raw_position") and imp.raw_position is not None:
                rx, ry = int(imp.raw_position[0]), int(imp.raw_position[1])
                cv2.circle(img, (rx, ry), 2, (255, 255, 255), -1, cv2.LINE_AA)

            # Label
            if hasattr(imp, "score_label") and imp.score_label is not None:
                cv2.putText(
                    img,
                    imp.score_label,
                    (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        return img

    def render_game_hud(
        self,
        img: np.ndarray,
        game: 'DemoGame',
        last_msg: str = ""
    ) -> np.ndarray:
        """
        Render game HUD (bottom-left of ROI).

        Args:
            img: Image to draw on
            game: Game instance
            last_msg: Last throw message

        Returns:
            Image with game HUD
        """
        from src.game.game import GameMode

        y0 = self.roi_size[1] - 60

        # Game mode
        mode_txt = "ATC" if game.mode == GameMode.ATC else "301"
        cv2.putText(
            img,
            f"Game: {mode_txt}",
            (10, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        y0 += 28

        # Game status
        if game.mode == GameMode.ATC:
            status_txt = "FINISH" if game.done else f"Target: {game.target}"
        else:
            status_txt = "FINISH" if game.done else f"Score: {game.score}"

        cv2.putText(
            img,
            status_txt,
            (10, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Last throw message
        if last_msg:
            cv2.putText(
                img,
                last_msg,
                (10, self.roi_size[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 220),
                2,
                cv2.LINE_AA
            )

        return img

    def render_debug_hud(
        self,
        img: np.ndarray,
        fps_stats: Optional[dict] = None,
        total_darts: int = 0,
        dart_config: Optional[object] = None
    ) -> np.ndarray:
        """
        Render debug information.

        Args:
            img: Image to draw on
            fps_stats: FPS statistics dict
            total_darts: Total darts detected
            dart_config: Dart detector config for motion tuning display

        Returns:
            Image with debug HUD
        """
        if fps_stats is not None:
            cv2.putText(
                img,
                f"FPS: {fps_stats.get('fps_median', 0):.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                img,
                f"Time: {fps_stats.get('frame_time_ms', 0):.1f}ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.putText(
                img,
                f"Darts: {total_darts}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        # Motion tuning parameters
        if dart_config is not None:
            cv2.putText(
                img,
                f"MotionTune  bias:{int(getattr(dart_config, 'motion_otsu_bias', 0)):+d}  "
                f"open:{int(getattr(dart_config, 'morph_open_ksize', 3))}  "
                f"close:{int(getattr(dart_config, 'morph_close_ksize', 5))}  "
                f"minWhite:{getattr(dart_config, 'motion_min_white_frac', 0.02) * 100:.1f}%",
                (10, self.roi_size[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

        return img

    def render_overlay_status(
        self,
        img: np.ndarray,
        overlay_mode: int,
        calibration: Optional['Calibration'] = None,
        unified_calibration: Optional[object] = None,
        current_preset: str = "balanced",
        align_auto: bool = False
    ) -> np.ndarray:
        """
        Render overlay mode status (top-right of ROI).

        Args:
            img: Image to draw on
            overlay_mode: Current overlay mode
            calibration: Board calibration
            unified_calibration: UnifiedCalibration object for scale calculation
            current_preset: Current detector preset name
            align_auto: Auto-align status (ALIGN mode)

        Returns:
            Image with status overlay
        """
        modes = {
            OVERLAY_MIN: "MIN",
            OVERLAY_RINGS: "RINGS",
            OVERLAY_FULL: "FULL",
            OVERLAY_ALIGN: "ALIGN"
        }

        cv2.putText(
            img,
            f"Overlay: {modes.get(overlay_mode, 'UNKNOWN')}",
            (self.roi_size[0] - 180, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
            cv2.LINE_AA
        )

        # Rotation and scale info
        if calibration is not None:
            if unified_calibration is not None:
                scale = float(calibration.r_outer_double_px) / float(
                    unified_calibration.metrics.roi_board_radius
                )
            else:
                scale = 1.0

            cv2.putText(
                img,
                f"rot:{calibration.rotation_deg:.1f}  scale:{scale:.3f}",
                (self.roi_size[0] - 180, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                2,
                cv2.LINE_AA
            )

        # Preset info
        cv2.putText(
            img,
            f"Preset: {current_preset}",
            (self.roi_size[0] - 180, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
            cv2.LINE_AA
        )

        # Auto-align status in ALIGN mode
        if overlay_mode == OVERLAY_ALIGN:
            cv2.putText(
                img,
                f"auto:{'ON' if align_auto else 'OFF'}",
                (self.roi_size[0] - 180, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                2,
                cv2.LINE_AA
            )

        return img

    def render_heatmaps(
        self,
        img: np.ndarray,
        heatmap_accumulator: Optional['HeatmapAccumulator'] = None,
        polar_heatmap: Optional['PolarHeatmap'] = None,
        heatmap_enabled: bool = False,
        polar_enabled: bool = False
    ) -> np.ndarray:
        """
        Render heatmap overlays.

        Args:
            img: Image to draw on
            heatmap_accumulator: Heatmap accumulator instance
            polar_heatmap: Polar heatmap instance
            heatmap_enabled: Whether heatmap overlay is enabled
            polar_enabled: Whether polar mini-panel is enabled

        Returns:
            Image with heatmap overlays
        """
        # Cartesian heatmap overlay
        if heatmap_accumulator is not None and heatmap_enabled:
            img = heatmap_accumulator.render_overlay(img, roi_mask=None)

        # Polar mini-panel (top-left)
        if polar_heatmap is not None and polar_enabled:
            img = polar_heatmap.overlay_panel(img, pos=(10, 110))

        return img

    def compose_canvas(
        self,
        main_panel: np.ndarray,
        roi_panel: np.ndarray,
        paused: bool = False
    ) -> np.ndarray:
        """
        Compose final canvas with main and ROI panels.

        Args:
            main_panel: Main camera view
            roi_panel: ROI panel
            paused: Whether system is paused

        Returns:
            Final composed canvas
        """
        canvas = self._canvas_background.copy()

        if main_panel.shape[0:2] != (self.main_size[1], self.main_size[0]):
            main_panel = cv2.resize(main_panel, self.main_size)
        if roi_panel.shape[0:2] != (self.roi_size[1], self.roi_size[0]):
            roi_panel = cv2.resize(roi_panel, self.roi_size)

        main_region = canvas[self._main_slice]
        canvas[self._main_slice] = cv2.addWeighted(main_panel, 0.92, main_region, 0.08, 0)

        roi_region = canvas[self._roi_slice]
        canvas[self._roi_slice] = cv2.addWeighted(roi_panel, 0.92, roi_region, 0.08, 0)

        self._draw_panel_border(canvas, self._main_panel_tl, self._main_panel_br)
        self._draw_panel_border(canvas, self._roi_panel_tl, self._roi_panel_br)

        if paused:
            cv2.putText(
                canvas,
                "PAUSED",
                (480, 56),
                cv2.FONT_HERSHEY_DUPLEX,
                1.4,
                (245, 210, 180),
                3,
            )

        return canvas
