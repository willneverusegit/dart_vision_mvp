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

from .typography import Typography, ChipDrawer, CardDrawer
from .hud_metrics import build_metric_chips, summarise_quality

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
        canvas_size: Tuple[int, int] = (1200, 600),
        metrics_sidebar_width: int = 0,
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
        self.sidebar_width = max(0, metrics_sidebar_width)

        width_needed = self.sidebar_width + self.main_size[0] + self.roi_size[0]
        height_needed = max(self.main_size[1], self.roi_size[1])
        canvas_width = max(canvas_size[0], width_needed)
        canvas_height = max(canvas_size[1], height_needed)
        self.canvas_size = (canvas_width, canvas_height)
        self.roi_center = (roi_size[0] // 2, roi_size[1] // 2)
        self._roi_y_offset = (self.canvas_size[1] - self.roi_size[1]) // 2
        self._main_panel_tl = (self.sidebar_width, 0)
        self._main_panel_br = (
            self.sidebar_width + self.main_size[0] - 1,
            self.main_size[1] - 1,
        )
        self._roi_panel_tl = (
            self.sidebar_width + self.main_size[0],
            self._roi_y_offset,
        )
        self._roi_panel_br = (
            self.sidebar_width + self.main_size[0] + self.roi_size[0] - 1,
            self._roi_y_offset + self.roi_size[1] - 1,
        )
        self._main_slice = np.s_[
            0:self.main_size[1],
            self.sidebar_width:self.sidebar_width + self.main_size[0],
        ]
        self._roi_slice = np.s_[
            self._roi_y_offset:self._roi_y_offset + self.roi_size[1],
            self.sidebar_width + self.main_size[0]:
            self.sidebar_width + self.main_size[0] + self.roi_size[0],
        ]
        self._sidebar_slice = None
        self._sidebar_panel_tl: Optional[Tuple[int, int]] = None
        self._sidebar_panel_br: Optional[Tuple[int, int]] = None
        if self.sidebar_width:
            self._sidebar_slice = np.s_[
                0:self.canvas_size[1],
                0:self.sidebar_width,
            ]
            inset = max(10, min(26, self.sidebar_width // 5))
            x_pad = max(6, inset // 2)
            self._sidebar_panel_tl = (x_pad, inset)
            right = max(x_pad + 1, self.sidebar_width - x_pad)
            right = min(self.sidebar_width - 1, right)
            bottom = max(inset + 1, self.canvas_size[1] - inset)
            bottom = min(self.canvas_size[1] - 1, bottom)
            self._sidebar_panel_br = (right, bottom)
        self._canvas_background = self._build_canvas_background()
        self._typography = Typography()
        self._chip_drawer = ChipDrawer(self._typography)
        self._card_drawer = CardDrawer(self._typography)

    def _build_canvas_background(self) -> np.ndarray:
        """Precompute the gradient canvas with glass panels."""
        background = self._create_gradient_background()
        if (
            self._sidebar_panel_tl is not None
            and self._sidebar_panel_br is not None
            and self._sidebar_panel_br[0] - self._sidebar_panel_tl[0] > 8
        ):
            self._add_glass_panel(
                background,
                self._sidebar_panel_tl,
                self._sidebar_panel_br,
                opacity=0.28,
            )
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
    ) -> np.ndarray:
        """
        Render main camera view with optional HUD.

        Args:
            frame: Raw camera frame
            overlay_mode: Current overlay mode (OVERLAY_MIN, RINGS, FULL, ALIGN)

        Returns:
            Resized and annotated main panel
        """
        return cv2.resize(frame, self.main_size)

    def draw_metric_sidebar(
        self,
        canvas: np.ndarray,
        hud_metrics: Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        """Draw compact metric chips in the dedicated sidebar area."""

        if hud_metrics is None or self.sidebar_width <= 0:
            return canvas

        if self._sidebar_panel_tl is not None:
            x = self._sidebar_panel_tl[0] + 4
            y = self._sidebar_panel_tl[1] + 4
        else:
            x = 12
            y = 18

        b_mean, f_var, e_pct = hud_metrics
        metric_chips = build_metric_chips(b_mean, f_var, e_pct)
        if not metric_chips:
            return canvas

        max_y = self.canvas_size[1] - 18
        chip_gap = 10

        for chip in metric_chips:
            _, height = self._chip_drawer.draw_metric_chip(
                canvas,
                origin=(x, y),
                label=chip.label,
                value=chip.value,
                status=chip.status,
                subtitle=chip.subtitle,
                compact=True,
            )
            y += height + chip_gap
            if y >= max_y:
                break

        overall_status = summarise_quality(metric_chips)
        _, summary_height = self._chip_drawer.draw_compact_chip(
            canvas,
            origin=(x, y),
            text=f"Quality {overall_status.upper()}",
            status=overall_status,
        )
        y += summary_height + 12

        if y < max_y:
            radius = 7
            spacing = radius * 2 + 8
            indicator_x = x + radius + 4
            indicator_y = y + radius
            labels = ["B", "F", "E"]
            for idx, chip in enumerate(metric_chips[:3]):
                palette = self._chip_drawer.get_palette(chip.status)
                color = palette.get("indicator", palette["fill"])
                cv2.circle(canvas, (indicator_x, indicator_y), radius, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (indicator_x, indicator_y), radius, palette["border"], 1, cv2.LINE_AA)
                letter = labels[idx]
                text_w, text_h, text_base = self._typography.measure(letter, 16)
                text_x = int(indicator_x - text_w / 2)
                text_y = int(indicator_y + (text_h - text_base) / 2)
                self._typography.draw(
                    canvas,
                    letter,
                    (text_x, text_y),
                    16,
                    palette["text_primary"],
                )
                indicator_x += spacing

        return canvas

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

    def render_info_cards(
        self,
        img: np.ndarray,
        game: 'DemoGame',
        last_msg: str = "",
        show_debug: bool = False,
        fps_stats: Optional[dict] = None,
        total_darts: int = 0,
        dart_config: Optional[object] = None,
        current_preset: str = "balanced",
    ) -> np.ndarray:
        """Render stacked cards summarising game and system state."""

        from src.game.game import GameMode

        if self.roi_size[0] <= 0 or self.roi_size[1] <= 0:
            return img

        pad = 20
        gap = 16
        bottom_margin = 18
        available_width = self.roi_size[0] - pad * 2
        if available_width <= 0:
            return img

        draw_debug = bool(show_debug)
        if draw_debug:
            paired_width = available_width - gap
            if paired_width <= 0:
                draw_debug = False
                game_width = max(available_width, 120)
            else:
                shared_width = max(int(paired_width // 2), 120)
                game_width = shared_width
        else:
            game_width = max(available_width, 120)

        mode_display = "Around the Clock" if game.mode == GameMode.ATC else "301"
        game_rows: List[dict] = [
            {"label": "Mode", "value": mode_display, "status": "accent"}
        ]

        if game.mode == GameMode.ATC:
            target = getattr(game, "target", None)
            if getattr(game, "done", False):
                objective_value = "Finished run"
                objective_status = "good"
            else:
                objective_value = f"Next: {target}" if target is not None else "Next: --"
                objective_status = "accent"
        else:
            score = getattr(game, "score", None)
            if getattr(game, "done", False):
                objective_value = "Checkout"
                objective_status = "good"
            else:
                objective_value = f"{score} left" if score is not None else "--"
                if score is not None and score <= 100:
                    objective_status = "warn"
                else:
                    objective_status = "accent"

        game_rows.append(
            {
                "label": "Objective",
                "value": objective_value,
                "status": objective_status,
            }
        )

        last_label = ""
        last_points = 0
        if hasattr(game, "last") and game.last is not None:
            last_label = getattr(game.last, "label", "") or ""
            last_points = int(getattr(game.last, "points", 0) or 0)

        if last_label:
            last_value = f"{last_label} ({last_points})" if last_points else last_label
        else:
            last_value = "Waiting for hit"

        last_hint: Optional[str] = None
        if last_points and last_label:
            last_hint = f"{last_points} pts"
        elif not draw_debug and total_darts:
            last_hint = f"{total_darts} impacts tracked"

        game_rows.append(
            {
                "label": "Last hit",
                "value": last_value,
                "status": "info",
                "hint": last_hint,
            }
        )

        footer_text = last_msg if last_msg else "Goal: reliable hit detection & scoring."
        game_card = self._card_drawer.prepare_card(game_width, "Game State", game_rows, footer=footer_text)

        base_line = self.roi_size[1] - bottom_margin
        game_y = max(int(base_line - game_card["height"]), 0)
        self._card_drawer.draw_card(img, (pad, game_y), game_card, status="accent")

        if not draw_debug:
            return img

        fps_value = None
        frame_time = None
        if fps_stats is not None:
            fps_value = fps_stats.get("fps_median")
            frame_time = fps_stats.get("frame_time_ms")

        fps_status = "info"
        fps_progress: Optional[float] = None
        if isinstance(fps_value, (int, float)):
            fps_value = float(fps_value)
            fps_progress = max(0.0, min(1.0, fps_value / 30.0))
            if fps_value >= 28.0:
                fps_status = "good"
            elif fps_value >= 22.0:
                fps_status = "warn"
            else:
                fps_status = "bad"

        frame_status = "info"
        if isinstance(frame_time, (int, float)):
            frame_time = float(frame_time)
            if frame_time <= 35.0:
                frame_status = "good"
            elif frame_time <= 45.0:
                frame_status = "warn"
            else:
                frame_status = "bad"

        debug_rows: List[dict] = [
            {
                "label": "Median FPS",
                "value": f"{fps_value:.1f}" if isinstance(fps_value, float) else "--",
                "status": fps_status,
                "progress": fps_progress,
                "hint": "≥30 keeps hits stable" if fps_progress is not None else None,
            },
            {
                "label": "Frame time",
                "value": f"{frame_time:.1f} ms" if isinstance(frame_time, float) else "--",
                "status": frame_status,
            },
            {
                "label": "Impacts logged",
                "value": str(int(total_darts)),
                "status": "info",
                "hint": "Session total" if total_darts else None,
            },
        ]

        if dart_config is not None:
            motion_parts: List[str] = []
            bias = getattr(dart_config, "motion_otsu_bias", None)
            if isinstance(bias, (int, float)):
                motion_parts.append(f"bias {int(round(bias)):+d}")
            open_k = getattr(dart_config, "morph_open_ksize", None)
            if isinstance(open_k, (int, float)):
                motion_parts.append(f"open {int(round(open_k))}")
            close_k = getattr(dart_config, "morph_close_ksize", None)
            if isinstance(close_k, (int, float)):
                motion_parts.append(f"close {int(round(close_k))}")
            min_white = getattr(dart_config, "motion_min_white_frac", None)
            if isinstance(min_white, (int, float)):
                motion_parts.append(f"white {min_white * 100:.0f}%")
            motion_summary = " | ".join(motion_parts) if motion_parts else "default"
            debug_rows.append(
                {
                    "label": "Motion tuning",
                    "value": motion_summary,
                    "status": "info",
                }
            )

        preset_display = current_preset.replace("_", " ").title()
        debug_footer = f"Preset: {preset_display}"

        debug_card = self._card_drawer.prepare_card(game_width, "System Health", debug_rows, footer=debug_footer)
        debug_y = max(int(base_line - debug_card["height"]), 0)
        debug_x = pad + game_width + gap
        self._card_drawer.draw_card(img, (debug_x, debug_y), debug_card, status=fps_status)

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

        mode_value = modes.get(overlay_mode, "UNKNOWN")
        mode_status = "accent"
        mode_subtitle = "HUD"
        if overlay_mode == OVERLAY_MIN:
            mode_subtitle = "CLEAN"
        elif overlay_mode == OVERLAY_RINGS:
            mode_subtitle = "TRACK"
        elif overlay_mode == OVERLAY_ALIGN:
            mode_subtitle = "AUTO" if align_auto else "MANUAL"
            mode_status = "accent" if align_auto else "info"

        chips = [
            {
                "label": "Mode",
                "value": mode_value,
                "status": mode_status,
                "subtitle": mode_subtitle,
            }
        ]

        if calibration is not None:
            rotation = float(getattr(calibration, "rotation_deg", 0.0))
            abs_rot = abs(rotation)
            if abs_rot <= 1.0:
                rot_status, rot_subtitle = "good", "LEVEL"
            elif abs_rot <= 3.0:
                rot_status, rot_subtitle = "warn", "OFFSET"
            else:
                rot_status, rot_subtitle = "bad", "FIX"
            chips.append(
                {
                    "label": "Rotation",
                    "value": f"{rotation:.1f}°",
                    "status": rot_status,
                    "subtitle": rot_subtitle,
                }
            )

            base_radius = float(getattr(calibration, "r_outer_double_px", 0.0))
            metrics = getattr(unified_calibration, "metrics", None) if unified_calibration is not None else None
            ref_radius = float(getattr(metrics, "roi_board_radius", 0.0)) if metrics is not None else 0.0
            if ref_radius > 1e-6:
                scale = base_radius / ref_radius
                scale_delta = abs(scale - 1.0)
                if scale_delta <= 0.03:
                    scale_status, scale_subtitle = "good", "ON-SPEC"
                elif scale_delta <= 0.07:
                    scale_status, scale_subtitle = "warn", "TUNE"
                else:
                    scale_status, scale_subtitle = "bad", "RESCALE"
                scale_value = f"{scale:.3f}"
            else:
                scale_status, scale_subtitle = "info", "NO REF"
                scale_value = "--"
            chips.append(
                {
                    "label": "Scale",
                    "value": scale_value,
                    "status": scale_status,
                    "subtitle": scale_subtitle,
                }
            )

        preset_display = current_preset.replace("_", " ").title()
        chips.append(
            {
                "label": "Preset",
                "value": preset_display,
                "status": "info",
                "subtitle": "DETECTOR",
            }
        )

        anchor_x = self.roi_size[0] - 16
        y = 18
        for chip in chips:
            _, height = self._chip_drawer.draw_metric_chip(
                img,
                origin=(anchor_x, y),
                label=chip["label"],
                value=chip["value"],
                status=chip["status"],
                subtitle=chip.get("subtitle"),
                align="right",
                compact=True,
            )
            y += height + 8

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
        if (
            self._sidebar_panel_tl is not None
            and self._sidebar_panel_br is not None
            and self.sidebar_width > 0
        ):
            self._draw_panel_border(canvas, self._sidebar_panel_tl, self._sidebar_panel_br)

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
