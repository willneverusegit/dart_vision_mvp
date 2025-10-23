"""
Visualization Helper - Simplified create_visualization using modular renderers.
"""

import cv2
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui import HUDRenderer, OverlayRenderer
    from src.board import BoardConfig, BoardMapper, Calibration
    from src.overlay.heatmap import HeatmapAccumulator
    from src.analytics.polar_heatmap import PolarHeatmap
    from src.game.game import DemoGame
    from src.vision import DartImpactDetector
    from src.capture import FPSCounter


def create_visualization_refactored(
    # Renderers
    hud_renderer: 'HUDRenderer',
    overlay_renderer: 'OverlayRenderer',

    # Input frames
    frame: np.ndarray,
    roi_frame: np.ndarray,
    motion_detected: bool,
    fg_mask: Optional[np.ndarray],

    # State
    overlay_mode: int,
    show_debug: bool,
    show_motion: bool,
    show_mask: bool,
    show_help: bool,
    annulus_mask: Optional[np.ndarray],

    # Components
    dart_detector: 'DartImpactDetector',
    board_mapper: Optional['BoardMapper'],
    board_cfg: Optional['BoardConfig'],
    calibration: 'Calibration',
    game: 'DemoGame',

    # Optional components
    heatmap_accumulator: Optional['HeatmapAccumulator'] = None,
    polar_heatmap: Optional['PolarHeatmap'] = None,
    heatmap_enabled: bool = False,
    polar_enabled: bool = False,

    # HUD data
    fps_counter: Optional['FPSCounter'] = None,
    total_darts: int = 0,
    last_msg: str = "",
    current_preset: str = "balanced",
    align_auto: bool = False,
    unified_calibration: Optional[object] = None,

    # System state
    paused: bool = False,

    # CLAHE flag
    use_clahe: bool = False
) -> np.ndarray:
    """
    Refactored visualization creation using modular renderers.

    Replaces the monolithic 200+ line create_visualization method.
    """
    from src.ui.overlay_renderer import OVERLAY_FULL, OVERLAY_ALIGN

    # ===== MAIN PANEL =====
    hud_metrics = None
    if overlay_mode == OVERLAY_FULL:
        gray_main = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_main = clahe.apply(gray_main)
        hud_metrics = hud_renderer.compute_metrics(gray_main)

    disp_main = overlay_renderer.render_main_panel(
        frame,
        overlay_mode,
    )

    # ===== ROI PANEL BASE =====
    disp_roi = overlay_renderer.render_roi_base(
        roi_frame,
        motion_detected,
        fg_mask,
        annulus_mask,
        show_motion,
        show_mask
    )

    # ===== BOARD RINGS & OVERLAYS =====
    disp_roi = overlay_renderer.render_overlay_rings(
        disp_roi,
        calibration,
        overlay_mode,
        board_mapper,
        board_cfg
    )

    # ===== HEATMAPS =====
    if overlay_mode == OVERLAY_ALIGN:
        disp_roi = overlay_renderer.render_heatmaps(
            disp_roi,
            heatmap_accumulator,
            polar_heatmap,
            heatmap_enabled,
            polar_enabled
        )

    # ===== IMPACT MARKERS =====
    disp_roi = overlay_renderer.render_impact_markers(disp_roi, dart_detector)

    fps_stats = None
    dart_config = None
    if show_debug and fps_counter is not None:
        stats_obj = fps_counter.get_stats()
        fps_stats = {
            'fps_median': stats_obj.fps_median,
            'frame_time_ms': stats_obj.frame_time_ms
        }
    if show_debug and dart_detector is not None:
        dart_config = dart_detector.config

    # ===== GAME & DEBUG CARDS =====
    disp_roi = overlay_renderer.render_info_cards(
        disp_roi,
        game,
        last_msg,
        show_debug=show_debug,
        fps_stats=fps_stats,
        total_darts=total_darts,
        dart_config=dart_config,
        current_preset=current_preset,
    )

    # ===== OVERLAY STATUS =====
    disp_roi = overlay_renderer.render_overlay_status(
        disp_roi,
        overlay_mode,
        calibration,
        unified_calibration,
        current_preset,
        align_auto
    )

    # ===== HELP OVERLAY =====
    if show_help:
        hud_renderer.draw_help_overlay(disp_roi)

    # ===== COMPOSE CANVAS =====
    canvas = overlay_renderer.compose_canvas(disp_main, disp_roi, paused=paused)

    if overlay_mode == OVERLAY_FULL and hud_metrics is not None:
        canvas = overlay_renderer.draw_metric_sidebar(canvas, hud_metrics)

    return canvas
