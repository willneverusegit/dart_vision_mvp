"""
Visualization Helper - Simplified create_visualization using modular renderers.
"""

import cv2
import math
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

    # Board calibration mode (new)
    board_calibration_mode: bool = False,

    # Game mode (on/off)
    game_mode: bool = True,

    # CLAHE flag
    use_clahe: bool = False
) -> np.ndarray:
    """
    Refactored visualization creation using modular renderers.

    Replaces the monolithic 200+ line create_visualization method.
    """
    from src.ui.overlay_renderer import OVERLAY_FULL, OVERLAY_ALIGN, OVERLAY_MIN

    # ===== MAIN PANEL =====
    # Note: hud_metrics (brightness/focus/edge) are NOT computed here
    # They are only shown in ROI calibration mode (via calibration_ui_manager)
    hud_metrics = None

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
    # Only show overlay when game mode is ON (unless in board calibration mode)
    effective_overlay_mode = overlay_mode if (game_mode or board_calibration_mode) else OVERLAY_MIN
    disp_roi = overlay_renderer.render_overlay_rings(
        disp_roi,
        calibration,
        effective_overlay_mode,
        board_mapper,
        board_cfg,
        board_calibration_mode
    )

    # ===== HEATMAPS =====
    # Note: Heatmaps disabled by default to keep ROI clean
    # Can be re-enabled via separate analytics module later
    if overlay_mode == OVERLAY_ALIGN and heatmap_enabled:
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

    motion_mask_ratio: Optional[float] = None
    if fg_mask is not None:
        mask = fg_mask
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        total = mask.size
        if total:
            motion_mask_ratio = float(np.count_nonzero(mask)) / float(total)

    candidate_progress: Optional[float] = None
    cooldown_count = 0
    if dart_detector is not None:
        config = getattr(dart_detector, "config", None)
        confirm_frames = getattr(config, "confirmation_frames", None)
        confirmation_count = getattr(dart_detector, "confirmation_count", None)
        current_candidate = getattr(dart_detector, "current_candidate", None)
        if (
            current_candidate is not None
            and isinstance(confirm_frames, (int, float))
            and confirm_frames > 0
            and isinstance(confirmation_count, (int, float))
        ):
            candidate_progress = max(
                0.0,
                min(1.0, float(confirmation_count) / float(confirm_frames))
            )
        cooldown_regions = getattr(dart_detector, "cooldown_regions", None)
        if isinstance(cooldown_regions, list):
            cooldown_count = len(cooldown_regions)

    board_offset_px: Optional[float] = None
    board_offset_pct: Optional[float] = None
    if calibration is not None:
        roi_cx, roi_cy = overlay_renderer.roi_center
        dx = float(calibration.cx) - float(roi_cx)
        dy = float(calibration.cy) - float(roi_cy)
        board_offset_px = math.hypot(dx, dy)
        if getattr(calibration, "r_outer_double_px", 0):
            board_offset_pct = board_offset_px / float(calibration.r_outer_double_px)

    sidebar_selection = overlay_renderer.prepare_sidebar_cards(
        overlay_mode=overlay_mode,
        paused=paused,
        show_motion=show_motion,
        show_debug=show_debug,
        game=game,
        last_msg=last_msg,
        fps_stats=fps_stats,
        total_darts=total_darts,
        dart_config=dart_config,
        current_preset=current_preset,
        motion_detected=motion_detected,
        motion_mask_ratio=motion_mask_ratio,
        candidate_progress=candidate_progress,
        cooldown_count=cooldown_count,
        board_offset_px=board_offset_px,
        board_offset_pct=board_offset_pct,
    )

    # ===== BOARD STATUS CHIPS (for sidebar) =====
    board_status_chips = overlay_renderer.build_board_status_chips(
        overlay_mode=effective_overlay_mode,
        calibration=calibration,
        unified_calibration=unified_calibration,
        current_preset=current_preset,
        align_auto=align_auto,
        board_calibration_mode=board_calibration_mode
    )

    # ===== OVERLAY STATUS (minimal mode badge in ROI) =====
    disp_roi = overlay_renderer.render_overlay_status(
        disp_roi,
        overlay_mode,
        align_auto,
        board_calibration_mode
    )

    # ===== HELP OVERLAY =====
    if show_help:
        hud_renderer.draw_help_overlay(disp_roi)

    # ===== COMPOSE CANVAS =====
    prefer_fast_blend = (
        overlay_mode == OVERLAY_MIN and not overlay_renderer.cards_enabled()
    )

    canvas = overlay_renderer.compose_canvas(
        disp_main,
        disp_roi,
        paused=paused,
        roi_top_cards=sidebar_selection.roi_top_cards,
        roi_bottom_cards=sidebar_selection.roi_bottom_cards,
        fast_blend=prefer_fast_blend,
    )

    should_draw_sidebar = (
        hud_metrics is not None
        or bool(sidebar_selection.cards)
        or bool(sidebar_selection.mode_chips)
        or bool(board_status_chips)
    )
    if should_draw_sidebar:
        canvas = overlay_renderer.draw_metric_sidebar(
            canvas,
            hud_metrics=hud_metrics,
            cards=sidebar_selection.cards,
            mode_chips=sidebar_selection.mode_chips,
            board_status_chips=board_status_chips,
        )

    return canvas
