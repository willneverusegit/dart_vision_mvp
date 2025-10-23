"""
Dart Vision Hotkey Bindings - Configures all hotkeys for DartVisionApp.
"""

import time
import cv2
import logging
import numpy as np
from typing import TYPE_CHECKING

from .hotkey_handler import HotkeyHandler, HotkeyCategory

if TYPE_CHECKING:
    from main import DartVisionApp  # Avoid circular import


logger = logging.getLogger(__name__)


# Key codes (from main.py)
VK_LEFT = 0x250000
VK_UP = 0x260000
VK_RIGHT = 0x270000
VK_DOWN = 0x280000

OCV_LEFT = 2424832
OCV_UP = 2490368
OCV_RIGHT = 2555904
OCV_DOWN = 2621440


class DartVisionHotkeys:
    """
    Configures and manages all hotkeys for DartVisionApp.

    Provides clean separation between hotkey definitions and app logic.
    """

    def __init__(self, app: 'DartVisionApp'):
        """
        Initialize hotkey bindings for app.

        Args:
            app: DartVisionApp instance to bind hotkeys to
        """
        self.app = app
        self.handler = HotkeyHandler()
        self._setup_all_hotkeys()

    def _setup_all_hotkeys(self):
        """Configure all hotkey bindings."""
        self._setup_system_hotkeys()
        self._setup_navigation_hotkeys()
        self._setup_overlay_hotkeys()
        self._setup_game_hotkeys()
        self._setup_debug_hotkeys()
        self._setup_calibration_hotkeys()
        self._setup_motion_tuning_hotkeys()
        self._setup_preset_hotkeys()
        self._setup_heatmap_hotkeys()

    # ========== SYSTEM ==========
    def _setup_system_hotkeys(self):
        """System-level hotkeys (quit, screenshot, help)."""
        self.handler.register(
            ord('q'),
            lambda: setattr(self.app, 'running', False),
            "Quit application",
            HotkeyCategory.SYSTEM
        )

        self.handler.register(
            ord('s'),
            self._screenshot,
            "Save screenshot",
            HotkeyCategory.SYSTEM
        )

        self.handler.register(
            ord('?'),
            lambda: self._toggle_attr('show_help', "HELP overlay"),
            "Toggle help overlay",
            HotkeyCategory.SYSTEM
        )

    def _screenshot(self):
        """Save screenshot with timestamp."""
        if hasattr(self.app, '_last_disp') and self.app._last_disp is not None:
            fn = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fn, self.app._last_disp)
            logger.info(f"Saved {fn}")

    # ========== NAVIGATION ==========
    def _setup_navigation_hotkeys(self):
        """Arrow key navigation for overlay adjustment."""
        # LEFT - Rotate CCW
        for key in (VK_LEFT, OCV_LEFT):
            self.handler.register(
                key,
                lambda: self._adjust_overlay_rotation(-0.5),
                "Rotate overlay CCW",
                HotkeyCategory.NAVIGATION
            )

        # RIGHT - Rotate CW
        for key in (VK_RIGHT, OCV_RIGHT):
            self.handler.register(
                key,
                lambda: self._adjust_overlay_rotation(+0.5),
                "Rotate overlay CW",
                HotkeyCategory.NAVIGATION
            )

        # UP - Scale up
        for key in (VK_UP, OCV_UP):
            self.handler.register(
                key,
                lambda: self._adjust_overlay_scale(1.01),
                "Scale overlay up (+1%)",
                HotkeyCategory.NAVIGATION
            )

        # DOWN - Scale down
        for key in (VK_DOWN, OCV_DOWN):
            self.handler.register(
                key,
                lambda: self._adjust_overlay_scale(1.0 / 1.01),
                "Scale overlay down (-1%)",
                HotkeyCategory.NAVIGATION
            )

        # JKLI - Fine center adjustment
        self.handler.register(ord('j'), lambda: self._adjust_center(-1, 0), "Move overlay left", HotkeyCategory.NAVIGATION)
        self.handler.register(ord('l'), lambda: self._adjust_center(+1, 0), "Move overlay right", HotkeyCategory.NAVIGATION)
        self.handler.register(ord('i'), lambda: self._adjust_center(0, -1), "Move overlay up", HotkeyCategory.NAVIGATION)
        self.handler.register(ord('k'), lambda: self._adjust_center(0, +1), "Move overlay down", HotkeyCategory.NAVIGATION)

    def _adjust_overlay_rotation(self, delta: float):
        """Adjust overlay rotation."""
        self.app.overlay_rot_deg += delta
        self.app._sync_mapper()
        if self.app.board_mapper:
            self.app.board_mapper.calib.rotation_deg = float(self.app.overlay_rot_deg)
        logger.info(f"[OVERLAY] rot={self.app.overlay_rot_deg:.2f} deg")

    def _adjust_overlay_scale(self, factor: float):
        """Adjust overlay scale."""
        self.app.overlay_scale = np.clip(self.app.overlay_scale * factor, 0.80, 1.20)
        self.app._sync_mapper()
        if self.app.board_mapper:
            from main import ROI_SIZE
            self.app.board_mapper.calib.r_outer_double_px = (
                float(self.app.roi_board_radius) * float(self.app.overlay_scale)
            )
        logger.info(f"[OVERLAY] scale={self.app.overlay_scale:.4f}")

    def _adjust_center(self, dx: float, dy: float):
        """Adjust overlay center."""
        self.app.overlay_center_dx += dx
        self.app.overlay_center_dy += dy
        self.app._sync_mapper()
        logger.info(f"[OVERLAY] center=({self.app.overlay_center_dx:+.1f}, {self.app.overlay_center_dy:+.1f})")

    # ========== OVERLAY ==========
    def _setup_overlay_hotkeys(self):
        """Overlay mode and adjustment hotkeys."""
        self.handler.register(
            ord('o'),
            self._cycle_overlay_mode,
            "Cycle overlay mode (MIN/RINGS/FULL/ALIGN)",
            HotkeyCategory.OVERLAY
        )

        self.handler.register(
            ord('R'),
            self._reset_overlay,
            "Reset overlay center/scale",
            HotkeyCategory.OVERLAY
        )

        self.handler.register(
            ord('X'),
            self._save_overlay,
            "Save overlay offsets",
            HotkeyCategory.OVERLAY
        )

    def _cycle_overlay_mode(self):
        """Cycle through overlay modes."""
        from src.ui.overlay_renderer import OVERLAY_MIN, OVERLAY_RINGS, OVERLAY_FULL, OVERLAY_ALIGN
        self.app.overlay_mode = (self.app.overlay_mode + 1) % 4
        modes = {OVERLAY_MIN: "MIN", OVERLAY_RINGS: "RINGS", OVERLAY_FULL: "FULL", OVERLAY_ALIGN: "ALIGN"}
        logger.info(f"[OVERLAY] mode -> {modes[self.app.overlay_mode]}")

    def _reset_overlay(self):
        """Reset overlay to defaults."""
        from main import ROI_CENTER
        self.app.overlay_center_dx = 0.0
        self.app.overlay_center_dy = 0.0
        self.app.overlay_scale = 1.0
        if self.app.board_mapper:
            self.app.board_mapper.calib.cx = float(ROI_CENTER[0])
            self.app.board_mapper.calib.cy = float(ROI_CENTER[1])
            self.app.board_mapper.calib.r_outer_double_px = float(self.app.roi_board_radius)
        logger.info("[OVERLAY] reset center/scale")

    def _save_overlay(self):
        """Save overlay adjustments to calibration file."""
        self.app._save_calibration_unified()
        logger.info("[OVERLAY] saved adjustments")

    # ========== GAME ==========
    def _setup_game_hotkeys(self):
        """Game control hotkeys."""
        self.handler.register(
            ord('g'),
            self._reset_game,
            "Reset game",
            HotkeyCategory.GAME
        )

        self.handler.register(
            ord('h'),
            self._switch_game_mode,
            "Switch game mode (ATC/301)",
            HotkeyCategory.GAME
        )

        self.handler.register(
            ord('r'),
            self._clear_darts,
            "Clear dart impacts",
            HotkeyCategory.GAME
        )

    def _reset_game(self):
        """Reset current game."""
        self.app.game.reset()
        self.app.last_msg = ""
        logger.info(f"[GAME] Reset {self.app.game.mode}")

    def _switch_game_mode(self):
        """Switch between game modes."""
        from src.game.game import GameMode
        new_mode = GameMode._301 if self.app.game.mode == GameMode.ATC else GameMode.ATC
        self.app.game.switch_mode(new_mode)
        self.app.last_msg = ""
        logger.info(f"[GAME] Switch to {self.app.game.mode}")

    def _clear_darts(self):
        """Clear all detected dart impacts."""
        self.app.dart.clear_impacts()
        self.app.total_darts = 0
        logger.info("[GAME] Cleared dart impacts")

    # ========== DEBUG ==========
    def _setup_debug_hotkeys(self):
        """Debug toggle hotkeys."""
        self.handler.register(
            ord('p'),
            lambda: self._toggle_attr('paused', "Pause"),
            "Pause/unpause",
            HotkeyCategory.DEBUG
        )

        self.handler.register(
            ord('d'),
            lambda: self._toggle_attr('show_debug', "Debug overlay"),
            "Toggle debug info",
            HotkeyCategory.DEBUG
        )

        self.handler.register(
            ord('m'),
            lambda: self._toggle_attr('show_motion', "Motion overlay"),
            "Toggle motion overlay",
            HotkeyCategory.DEBUG
        )

        self.handler.register(
            ord('M'),
            lambda: self._toggle_attr('show_mask', "Mask overlay"),
            "Toggle mask overlay",
            HotkeyCategory.DEBUG
        )

        self.handler.register(
            ord('V'),
            self._toggle_mask_debug,
            "Toggle processed mask debug window",
            HotkeyCategory.DEBUG
        )

    def _toggle_attr(self, attr: str, name: str):
        """Toggle a boolean attribute and log."""
        current = getattr(self.app, attr, False)
        setattr(self.app, attr, not current)
        logger.info(f"{name}: {'ON' if not current else 'OFF'}")

    def _toggle_mask_debug(self):
        """Toggle mask debug window."""
        self.app.show_mask_debug = not getattr(self.app, 'show_mask_debug', False)
        if not self.app.show_mask_debug:
            try:
                cv2.destroyWindow("MotionMask (processed)")
            except Exception:
                pass
        logger.info(f"Processed MotionMask window: {'ON' if self.app.show_mask_debug else 'OFF'}")

    # ========== CALIBRATION ==========
    def _setup_calibration_hotkeys(self):
        """Calibration and Hough alignment hotkeys."""
        self.handler.register(
            ord('c'),
            lambda: self.app._recalibrate_and_apply(),
            "Recalibrate ROI",
            HotkeyCategory.CALIBRATION
        )

        self.handler.register(
            ord('C'),
            self._toggle_board_calibration,
            "Toggle board overlay calibration (colored dartboard)",
            HotkeyCategory.CALIBRATION
        )

        self.handler.register(
            ord('t'),
            self._hough_once,
            "Hough alignment (once)",
            HotkeyCategory.CALIBRATION
        )

        self.handler.register(
            ord('T'),
            self._hough_once,  # Same as 't'
            "Hough alignment (once)",
            HotkeyCategory.CALIBRATION
        )

        self.handler.register(
            ord('z'),
            self._toggle_auto_align,
            "Toggle auto-align",
            HotkeyCategory.CALIBRATION
        )

    def _hough_once(self):
        """Run Hough ring detection once."""
        if not hasattr(self.app, '_last_roi_frame') or self.app._last_roi_frame is None:
            logger.info("[HoughRings] No ROI frame available")
            return

        res = self.app._hough_refine_rings(self.app._last_roi_frame)
        if res is None:
            logger.info("[HoughRings] no circles detected in current ROI")
            return

        # Update calibration (simplified from original code)
        cx, cy, r_out = res
        if self.app.uc is None:
            logger.warning("[HoughRings] Unified calib not present in memory.")
            return

        base_cx, base_cy = self.app.uc.metrics.center_px
        self.app.uc.overlay_adjust.center_dx_px = float(cx) - float(base_cx)
        self.app.uc.overlay_adjust.center_dy_px = float(cy) - float(base_cy)
        self.app.uc.overlay_adjust.r_outer_double_px = float(r_out)

        # Sync
        from src.calibration.calib_io import compute_effective_H, save_unified_calibration
        self.app.homography_eff = compute_effective_H(self.app.uc)
        if hasattr(self.app, "_sync_mapper_from_unified"):
            self.app._sync_mapper_from_unified()
            self.app._roi_annulus_mask = None
            self.app._ensure_roi_annulus_mask()
            if self.app.dart is not None and self.app.board_mapper is not None:
                self.app.dart.config.cal_cx = float(self.app.board_mapper.calib.cx)
                self.app.dart.config.cal_cy = float(self.app.board_mapper.calib.cy)

        save_unified_calibration(self.app.calib_path, self.app.uc)
        logger.info(f"[HoughRings] cx={cx:.1f} cy={cy:.1f} rOD={r_out:.1f}")

    def _toggle_auto_align(self):
        """Toggle auto-alignment mode."""
        self.app.align_auto = not self.app.align_auto
        logger.info(f"[ALIGN] auto={'ON' if self.app.align_auto else 'OFF'} (mode must be ALIGN to run)")

    def _toggle_board_calibration(self):
        """Toggle board overlay calibration mode (colored dartboard)."""
        from src.ui.overlay_renderer import OVERLAY_FULL
        self.app.board_calibration_mode = not self.app.board_calibration_mode

        # Auto-switch to FULL mode when enabling calibration
        if self.app.board_calibration_mode and self.app.overlay_mode != OVERLAY_FULL:
            self.app.overlay_mode = OVERLAY_FULL
            logger.info("[OVERLAY] Switched to FULL mode for board calibration")

        status = "ON" if self.app.board_calibration_mode else "OFF"
        logger.info(f"[BOARD CAL] Colored dartboard overlay: {status}")
        if self.app.board_calibration_mode:
            logger.info("[BOARD CAL] Use arrow keys to adjust rotation/scale, j/k/l/i for center, X to save")

    # ========== MOTION TUNING ==========
    def _setup_motion_tuning_hotkeys(self):
        """Live motion parameter tuning hotkeys."""
        # Otsu bias
        self.handler.register(ord('b'), lambda: self._adjust_motion_param('motion_otsu_bias', -1, -32, 64, "Otsu bias"), "Decrease Otsu bias", HotkeyCategory.MOTION_TUNING)
        self.handler.register(ord('B'), lambda: self._adjust_motion_param('motion_otsu_bias', +1, -32, 64, "Otsu bias"), "Increase Otsu bias", HotkeyCategory.MOTION_TUNING)

        # Morph open
        self.handler.register(ord('f'), lambda: self._adjust_morph('morph_open_ksize', -2, "Morph OPEN"), "Decrease morph open", HotkeyCategory.MOTION_TUNING)
        self.handler.register(ord('F'), lambda: self._adjust_morph('morph_open_ksize', +2, "Morph OPEN"), "Increase morph open", HotkeyCategory.MOTION_TUNING)

        # Morph close
        self.handler.register(ord('n'), lambda: self._adjust_morph('morph_close_ksize', -2, "Morph CLOSE"), "Decrease morph close", HotkeyCategory.MOTION_TUNING)
        self.handler.register(ord('N'), lambda: self._adjust_morph('morph_close_ksize', +2, "Morph CLOSE"), "Increase morph close", HotkeyCategory.MOTION_TUNING)

        # Min white fraction
        self.handler.register(ord('w'), lambda: self._adjust_white_frac(-0.002, "minWhite"), "Decrease min white fraction", HotkeyCategory.MOTION_TUNING)
        self.handler.register(ord('W'), lambda: self._adjust_white_frac(+0.002, "minWhite"), "Increase min white fraction", HotkeyCategory.MOTION_TUNING)

    def _adjust_motion_param(self, param: str, delta: int, min_val: int, max_val: int, name: str):
        """Adjust integer motion parameter."""
        if self.app.dart is None:
            return
        current = int(getattr(self.app.dart.config, param, 0))
        new_val = np.clip(current + delta, min_val, max_val)
        setattr(self.app.dart.config, param, int(new_val))
        logger.info(f"[Tune] {name} = {new_val:+d}")

    def _adjust_morph(self, param: str, delta: int, name: str):
        """Adjust morphology kernel size (must be odd)."""
        if self.app.dart is None:
            return
        current = int(getattr(self.app.dart.config, param, 3))
        new_val = max(1, current + delta)
        if new_val % 2 == 0:
            new_val += 1 if delta > 0 else -1
        new_val = np.clip(new_val, 1, 31)
        setattr(self.app.dart.config, param, int(new_val))
        logger.info(f"[Tune] {name} = {new_val}")

    def _adjust_white_frac(self, delta: float, name: str):
        """Adjust min white fraction."""
        if self.app.dart is None:
            return
        current = float(getattr(self.app.dart.config, 'motion_min_white_frac', 0.02))
        new_val = np.clip(current + delta, 0.0, 0.20)
        self.app.dart.config.motion_min_white_frac = round(new_val, 4)
        logger.info(f"[Tune] {name} = {new_val * 100:.2f}%")

    # ========== PRESETS ==========
    def _setup_preset_hotkeys(self):
        """Detector preset hotkeys."""
        self.handler.register(
            ord('1'),
            lambda: self._apply_preset("aggressive"),
            "Apply aggressive preset",
            HotkeyCategory.PRESETS
        )

        self.handler.register(
            ord('2'),
            lambda: self._apply_preset("balanced"),
            "Apply balanced preset",
            HotkeyCategory.PRESETS
        )

        self.handler.register(
            ord('3'),
            lambda: self._apply_preset("stable"),
            "Apply stable preset",
            HotkeyCategory.PRESETS
        )

    def _apply_preset(self, preset_name: str):
        """Apply detector preset."""
        from src.vision.dart_impact_detector import apply_detector_preset
        self.app.dart.config = apply_detector_preset(self.app.dart.config, preset_name)
        self.app.current_preset = preset_name
        logger.info(f"[PRESET] detector -> {preset_name}")

    # ========== HEATMAP ==========
    def _setup_heatmap_hotkeys(self):
        """Heatmap toggle hotkeys."""
        self.handler.register(
            ord('H'),
            lambda: self._toggle_attr('heatmap_enabled', "HEATMAP image-space overlay"),
            "Toggle cartesian heatmap",
            HotkeyCategory.HEATMAP
        )

        self.handler.register(
            ord('P'),
            lambda: self._toggle_attr('polar_enabled', "HEATMAP polar panel"),
            "Toggle polar heatmap",
            HotkeyCategory.HEATMAP
        )

    def handle_key(self, key: int) -> bool:
        """
        Handle a key press.

        Args:
            key: Key code from cv2.waitKeyEx()

        Returns:
            True if key was handled, False otherwise
        """
        return self.handler.handle(key)

    def print_help(self):
        """Print all hotkeys to console."""
        self.handler.print_help()
