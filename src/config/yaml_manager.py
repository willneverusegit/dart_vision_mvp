"""
YAML Configuration Manager - Single Source of Truth

Centralized YAML loader with atomic write support for safe configuration updates.

Configuration Files (all in main root):
- board.yaml: Board geometry (radii, sectors, angles)
- calibration_unified.yaml: Calibration data (homography, center, rotation, shifts)
- overlay_config.yaml: Overlay styling (colors, text, transparency)

Key Features:
- Atomic writes (temp file + os.replace) to prevent corruption
- Single source of truth for all configurations
- Type-safe access with validation
"""

import os
import yaml
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


# === YAML PATHS (all in main root) ===
ROOT_DIR = Path(__file__).parent.parent.parent  # dart_vision_mvp/

BOARD_YAML = ROOT_DIR / "board.yaml"
CALIBRATION_YAML = ROOT_DIR / "calibration_unified.yaml"
OVERLAY_YAML = ROOT_DIR / "overlay_config.yaml"


# === ATOMIC WRITE HELPER ===

def atomic_write_yaml(file_path: Path, data: Dict[str, Any]) -> None:
    """
    Atomically write YAML data to file using temp file + os.replace.

    This prevents corruption from partial writes or race conditions:
    1. Write to temporary file in same directory
    2. Flush and sync to disk
    3. Atomically replace original file

    Args:
        file_path: Path to target YAML file
        data: Dictionary to write as YAML

    Raises:
        OSError: If write fails
        yaml.YAMLError: If data cannot be serialized
    """
    file_path = Path(file_path)

    # Create temp file in same directory (required for atomic replace)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=file_path.parent,
        prefix=f".{file_path.name}.",
        suffix=".tmp"
    )

    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,  # Preserve insertion order
                indent=2
            )
            f.flush()
            os.fsync(f.fileno())  # Force write to disk

        # Atomic replace (POSIX-compliant)
        os.replace(temp_path, file_path)

    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


# === CENTRAL YAML LOADER ===

class YAMLConfigManager:
    """
    Central manager for all YAML configurations.

    Provides:
    - Lazy loading with caching
    - Atomic writes for updates
    - Single source of truth enforcement
    """

    def __init__(self):
        self._board_config: Optional[Dict[str, Any]] = None
        self._calibration_config: Optional[Dict[str, Any]] = None
        self._overlay_config: Optional[Dict[str, Any]] = None

    # === BOARD CONFIGURATION ===

    def load_board_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load board geometry configuration.

        Contains: radii, sectors, angles
        Source: board.yaml

        Args:
            force_reload: Clear cache and reload from disk

        Returns:
            Board configuration dictionary
        """
        if self._board_config is None or force_reload:
            with open(BOARD_YAML, 'r', encoding='utf-8') as f:
                self._board_config = yaml.safe_load(f)
        return self._board_config

    def save_board_config(self, config: Dict[str, Any]) -> None:
        """
        Save board configuration atomically.

        Args:
            config: Complete board configuration dictionary
        """
        atomic_write_yaml(BOARD_YAML, config)
        self._board_config = config  # Update cache

    # === CALIBRATION CONFIGURATION ===

    def load_calibration_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load calibration data.

        Contains: homography, center, rotation, shifts
        Source: calibration_unified.yaml

        Args:
            force_reload: Clear cache and reload from disk

        Returns:
            Calibration configuration dictionary
        """
        if self._calibration_config is None or force_reload:
            with open(CALIBRATION_YAML, 'r', encoding='utf-8') as f:
                self._calibration_config = yaml.safe_load(f)
        return self._calibration_config

    def save_calibration_config(self, config: Dict[str, Any]) -> None:
        """
        Save calibration configuration atomically.

        Args:
            config: Complete calibration configuration dictionary
        """
        atomic_write_yaml(CALIBRATION_YAML, config)
        self._calibration_config = config  # Update cache

    # === OVERLAY CONFIGURATION ===

    def load_overlay_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load overlay styling configuration.

        Contains: colors, text settings, transparency
        Source: overlay_config.yaml

        Args:
            force_reload: Clear cache and reload from disk

        Returns:
            Overlay configuration dictionary
        """
        if self._overlay_config is None or force_reload:
            if not OVERLAY_YAML.exists():
                # Create default config if missing
                self._overlay_config = self._create_default_overlay_config()
                self.save_overlay_config(self._overlay_config)
            else:
                with open(OVERLAY_YAML, 'r', encoding='utf-8') as f:
                    self._overlay_config = yaml.safe_load(f)
        return self._overlay_config

    def save_overlay_config(self, config: Dict[str, Any]) -> None:
        """
        Save overlay configuration atomically.

        Args:
            config: Complete overlay configuration dictionary
        """
        atomic_write_yaml(OVERLAY_YAML, config)
        self._overlay_config = config  # Update cache

    # === HELPER: UPDATE NESTED VALUES ===

    def update_calibration_overlay_adjust(
        self,
        rotation_deg: Optional[float] = None,
        r_outer_double_px: Optional[float] = None,
        center_dx_px: Optional[float] = None,
        center_dy_px: Optional[float] = None
    ) -> None:
        """
        Update overlay adjustment values in calibration config.

        Convenience method for updating rotation/scale/shift.
        Atomically writes back to calibration_unified.yaml.

        Args:
            rotation_deg: Board rotation in degrees
            r_outer_double_px: Outer double ring radius in pixels
            center_dx_px: X offset from ROI center
            center_dy_px: Y offset from ROI center
        """
        calib = self.load_calibration_config()

        overlay_adjust = calib.setdefault('calibration', {}).setdefault('overlay_adjust', {})

        if rotation_deg is not None:
            overlay_adjust['rotation_deg'] = rotation_deg
        if r_outer_double_px is not None:
            overlay_adjust['r_outer_double_px'] = r_outer_double_px
        if center_dx_px is not None:
            overlay_adjust['center_dx_px'] = center_dx_px
        if center_dy_px is not None:
            overlay_adjust['center_dy_px'] = center_dy_px

        self.save_calibration_config(calib)

    # === DEFAULTS ===

    def _create_default_overlay_config(self) -> Dict[str, Any]:
        """
        Create default overlay configuration.

        Returns:
            Default overlay config with colors, text, transparency
        """
        return {
            'version': 1,

            # Color definitions (BGR format for OpenCV)
            'colors': {
                # Triple ring (alternating red/green per sector)
                'triple_red': [0, 0, 200],
                'triple_green': [0, 180, 0],

                # Double ring (alternating red/green per sector)
                'double_red': [0, 0, 200],
                'double_green': [0, 180, 0],

                # Single fields (alternating dark/light per sector)
                'single_dark': [20, 20, 20],      # Black
                'single_light': [200, 220, 240],  # Cream/Beige

                # Bullseye
                'bull_inner': [0, 0, 220],   # Red (50 points)
                'bull_outer': [0, 200, 0],   # Green (25 points)

                # Text colors
                'text_on_dark': [240, 240, 240],   # Light text
                'text_on_light': [30, 30, 30],     # Dark text

                # Calibration guides
                'guide_crosshair': [0, 255, 255],  # Cyan
                'guide_circles': [100, 100, 255],  # Light red
            },

            # Transparency settings
            'transparency': {
                'dartboard_overlay': 0.40,  # Colored dartboard alpha
                'calibration_mode': 0.45,   # Slightly more visible during calibration
            },

            # Text rendering
            'text': {
                'sector_numbers': {
                    'font': 'FONT_HERSHEY_SIMPLEX',
                    'font_scale': 0.7,
                    'thickness': 2,
                    'outline_thickness': 4,  # thickness + 2
                    'show_in_calibration': True,
                    'show_in_normal': True,
                },
            },

            # Calibration guides
            'calibration_guides': {
                'show_crosshair': True,
                'show_circles': True,
                'crosshair_size': 20,
                'circle_radii': [0.25, 0.5, 0.75, 1.0],  # Fractions of board radius
            },

            # Sector rendering
            'sectors': {
                'num_arc_points': 10,  # Minimum arc points for smooth rendering
                'use_antialiasing': True,
            },
        }


# === GLOBAL INSTANCE ===

# Singleton instance for global access
_config_manager: Optional[YAMLConfigManager] = None


def get_config_manager() -> YAMLConfigManager:
    """
    Get global YAMLConfigManager instance (singleton).

    Returns:
        YAMLConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = YAMLConfigManager()
    return _config_manager
