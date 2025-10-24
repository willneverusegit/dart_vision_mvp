"""
Colored Dartboard Overlay - Renders realistic dartboard with colors.

Provides 1:1 visual representation of scoring zones for calibration:
- Each SECTOR (pie slice) colored individually
- Triple/Double rings: Alternating Red/Green per sector
- Single fields: Alternating Dark/Light (Black/Cream) per sector
- Bullseye: Inner Red (50pts), Outer Green (25pts)
- Centered numbers in single fields

All colors, transparency, and styling loaded from overlay_config.yaml.
"""

import cv2
import numpy as np
import math
from typing import Tuple, Optional, TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from src.board import BoardMapper

from src.config.yaml_manager import get_config_manager

# Standard dartboard sector arrangement (starting at top, clockwise)
# Loaded from board.yaml at runtime, but hardcoded here for convenience
DARTBOARD_SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


# === CONFIGURATION HELPERS ===

def _get_overlay_config() -> Dict[str, Any]:
    """Get overlay configuration from YAML."""
    return get_config_manager().load_overlay_config()


def _get_colors() -> Dict[str, Tuple[int, int, int]]:
    """
    Get color definitions from overlay_config.yaml.

    Returns:
        Dictionary mapping color names to BGR tuples
    """
    config = _get_overlay_config()
    colors_list = config.get('colors', {})

    # Convert list format [B, G, R] to tuple (B, G, R)
    return {
        name: tuple(bgr) for name, bgr in colors_list.items()
    }


def _get_alpha(calibration_mode: bool = False) -> float:
    """
    Get transparency value from overlay_config.yaml.

    Args:
        calibration_mode: If True, use calibration mode transparency

    Returns:
        Alpha value (0.0-1.0)
    """
    config = _get_overlay_config()
    transparency = config.get('transparency', {})

    if calibration_mode:
        return transparency.get('calibration_mode', 0.45)
    else:
        return transparency.get('dartboard_overlay', 0.40)


def draw_filled_sector(
    img: np.ndarray,
    center: Tuple[int, int],
    radius_inner: float,
    radius_outer: float,
    angle_start_deg: float,
    angle_end_deg: float,
    color: Tuple[int, int, int],
    alpha: float
) -> np.ndarray:
    """
    Draw a filled sector (pie slice) between two radii.

    Args:
        img: Image to draw on
        center: Center point (cx, cy)
        radius_inner: Inner radius
        radius_outer: Outer radius
        angle_start_deg: Start angle in degrees (0° = right, counter-clockwise in OpenCV)
        angle_end_deg: End angle in degrees
        color: BGR color tuple
        alpha: Transparency (0.0 = transparent, 1.0 = opaque)

    Returns:
        Image with sector drawn
    """
    # Create overlay for alpha blending
    overlay = img.copy()

    # Create points for filled polygon
    # We need to draw the sector as a polygon with many points

    # Number of points along the arc (more = smoother)
    num_points = max(10, int(abs(angle_end_deg - angle_start_deg)))

    # Outer arc points
    outer_points = []
    for i in range(num_points + 1):
        angle = angle_start_deg + (angle_end_deg - angle_start_deg) * i / num_points
        rad = math.radians(angle)
        x = int(center[0] + radius_outer * math.cos(rad))
        y = int(center[1] + radius_outer * math.sin(rad))
        outer_points.append([x, y])

    # Inner arc points (reversed)
    inner_points = []
    for i in range(num_points + 1):
        angle = angle_end_deg - (angle_end_deg - angle_start_deg) * i / num_points
        rad = math.radians(angle)
        x = int(center[0] + radius_inner * math.cos(rad))
        y = int(center[1] + radius_inner * math.sin(rad))
        inner_points.append([x, y])

    # Combine to form closed polygon
    points = np.array(outer_points + inner_points, dtype=np.int32)

    # Draw filled polygon
    cv2.fillPoly(overlay, [points], color, cv2.LINE_AA)

    # Blend with original image
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    return img


def draw_sector_number(
    img: np.ndarray,
    center: Tuple[int, int],
    radius_inner: float,
    radius_outer: float,
    angle_mid_deg: float,
    number: int,
    is_dark_background: bool,
    colors: Dict[str, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Draw sector number centered in single field.

    Args:
        img: Image to draw on
        center: Board center (cx, cy)
        radius_inner: Inner radius of single field (triple outer)
        radius_outer: Outer radius of single field (double inner)
        angle_mid_deg: Middle angle of sector (OpenCV convention: 0° = right)
        number: Sector number to draw
        is_dark_background: True if background is dark (use light text)
        colors: Color dictionary from config

    Returns:
        Image with number drawn
    """
    # Calculate position: midpoint between inner and outer radius
    radius_text = (radius_inner + radius_outer) / 2.0

    # Convert angle to radians
    angle_rad = math.radians(angle_mid_deg)

    # Calculate text position
    text_x = int(center[0] + radius_text * math.cos(angle_rad))
    text_y = int(center[1] + radius_text * math.sin(angle_rad))

    # Choose text color based on background
    text_color = colors["text_on_dark"] if is_dark_background else colors["text_on_light"]

    # Get text settings from config
    config = _get_overlay_config()
    text_cfg = config.get('text', {}).get('sector_numbers', {})

    # Measure text size for centering
    text = str(number)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = text_cfg.get('font_scale', 0.7)
    thickness = text_cfg.get('thickness', 2)
    outline_thickness = text_cfg.get('outline_thickness', 4)

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Center text
    text_x -= text_width // 2
    text_y += text_height // 2

    # Draw text with outline for better visibility
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)  # Outline
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)  # Text

    return img


def draw_colored_dartboard_overlay(
    img: np.ndarray,
    board_mapper: 'BoardMapper',
    alpha: Optional[float] = None,
    show_numbers: bool = True,
    calibration_mode: bool = False
) -> np.ndarray:
    """
    Draw full colored dartboard overlay matching real dartboard colors.

    Each SECTOR (pie slice) is colored individually:
    - Sector 20 at top (12 o'clock) - starts dark/black
    - Alternating dark/light for singles
    - Alternating red/green for triples/doubles

    All colors and settings loaded from overlay_config.yaml.

    Args:
        img: Image to draw on (ROI frame)
        board_mapper: BoardMapper with calibration data
        alpha: Overall transparency (0.0-1.0). If None, uses config default.
        show_numbers: Whether to draw sector numbers
        calibration_mode: If True, use calibration mode transparency

    Returns:
        Image with colored overlay
    """
    # Load configuration
    colors = _get_colors()
    if alpha is None:
        alpha = _get_alpha(calibration_mode)

    calib = board_mapper.calib
    cfg = board_mapper.cfg

    # Extract calibration parameters
    cx = int(round(calib.cx))
    cy = int(round(calib.cy))
    r_base = calib.r_outer_double_px  # Outer double ring radius

    # Calculate all ring radii (relative to outer double)
    r_double_outer = r_base
    r_double_inner = r_base * cfg.radii.r_double_inner
    r_triple_outer = r_base * cfg.radii.r_triple_outer
    r_triple_inner = r_base * cfg.radii.r_triple_inner
    r_bull_outer = r_base * cfg.radii.r_bull_outer
    r_bull_inner = r_base * cfg.radii.r_bull_inner

    # Angle per sector (360° / 20 sectors = 18°)
    angle_per_sector = 18.0

    # Starting angle: Sector 20 at top (12 o'clock)
    # IMPORTANT: OpenCV has inverted Y-axis (Y grows downward)!
    # - 0° = right (3 o'clock)
    # - 90° = BOTTOM (6 o'clock) - because Y-axis is inverted!
    # - 180° = left (9 o'clock)
    # - 270° = TOP (12 o'clock) - because Y-axis is inverted!
    # Sector boundaries at ±9° from sector center

    # Draw each sector
    for sector_idx, sector_num in enumerate(DARTBOARD_SECTORS):
        # Calculate sector angles (OpenCV convention with inverted Y-axis)
        # Sector 0 (20) centered at 270° (top/12 o'clock)
        # Sector 1 (1) centered at 270° - 18° = 252°
        # etc. (clockwise around board because of inverted Y)
        angle_center = 270.0 - sector_idx * angle_per_sector
        angle_start = angle_center - 9.0  # Half sector width
        angle_end = angle_center + 9.0

        # Determine colors (alternating pattern)
        # Sector 0 (20) = dark single, red triple/double
        is_dark_single = (sector_idx % 2 == 0)
        is_red = (sector_idx % 2 == 0)

        triple_color = colors["triple_red"] if is_red else colors["triple_green"]
        double_color = colors["double_red"] if is_red else colors["double_green"]
        single_color = colors["single_dark"] if is_dark_single else colors["single_light"]

        # Draw DOUBLE ring sector (outermost)
        draw_filled_sector(
            img, (cx, cy),
            r_double_inner, r_double_outer,
            angle_start, angle_end,
            double_color, alpha
        )

        # Draw SINGLE field sector (between double and triple)
        draw_filled_sector(
            img, (cx, cy),
            r_triple_outer, r_double_inner,
            angle_start, angle_end,
            single_color, alpha
        )

        # Draw TRIPLE ring sector
        draw_filled_sector(
            img, (cx, cy),
            r_triple_inner, r_triple_outer,
            angle_start, angle_end,
            triple_color, alpha
        )

        # Draw INNER SINGLE (between triple and bullseye) - same color as outer single
        draw_filled_sector(
            img, (cx, cy),
            r_bull_outer, r_triple_inner,
            angle_start, angle_end,
            single_color, alpha
        )

        # Draw sector number in OUTER single field
        if show_numbers:
            draw_sector_number(
                img, (cx, cy),
                r_triple_outer,  # Inner radius of outer single field
                r_double_inner,  # Outer radius of outer single field
                angle_center,
                sector_num,
                is_dark_single,
                colors
            )

    # Draw BULLSEYE (outer green ring - 25 points)
    overlay_bull = img.copy()
    cv2.circle(overlay_bull, (cx, cy), int(r_bull_outer), colors["bull_outer"], -1, cv2.LINE_AA)
    cv2.addWeighted(overlay_bull, alpha, img, 1 - alpha, 0, img)

    # Draw BULLSEYE (inner red circle - 50 points)
    overlay_bull_inner = img.copy()
    cv2.circle(overlay_bull_inner, (cx, cy), int(r_bull_inner), colors["bull_inner"], -1, cv2.LINE_AA)
    cv2.addWeighted(overlay_bull_inner, alpha, img, 1 - alpha, 0, img)

    return img


def draw_calibration_guides(
    img: np.ndarray,
    center: Tuple[int, int],
    radius: float,
    show_crosshair: Optional[bool] = None,
    show_circles: Optional[bool] = None
) -> np.ndarray:
    """
    Draw calibration guide overlays for precise alignment.

    All settings loaded from overlay_config.yaml.

    Args:
        img: Image to draw on
        center: Center point (cx, cy)
        radius: Reference radius
        show_crosshair: Show center crosshair (None = use config default)
        show_circles: Show concentric circles (None = use config default)

    Returns:
        Image with guides
    """
    # Load configuration
    config = _get_overlay_config()
    guide_cfg = config.get('calibration_guides', {})
    colors = _get_colors()

    # Use config defaults if not specified
    if show_crosshair is None:
        show_crosshair = guide_cfg.get('show_crosshair', True)
    if show_circles is None:
        show_circles = guide_cfg.get('show_circles', True)

    cx, cy = center

    # Crosshair at center
    if show_crosshair:
        crosshair_size = guide_cfg.get('crosshair_size', 20)
        color_cross = colors.get('guide_crosshair', (0, 255, 255))
        thickness = 2

        # Horizontal line
        cv2.line(img, (cx - crosshair_size, cy), (cx + crosshair_size, cy), color_cross, thickness, cv2.LINE_AA)
        # Vertical line
        cv2.line(img, (cx, cy - crosshair_size), (cx, cy + crosshair_size), color_cross, thickness, cv2.LINE_AA)

        # Center dot
        cv2.circle(img, (cx, cy), 3, color_cross, -1, cv2.LINE_AA)

    # Concentric circles for radius reference
    if show_circles:
        color_circle = colors.get('guide_circles', (100, 100, 255))
        radii = guide_cfg.get('circle_radii', [0.25, 0.5, 0.75, 1.0])

        for r_frac in radii:
            r = int(radius * r_frac)
            cv2.circle(img, (cx, cy), r, color_circle, 1, cv2.LINE_AA)

    return img
