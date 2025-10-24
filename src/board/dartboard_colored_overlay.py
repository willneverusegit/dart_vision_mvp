"""
Colored Dartboard Overlay - Renders realistic dartboard with colors.

Provides 1:1 visual representation of scoring zones for calibration:
- Each SECTOR (pie slice) colored individually
- Triple/Double rings: Alternating Red/Green per sector
- Single fields: Alternating Dark/Light (Black/Cream) per sector
- Bullseye: Inner Red (50pts), Outer Green (25pts)
- Centered numbers in single fields
"""

import cv2
import numpy as np
import math
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.board import BoardMapper

# Standard dartboard sector arrangement (starting at top, clockwise)
DARTBOARD_SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# Colors (BGR format for OpenCV) - matching real dartboard
COLORS_BGR = {
    # Triple ring (alternating red/green)
    "triple_red": (0, 0, 200),
    "triple_green": (0, 180, 0),

    # Double ring (alternating red/green)
    "double_red": (0, 0, 200),
    "double_green": (0, 180, 0),

    # Single fields (alternating dark/light)
    "single_dark": (20, 20, 20),        # Black
    "single_light": (200, 220, 240),    # Cream/Beige

    # Bullseye
    "bull_inner": (0, 0, 220),   # Red (50 points)
    "bull_outer": (0, 200, 0),   # Green (25 points)

    # Text colors
    "text_on_dark": (240, 240, 240),
    "text_on_light": (30, 30, 30),
}

# Default alpha (transparency)
DEFAULT_ALPHA = 0.40


def draw_filled_sector(
    img: np.ndarray,
    center: Tuple[int, int],
    radius_inner: float,
    radius_outer: float,
    angle_start_deg: float,
    angle_end_deg: float,
    color: Tuple[int, int, int],
    alpha: float = DEFAULT_ALPHA
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
    is_dark_background: bool
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
    text_color = COLORS_BGR["text_on_dark"] if is_dark_background else COLORS_BGR["text_on_light"]

    # Measure text size for centering
    text = str(number)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Center text
    text_x -= text_width // 2
    text_y += text_height // 2

    # Draw text with outline for better visibility
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)  # Outline
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)  # Text

    return img


def draw_colored_dartboard_overlay(
    img: np.ndarray,
    board_mapper: 'BoardMapper',
    alpha: float = DEFAULT_ALPHA,
    show_numbers: bool = True
) -> np.ndarray:
    """
    Draw full colored dartboard overlay matching real dartboard colors.

    Each SECTOR (pie slice) is colored individually:
    - Sector 20 at top (12 o'clock) - starts dark/black
    - Alternating dark/light for singles
    - Alternating red/green for triples/doubles

    Args:
        img: Image to draw on (ROI frame)
        board_mapper: BoardMapper with calibration data
        alpha: Overall transparency (0.0-1.0)
        show_numbers: Whether to draw sector numbers

    Returns:
        Image with colored overlay
    """
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
    # In OpenCV: 0° = right (3 o'clock), counter-clockwise
    # 12 o'clock = 90° in OpenCV convention
    # Sector boundaries at ±9° from sector center
    # So sector 20 goes from 81° to 99° (centered at 90°)

    # Draw each sector
    for sector_idx, sector_num in enumerate(DARTBOARD_SECTORS):
        # Calculate sector angles (OpenCV convention: 0° = right, CCW)
        # Sector 0 (20) centered at 90° (top)
        # Sector 1 (1) centered at 90° - 18° = 72°
        # etc.
        angle_center = 90.0 - sector_idx * angle_per_sector  # CCW from top
        angle_start = angle_center - 9.0  # Half sector width
        angle_end = angle_center + 9.0

        # Determine colors (alternating pattern)
        # Sector 0 (20) = dark single, red triple/double
        is_dark_single = (sector_idx % 2 == 0)
        is_red = (sector_idx % 2 == 0)

        triple_color = COLORS_BGR["triple_red"] if is_red else COLORS_BGR["triple_green"]
        double_color = COLORS_BGR["double_red"] if is_red else COLORS_BGR["double_green"]
        single_color = COLORS_BGR["single_dark"] if is_dark_single else COLORS_BGR["single_light"]

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
                is_dark_single
            )

    # Draw BULLSEYE (outer green ring - 25 points)
    overlay_bull = img.copy()
    cv2.circle(overlay_bull, (cx, cy), int(r_bull_outer), COLORS_BGR["bull_outer"], -1, cv2.LINE_AA)
    cv2.addWeighted(overlay_bull, alpha, img, 1 - alpha, 0, img)

    # Draw BULLSEYE (inner red circle - 50 points)
    overlay_bull_inner = img.copy()
    cv2.circle(overlay_bull_inner, (cx, cy), int(r_bull_inner), COLORS_BGR["bull_inner"], -1, cv2.LINE_AA)
    cv2.addWeighted(overlay_bull_inner, alpha, img, 1 - alpha, 0, img)

    return img


def draw_calibration_guides(
    img: np.ndarray,
    center: Tuple[int, int],
    radius: float,
    show_crosshair: bool = True,
    show_circles: bool = True
) -> np.ndarray:
    """
    Draw calibration guide overlays for precise alignment.

    Args:
        img: Image to draw on
        center: Center point (cx, cy)
        radius: Reference radius
        show_crosshair: Show center crosshair
        show_circles: Show concentric circles

    Returns:
        Image with guides
    """
    cx, cy = center

    # Crosshair at center
    if show_crosshair:
        crosshair_size = 20
        color_cross = (0, 255, 255)  # Cyan
        thickness = 2

        # Horizontal line
        cv2.line(img, (cx - crosshair_size, cy), (cx + crosshair_size, cy), color_cross, thickness, cv2.LINE_AA)
        # Vertical line
        cv2.line(img, (cx, cy - crosshair_size), (cx, cy + crosshair_size), color_cross, thickness, cv2.LINE_AA)

        # Center dot
        cv2.circle(img, (cx, cy), 3, color_cross, -1, cv2.LINE_AA)

    # Concentric circles for radius reference
    if show_circles:
        color_circle = (100, 100, 255)  # Light red
        radii = [0.25, 0.5, 0.75, 1.0]

        for r_frac in radii:
            r = int(radius * r_frac)
            cv2.circle(img, (cx, cy), r, color_circle, 1, cv2.LINE_AA)

    return img
