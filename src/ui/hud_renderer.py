"""
HUD Renderer - Heads-Up-Display components for Dart Vision MVP.

Handles all HUD rendering including:
- Image quality metrics (brightness, focus, edge density)
- Traffic light status indicators
- Help overlay
- Debug information
"""

import cv2
import numpy as np
from collections import deque
from typing import Tuple, List, Optional


class HUDRenderer:
    """Renders HUD elements for image quality monitoring and user feedback."""

    def __init__(self, smoothing_window: int = 15):
        """
        Initialize HUD renderer with smoothing buffers.

        Args:
            smoothing_window: Number of frames to smooth over for metrics
        """
        self.smoothing_window = smoothing_window

        # Smoothing buffers (EMA-like)
        self._hud_b = deque(maxlen=smoothing_window)  # Brightness
        self._hud_f = deque(maxlen=smoothing_window)  # Focus (Laplacian Var)
        self._hud_e = deque(maxlen=smoothing_window)  # Edge density %

        # Thresholds for quality assessment
        self.brightness_range = (120.0, 170.0)  # Good range
        self.brightness_acceptable = (110.0, 180.0)  # Yellow range
        self.focus_good = 1500.0
        self.focus_acceptable = 800.0
        self.edge_range = (5.0, 10.0)  # Good range
        self.edge_acceptable = (3.5, 15.0)  # Yellow range

    def compute_metrics(self, gray: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute image quality metrics from grayscale image.

        Args:
            gray: Grayscale image (numpy array)

        Returns:
            Tuple of (brightness, focus_variance, edge_density_percent)
        """
        # Brightness (mean intensity)
        b = float(np.mean(gray))

        # Focus proxy (Laplacian variance - higher = sharper)
        f = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Edge density (Canny edges as percentage)
        edges = cv2.Canny(gray, 80, 160)
        e = 100.0 * float(np.mean(edges > 0))

        # Smooth with moving average
        self._hud_b.append(b)
        self._hud_f.append(f)
        self._hud_e.append(e)

        b_smooth = float(np.mean(self._hud_b))
        f_smooth = float(np.mean(self._hud_f))
        e_smooth = float(np.mean(self._hud_e))

        return b_smooth, f_smooth, e_smooth

    def get_metrics_color(
        self,
        b: float,
        f: float,
        e: float,
        charuco_corners: int = 0
    ) -> Tuple[int, int, int]:
        """
        Determine color based on metric quality (green = good, yellow = warning).

        Args:
            b: Brightness value
            f: Focus variance value
            e: Edge density percentage
            charuco_corners: Number of ChArUco corners detected (calibration only)

        Returns:
            BGR color tuple
        """
        ok_b = self.brightness_range[0] <= b <= self.brightness_range[1]
        ok_f = f >= self.focus_good
        ok_e = self.edge_range[0] <= e <= self.edge_range[1]
        ok_c = (charuco_corners >= 8)  # Only relevant during calibration

        ok = ok_b and ok_f and ok_e and ok_c
        return (0, 255, 0) if ok else (0, 200, 200)

    def draw_text_lines(
        self,
        img: np.ndarray,
        lines: List[str],
        color: Tuple[int, int, int] = (0, 255, 0),
        org: Tuple[int, int] = (12, 24)
    ):
        """
        Draw multiple text lines on image.

        Args:
            img: Image to draw on (modified in-place)
            lines: List of text strings
            color: BGR color tuple
            org: Starting position (x, y)
        """
        x, y = org
        for ln in lines:
            cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            y += 22

    def draw_traffic_light(
        self,
        img: np.ndarray,
        b: float,
        f: float,
        e: float,
        org: Tuple[int, int] = (12, 105)
    ):
        """
        Draw traffic light status indicators for B/F/E metrics.

        Green = optimal, Yellow = acceptable, Red = poor

        Args:
            img: Image to draw on (modified in-place)
            b: Brightness value
            f: Focus variance
            e: Edge density percentage
            org: Top-left position (x, y)
        """
        def status_brightness(v: float) -> str:
            if self.brightness_range[0] <= v <= self.brightness_range[1]:
                return 'G'
            elif self.brightness_acceptable[0] <= v <= self.brightness_acceptable[1]:
                return 'Y'
            else:
                return 'R'

        def status_focus(v: float) -> str:
            if v >= self.focus_good:
                return 'G'
            elif v >= self.focus_acceptable:
                return 'Y'
            else:
                return 'R'

        def status_edge(v: float) -> str:
            if self.edge_range[0] <= v <= self.edge_range[1]:
                return 'G'
            elif self.edge_acceptable[0] <= v <= self.edge_acceptable[1]:
                return 'Y'
            else:
                return 'R'

        colors = {
            'R': (36, 36, 255),   # Red (BGR)
            'Y': (0, 255, 255),   # Yellow
            'G': (0, 200, 0)      # Green
        }

        states = [
            ("B", status_brightness(b)),
            ("F", status_focus(f)),
            ("E", status_edge(e))
        ]

        x, y = org
        w, h, pad = 18, 18, 8

        # Title
        cv2.putText(img, "Status:", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)

        # Individual indicators
        for i, (label, status) in enumerate(states):
            p1 = (x + i * (w + pad), y)
            p2 = (p1[0] + w, p1[1] + h)

            # Filled rectangle with status color
            cv2.rectangle(img, p1, p2, colors[status], -1)
            cv2.rectangle(img, p1, p2, (20, 20, 20), 1)  # Border

            # Label below
            cv2.putText(img, label, (p1[0], p2[1] + 16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

        # Overall status (ALL)
        status_values = [s for _, s in states]
        if all(s == 'G' for s in status_values):
            overall = 'G'
        elif 'R' in status_values:
            overall = 'R'
        else:
            overall = 'Y'

        X = x + 3 * (w + pad) + 20
        cv2.putText(img, "ALL", (X, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.rectangle(img, (X, y), (X + w, y + h), colors[overall], -1)
        cv2.rectangle(img, (X, y), (X + w, y + h), (20, 20, 20), 1)

    def draw_help_overlay(self, img: np.ndarray):
        """
        Draw compact help overlay in bottom-right corner.

        Uses local ROI blending for safe rendering across all OpenCV backends.

        Args:
            img: Image to draw on (modified in-place)
        """
        pad = 10
        lines = [
            "Help / Controls",
            "1/2/3: Preset Agg/Bal/Stable",
            "o: Overlay (MIN/RINGS/FULL/ALIGN)",
            "t: Hough once   z: Auto-Hough",
            "Arrows: rot/scale overlay",
            "X: Save overlay offsets",
            "c: Recalibrate   s: Screenshot",
            "g: Game reset    h: Switch game",
            "?: Toggle help",
        ]

        # Calculate box dimensions
        w = 310
        h = 22 * len(lines) + 2 * pad
        H, W = img.shape[:2]
        x = max(0, W - w - pad)
        y = max(0, H - h - pad)

        # Extract ROI
        roi = img[y:y + h, x:x + w]
        if roi.size == 0:
            return  # Edge case: image too small

        # Create semi-transparent background
        bg = roi.copy()
        cv2.rectangle(bg, (0, 0), (w, h), (20, 20, 20), -1)

        # Alpha blend (safe, no in-place parameter)
        blended = cv2.addWeighted(bg, 0.6, roi, 0.4, 0.0)
        roi[:] = blended  # Write back to image

        # Draw text on the now-tinted ROI
        ytxt = pad + 18
        cv2.putText(roi, lines[0], (pad, ytxt),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        ytxt += 10

        for ln in lines[1:]:
            ytxt += 18
            cv2.putText(roi, ln, (pad, ytxt),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    def reset_buffers(self):
        """Clear all smoothing buffers (e.g., after scene change)."""
        self._hud_b.clear()
        self._hud_f.clear()
        self._hud_e.clear()
