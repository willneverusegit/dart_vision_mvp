"""Utility helpers for consistent typography and chip styling in the overlays."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

try:  # Optional FreeType support (opencv-contrib)
    import cv2.freetype as cv2_freetype  # type: ignore
except (ImportError, AttributeError):  # pragma: no cover - depends on build
    cv2_freetype = None  # type: ignore

# Default location for a custom UI font. If the file is missing or OpenCV was
# built without freetype support we gracefully fall back to Hershey fonts.
_DEFAULT_FONT_PATH = Path(__file__).resolve().parent / "assets" / "fonts" / "Inter-SemiBold.ttf"


class Typography:
    """Wrapper around OpenCV text rendering with optional FreeType support."""

    def __init__(self, font_path: Optional[Path] = None) -> None:
        self._font_path = Path(font_path) if font_path else _DEFAULT_FONT_PATH
        self._freetype = self._load_freetype_font()
        self._fallback_font = cv2.FONT_HERSHEY_DUPLEX

    def _load_freetype_font(self):  # pragma: no cover - requires optional dep
        if cv2_freetype is None or not self._font_path.exists():
            return None
        try:
            ft2 = cv2_freetype.createFreeType2()
            ft2.loadFontData(str(self._font_path), 0)
            return ft2
        except Exception:  # Defensive: avoid crashing HUD when font missing
            return None

    def measure(self, text: str, font_height: int, thickness: int = -1) -> Tuple[int, int, int]:
        """Return (width, height, baseline) for the given text."""
        if not text:
            return 0, 0, 0

        if self._freetype is not None:
            (w, h), baseline = self._freetype.getTextSize(text, font_height, thickness)
            return int(w), int(h), int(baseline)

        scale = max(font_height / 30.0, 0.2)
        thickness_px = max(1, int(round(scale * 2)))
        (w, h), baseline = cv2.getTextSize(text, self._fallback_font, scale, thickness_px)
        return int(w), int(h), int(baseline)

    def draw(
        self,
        canvas: np.ndarray,
        text: str,
        org: Tuple[int, int],
        font_height: int,
        color: Tuple[int, int, int],
        thickness: int = -1,
    ) -> None:
        """Draw text using FreeType if available, otherwise Hershey fonts."""
        if not text:
            return

        if self._freetype is not None:  # pragma: no cover - optional runtime path
            self._freetype.putText(
                canvas,
                text,
                org,
                font_height,
                color,
                thickness,
                cv2.LINE_AA,
                False,
            )
            return

        scale = max(font_height / 30.0, 0.2)
        thickness_px = max(1, int(round(scale * 2)))
        cv2.putText(
            canvas,
            text,
            org,
            self._fallback_font,
            scale,
            color,
            thickness_px,
            cv2.LINE_AA,
        )


class ChipDrawer:
    """Utility for drawing rounded information chips with consistent styling."""

    _STATUS_PALETTE = {
        "good": {
            "fill": (82, 122, 96),
            "border": (136, 210, 182),
            "text_primary": (250, 250, 252),
            "text_secondary": (222, 240, 232),
            "indicator": (120, 220, 180),
        },
        "warn": {
            "fill": (92, 180, 230),
            "border": (144, 215, 255),
            "text_primary": (255, 255, 255),
            "text_secondary": (240, 245, 250),
            "indicator": (120, 225, 255),
        },
        "bad": {
            "fill": (72, 80, 210),
            "border": (132, 142, 255),
            "text_primary": (255, 240, 240),
            "text_secondary": (235, 218, 218),
            "indicator": (120, 120, 255),
        },
        "info": {
            "fill": (88, 88, 138),
            "border": (140, 140, 200),
            "text_primary": (238, 238, 248),
            "text_secondary": (214, 214, 234),
            "indicator": (170, 170, 230),
        },
        "accent": {
            "fill": (74, 112, 182),
            "border": (128, 176, 242),
            "text_primary": (248, 248, 255),
            "text_secondary": (224, 232, 248),
            "indicator": (156, 192, 255),
        },
    }

    def __init__(self, typography: Typography) -> None:
        self._typography = typography

    def draw_metric_chip(
        self,
        canvas: np.ndarray,
        origin: Tuple[int, int],
        label: str,
        value: str,
        status: str,
        subtitle: Optional[str] = None,
        align: str = "left",
        compact: bool = False,
        alpha: float = 0.82,
    ) -> Tuple[int, int]:
        """Draw a multi-line chip and return its (width, height)."""
        palette = self._STATUS_PALETTE.get(status, self._STATUS_PALETTE["info"])

        if compact:
            pad_x = 14
            pad_top = 8
            pad_mid = 4
            pad_bottom = 10 if subtitle else 8
            label_height = 18
            value_height = 28
            subtitle_height = 16
            indicator_radius = 6
        else:
            pad_x = 18
            pad_top = 12
            pad_mid = 6
            pad_bottom = 12 if subtitle else 10
            label_height = 20
            value_height = 36
            subtitle_height = 18
            indicator_radius = 7

        indicator_gap = 10
        label_text = label.upper()
        value_text = value
        subtitle_text = subtitle.upper() if subtitle else ""

        label_w, label_h, label_base = self._typography.measure(label_text, label_height)
        value_w, value_h, value_base = self._typography.measure(value_text, value_height)
        subtitle_w, subtitle_h, subtitle_base = self._typography.measure(subtitle_text, subtitle_height)

        show_indicator = palette.get("indicator") is not None and status in ("good", "warn", "bad")
        indicator_space = (indicator_radius * 2 + indicator_gap) if show_indicator else 0

        text_block_width = max(label_w, value_w, subtitle_w)
        chip_width = pad_x * 2 + indicator_space + text_block_width
        chip_height = pad_top + label_h + label_base + pad_mid + value_h + value_base
        if subtitle_text:
            chip_height += pad_mid + subtitle_h + subtitle_base
        chip_height += pad_bottom

        if align == "right":
            x = int(origin[0] - chip_width)
        else:
            x = int(origin[0])
        y = int(origin[1])
        x2 = x + chip_width
        y2 = y + chip_height

        if x2 <= 0 or y2 <= 0 or x >= canvas.shape[1] or y >= canvas.shape[0]:
            return chip_width, chip_height

        self._rounded_rect(canvas, (x, y, x2, y2), indicator_radius + 4, palette["fill"], alpha)
        self._rounded_rect_outline(canvas, (x, y, x2, y2), indicator_radius + 4, palette["border"], 1)

        text_x = x + pad_x + (indicator_space if show_indicator else 0)
        label_baseline_y = y + pad_top + label_h
        value_baseline_y = label_baseline_y + label_base + pad_mid + value_h
        subtitle_baseline_y = 0
        if subtitle_text:
            subtitle_baseline_y = value_baseline_y + value_base + pad_mid + subtitle_h

        self._typography.draw(canvas, label_text, (text_x, label_baseline_y), label_height, palette["text_secondary"])
        self._typography.draw(canvas, value_text, (text_x, value_baseline_y), value_height, palette["text_primary"])
        if subtitle_text:
            self._typography.draw(
                canvas,
                subtitle_text,
                (text_x, subtitle_baseline_y),
                subtitle_height,
                palette["text_secondary"],
            )

        if show_indicator:
            center = (x + pad_x + indicator_radius, label_baseline_y - label_h // 2)
            cv2.circle(canvas, center, indicator_radius, palette["indicator"], -1, cv2.LINE_AA)
            cv2.circle(canvas, center, indicator_radius, (24, 24, 24), 1, cv2.LINE_AA)

        return chip_width, chip_height

    def draw_compact_chip(
        self,
        canvas: np.ndarray,
        origin: Tuple[int, int],
        text: str,
        status: str = "info",
        align: str = "left",
        alpha: float = 0.78,
    ) -> Tuple[int, int]:
        """Draw a single-line pill chip."""
        palette = self._STATUS_PALETTE.get(status, self._STATUS_PALETTE["info"])
        pad_x = 14
        pad_y = 8
        text_height = 24
        indicator_radius = 5
        indicator_gap = 8

        text_w, text_h, text_base = self._typography.measure(text, text_height)
        show_indicator = status in ("good", "warn", "bad")
        indicator_space = indicator_radius * 2 + indicator_gap if show_indicator else 0

        chip_width = pad_x * 2 + indicator_space + text_w
        chip_height = pad_y * 2 + text_h + text_base

        if align == "right":
            x = int(origin[0] - chip_width)
        else:
            x = int(origin[0])
        y = int(origin[1])
        x2, y2 = x + chip_width, y + chip_height

        if x2 <= 0 or y2 <= 0 or x >= canvas.shape[1] or y >= canvas.shape[0]:
            return chip_width, chip_height

        self._rounded_rect(canvas, (x, y, x2, y2), indicator_radius + 6, palette["fill"], alpha)
        self._rounded_rect_outline(canvas, (x, y, x2, y2), indicator_radius + 6, palette["border"], 1)

        text_x = x + pad_x + (indicator_space if show_indicator else 0)
        baseline_y = y + pad_y + text_h
        self._typography.draw(canvas, text, (text_x, baseline_y), text_height, palette["text_primary"])

        if show_indicator:
            center = (x + pad_x + indicator_radius, baseline_y - text_h // 2)
            cv2.circle(canvas, center, indicator_radius, palette["indicator"], -1, cv2.LINE_AA)
            cv2.circle(canvas, center, indicator_radius, (24, 24, 24), 1, cv2.LINE_AA)

        return chip_width, chip_height

    @staticmethod
    def _rounded_rect(
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        radius: int,
        color: Tuple[int, int, int],
        alpha: float,
    ) -> None:
        x1, y1, x2, y2 = rect
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, canvas.shape[1])
        y2 = min(y2, canvas.shape[0])
        if x2 <= x1 or y2 <= y1:
            return

        width = x2 - x1
        height = y2 - y1
        radius = int(max(1, min(radius, min(width, height) // 2)))

        overlay = np.zeros((height, width, 3), dtype=canvas.dtype)
        cv2.rectangle(overlay, (radius, 0), (width - radius - 1, height - 1), color, -1, cv2.LINE_AA)
        cv2.rectangle(overlay, (0, radius), (width - 1, height - radius - 1), color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (radius, radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (width - radius - 1, radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (radius, height - radius - 1), radius, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (width - radius - 1, height - radius - 1), radius, color, -1, cv2.LINE_AA)

        region = canvas[y1:y2, x1:x2]
        cv2.addWeighted(overlay, alpha, region, 1.0 - alpha, 0.0, dst=region)

    @staticmethod
    def _rounded_rect_outline(
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        radius: int,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        x1, y1, x2, y2 = rect
        radius = int(max(1, min(radius, (min(x2 - x1, y2 - y1) // 2))))
        cv2.line(canvas, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1 + radius, y2 - 1), (x2 - radius, y2 - 1), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x2 - 1, y1 + radius), (x2 - 1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (x2 - radius - 1, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (x1 + radius, y2 - radius - 1), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (x2 - radius - 1, y2 - radius - 1), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
