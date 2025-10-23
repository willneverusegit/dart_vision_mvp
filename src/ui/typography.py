"""Utility helpers for consistent typography and chip styling in the overlays."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

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

    @classmethod
    def get_palette(cls, status: str) -> Dict[str, Tuple[int, int, int]]:
        """Return a color palette for the given status."""
        return cls._STATUS_PALETTE.get(status, cls._STATUS_PALETTE["info"])

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
        palette = self.get_palette(status)

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
        palette = self.get_palette(status)
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


class CardDrawer:
    """Render stacked information cards with consistent styling."""

    _BASE_FILL = (46, 48, 84)
    _BASE_BORDER = (88, 96, 146)
    _BASE_TITLE = (236, 238, 252)
    _BASE_LABEL = (204, 210, 232)
    _BASE_VALUE = (248, 250, 255)
    _BASE_HINT = (198, 204, 226)
    _BASE_FOOTER = (206, 210, 234)
    _BASE_ACCENT = (132, 168, 236)
    _BASE_PROGRESS_BG = (58, 60, 92)

    def __init__(self, typography: Typography) -> None:
        self._typography = typography
        self._pad_x = 18
        self._pad_top = 18
        self._pad_bottom = 18
        self._row_gap = 14
        self._label_value_gap = 4
        self._progress_height = 8
        self._progress_gap = 12
        self._title_height = 24
        self._label_height = 16
        self._value_height = 28
        self._footer_height = 16

    def prepare_card(
        self,
        width: int,
        title: str,
        rows: List[Dict[str, Any]],
        footer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pre-compute layout metrics for a card."""
        width = max(int(width), 96)
        content = self._prepare_content(width, title, rows, footer)
        content["width"] = width
        content["height"] = self._compute_height(content)
        return content

    def draw_card(
        self,
        canvas: np.ndarray,
        origin: Tuple[int, int],
        card: Dict[str, Any],
        status: str = "info",
        align: str = "left",
        alpha: float = 0.82,
    ) -> Tuple[int, int]:
        """Draw a prepared card and return its size."""
        width = int(card.get("width", 0))
        height = int(card.get("height", 0))
        if width <= 0 or height <= 0:
            return width, height

        if align == "right":
            x1 = int(origin[0] - width)
        else:
            x1 = int(origin[0])
        y1 = int(origin[1])
        x2 = x1 + width
        y2 = y1 + height

        if x2 <= 0 or y2 <= 0 or x1 >= canvas.shape[1] or y1 >= canvas.shape[0]:
            return width, height

        style = self._resolve_style(status)

        ChipDrawer._rounded_rect(canvas, (x1, y1, x2, y2), 18, style["fill"], alpha)
        ChipDrawer._rounded_rect_outline(canvas, (x1, y1, x2, y2), 18, style["border"], 1)
        cv2.line(canvas, (x1 + 20, y1 + 10), (x2 - 20, y1 + 10), style["accent"], 2, cv2.LINE_AA)

        text_x = x1 + self._pad_x
        cursor_y = y1 + self._pad_top

        title_block = card.get("title")
        if title_block is not None:
            _, txt_h, txt_base = title_block["metrics"]
            baseline = cursor_y + txt_h
            self._typography.draw(canvas, title_block["text"], (text_x, baseline), self._title_height, style["title"])
            cursor_y = baseline + txt_base + self._row_gap

        rows = card.get("rows", [])
        for idx, row in enumerate(rows):
            label_block = row.get("label")
            if label_block is not None:
                _, lbl_h, lbl_base = label_block["metrics"]
                baseline = cursor_y + lbl_h
                self._typography.draw(
                    canvas,
                    label_block["text"],
                    (text_x, baseline),
                    self._label_height,
                    style["label"],
                )
                cursor_y = baseline + lbl_base + self._label_value_gap

            value_block = row.get("value")
            value_palette = ChipDrawer.get_palette(row.get("status", "info"))
            value_color = self._mix_color(style["value"], value_palette.get("text_primary", style["value"]), 0.45)
            _, val_h, val_base = value_block["metrics"]
            baseline = cursor_y + val_h
            self._typography.draw(
                canvas,
                value_block["text"],
                (text_x, baseline),
                self._value_height,
                value_color,
            )
            cursor_y = baseline + val_base

            progress = row.get("progress")
            if progress is not None:
                bar_x1 = text_x
                bar_x2 = x2 - self._pad_x
                bar_y1 = cursor_y + 6
                bar_y2 = bar_y1 + self._progress_height
                cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), style["progress_bg"], -1, cv2.LINE_AA)
                progress_width = int((bar_x2 - bar_x1) * progress)
                if progress_width > 0:
                    cv2.rectangle(
                        canvas,
                        (bar_x1, bar_y1),
                        (bar_x1 + progress_width, bar_y2),
                        style["progress_fg"],
                        -1,
                        cv2.LINE_AA,
                    )
                cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), style["progress_border"], 1, cv2.LINE_AA)
                cursor_y = bar_y2 + self._progress_gap

            hint_block = row.get("hint")
            if hint_block is not None:
                _, hint_h, hint_base = hint_block["metrics"]
                baseline = cursor_y + hint_h
                self._typography.draw(
                    canvas,
                    hint_block["text"],
                    (text_x, baseline),
                    self._label_height,
                    style["hint"],
                )
                cursor_y = baseline + hint_base

            if idx != len(rows) - 1 or card.get("footer") is not None:
                cursor_y += self._row_gap

        footer_block = card.get("footer")
        if footer_block is not None:
            _, foot_h, foot_base = footer_block["metrics"]
            baseline = cursor_y + foot_h
            self._typography.draw(canvas, footer_block["text"], (text_x, baseline), self._footer_height, style["footer"])

        return width, height

    def _prepare_content(
        self,
        width: int,
        title: str,
        rows: List[Dict[str, Any]],
        footer: Optional[str],
    ) -> Dict[str, Any]:
        max_text_width = max(width - 2 * self._pad_x, 48)

        title_block = None
        if title:
            title_text = self._ellipsize(title.upper(), max_text_width, self._title_height)
            title_block = {
                "text": title_text,
                "metrics": self._typography.measure(title_text, self._title_height),
            }

        prepared_rows: List[Dict[str, Any]] = []
        for row in rows:
            label_block = None
            label_text = row.get("label", "")
            if label_text:
                label_text = self._ellipsize(label_text.upper(), max_text_width, self._label_height)
                label_block = {
                    "text": label_text,
                    "metrics": self._typography.measure(label_text, self._label_height),
                }

            value_text = self._ellipsize(str(row.get("value", "")), max_text_width, self._value_height)
            value_block = {
                "text": value_text,
                "metrics": self._typography.measure(value_text, self._value_height),
            }

            hint_block = None
            hint_text = row.get("hint")
            if hint_text:
                hint_text = self._ellipsize(str(hint_text), max_text_width, self._label_height)
                hint_block = {
                    "text": hint_text,
                    "metrics": self._typography.measure(hint_text, self._label_height),
                }

            progress = row.get("progress")
            if progress is not None:
                try:
                    progress_val = float(progress)
                except (TypeError, ValueError):
                    progress_val = 0.0
                progress = max(0.0, min(1.0, progress_val))

            prepared_rows.append(
                {
                    "label": label_block,
                    "value": value_block,
                    "hint": hint_block,
                    "status": row.get("status", "info"),
                    "progress": progress,
                }
            )

        footer_block = None
        if footer:
            footer_text = self._ellipsize(str(footer), max_text_width, self._footer_height)
            footer_block = {
                "text": footer_text,
                "metrics": self._typography.measure(footer_text, self._footer_height),
            }

        return {
            "title": title_block,
            "rows": prepared_rows,
            "footer": footer_block,
        }

    def _compute_height(self, card: Dict[str, Any]) -> int:
        height = self._pad_top + self._pad_bottom

        title_block = card.get("title")
        rows: List[Dict[str, Any]] = card.get("rows", [])
        footer_block = card.get("footer")

        if title_block is not None:
            _, h, base = title_block["metrics"]
            height += h + base
            if rows or footer_block:
                height += self._row_gap

        for idx, row in enumerate(rows):
            label_block = row.get("label")
            if label_block is not None:
                _, h, base = label_block["metrics"]
                height += h + base + self._label_value_gap

            value_block = row.get("value")
            _, h, base = value_block["metrics"]
            height += h + base

            if row.get("progress") is not None:
                height += self._progress_height + self._progress_gap

            hint_block = row.get("hint")
            if hint_block is not None:
                _, h, base = hint_block["metrics"]
                height += h + base

            if idx != len(rows) - 1 or footer_block is not None:
                height += self._row_gap

        if footer_block is not None:
            _, h, base = footer_block["metrics"]
            height += h + base

        return int(height)

    def _resolve_style(self, status: str) -> Dict[str, Tuple[int, int, int]]:
        palette = ChipDrawer.get_palette(status)
        return {
            "fill": self._mix_color(self._BASE_FILL, palette.get("fill", self._BASE_FILL), 0.45),
            "border": self._mix_color(self._BASE_BORDER, palette.get("border", self._BASE_BORDER), 0.55),
            "title": self._BASE_TITLE,
            "label": self._mix_color(self._BASE_LABEL, palette.get("text_secondary", self._BASE_LABEL), 0.3),
            "value": self._mix_color(self._BASE_VALUE, palette.get("text_primary", self._BASE_VALUE), 0.4),
            "hint": self._BASE_HINT,
            "footer": self._BASE_FOOTER,
            "accent": self._mix_color(self._BASE_ACCENT, palette.get("indicator", self._BASE_ACCENT), 0.6),
            "progress_bg": self._BASE_PROGRESS_BG,
            "progress_fg": self._mix_color(self._BASE_ACCENT, palette.get("indicator", self._BASE_ACCENT), 0.7),
            "progress_border": self._mix_color(self._BASE_BORDER, palette.get("border", self._BASE_BORDER), 0.4),
        }

    @staticmethod
    def _mix_color(base: Tuple[int, int, int], accent: Tuple[int, int, int], weight: float) -> Tuple[int, int, int]:
        weight = max(0.0, min(1.0, weight))
        return tuple(
            int(round(base[i] * (1.0 - weight) + accent[i] * weight))
            for i in range(3)
        )

    def _ellipsize(self, text: str, max_width: int, font_height: int) -> str:
        if not text:
            return ""

        width, _, _ = self._typography.measure(text, font_height)
        if width <= max_width:
            return text

        ellipsis = "…"
        ell_w, _, _ = self._typography.measure(ellipsis, font_height)
        if ell_w >= max_width:
            return ellipsis

        low, high = 0, len(text)
        best = ""
        while low <= high:
            mid = (low + high) // 2
            candidate = text[:mid]
            cand_w, _, _ = self._typography.measure(candidate, font_height)
            if cand_w + ell_w <= max_width:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1

        return best + ellipsis if best else ellipsis
