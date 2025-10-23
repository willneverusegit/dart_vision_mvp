"""Shared helpers to classify HUD metrics for consistent chip styling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

BRIGHTNESS_GOOD_RANGE = (120.0, 170.0)
BRIGHTNESS_WARN_RANGE = (110.0, 180.0)
FOCUS_GOOD_THRESHOLD = 1500.0
FOCUS_WARN_THRESHOLD = 800.0
EDGE_GOOD_RANGE = (5.0, 10.0)
EDGE_WARN_RANGE = (3.5, 15.0)


@dataclass(frozen=True)
class MetricChip:
    """Descriptor for a metric chip to render."""

    key: str
    label: str
    value: str
    status: str
    subtitle: str


def _brightness_status(value: float) -> MetricChip:
    if BRIGHTNESS_GOOD_RANGE[0] <= value <= BRIGHTNESS_GOOD_RANGE[1]:
        return MetricChip("brightness", "Brightness", f"{value:.0f}", "good", "OK")
    if BRIGHTNESS_WARN_RANGE[0] <= value <= BRIGHTNESS_WARN_RANGE[1]:
        return MetricChip("brightness", "Brightness", f"{value:.0f}", "warn", "ADJUST")
    return MetricChip("brightness", "Brightness", f"{value:.0f}", "bad", "FIX")


def _focus_status(value: float) -> MetricChip:
    if value >= FOCUS_GOOD_THRESHOLD:
        return MetricChip("focus", "Focus", f"{int(value):d}", "good", "SHARP")
    if value >= FOCUS_WARN_THRESHOLD:
        return MetricChip("focus", "Focus", f"{int(value):d}", "warn", "TUNE")
    return MetricChip("focus", "Focus", f"{int(value):d}", "bad", "SOFT")


def _edge_status(value: float) -> MetricChip:
    if EDGE_GOOD_RANGE[0] <= value <= EDGE_GOOD_RANGE[1]:
        return MetricChip("edge", "Edges", f"{value:.1f}%", "good", "OK")
    if EDGE_WARN_RANGE[0] <= value <= EDGE_WARN_RANGE[1]:
        label = "Edges"
        return MetricChip("edge", label, f"{value:.1f}%", "warn", "TUNE")
    return MetricChip("edge", "Edges", f"{value:.1f}%", "bad", "FIX")


def build_metric_chips(brightness: float, focus: float, edge: float) -> List[MetricChip]:
    """Return descriptors for brightness/focus/edge chips."""
    return [
        _brightness_status(brightness),
        _focus_status(focus),
        _edge_status(edge),
    ]


def summarise_quality(chips: List[MetricChip]) -> str:
    """Return aggregate quality flag (good/warn/bad)."""
    states = {chip.status for chip in chips}
    if "bad" in states:
        return "bad"
    if "warn" in states:
        return "warn"
    return "good"
