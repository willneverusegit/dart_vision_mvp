"""Shared helpers to classify HUD metrics for consistent chip styling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.game.game import DemoGame

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


class HudCardMode(Enum):
    """Logical groups for sidebar cards."""

    GAME = "game"
    MOTION = "motion"
    DEBUG = "debug"
    FULL = "full"


class HudCardPlacement(Enum):
    """Target location for rendering a HUD card."""

    SIDEBAR = "sidebar"
    ROI_TOP = "roi_top"
    ROI_BOTTOM = "roi_bottom"


@dataclass
class HudCardData:
    """Runtime data needed to populate HUD cards."""

    game: Optional['DemoGame'] = None
    last_msg: str = ""
    fps_stats: Optional[Dict[str, float]] = None
    total_darts: int = 0
    dart_config: Optional[object] = None
    current_preset: str = "balanced"
    motion_enabled: bool = False
    motion_detected: bool = False
    debug_enabled: bool = False


@dataclass(frozen=True)
class HudCardPayload:
    """Prepared payload for drawing a sidebar card."""

    key: str
    title: str
    rows: List[Dict[str, object]]
    footer: Optional[str] = None
    status: str = "info"


@dataclass(frozen=True)
class HudCardDefinition:
    """Static definition describing how to build a HUD card."""

    key: str
    modes: Tuple[HudCardMode, ...]
    builder: Callable[[HudCardData], Optional[HudCardPayload]]
    default_active: bool = True
    placement: HudCardPlacement = HudCardPlacement.SIDEBAR


@dataclass(frozen=True)
class HudSidebarSelection:
    """Result of resolving cards for the current frame."""

    cards: List[HudCardPayload]
    mode: Optional[HudCardMode]
    mode_chips: List[MetricChip]
    roi_top_cards: List[HudCardPayload] = field(default_factory=list)
    roi_bottom_cards: List[HudCardPayload] = field(default_factory=list)


MODE_ORDER: Tuple[HudCardMode, ...] = (
    HudCardMode.GAME,
    HudCardMode.MOTION,
    HudCardMode.DEBUG,
    HudCardMode.FULL,
)

MODE_LABELS: Dict[HudCardMode, str] = {
    HudCardMode.GAME: "Game",
    HudCardMode.MOTION: "Motion",
    HudCardMode.DEBUG: "Debug",
    HudCardMode.FULL: "Full",
}


def _build_game_card(data: HudCardData) -> Optional[HudCardPayload]:
    """Create the game status card."""

    game = data.game
    if game is None:
        return None

    from src.game.game import GameMode

    rows: List[Dict[str, object]] = []
    mode_display = "Around the Clock" if game.mode == GameMode.ATC else "301"
    rows.append({"label": "Mode", "value": mode_display, "status": "accent"})

    if game.mode == GameMode.ATC:
        target = getattr(game, "target", None)
        if getattr(game, "done", False):
            objective_value = "Finished run"
            objective_status = "good"
        else:
            objective_value = f"Next: {target}" if target is not None else "Next: --"
            objective_status = "accent"
    else:
        score = getattr(game, "score", None)
        if getattr(game, "done", False):
            objective_value = "Checkout"
            objective_status = "good"
        else:
            objective_value = f"{score} left" if score is not None else "--"
            if score is not None and score <= 100:
                objective_status = "warn"
            else:
                objective_status = "accent"

    rows.append(
        {
            "label": "Objective",
            "value": objective_value,
            "status": objective_status,
        }
    )

    last_label = ""
    last_points = 0
    if hasattr(game, "last") and game.last is not None:
        last_label = getattr(game.last, "label", "") or ""
        last_points = int(getattr(game.last, "points", 0) or 0)

    if last_label:
        last_value = f"{last_label} ({last_points})" if last_points else last_label
    else:
        last_value = "Waiting for hit"

    last_hint: Optional[str] = None
    if last_points and last_label:
        last_hint = f"{last_points} pts"
    elif not data.debug_enabled and data.total_darts:
        last_hint = f"{data.total_darts} impacts tracked"

    rows.append(
        {
            "label": "Last hit",
            "value": last_value,
            "status": "info",
            "hint": last_hint,
        }
    )

    footer_text = data.last_msg if data.last_msg else "Goal: reliable hit detection & scoring."

    return HudCardPayload(
        key="game",
        title="Game State",
        rows=rows,
        footer=footer_text,
        status="accent",
    )


def _build_debug_card(data: HudCardData) -> Optional[HudCardPayload]:
    """Create the system health card."""

    fps_value: Optional[float] = None
    frame_time: Optional[float] = None
    if data.fps_stats is not None:
        raw_fps = data.fps_stats.get("fps_median")
        raw_ft = data.fps_stats.get("frame_time_ms")
        if isinstance(raw_fps, (int, float)):
            fps_value = float(raw_fps)
        if isinstance(raw_ft, (int, float)):
            frame_time = float(raw_ft)

    fps_status = "info"
    fps_progress: Optional[float] = None
    if fps_value is not None:
        fps_progress = max(0.0, min(1.0, fps_value / 30.0))
        if fps_value >= 28.0:
            fps_status = "good"
        elif fps_value >= 22.0:
            fps_status = "warn"
        else:
            fps_status = "bad"

    frame_status = "info"
    if frame_time is not None:
        if frame_time <= 35.0:
            frame_status = "good"
        elif frame_time <= 45.0:
            frame_status = "warn"
        else:
            frame_status = "bad"

    rows: List[Dict[str, object]] = [
        {
            "label": "Median FPS",
            "value": f"{fps_value:.1f}" if fps_value is not None else "--",
            "status": fps_status,
            "progress": fps_progress,
            "hint": "≥30 keeps hits stable" if fps_progress is not None else None,
        },
        {
            "label": "Frame time",
            "value": f"{frame_time:.1f} ms" if frame_time is not None else "--",
            "status": frame_status,
        },
        {
            "label": "Impacts logged",
            "value": str(int(data.total_darts)),
            "status": "info",
            "hint": "Session total" if data.total_darts else None,
        },
    ]

    preset_display = data.current_preset.replace("_", " ").title()
    footer = f"Preset: {preset_display}"

    return HudCardPayload(
        key="debug",
        title="System Health",
        rows=rows,
        footer=footer,
        status=fps_status,
    )


def _build_motion_card(data: HudCardData) -> Optional[HudCardPayload]:
    """Create a motion diagnostics card."""

    has_config = data.dart_config is not None
    if not data.motion_enabled and not data.motion_detected and not has_config:
        return None

    rows: List[Dict[str, object]] = []
    rows.append(
        {
            "label": "Overlay",
            "value": "ON" if data.motion_enabled else "OFF",
            "status": "good" if data.motion_enabled else "info",
            "hint": "Hotkey: m",
        }
    )

    rows.append(
        {
            "label": "Detection",
            "value": "Motion" if data.motion_detected else "Idle",
            "status": "accent" if data.motion_detected else "info",
        }
    )

    if has_config:
        motion_parts: List[str] = []
        bias = getattr(data.dart_config, "motion_otsu_bias", None)
        if isinstance(bias, (int, float)):
            motion_parts.append(f"bias {int(round(bias)):+d}")
        open_k = getattr(data.dart_config, "morph_open_ksize", None)
        if isinstance(open_k, (int, float)):
            motion_parts.append(f"open {int(round(open_k))}")
        close_k = getattr(data.dart_config, "morph_close_ksize", None)
        if isinstance(close_k, (int, float)):
            motion_parts.append(f"close {int(round(close_k))}")
        min_white = getattr(data.dart_config, "motion_min_white_frac", None)
        if isinstance(min_white, (int, float)):
            motion_parts.append(f"white {min_white * 100:.0f}%")
        motion_summary = " | ".join(motion_parts) if motion_parts else "default"
        rows.append(
            {
                "label": "Tuning",
                "value": motion_summary,
                "status": "info",
            }
        )

    footer = "Keep motion stable to lock hits."
    status = "accent" if data.motion_enabled else "info"

    return HudCardPayload(
        key="motion",
        title="Motion Diagnostics",
        rows=rows,
        footer=footer,
        status=status,
    )


DEFAULT_CARD_DEFINITIONS: Tuple[HudCardDefinition, ...] = (
    HudCardDefinition(
        key="game",
        modes=(HudCardMode.GAME, HudCardMode.FULL),
        builder=_build_game_card,
        default_active=True,
        placement=HudCardPlacement.ROI_TOP,
    ),
    HudCardDefinition(
        key="motion",
        modes=(HudCardMode.MOTION, HudCardMode.DEBUG, HudCardMode.FULL),
        builder=_build_motion_card,
        default_active=True,
        placement=HudCardPlacement.ROI_BOTTOM,
    ),
    HudCardDefinition(
        key="debug",
        modes=(HudCardMode.DEBUG, HudCardMode.FULL),
        builder=_build_debug_card,
        default_active=True,
    ),
)


class CardManager:
    """Manage HUD card availability and tab navigation."""

    def __init__(
        self,
        definitions: Iterable[HudCardDefinition] = DEFAULT_CARD_DEFINITIONS,
        full_mode_values: Optional[Iterable[int]] = None,
    ) -> None:
        self._definitions: Tuple[HudCardDefinition, ...] = tuple(definitions)
        self._mode_enabled: Dict[HudCardMode, bool] = {mode: True for mode in MODE_ORDER}
        for definition in self._definitions:
            for mode in definition.modes:
                if mode not in self._mode_enabled:
                    self._mode_enabled[mode] = definition.default_active
        self._current_mode: Optional[HudCardMode] = None
        self._available_modes: List[HudCardMode] = []
        self._full_mode_values = set(full_mode_values or ())

    def focus_next_mode(self) -> None:
        """Advance to the next available mode tab."""

        self._cycle_mode(+1)

    def focus_prev_mode(self) -> None:
        """Go back to the previous available mode tab."""

        self._cycle_mode(-1)

    def _cycle_mode(self, direction: int) -> None:
        if not self._available_modes:
            return
        if self._current_mode not in self._available_modes:
            self._current_mode = self._available_modes[0]
            return
        idx = self._available_modes.index(self._current_mode)
        idx = (idx + direction) % len(self._available_modes)
        self._current_mode = self._available_modes[idx]

    def for_state(
        self,
        game_running: bool,
        motion_enabled: bool,
        debug_enabled: bool,
        overlay_mode: object,
        hud_data: Optional[HudCardData] = None,
    ) -> HudSidebarSelection:
        """Resolve which cards to draw for the current frame."""

        data = HudCardData(
            game=hud_data.game if hud_data else None,
            last_msg=hud_data.last_msg if hud_data else "",
            fps_stats=hud_data.fps_stats if hud_data else None,
            total_darts=hud_data.total_darts if hud_data else 0,
            dart_config=hud_data.dart_config if hud_data else None,
            current_preset=hud_data.current_preset if hud_data else "balanced",
            motion_enabled=bool(motion_enabled),
            motion_detected=hud_data.motion_detected if hud_data else False,
            debug_enabled=bool(debug_enabled),
        )

        active_modes = []
        if game_running:
            active_modes.append(HudCardMode.GAME)
        if motion_enabled:
            active_modes.append(HudCardMode.MOTION)
        if debug_enabled:
            active_modes.append(HudCardMode.DEBUG)

        full_active = False
        if isinstance(overlay_mode, bool):
            full_active = bool(overlay_mode)
        elif isinstance(overlay_mode, (int, float)):
            full_active = int(overlay_mode) in self._full_mode_values
        if full_active:
            active_modes.append(HudCardMode.FULL)

        self._available_modes = [
            mode
            for mode in MODE_ORDER
            if mode in active_modes and self._mode_enabled.get(mode, True)
        ]

        if not self._available_modes:
            self._current_mode = None
            return HudSidebarSelection(
                cards=[],
                mode=None,
                mode_chips=self._build_mode_chips(None),
                roi_top_cards=[],
                roi_bottom_cards=[],
            )

        if self._current_mode not in self._available_modes:
            self._current_mode = self._available_modes[0]

        sidebar_cards: List[HudCardPayload] = []
        roi_top_cards: List[HudCardPayload] = []
        roi_bottom_cards: List[HudCardPayload] = []
        for definition in self._definitions:
            if self._current_mode not in definition.modes:
                continue
            payload = definition.builder(data)
            if payload is not None:
                if definition.placement == HudCardPlacement.ROI_TOP:
                    roi_top_cards.append(payload)
                elif definition.placement == HudCardPlacement.ROI_BOTTOM:
                    roi_bottom_cards.append(payload)
                else:
                    sidebar_cards.append(payload)

        chips = self._build_mode_chips(self._current_mode)
        return HudSidebarSelection(
            cards=sidebar_cards,
            mode=self._current_mode,
            mode_chips=chips,
            roi_top_cards=roi_top_cards,
            roi_bottom_cards=roi_bottom_cards,
        )

    def _build_mode_chips(self, selected: Optional[HudCardMode]) -> List[MetricChip]:
        chips: List[MetricChip] = []
        available = set(self._available_modes)
        for mode in MODE_ORDER:
            label = MODE_LABELS[mode]
            if mode == selected:
                chips.append(MetricChip(mode.value, label, "ACTIVE", "accent", "[ ] to cycle"))
            elif mode in available:
                chips.append(MetricChip(mode.value, label, "READY", "good", None))
            else:
                chips.append(MetricChip(mode.value, label, "OFF", "info", None))
        return chips

    def current_mode_label(self) -> Optional[str]:
        """Return the human-readable label of the selected mode."""

        if self._current_mode is None:
            return None
        return MODE_LABELS.get(self._current_mode, self._current_mode.value.title())
