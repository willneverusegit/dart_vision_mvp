"""Guided parameter tuner with atomic config integration."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.vision.motion_detector import MotionDetector, MotionConfig
from src.vision.dart_impact_detector import DartImpactDetector, DartDetectorConfig, DartImpact
from src.vision.detector_config_manager import DetectorConfigManager
from src.vision.environment_optimizer import EnvironmentOptimizer, EnvironmentProfile

logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    video_source: str = ""
    webcam_index: int = 0
    use_webcam: bool = False
    start_frame: int = 0
    loop_video: bool = True
    window_width: int = 1600
    window_height: int = 900
    detector_config: Path = Path("config/detectors.yaml")
    optimize_samples: int = 120


@dataclass
class TrackbarSpec:
    label: str
    attr: str
    min_value: float
    max_value: float
    step: float
    target: str  # 'motion' or 'dart'
    dtype: str = "float"
    to_config: Optional[Callable[[float], float]] = None
    from_config: Optional[Callable[[float], float]] = None

    @property
    def max_position(self) -> int:
        return int(round((self.max_value - self.min_value) / self.step))

    def to_position(self, value: float) -> int:
        if self.from_config:
            value = self.from_config(value)
        if self.dtype == "bool":
            return 1 if bool(value) else 0
        pos = int(round((value - self.min_value) / self.step))
        return int(np.clip(pos, 0, self.max_position))

    def to_value(self, position: int) -> float:
        position = int(np.clip(position, 0, self.max_position))
        if self.dtype == "bool":
            value = bool(position)
        else:
            value = self.min_value + position * self.step
            if self.dtype == "int":
                value = int(round(value))
            else:
                value = float(value)
        if self.to_config:
            value = self.to_config(value)
        return value


@dataclass
class ParameterGroup:
    name: str
    description: str
    specs: List[TrackbarSpec]


class ParameterTuner:
    """Interactive guided tuner with presets and auto-optimization."""

    def __init__(self, config: TuningConfig):
        self.config = config
        self.logger = logging.getLogger("ParameterTuner")

        self.manager = DetectorConfigManager(config.detector_config)
        self.optimizer = EnvironmentOptimizer()

        self.motion_cfg: MotionConfig
        self.dart_cfg: DartDetectorConfig
        self.motion_detector: Optional[MotionDetector] = None
        self.dart_detector: Optional[DartImpactDetector] = None

        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 30.0
        self.paused = True
        self.frame_step = 0
        self.loop_video = config.loop_video
        self.playback_speed = 1.0

        self.main_window = "Dart Detection"
        self.controls_window = "Parameter Controls"
        self.playback_window = "Playback"
        self.dashboard_window = "Dashboard"

        self.groups: List[ParameterGroup] = []
        self.group_index = 0
        self.trackbar_specs: Dict[str, TrackbarSpec] = {}

        self.last_changed_param: Optional[str] = None
        self.param_change_time: float = 0.0
        self.config_dirty = False
        self.env_profile: Optional[EnvironmentProfile] = None

        self.stats = {
            "frames": 0,
            "motion": 0,
            "darts": 0,
        }

        self.last_frame: Optional[np.ndarray] = None
        self.last_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        if not self._open_source():
            return False

        self.motion_cfg, self.dart_cfg = self.manager.get_configs()
        self._define_groups()
        self._update_detectors(force=True)

        cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.main_window, self.config.window_width, self.config.window_height)

        self._create_playback_controls()
        self._create_group_controls()
        cv2.namedWindow(self.dashboard_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.dashboard_window, 520, 820)

        if self.config.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.start_frame)
            self.current_frame_idx = self.config.start_frame

        self.logger.info("Initialization complete. SPACE=play/pause, arrows=switch groups, o=optimize")
        return True

    def _open_source(self) -> bool:
        if self.config.use_webcam:
            self.cap = cv2.VideoCapture(self.config.webcam_index)
        else:
            if not Path(self.config.video_source).exists():
                self.logger.error("Video file not found: %s", self.config.video_source)
                return False
            self.cap = cv2.VideoCapture(self.config.video_source)

        if not self.cap or not self.cap.isOpened():
            self.logger.error("Failed to open video source")
            return False

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        return True

    def _define_groups(self) -> None:
        motion_group = ParameterGroup(
            name="Motion Sensitivity",
            description="Steuert, wann Bewegung als relevant gilt.",
            specs=[
                TrackbarSpec("Var Threshold", "var_threshold", 10, 200, 1, "motion", "int"),
                TrackbarSpec("Min Motion Pixels", "motion_pixel_threshold", 100, 5000, 50, "motion", "int"),
                TrackbarSpec("Min Contour Area", "min_contour_area", 10, 1000, 10, "motion", "int"),
                TrackbarSpec("Max Contour Area", "max_contour_area", 500, 30000, 250, "motion", "int"),
                TrackbarSpec(
                    "Morph Kernel", "morph_kernel_size", 1, 15, 2, "motion", "int",
                    to_config=lambda v: int(v) | 1,
                    from_config=lambda v: v
                ),
            ],
        )

        adaptive_group = ParameterGroup(
            name="Motion Adaptivity",
            description="Anpassung an Licht & Ruhephasen.",
            specs=[
                TrackbarSpec("Adaptive Otsu", "adaptive_otsu_enabled", 0, 1, 1, "motion", "bool"),
                TrackbarSpec("Dark Threshold", "brightness_dark_threshold", 20, 120, 5, "motion"),
                TrackbarSpec("Bright Threshold", "brightness_bright_threshold", 140, 240, 5, "motion"),
                TrackbarSpec("Bias Dark", "otsu_bias_dark", -40, 0, 1, "motion", "int"),
                TrackbarSpec("Bias Normal", "otsu_bias_normal", -20, 20, 1, "motion", "int"),
                TrackbarSpec("Bias Bright", "otsu_bias_bright", 0, 40, 1, "motion", "int"),
                TrackbarSpec("Search Trigger", "search_mode_trigger_frames", 30, 240, 10, "motion", "int"),
                TrackbarSpec("Search Drop", "search_mode_threshold_drop", 50, 300, 10, "motion", "int"),
            ],
        )

        dart_shape = ParameterGroup(
            name="Dart Shape",
            description="Konturen- und Formfilter.",
            specs=[
                TrackbarSpec("Min Area", "min_area", 5, 1500, 5, "dart", "int"),
                TrackbarSpec("Max Area", "max_area", 200, 4000, 20, "dart", "int"),
                TrackbarSpec(
                    "Min AR (x100)", "min_aspect_ratio", 10, 80, 1, "dart",
                    to_config=lambda v: round(v / 100.0, 2),
                    from_config=lambda v: v * 100.0
                ),
                TrackbarSpec(
                    "Max AR (x100)", "max_aspect_ratio", 120, 400, 1, "dart",
                    to_config=lambda v: round(v / 100.0, 2),
                    from_config=lambda v: v * 100.0
                ),
                TrackbarSpec(
                    "Min Solidity", "min_solidity", 5, 95, 1, "dart",
                    to_config=lambda v: round(v / 100.0, 2),
                    from_config=lambda v: v * 100.0
                ),
                TrackbarSpec(
                    "Max Solidity", "max_solidity", 50, 99, 1, "dart",
                    to_config=lambda v: round(v / 100.0, 2),
                    from_config=lambda v: v * 100.0
                ),
                TrackbarSpec(
                    "Min Extent", "min_extent", 5, 60, 1, "dart",
                    to_config=lambda v: round(v / 100.0, 2),
                    from_config=lambda v: v * 100.0
                ),
                TrackbarSpec(
                    "Max Extent", "max_extent", 30, 90, 1, "dart",
                    to_config=lambda v: round(v / 100.0, 2),
                    from_config=lambda v: v * 100.0
                ),
                TrackbarSpec(
                    "Convexity (x100)", "convexity_min_ratio", 50, 95, 1, "dart",
                    to_config=lambda v: round(v / 100.0, 2),
                    from_config=lambda v: v * 100.0
                ),
            ],
        )

        temporal_group = ParameterGroup(
            name="Temporal Logic",
            description="Bestätigung & Cooldown.",
            specs=[
                TrackbarSpec("Confirm Frames", "confirmation_frames", 1, 8, 1, "dart", "int"),
                TrackbarSpec("Pos Tolerance", "position_tolerance_px", 10, 50, 2, "dart", "int"),
                TrackbarSpec("Cooldown Frames", "cooldown_frames", 10, 120, 5, "dart", "int"),
                TrackbarSpec("Cooldown Radius", "cooldown_radius_px", 20, 120, 5, "dart", "int"),
                TrackbarSpec("History Size", "candidate_history_size", 10, 80, 5, "dart", "int"),
            ],
        )

        refine_group = ParameterGroup(
            name="Refinement",
            description="Feinjustierung der Trefferposition.",
            specs=[
                TrackbarSpec("Refine Enabled", "refine_enabled", 0, 1, 1, "dart", "bool"),
                TrackbarSpec(
                    "Refine Threshold", "refine_threshold", 10, 90, 1, "dart",
                    to_config=lambda v: round(v / 100.0, 2),
                    from_config=lambda v: v * 100.0
                ),
                TrackbarSpec("Refine ROI", "refine_roi_size_px", 40, 160, 2, "dart", "int"),
                TrackbarSpec("Tip Enabled", "tip_refine_enabled", 0, 1, 1, "dart", "bool"),
                TrackbarSpec("Tip ROI", "tip_roi_px", 16, 80, 2, "dart", "int"),
                TrackbarSpec("Tip Search", "tip_search_px", 6, 40, 1, "dart", "int"),
                TrackbarSpec(
                    "Tip Edge", "tip_edge_weight", 20, 90, 1, "dart",
                    to_config=lambda v: round(v / 100.0, 2),
                    from_config=lambda v: v * 100.0
                ),
                TrackbarSpec(
                    "Tip Dark", "tip_dark_weight", 10, 90, 1, "dart",
                    to_config=lambda v: round(v / 100.0, 2),
                    from_config=lambda v: v * 100.0
                ),
            ],
        )

        self.groups = [motion_group, adaptive_group, dart_shape, temporal_group, refine_group]

    def _create_playback_controls(self) -> None:
        cv2.namedWindow(self.playback_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.playback_window, 420, 120)
        cv2.createTrackbar("Paused", self.playback_window, 1, 1, lambda x: self._set_paused(bool(x)))
        if self.total_frames > 0:
            cv2.createTrackbar("Frame", self.playback_window, self.current_frame_idx,
                               max(1, self.total_frames - 1), self._seek_frame)
        cv2.createTrackbar("Speed x10", self.playback_window, 10, 30, self._change_speed)

    def _create_group_controls(self) -> None:
        try:
            cv2.destroyWindow(self.controls_window)
        except cv2.error:
            pass
        cv2.namedWindow(self.controls_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.controls_window, 400, 640)

        self.trackbar_specs.clear()
        group = self.groups[self.group_index]
        for spec in group.specs:
            target = self.motion_cfg if spec.target == "motion" else self.dart_cfg
            current_value = getattr(target, spec.attr)
            pos = spec.to_position(current_value)
            cv2.createTrackbar(spec.label, self.controls_window, pos, spec.max_position,
                               self._make_trackbar_callback(spec))
            self.trackbar_specs[spec.label] = spec

    def _make_trackbar_callback(self, spec: TrackbarSpec) -> Callable[[int], None]:
        def _callback(pos: int) -> None:
            value = spec.to_value(pos)
            target = self.motion_cfg if spec.target == "motion" else self.dart_cfg
            setattr(target, spec.attr, value)
            self.config_dirty = True
            self.last_changed_param = spec.label
            self.param_change_time = time.time()
        return _callback

    def _set_paused(self, paused: bool) -> None:
        self.paused = paused

    def _seek_frame(self, value: int) -> None:
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
        self.current_frame_idx = value
        self.frame_step = 1

    def _change_speed(self, value: int) -> None:
        self.playback_speed = max(0.1, value / 10.0)

    # ------------------------------------------------------------------
    # Runtime loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        if not self.initialize():
            return

        while True:
            frame = self._read_frame()
            if frame is None:
                break

            if self.config_dirty:
                self._update_detectors()
                self._sync_trackbars()

            motion_detected, impact, mask = self._process_frame(frame)
            self._render_dashboard(motion_detected, impact)

            display = self._compose_display(frame, mask, impact)
            cv2.imshow(self.main_window, display)

            key = cv2.waitKeyEx(1)
            if key != -1:
                if self._handle_key(key):
                    break

        self.cleanup()

    def _read_frame(self) -> Optional[np.ndarray]:
        if not self.cap:
            return None

        if not self.paused or self.frame_step > 0:
            ret, frame = self.cap.read()
            if not ret:
                if not self.config.use_webcam and self.loop_video:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_idx = 0
                    ret, frame = self.cap.read()
                    if not ret:
                        return None
                else:
                    return None
            self.current_frame_idx += 1
            self.frame_step = max(0, self.frame_step - 1)
            if self.total_frames > 0:
                cv2.setTrackbarPos("Frame", self.playback_window,
                                   min(self.current_frame_idx, self.total_frames - 1))
            self.last_frame = frame.copy()
            return frame
        return self.last_frame

    def _update_detectors(self, force: bool = False) -> None:
        if not force and not self.config_dirty:
            return
        self.motion_detector = MotionDetector(self.motion_cfg)
        self.dart_detector = DartImpactDetector(self.dart_cfg)
        self.config_dirty = False

    def _sync_trackbars(self) -> None:
        group = self.groups[self.group_index]
        for spec in group.specs:
            current_value = getattr(self.motion_cfg if spec.target == "motion" else self.dart_cfg, spec.attr)
            pos = spec.to_position(current_value)
            cv2.setTrackbarPos(spec.label, self.controls_window, pos)

    def _process_frame(self, frame: np.ndarray) -> Tuple[bool, Optional[DartImpact], Optional[np.ndarray]]:
        if not self.motion_detector or not self.dart_detector:
            return False, None, None

        timestamp = self.current_frame_idx / max(self.fps, 1e-3)
        motion, event, mask = self.motion_detector.detect_motion(frame, self.current_frame_idx, timestamp)
        if motion:
            self.stats["motion"] += 1
        impact = None
        if motion:
            impact = self.dart_detector.detect_dart(frame, mask, self.current_frame_idx, timestamp)
            if impact:
                self.stats["darts"] += 1
        self.stats["frames"] += 1
        self.last_mask = mask
        return motion, impact, mask

    def _compose_display(self, frame: np.ndarray, mask: Optional[np.ndarray], impact: Optional[DartImpact]) -> np.ndarray:
        display = frame.copy()
        if impact:
            cv2.circle(display, impact.position, 10, (0, 255, 0), 2)
            cv2.putText(display, f"Dart {impact.confidence:.2f}", (impact.position[0] + 5, impact.position[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        if mask is not None:
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_rgb = cv2.applyColorMap(mask, cv2.COLORMAP_MAGMA)
            mask_rgb = cv2.resize(mask_rgb, (frame.shape[1] // 3, frame.shape[0] // 3))
            h, w, _ = mask_rgb.shape
            display[0:h, 0:w] = mask_rgb
        preset_names = ", ".join(self.manager.get_presets().keys()) or "none"
        overlay_text = (
            f"Presets: {preset_names} | Group [{self.group_index + 1}/{len(self.groups)}]"
        )
        cv2.putText(display, overlay_text, (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return display

    def _render_dashboard(self, motion_detected: bool, impact: Optional[DartImpact]) -> None:
        canvas = np.zeros((820, 520, 3), dtype=np.uint8)
        canvas[:] = (25, 25, 25)
        y = 30
        cv2.putText(canvas, "PARAMETER DASHBOARD", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += 30

        group = self.groups[self.group_index]
        cv2.putText(canvas, f"Group: {group.name}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (180, 220, 255), 2, cv2.LINE_AA)
        y += 28
        cv2.putText(canvas, group.description, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180), 1, cv2.LINE_AA)
        y += 32

        for spec in group.specs:
            current_value = getattr(self.motion_cfg if spec.target == "motion" else self.dart_cfg, spec.attr)
            label = spec.label
            if label.endswith("x100") or label.startswith("Tip Edge") or label.startswith("Tip Dark") or label == "Refine Threshold":
                display_value = f"{current_value:.2f}"
            else:
                display_value = f"{current_value}"
            highlight = (self.last_changed_param == spec.label and
                         (time.time() - self.param_change_time) < 2.0)
            color = (0, 255, 0) if highlight else (220, 220, 220)
            cv2.putText(canvas, f"{label:>20}: {display_value}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1 + int(highlight), cv2.LINE_AA)
            y += 24

        y += 12
        cv2.line(canvas, (20, y), (500, y), (80, 80, 80), 1)
        y += 24
        status_text = f"Frame {self.current_frame_idx} | Motion={'YES' if motion_detected else 'no'} | Darts={self.stats['darts']}"
        cv2.putText(canvas, status_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        y += 26

        if self.env_profile:
            cv2.putText(canvas, "Environment", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (100, 200, 255), 2, cv2.LINE_AA)
            y += 24
            cv2.putText(canvas,
                        f"Brightness: {self.env_profile.mean_brightness:.1f} ± {self.env_profile.brightness_std:.1f}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            y += 22
            cv2.putText(canvas, f"Motion score: {self.env_profile.motion_score:.2f} | Noise: {self.env_profile.noise_level:.2f}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            y += 22
            cv2.putText(canvas, f"Suggested preset: {self.env_profile.recommended_preset}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            y += 26

        tips = [
            "SPACE: Play/Pause | Arrows: Group | 1-5: Jump group",
            "S: Save YAML | R: Reload YAML | O: Auto optimize",
            "P: Cycle presets | Q: Quit",
            "Ziel: Zuverlässige Treffererkennung & korrekte Punkte!",
        ]
        for tip in tips:
            cv2.putText(canvas, tip, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (160, 160, 160), 1, cv2.LINE_AA)
            y += 20

        cv2.imshow(self.dashboard_window, canvas)

    def _handle_key(self, key: int) -> bool:
        if key in (ord('q'), 27):
            return True
        if key == ord(' '):
            self.paused = not self.paused
            cv2.setTrackbarPos("Paused", self.playback_window, int(self.paused))
        elif key == ord('.'):
            self.frame_step = 1
            self.paused = False
        elif key == ord(','):
            if self.cap and self.current_frame_idx > 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, self.current_frame_idx - 2))
                self.current_frame_idx = max(0, self.current_frame_idx - 2)
                self.frame_step = 1
                self.paused = False
        elif key in (ord('['), 2424832):  # left arrow or '['
            self.group_index = (self.group_index - 1) % len(self.groups)
            self._create_group_controls()
        elif key in (ord(']'), 2555904):  # right arrow or ']'
            self.group_index = (self.group_index + 1) % len(self.groups)
            self._create_group_controls()
        elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            idx = key - ord('1')
            if 0 <= idx < len(self.groups):
                self.group_index = idx
                self._create_group_controls()
        elif key == ord('s'):
            self._save_config()
        elif key == ord('r'):
            self._reload_config()
        elif key == ord('o'):
            self._auto_optimize()
        elif key == ord('p'):
            self._cycle_preset()
        return False

    def _auto_optimize(self) -> None:
        frames: List[np.ndarray] = []
        if self.last_frame is not None:
            frames.append(self.last_frame)
        sample_target = max(30, self.config.optimize_samples)
        while len(frames) < sample_target:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        if not frames:
            self.logger.warning("Auto optimize skipped (no frames)")
            return
        motion_cfg, dart_cfg, profile = self.optimizer.optimize(frames, self.motion_cfg, self.dart_cfg)
        self.motion_cfg = motion_cfg
        self.dart_cfg = dart_cfg
        self.env_profile = profile
        self.config_dirty = True
        self.logger.info("Auto optimization applied: preset %s", profile.recommended_preset)

        if not self.config.use_webcam and self.cap:
            rewind = max(0, self.current_frame_idx - len(frames))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, rewind)
            self.current_frame_idx = rewind
            self.frame_step = 1

    def _cycle_preset(self) -> None:
        presets = list(self.manager.get_presets().keys())
        if not presets:
            self.logger.warning("No presets available in detector config")
            return
        if not hasattr(self, "_preset_index"):
            self._preset_index = 0
        else:
            self._preset_index = (self._preset_index + 1) % len(presets)
        name = presets[self._preset_index]
        self.dart_cfg = self.manager.apply_preset(self.dart_cfg, name)
        self.config_dirty = True
        self.logger.info("Preset applied: %s", name)

    def _save_config(self) -> None:
        self.manager.save(self.motion_cfg, self.dart_cfg)
        self.logger.info("Configuration saved to %s", self.manager.config_path)

    def _reload_config(self) -> None:
        cfg = self.manager.refresh()
        self.motion_cfg = MotionConfig.from_schema(cfg.motion)
        self.dart_cfg = DartDetectorConfig.from_schema(cfg.dart_detector)
        self.config_dirty = True
        self.logger.info("Configuration reloaded from disk")

    def cleanup(self) -> None:
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Tuning session ended")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guided Dart Detection Parameter Tuner")
    parser.add_argument("--video", "-v", type=str, help="Video file path")
    parser.add_argument("--webcam", "-w", type=int, default=0, help="Webcam index")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame")
    parser.add_argument("--detector-config", type=str, default="config/detectors.yaml",
                        help="Detector config YAML")
    parser.add_argument("--optimize-samples", type=int, default=120, help="Frames for auto optimizer")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    tuning_cfg = TuningConfig(
        video_source=args.video or "",
        webcam_index=args.webcam,
        use_webcam=(args.video is None),
        start_frame=args.start_frame,
        detector_config=Path(args.detector_config),
        optimize_samples=args.optimize_samples,
    )

    tuner = ParameterTuner(tuning_cfg)
    tuner.run()


if __name__ == "__main__":
    main()
