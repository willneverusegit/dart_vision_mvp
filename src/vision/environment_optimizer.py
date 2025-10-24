"""Environment-aware optimizer for detector parameters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from .motion_detector import MotionConfig
from .dart_impact_detector import DartDetectorConfig

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentProfile:
    """Summary statistics describing current lighting and motion."""

    mean_brightness: float
    brightness_std: float
    motion_score: float
    noise_level: float
    frame_count: int
    recommended_preset: str


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class EnvironmentOptimizer:
    """Heuristic optimizer that adapts configs to observed environment."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze(self, frames: Iterable[np.ndarray]) -> EnvironmentProfile:
        """Compute brightness and motion statistics from sample frames."""
        frames_list: List[np.ndarray] = []
        for frame in frames:
            if frame is None:
                continue
            frames_list.append(frame)

        if not frames_list:
            raise ValueError("No frames supplied for environment analysis")

        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames_list]
        brightness = np.array([float(np.mean(g)) for g in grays], dtype=np.float32)
        diffs = []
        for prev, curr in zip(grays, grays[1:]):
            diff = cv2.absdiff(curr, prev)
            diffs.append(float(np.mean(diff)))
        motion_score = float(np.mean(diffs)) if diffs else 0.0
        noise_level = float(np.std(diffs)) if diffs else 0.0

        mean_brightness = float(np.mean(brightness))
        brightness_std = float(np.std(brightness))

        if mean_brightness < 75:
            preset = "aggressive"
        elif mean_brightness > 150:
            preset = "stable"
        else:
            preset = "balanced"

        return EnvironmentProfile(
            mean_brightness=mean_brightness,
            brightness_std=brightness_std,
            motion_score=motion_score,
            noise_level=noise_level,
            frame_count=len(frames_list),
            recommended_preset=preset,
        )

    def adjust(
        self,
        profile: EnvironmentProfile,
        motion_cfg: MotionConfig,
        dart_cfg: DartDetectorConfig,
    ) -> Tuple[MotionConfig, DartDetectorConfig]:
        """Return configs adjusted to better match observed environment."""
        motion = MotionConfig(**motion_cfg.to_dict())
        dart = DartDetectorConfig(**dart_cfg.to_dict())

        # Lighting adjustments
        if profile.mean_brightness < 75:
            motion.var_threshold = int(_clamp(motion.var_threshold * 0.9, 10, 200))
            motion.motion_pixel_threshold = int(_clamp(motion.motion_pixel_threshold * 0.8, 100, 5000))
            dart.motion_otsu_bias = int(_clamp(dart.motion_otsu_bias * 0.8, -30, 30))
            dart.min_area = int(_clamp(dart.min_area * 0.9, 5, dart.max_area))
        elif profile.mean_brightness > 150:
            motion.var_threshold = int(_clamp(motion.var_threshold * 1.1, 10, 200))
            motion.motion_pixel_threshold = int(_clamp(motion.motion_pixel_threshold * 1.2, 100, 5000))
            dart.motion_otsu_bias = int(_clamp(dart.motion_otsu_bias * 1.2, -30, 30))
            dart.min_area = int(_clamp(dart.min_area * 1.1, 5, dart.max_area))

        # Noise adjustments
        if profile.noise_level > 12.0:
            motion.min_contour_area = int(_clamp(motion.min_contour_area * 1.2, 10, motion.max_contour_area))
            dart.min_solidity = float(_clamp(dart.min_solidity * 1.05, 0.05, dart.max_solidity))
            dart.confirmation_frames = int(_clamp(dart.confirmation_frames + 1, 1, 10))
        elif profile.noise_level < 5.0:
            motion.min_contour_area = int(_clamp(motion.min_contour_area * 0.9, 10, motion.max_contour_area))
            dart.confirmation_frames = int(_clamp(dart.confirmation_frames - 1, 1, 10))

        # Motion score influences cooldown
        if profile.motion_score > 10.0:
            dart.cooldown_frames = int(_clamp(dart.cooldown_frames * 1.2, 5, 100))
        elif profile.motion_score < 3.0:
            dart.cooldown_frames = int(_clamp(dart.cooldown_frames * 0.9, 5, 100))

        return motion, dart

    def optimize(
        self,
        frames: Iterable[np.ndarray],
        motion_cfg: MotionConfig,
        dart_cfg: DartDetectorConfig,
    ) -> Tuple[MotionConfig, DartDetectorConfig, EnvironmentProfile]:
        """Shortcut: analyze frames and return adjusted configs + profile."""
        profile = self.analyze(frames)
        motion, dart = self.adjust(profile, motion_cfg, dart_cfg)
        self.logger.info(
            "Environment profile → brightness=%.1f±%.1f, motion=%.2f, noise=%.2f, preset=%s",
            profile.mean_brightness,
            profile.brightness_std,
            profile.motion_score,
            profile.noise_level,
            profile.recommended_preset,
        )
        return motion, dart, profile
