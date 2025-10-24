"""Detector configuration manager with atomic persistence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .motion_detector import MotionConfig
from .dart_impact_detector import DartDetectorConfig
from .config_schema import (
    DetectorConfigFile,
    MotionConfigSchema,
    DartDetectorConfigSchema,
    validate_config_file,
    save_config_atomic,
    create_default_config,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)


class DetectorConfigManager:
    """Load and persist detector configs via atomic YAML writes."""

    def __init__(self, config_path: Path | str = Path("config/detectors.yaml")):
        self.config_path = Path(config_path)
        self._config: Optional[DetectorConfigFile] = None

    def load(self) -> DetectorConfigFile:
        """Load config from disk (creating defaults if missing)."""
        if self._config is not None:
            return self._config

        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            create_default_config(self.config_path)

        self._config = validate_config_file(self.config_path)
        return self._config

    def refresh(self) -> DetectorConfigFile:
        """Force reload from disk."""
        self._config = None
        return self.load()

    def get_configs(self) -> Tuple[MotionConfig, DartDetectorConfig]:
        """Return Motion/Dart configs as dataclasses."""
        cfg = self.load()
        motion = MotionConfig.from_schema(cfg.motion)
        dart = DartDetectorConfig.from_schema(cfg.dart_detector)
        return motion, dart

    def get_presets(self) -> Dict[str, Dict[str, Any]]:
        """Return available presets (lower-case keys)."""
        cfg = self.load()
        presets = cfg.presets or {}
        return {name.lower(): values for name, values in presets.items()}

    def apply_preset(
        self,
        dart_config: DartDetectorConfig,
        preset_name: Optional[str],
    ) -> DartDetectorConfig:
        """Merge preset overrides into dart config."""
        if not preset_name:
            return dart_config

        presets = self.get_presets()
        preset = presets.get(preset_name.lower())
        if preset is None:
            logger.warning("Unknown detector preset '%s'", preset_name)
            return dart_config

        merged = dart_config.to_dict()
        merged.update(preset)
        return DartDetectorConfig(**merged)

    def save(
        self,
        motion_config: MotionConfig,
        dart_config: DartDetectorConfig,
        presets: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> DetectorConfigFile:
        """Persist configs atomically and update cache."""
        cfg = self.load()
        payload = DetectorConfigFile(
            schema_version=SCHEMA_VERSION,
            motion=MotionConfigSchema(**motion_config.to_dict()),
            dart_detector=DartDetectorConfigSchema(**dart_config.to_dict()),
            presets=presets if presets is not None else cfg.presets,
        )
        save_config_atomic(payload, self.config_path)
        self._config = payload
        logger.info("Detector configuration saved: %s", self.config_path)
        return payload

    def save_motion(self, motion_config: MotionConfig) -> DetectorConfigFile:
        """Persist only motion config."""
        _, dart = self.get_configs()
        return self.save(motion_config, dart)

    def save_dart(self, dart_config: DartDetectorConfig) -> DetectorConfigFile:
        """Persist only dart config."""
        motion, _ = self.get_configs()
        return self.save(motion, dart_config)
