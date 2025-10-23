"""
Preset Manager - Load, save, and manage detection parameter presets.

Provides persistent storage and management of detection configurations:
- Save custom parameter sets
- Load presets from YAML files
- Compare presets
- Export/import between systems
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


logger = logging.getLogger(__name__)


class PresetManager:
    """
    Manage detection parameter presets.

    Features:
    - Save/load presets from YAML files
    - Organize presets by category
    - Version control for presets
    - Preset comparison and diff
    """

    def __init__(self, preset_dir: Path):
        """
        Initialize preset manager.

        Args:
            preset_dir: Directory for storing presets
        """
        self.preset_dir = Path(preset_dir)
        self.preset_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("PresetManager")
        self.logger.info(f"Preset directory: {self.preset_dir}")

    def save_preset(self, name: str, preset_data: Dict[str, Any]) -> Path:
        """
        Save preset to YAML file.

        Args:
            name: Preset name (will be sanitized for filename)
            preset_data: Dictionary with 'motion' and 'dart' configs

        Returns:
            Path to saved preset file
        """
        # Sanitize name for filename
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        filename = f"{safe_name}.yaml"
        filepath = self.preset_dir / filename

        # Add metadata
        preset_with_meta = {
            "name": name,
            "created": datetime.now().isoformat(),
            "version": "1.0",
            **preset_data,
        }

        # Save to YAML
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(preset_with_meta, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"Saved preset: {filepath}")
        return filepath

    def load_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load preset from YAML file.

        Args:
            name: Preset name or filename

        Returns:
            Preset data dictionary or None if not found
        """
        # Try direct filename first
        filepath = self.preset_dir / name
        if not filepath.exists():
            # Try with .yaml extension
            filepath = self.preset_dir / f"{name}.yaml"

        if not filepath.exists():
            self.logger.warning(f"Preset not found: {name}")
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                preset_data = yaml.safe_load(f)
            self.logger.info(f"Loaded preset: {filepath}")
            return preset_data
        except Exception as e:
            self.logger.error(f"Failed to load preset {name}: {e}")
            return None

    def list_presets(self) -> List[str]:
        """
        List all available presets.

        Returns:
            List of preset names (without .yaml extension)
        """
        presets = []
        for filepath in self.preset_dir.glob("*.yaml"):
            presets.append(filepath.stem)
        return sorted(presets)

    def delete_preset(self, name: str) -> bool:
        """
        Delete a preset file.

        Args:
            name: Preset name

        Returns:
            True if deleted, False if not found
        """
        filepath = self.preset_dir / f"{name}.yaml"
        if filepath.exists():
            filepath.unlink()
            self.logger.info(f"Deleted preset: {filepath}")
            return True
        return False

    def export_preset(self, name: str, output_path: Path) -> bool:
        """
        Export preset to a different location.

        Args:
            name: Preset name
            output_path: Destination file path

        Returns:
            True if successful
        """
        preset_data = self.load_preset(name)
        if preset_data is None:
            return False

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(preset_data, f, default_flow_style=False)
            self.logger.info(f"Exported preset to: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export preset: {e}")
            return False

    def import_preset(self, source_path: Path, name: Optional[str] = None) -> bool:
        """
        Import preset from external file.

        Args:
            source_path: Source file path
            name: Optional new name (defaults to source filename)

        Returns:
            True if successful
        """
        if not source_path.exists():
            self.logger.error(f"Source file not found: {source_path}")
            return False

        try:
            with open(source_path, "r", encoding="utf-8") as f:
                preset_data = yaml.safe_load(f)

            preset_name = name or source_path.stem
            self.save_preset(preset_name, preset_data)
            return True
        except Exception as e:
            self.logger.error(f"Failed to import preset: {e}")
            return False

    def compare_presets(self, name1: str, name2: str) -> Dict[str, Any]:
        """
        Compare two presets and return differences.

        Args:
            name1: First preset name
            name2: Second preset name

        Returns:
            Dictionary with differences
        """
        preset1 = self.load_preset(name1)
        preset2 = self.load_preset(name2)

        if preset1 is None or preset2 is None:
            return {"error": "One or both presets not found"}

        differences = {}

        # Compare motion configs
        if "motion" in preset1 and "motion" in preset2:
            motion_diff = self._dict_diff(preset1["motion"], preset2["motion"])
            if motion_diff:
                differences["motion"] = motion_diff

        # Compare dart configs
        if "dart" in preset1 and "dart" in preset2:
            dart_diff = self._dict_diff(preset1["dart"], preset2["dart"])
            if dart_diff:
                differences["dart"] = dart_diff

        return differences

    def _dict_diff(self, dict1: Dict, dict2: Dict) -> Dict[str, Any]:
        """
        Find differences between two dictionaries.

        Args:
            dict1: First dictionary
            dict2: Second dictionary

        Returns:
            Dictionary with differences
        """
        diff = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())

        for key in all_keys:
            val1 = dict1.get(key)
            val2 = dict2.get(key)

            if val1 != val2:
                diff[key] = {"preset1": val1, "preset2": val2}

        return diff

    def create_default_presets(self):
        """Create default presets for quick start"""
        # These match the built-in presets in dart_impact_detector.py
        default_presets = {
            "aggressive": {
                "motion": {
                    "var_threshold": 40,
                    "motion_pixel_threshold": 400,
                    "morph_kernel_size": 3,
                },
                "dart": {
                    "min_area": 6,
                    "max_area": 1600,
                    "min_aspect_ratio": 0.25,
                    "max_aspect_ratio": 3.6,
                    "min_solidity": 0.08,
                    "max_solidity": 0.98,
                    "confirmation_frames": 2,
                    "convexity_min_ratio": 0.65,
                },
            },
            "balanced": {
                "motion": {
                    "var_threshold": 50,
                    "motion_pixel_threshold": 500,
                    "morph_kernel_size": 5,
                },
                "dart": {
                    "min_area": 10,
                    "max_area": 1100,
                    "min_aspect_ratio": 0.30,
                    "max_aspect_ratio": 3.0,
                    "min_solidity": 0.10,
                    "max_solidity": 0.95,
                    "confirmation_frames": 3,
                    "convexity_min_ratio": 0.70,
                },
            },
            "stable": {
                "motion": {
                    "var_threshold": 60,
                    "motion_pixel_threshold": 600,
                    "morph_kernel_size": 7,
                },
                "dart": {
                    "min_area": 14,
                    "max_area": 900,
                    "min_aspect_ratio": 0.34,
                    "max_aspect_ratio": 2.6,
                    "min_solidity": 0.12,
                    "max_solidity": 0.92,
                    "confirmation_frames": 4,
                    "convexity_min_ratio": 0.75,
                },
            },
        }

        for name, data in default_presets.items():
            self.save_preset(name, data)

        self.logger.info("Created default presets")
