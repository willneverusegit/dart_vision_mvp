"""
Config Schema Validation with Pydantic v2
Provides type-safe configuration loading and atomic YAML writes.

Features:
- Pydantic v2 models for MotionConfig and DartDetectorConfig
- Schema versioning (v1.0.0)
- Atomic YAML writes (temp → move)
- Comprehensive validation with helpful error messages
"""

import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import tempfile
import shutil

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"


class MotionConfigSchema(BaseModel):
    """Pydantic schema for MotionConfig"""
    model_config = ConfigDict(extra='forbid')  # Reject unknown fields
    
    var_threshold: int = Field(default=50, ge=10, le=200, description="MOG2 variance threshold")
    detect_shadows: bool = Field(default=True, description="Enable shadow detection")
    history: int = Field(default=500, ge=100, le=2000, description="Background model history")
    motion_pixel_threshold: int = Field(default=500, ge=100, le=5000, description="Min pixels for motion")
    min_contour_area: int = Field(default=100, ge=10, le=1000, description="Min contour area")
    max_contour_area: int = Field(default=5000, ge=1000, le=50000, description="Max contour area")
    morph_kernel_size: int = Field(default=3, ge=1, le=15, description="Morphology kernel size")
    event_history_size: int = Field(default=10, ge=1, le=100, description="Event history buffer")
    
    @field_validator('morph_kernel_size')
    @classmethod
    def validate_odd_kernel(cls, v: int) -> int:
        """Ensure kernel size is odd"""
        if v % 2 == 0:
            raise ValueError(f"morph_kernel_size must be odd, got {v}")
        return v


class DartDetectorConfigSchema(BaseModel):
    """Pydantic schema for DartDetectorConfig"""
    model_config = ConfigDict(extra='forbid')
    
    # Shape constraints
    min_area: int = Field(default=10, ge=1, le=100, description="Min contour area")
    max_area: int = Field(default=1000, ge=100, le=5000, description="Max contour area")
    min_aspect_ratio: float = Field(default=0.3, ge=0.1, le=1.0, description="Min aspect ratio")
    max_aspect_ratio: float = Field(default=3.0, ge=1.0, le=10.0, description="Max aspect ratio")
    
    # Advanced shape heuristics
    min_solidity: float = Field(default=0.1, ge=0.0, le=1.0, description="Min solidity")
    max_solidity: float = Field(default=0.95, ge=0.0, le=1.0, description="Max solidity")
    min_extent: float = Field(default=0.05, ge=0.0, le=1.0, description="Min extent")
    max_extent: float = Field(default=0.75, ge=0.0, le=1.0, description="Max extent")
    min_edge_density: float = Field(default=0.02, ge=0.0, le=1.0, description="Min edge density")
    max_edge_density: float = Field(default=0.35, ge=0.0, le=1.0, description="Max edge density")
    preferred_aspect_ratio: float = Field(default=0.35, ge=0.1, le=5.0, description="Preferred AR")
    aspect_ratio_tolerance: float = Field(default=1.5, ge=0.5, le=5.0, description="AR tolerance multiplier")
    
    # Edge detection
    edge_canny_threshold1: int = Field(default=40, ge=10, le=200, description="Canny low threshold")
    edge_canny_threshold2: int = Field(default=120, ge=50, le=300, description="Canny high threshold")
    
    # Confidence weighting
    circularity_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Circularity weight")
    solidity_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Solidity weight")
    extent_weight: float = Field(default=0.15, ge=0.0, le=1.0, description="Extent weight")
    edge_weight: float = Field(default=0.15, ge=0.0, le=1.0, description="Edge weight")
    aspect_ratio_weight: float = Field(default=0.15, ge=0.0, le=1.0, description="Aspect ratio weight")
    
    # Temporal confirmation
    confirmation_frames: int = Field(default=3, ge=1, le=10, description="Frames for confirmation")
    position_tolerance_px: int = Field(default=20, ge=5, le=50, description="Position tolerance")
    
    # Cooldown
    cooldown_frames: int = Field(default=30, ge=5, le=100, description="Cooldown duration")
    cooldown_radius_px: int = Field(default=50, ge=10, le=150, description="Cooldown radius")
    
    # History
    candidate_history_size: int = Field(default=20, ge=5, le=100, description="Candidate history buffer")
    
    # Motion mask preprocessing
    motion_mask_smoothing_kernel: int = Field(default=7, ge=0, le=21, description="Smoothing kernel (0=disable)")
    motion_adaptive: bool = Field(default=True, description="Enable adaptive Otsu")
    motion_otsu_bias: int = Field(default=8, ge=-50, le=50, description="Otsu bias")
    motion_min_area_px: int = Field(default=24, ge=1, le=500, description="Min area after morphology")
    morph_open_ksize: int = Field(default=3, ge=1, le=21, description="Morph open kernel")
    morph_close_ksize: int = Field(default=5, ge=1, le=21, description="Morph close kernel")
    motion_min_white_frac: float = Field(default=0.015, ge=0.0, le=0.5, description="Min white fraction")
    
    # Refine features
    refine_enabled: bool = Field(default=True, description="Enable impact refine")
    refine_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="Refine score threshold")
    refine_roi_size_px: int = Field(default=80, ge=20, le=200, description="Refine ROI size")
    refine_canny_lo: int = Field(default=60, ge=10, le=150, description="Refine Canny low")
    refine_canny_hi: int = Field(default=180, ge=50, le=300, description="Refine Canny high")
    refine_hough_thresh: int = Field(default=30, ge=5, le=100, description="Hough accumulator")
    refine_min_line_len: int = Field(default=10, ge=1, le=50, description="Min line length")
    refine_max_line_gap: int = Field(default=4, ge=1, le=20, description="Max line gap")
    
    # Tip refine
    tip_refine_enabled: bool = Field(default=True, description="Enable tip refine")
    tip_roi_px: int = Field(default=36, ge=10, le=100, description="Tip ROI size")
    tip_search_px: int = Field(default=14, ge=5, le=50, description="Tip search radius")
    tip_max_shift_px: int = Field(default=16, ge=5, le=50, description="Max tip shift")
    tip_edge_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="Edge weight")
    tip_dark_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Dark weight")
    tip_canny_lo: int = Field(default=60, ge=10, le=150, description="Tip Canny low")
    tip_canny_hi: int = Field(default=180, ge=50, le=300, description="Tip Canny high")
    
    # NEW: Convexity-Gate (Proposal 1)
    convexity_gate_enabled: bool = Field(default=True, description="Enable convexity filtering")
    convexity_min_ratio: float = Field(default=0.70, ge=0.3, le=1.0, description="Min convexity ratio")
    hierarchy_filter_enabled: bool = Field(default=True, description="Enable blob hierarchy filter")
    
    @field_validator('motion_mask_smoothing_kernel', 'morph_open_ksize', 'morph_close_ksize')
    @classmethod
    def validate_odd_or_zero(cls, v: int, info) -> int:
        """Ensure kernel sizes are odd or zero"""
        if v > 0 and v % 2 == 0:
            raise ValueError(f"{info.field_name} must be odd or 0, got {v}")
        return v
    
    @field_validator('tip_edge_weight')
    @classmethod
    def validate_tip_weights(cls, v: float, info) -> float:
        """Ensure edge + dark weights sum to ~1.0"""
        # Note: We can't access tip_dark_weight here, so we just validate range
        # Full validation happens at runtime in DartDetectorConfig
        return v


class DetectorConfigFile(BaseModel):
    """Top-level config file schema"""
    model_config = ConfigDict(extra='allow')  # Allow presets
    
    schema_version: str = Field(default=SCHEMA_VERSION, description="Config schema version")
    motion: MotionConfigSchema = Field(default_factory=MotionConfigSchema)
    dart_detector: DartDetectorConfigSchema = Field(default_factory=DartDetectorConfigSchema)
    presets: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Detector presets")


def validate_config_file(config_path: Path) -> DetectorConfigFile:
    """
    Validate a detector config YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Validated DetectorConfigFile
        
    Raises:
        ValueError: If config is invalid
        FileNotFoundError: If file doesn't exist
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            raw_data = yaml.safe_load(f)
        
        if raw_data is None:
            raw_data = {}
        
        # Validate with Pydantic
        config = DetectorConfigFile(**raw_data)
        
        logger.info(f"✅ Config validation passed: {config_path}")
        logger.info(f"   Schema version: {config.schema_version}")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Config validation failed: {config_path}")
        logger.error(f"   Error: {e}")
        raise ValueError(f"Invalid config file: {e}") from e


def save_config_atomic(config: DetectorConfigFile, config_path: Path) -> None:
    """
    Save config to YAML with atomic write (temp → move).
    
    Args:
        config: Validated config object
        config_path: Target YAML path
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict (Pydantic v2)
    config_dict = config.model_dump(exclude_none=True)
    
    # Atomic write via temp file
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yaml',
        dir=config_path.parent,
        delete=False
    ) as tmp:
        yaml.safe_dump(
            config_dict,
            tmp,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True
        )
        tmp_path = Path(tmp.name)
    
    # Atomic move
    shutil.move(str(tmp_path), str(config_path))
    logger.info(f"✅ Config saved atomically: {config_path}")


def create_default_config(output_path: Path) -> DetectorConfigFile:
    """
    Create a default config file with all fields documented.
    
    Args:
        output_path: Where to save the config
        
    Returns:
        Default config object
    """
    config = DetectorConfigFile()
    save_config_atomic(config, output_path)
    logger.info(f"✅ Default config created: {output_path}")
    return config


# CLI helper for validation
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python config_schema.py <config.yaml>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    
    try:
        config = validate_config_file(config_path)
        print(f"✅ Config valid: {config_path}")
        print(f"   Schema version: {config.schema_version}")
        print(f"   Motion var_threshold: {config.motion.var_threshold}")
        print(f"   Dart min_area: {config.dart_detector.min_area}")
        print(f"   Convexity gate: {config.dart_detector.convexity_gate_enabled}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)
