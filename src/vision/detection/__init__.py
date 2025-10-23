"""
Vision detection modules package.

Provides both original monolithic detectors and refactored modular components:

Original (for backward compatibility):
- MotionDetector (439 lines)
- DartImpactDetector (600 lines)

Refactored (modular architecture):
- MotionDetectorRefactored (orchestrator)
  - BackgroundSubtractor
  - AdaptiveThreshold
  - MotionFilter
  - TemporalGate

- DartDetectorRefactored (orchestrator)
  - MaskPreprocessor
  - ShapeAnalyzer
  - ConfirmationTracker
  - CooldownManager
"""

# Original detectors (backward compatibility)
from .motion_detector import MotionDetector, MotionConfig, MotionEvent
from .dart_impact_detector import (
    DartImpactDetector,
    DartDetectorConfig,
    DartImpact,
    DartCandidate,
    apply_detector_preset,
)

# Refactored Motion Detection
from .motion_detector_refactored import MotionDetectorRefactored
from .background_subtractor import BackgroundSubtractor
from .adaptive_threshold import AdaptiveThreshold
from .motion_filter import MotionFilter
from .temporal_gate import TemporalGate

# Refactored Dart Detection
from .dart_detector_refactored import (
    DartDetectorRefactored,
    DartDetectorConfig as DartDetectorConfigRefactored,
    apply_detector_preset as apply_dart_preset,
)
from .mask_preprocessor import MaskPreprocessor
from .shape_analyzer import ShapeAnalyzer, ShapeMetrics
from .confirmation_tracker import (
    ConfirmationTracker,
    DartCandidate as DartCandidateRefactored,
    DartImpact as DartImpactRefactored,
)
from .cooldown_manager import CooldownManager

__all__ = [
    # Original
    "MotionDetector",
    "MotionConfig",
    "MotionEvent",
    "DartImpactDetector",
    "DartDetectorConfig",
    "DartImpact",
    "DartCandidate",
    "apply_detector_preset",
    # Refactored Motion
    "MotionDetectorRefactored",
    "BackgroundSubtractor",
    "AdaptiveThreshold",
    "MotionFilter",
    "TemporalGate",
    # Refactored Dart
    "DartDetectorRefactored",
    "DartDetectorConfigRefactored",
    "apply_dart_preset",
    "MaskPreprocessor",
    "ShapeAnalyzer",
    "ShapeMetrics",
    "ConfirmationTracker",
    "DartCandidateRefactored",
    "DartImpactRefactored",
    "CooldownManager",
]
