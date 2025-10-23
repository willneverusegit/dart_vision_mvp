"""
Vision detection modules package.
"""

from .motion_detector import MotionDetector, MotionConfig, MotionEvent
from .dart_impact_detector import (
    DartImpactDetector,
    DartDetectorConfig,
    DartImpact,
    DartCandidate,
    apply_detector_preset,
)

__all__ = [
    "MotionDetector",
    "MotionConfig",
    "MotionEvent",
    "DartImpactDetector",
    "DartDetectorConfig",
    "DartImpact",
    "DartCandidate",
    "apply_detector_preset",
]
