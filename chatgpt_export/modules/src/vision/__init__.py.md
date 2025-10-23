# Module: `src\vision\__init__.py`
Hash: `1668bb9eb6a7` · LOC: 1 · Main guard: false

## Imports
—

## From-Imports
- `from motion_detector import MotionDetector, MotionConfig, MotionEvent`\n- `from dart_impact_detector import DartImpactDetector, DartDetectorConfig, DartImpact, FieldMapper, FieldMapperConfig`

## Classes
—

## Functions
—

## Intra-module calls (heuristic)
—

## Code
```python
"""Vision processing modules"""

from .motion_detector import MotionDetector, MotionConfig, MotionEvent
from .dart_impact_detector import (
    DartImpactDetector,
    DartDetectorConfig,
    DartImpact,
    FieldMapper,
    FieldMapperConfig
)

__all__ = [
    'MotionDetector',
    'MotionConfig',
    'MotionEvent',
    'DartImpactDetector',
    'DartDetectorConfig',
    'DartImpact',
    'FieldMapper',
    'FieldMapperConfig',
]
```
