# Module: `src\calibration\__init__.py`
Hash: `3d0079bc312f` · LOC: 1 · Main guard: false

## Imports
—

## From-Imports
- `from roi_processor import ROIProcessor, ROIConfig`\n- `from calibration_manager import CalibrationManager, CalibrationData`\n- `from unified_calibrator import UnifiedCalibrator, CalibrationMethod, CalibrationResult`

## Classes
—

## Functions
—

## Intra-module calls (heuristic)
—

## Code
```python
"""Calibration and ROI processing modules"""

from .roi_processor import ROIProcessor, ROIConfig
from .calibration_manager import CalibrationManager, CalibrationData
from .unified_calibrator import UnifiedCalibrator, CalibrationMethod, CalibrationResult

__all__ = [
    'ROIProcessor',
    'ROIConfig',
    'CalibrationManager',
    'CalibrationData',
    'UnifiedCalibrator',
    'CalibrationMethod',
    'CalibrationResult'
]

```
