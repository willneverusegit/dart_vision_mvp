# Module: `src\capture\__init__.py`
Hash: `981593b9a4f1` · LOC: 1 · Main guard: false

## Imports
—

## From-Imports
- `from threaded_camera import ThreadedCamera, CameraConfig`\n- `from fps_counter import FPSCounter, FPSStats`

## Classes
—

## Functions
—

## Intra-module calls (heuristic)
—

## Code
```python
"""Video capture and FPS monitoring"""

from .threaded_camera import ThreadedCamera, CameraConfig
from .fps_counter import FPSCounter, FPSStats

__all__ = [
    'ThreadedCamera',
    'CameraConfig',
    'FPSCounter',
    'FPSStats'
]

```
