# Module: `src\pipeline\frame_result.py`
Hash: `93a155d3ad50` · LOC: 1 · Main guard: false

## Imports
- `numpy`

## From-Imports
- `from __future__ import annotations`\n- `from dataclasses import dataclass`\n- `from typing import Optional, Dict, Any`

## Classes
- `FrameResult` (L7): Container returned by `process_frame`.

## Functions
—

## Intra-module calls (heuristic)
—

## Code
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class FrameResult:
    """Container returned by `process_frame`."""
    roi: np.ndarray
    motion_detected: bool
    fg_mask: np.ndarray
    impact: Optional[Dict[str, Any]]
    hud: Optional[Dict[str, Any]] = None

```
