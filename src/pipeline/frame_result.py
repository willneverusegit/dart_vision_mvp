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
