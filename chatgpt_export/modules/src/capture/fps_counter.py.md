# Module: `src\capture\fps_counter.py`
Hash: `ad283e1882c1` · LOC: 1 · Main guard: false

## Imports
- `numpy`\n- `time`

## From-Imports
- `from collections import deque`\n- `from dataclasses import dataclass`\n- `from typing import Optional`

## Classes
- `FPSStats` (L14): FPS statistics container\n- `FPSCounter` (L24): Real-time FPS measurement with statistical analysis.

## Functions
- `__init__()` (L34): Args:\n- `update()` (L45): Update FPS counter (call once per frame)\n- `get_fps()` (L63): Get current median FPS\n- `get_stats()` (L71): Get comprehensive FPS statistics\n- `get_overall_fps()` (L92): Get overall average FPS since start\n- `reset()` (L97): Reset all counters

## Intra-module calls (heuristic)
FPSStats, append, array, clear, deque, len, mean, median, percentile, perf_counter

## Code
```python
"""
FPS Counter with Statistical Analysis
Provides real-time performance metrics for validation.
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class FPSStats:
    """FPS statistics container"""
    fps_median: float
    fps_p95: float  # 95th percentile
    fps_mean: float
    frame_time_ms: float
    frame_time_p95_ms: float
    samples: int


class FPSCounter:
    """
    Real-time FPS measurement with statistical analysis.

    Metrics:
    - Median FPS: Primary performance indicator
    - P95 FPS: Worst-case performance (95th percentile)
    - Frame time: Processing latency
    """

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of samples for rolling statistics
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time: Optional[float] = None
        self.frame_count = 0
        self.start_time = time.perf_counter()

    def update(self) -> float:
        """
        Update FPS counter (call once per frame)

        Returns:
            Current frame duration in seconds
        """
        current_time = time.perf_counter()

        if self.last_time is not None:
            frame_duration = current_time - self.last_time
            self.frame_times.append(frame_duration)

        self.last_time = current_time
        self.frame_count += 1

        return current_time - self.start_time

    def get_fps(self) -> float:
        """Get current median FPS"""
        if len(self.frame_times) < 2:
            return 0.0

        median_duration = np.median(self.frame_times)
        return 1.0 / median_duration if median_duration > 0 else 0.0

    def get_stats(self) -> FPSStats:
        """Get comprehensive FPS statistics"""
        if len(self.frame_times) < 2:
            return FPSStats(0, 0, 0, 0, 0, 0)

        durations = np.array(self.frame_times)

        # Calculate FPS from frame durations
        median_duration = np.median(durations)
        p95_duration = np.percentile(durations, 95)
        mean_duration = np.mean(durations)

        return FPSStats(
            fps_median=1.0 / median_duration if median_duration > 0 else 0,
            fps_p95=1.0 / p95_duration if p95_duration > 0 else 0,
            fps_mean=1.0 / mean_duration if mean_duration > 0 else 0,
            frame_time_ms=median_duration * 1000,
            frame_time_p95_ms=p95_duration * 1000,
            samples=len(self.frame_times)
        )

    def get_overall_fps(self) -> float:
        """Get overall average FPS since start"""
        elapsed = time.perf_counter() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0

    def reset(self):
        """Reset all counters"""
        self.frame_times.clear()
        self.last_time = None
        self.frame_count = 0
        self.start_time = time.perf_counter()
```
