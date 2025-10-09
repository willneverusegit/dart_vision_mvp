"""
Performance profiler for pipeline optimization
"""

import time
import numpy as np
from collections import defaultdict
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Track execution time of pipeline stages"""

    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.enabled = True

    def get_report(self) -> str:
        """Generate performance report"""
        if not self.timings:
            return "\nNo profiling data available"

        report = ["\n" + "=" * 60]
        report.append("PERFORMANCE PROFILE")
        report.append("=" * 60)

        total_avg = 0
        for stage, times in sorted(self.timings.items()):
            if not times:
                continue

            avg = np.mean(times)
            median = np.median(times)
            p95 = np.percentile(times, 95)
            min_time = np.min(times)
            max_time = np.max(times)
            total_avg += avg

            report.append(f"\n{stage}:")
            report.append(f"  Avg:    {avg:>6.2f}ms")
            report.append(f"  Median: {median:>6.2f}ms")
            report.append(f"  P95:    {p95:>6.2f}ms")
            report.append(f"  Min:    {min_time:>6.2f}ms")
            report.append(f"  Max:    {max_time:>6.2f}ms")
            report.append(f"  Samples: {len(times)}")

        report.append("\n" + "-" * 60)
        report.append(f"Total Pipeline Avg: {total_avg:>6.2f}ms")
        report.append(f"Expected FPS:       {1000/total_avg:>6.1f}")
        report.append("=" * 60 + "\n")

        return "\n".join(report)