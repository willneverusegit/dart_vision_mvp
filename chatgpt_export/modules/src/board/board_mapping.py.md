# Module: `src\board\board_mapping.py`
Hash: `220b28f1bdd3` · LOC: 1 · Main guard: false

## Imports
- `math`\n- `numpy`

## From-Imports
- `from __future__ import annotations`\n- `from dataclasses import dataclass`\n- `from typing import Tuple, Optional`\n- `from config_models import BoardConfig`

## Classes
- `Calibration` (L14)\n- `BoardMapper` (L20)

## Functions
- `__init__()` (L21)\n- `cart2polar_norm()` (L26): Convert image px (x,y) to normalized polar (r_norm in [0,1], theta_deg). \n- `rel2screen()` (L50): Inverse of cart2polar angle rebasing: relative->screen degrees.\n- `classify_ring()` (L61)\n- `classify_sector()` (L79): Return sector value (1..20) by angle from 0..360 relative to sector 20 centerline at 0°.\n- `score_from_hit()` (L89): Return ring label, sector value (or None), and pretty label like 'T20', 'D5', 'S12', 'Bull', 'Outer Bull', 'Miss'.\n- `batch_score()` (L111)

## Intra-module calls (heuristic)
append, array, atan2, cart2polar_norm, classify_ring, classify_sector, degrees, float, hypot, int, max, score_from_hit, theta0_effective

## Code
```python
# Board mapping utilities: px -> polar -> ring/sector -> score
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from .config_models import BoardConfig

SECTOR_VALUES = [20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5]

@dataclass
class Calibration:
    cx: float
    cy: float
    r_outer_double_px: float  # radius of the outer double ring outer edge in pixels
    rotation_deg: float = 0.0 # additional rotation from calibration (if any); positive is CCW

class BoardMapper:
    def __init__(self, cfg: BoardConfig, calib: Calibration):
        self.cfg = cfg
        self.calib = calib

    # --- coordinate transforms ---
    def cart2polar_norm(self, x: float, y: float) -> Tuple[float, float]:
        """Convert image px (x,y) to normalized polar (r_norm in [0,1], theta_deg). 
        theta accounts for theta0, clockwise, and calibration rotation.
        Image origin assumed top-left; x right, y down.
        """
        dx = x - self.calib.cx
        dy = y - self.calib.cy
        r_px = math.hypot(dx, dy)
        r_norm = r_px / max(self.calib.r_outer_double_px, 1e-6)

        # atan2 in screen coords: y-down; convert to mathematical coords by negating dy
        theta = math.degrees(math.atan2(-dy, dx))  # 0° at +x, CCW
        # Apply calibration rotation (positive CCW), then rebase to theta0 and direction
        theta = (theta - self.calib.rotation_deg) % 360.0

        # Rebase to effective theta0 (Grenze oder Mitte je nach config)
        theta0 = self.cfg.angles.theta0_effective(self.cfg.sectors)
        theta_rel = (theta - theta0) % 360.0

        if self.cfg.angles.clockwise:
            theta_rel = (360.0 - theta_rel) % 360.0

        return r_norm, theta_rel

    def rel2screen(self, theta_rel_deg: float) -> float:
        """Inverse of cart2polar angle rebasing: relative->screen degrees."""
        theta = theta_rel_deg
        if self.cfg.angles.clockwise:
            theta = (360.0 - theta) % 360.0
        theta0 = self.cfg.angles.theta0_effective(self.cfg.sectors)
        theta = (theta + theta0 + self.calib.rotation_deg) % 360

        return theta

    # --- ring classification ---
    def classify_ring(self, r_norm: float) -> str:
        r = self.cfg.radii
        tol = r.radial_tolerance
        if r_norm <= r.r_bull_inner + tol:
            return "inner_bull"
        if r_norm <= r.r_bull_outer + tol:
            return "outer_bull"
        if r.r_triple_inner - tol <= r_norm <= r.r_triple_outer + tol:
            return "triple"
        if r.r_double_inner - tol <= r_norm <= r.r_double_outer + tol:
            return "double"
        if r_norm < r.r_triple_inner - tol:
            return "single_inner"
        if r.r_triple_outer + tol < r_norm < r.r_double_inner - tol:
            return "single_outer"
        return "miss"

    # --- sector classification ---
    def classify_sector(self, theta_rel_deg: float) -> int:
        """Return sector value (1..20) by angle from 0..360 relative to sector 20 centerline at 0°."""
        width = self.cfg.sectors.width_deg
        half = width / 2.0
        # sector index 0..19 where 0 is sector 20 (centerline at 0° after rebasing)
        # Map angle to nearest sector centerline
        idx = int(((theta_rel_deg + half) // width) % 20)
        return self.cfg.sectors.order[idx]

    # --- final scoring ---
    def score_from_hit(self, x: float, y: float) -> Tuple[str, Optional[int], str]:
        """Return ring label, sector value (or None), and pretty label like 'T20', 'D5', 'S12', 'Bull', 'Outer Bull', 'Miss'."""
        r_norm, theta = self.cart2polar_norm(x, y)
        ring = self.classify_ring(r_norm)
        if ring == "miss":
            return ring, None, "Miss"
        if ring == "inner_bull":
            return ring, None, "Bull (50)"
        if ring == "outer_bull":
            return ring, None, "Outer Bull (25)"
        sector = self.classify_sector(theta)
        if ring == "triple":
            label = f"T{sector}"
            return ring, sector, label
        if ring == "double":
            label = f"D{sector}"
            return ring, sector, label
        # singles
        label = f"S{sector}"
        return ring, sector, label

    # --- convenience: vectorized for Nx2 points ---
    def batch_score(self, points_xy: np.ndarray) -> np.ndarray:
        out = []
        for x, y in points_xy:
            ring, sec, label = self.score_from_hit(float(x), float(y))
            out.append([ring, sec if sec is not None else 0, label])
        return np.array(out, dtype=object)

```
