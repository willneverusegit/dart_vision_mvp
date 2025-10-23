# Module: `src\analytics\stats_accumulator.py`
Hash: `c17d9cdd4746` · LOC: 1 · Main guard: false

## Imports
- `csv`\n- `json`\n- `numpy`\n- `os`\n- `time`

## From-Imports
- `from __future__ import annotations`\n- `from dataclasses import dataclass, asdict`\n- `from typing import Optional, Dict, Any, Tuple`

## Classes
- `HitRecord` (L12)\n- `StatsAccumulator` (L20)

## Functions
- `__init__()` (L21)\n- `_label()` (L35)\n- `add()` (L43)\n- `ring_distribution()` (L70)\n- `sector_distribution()` (L77)\n- `ring_sector_matrix()` (L81)\n- `summary()` (L90)\n- `export_json()` (L106)\n- `export_csv_dists()` (L111)

## Intra-module calls (heuristic)
HitRecord, _label, dirname, dump, enumerate, float, get, int, items, len, makedirs, max, open, range, ring_distribution, ring_sector_matrix, sector_distribution, startswith, str, sum, summary, time, tolist, writer, writerow, zeros

## Code
```python
# Lightweight session stats for darts: ring/sector distributions, streaks, summary exports
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
import json, csv, os, time
import numpy as np

RINGS = ["single_inner","triple","single_outer","double","outer_bull","inner_bull","miss"]
RING_IDX = {r:i for i,r in enumerate(RINGS)}

@dataclass
class HitRecord:
    t_wall: float
    ring: str
    sector: Optional[int]  # 1..20 or None (bulls/miss)
    points: int
    cx: Optional[float] = None
    cy: Optional[float] = None

class StatsAccumulator:
    def __init__(self):
        self.t0 = time.time()
        self.n_hits = 0
        self.points_total = 0
        self.last_label: Optional[str] = None
        # 7x20 matrix (bulls/miss share sector=None -> stored separately)
        self.grid = np.zeros((len(RINGS), 20), dtype=np.int32)
        self.bulls = np.zeros(2, dtype=np.int32)  # OB, IB
        self.miss = 0
        # simple streaks (non-miss streak)
        self.current_streak = 0
        self.best_streak = 0

    @staticmethod
    def _label(ring: str, sector: Optional[int]) -> str:
        if ring == "inner_bull": return "IB"
        if ring == "outer_bull": return "OB"
        if ring == "miss": return "M"
        if sector is None: return "S?"
        prefix = "T" if ring=="triple" else "D" if ring=="double" else "S"
        return f"{prefix}{int(sector)}"

    def add(self, ring: str, sector: Optional[int], points: int,
            cx: Optional[float]=None, cy: Optional[float]=None) -> HitRecord:
        self.n_hits += 1
        self.points_total += int(points)
        rec = HitRecord(time.time(), ring, sector, int(points), cx, cy)

        if ring == "outer_bull":
            self.bulls[0] += 1
            self.current_streak += 1
        elif ring == "inner_bull":
            self.bulls[1] += 1
            self.current_streak += 1
        elif ring == "miss" or sector is None:
            self.miss += 1
            self.current_streak = 0
        else:
            r = RING_IDX.get(ring, None)
            if r is not None:
                s = (int(sector) - 1) % 20
                self.grid[r, s] += 1
            self.current_streak += 1

        self.best_streak = max(self.best_streak, self.current_streak)
        self.last_label = self._label(ring, sector)
        return rec

    # --- aggregates ---
    def ring_distribution(self) -> Dict[str, int]:
        out: Dict[str,int] = {r:int(self.grid[i,:].sum()) for r,i in RING_IDX.items() if r.startswith("single") or r in ("double","triple")}
        out["outer_bull"] = int(self.bulls[0])
        out["inner_bull"] = int(self.bulls[1])
        out["miss"] = int(self.miss)
        return out

    def sector_distribution(self) -> Dict[str, int]:
        s = self.grid.sum(axis=0)  # over ring rows (excl bulls/miss)
        return {str(i+1): int(v) for i, v in enumerate(s.tolist())}

    def ring_sector_matrix(self) -> np.ndarray:
        # returns 7x20 with bulls/miss placed in extra rows 4..6 similar to PolarHeatmap
        full = np.zeros((7,20), dtype=np.int32)
        full[0:4,:] = self.grid[0:4,:]
        full[4,:] = int(self.bulls[0])
        full[5,:] = int(self.bulls[1])
        full[6,:] = int(self.miss)
        return full

    def summary(self) -> Dict[str, Any]:
        dur_s = max(1e-6, time.time() - self.t0)
        per_min = self.n_hits / (dur_s/60.0)
        return {
            "hits": int(self.n_hits),
            "points_total": int(self.points_total),
            "points_per_hit": float(self.points_total / max(1, self.n_hits)),
            "hits_per_min": float(per_min),
            "miss_rate": float(self.miss / max(1, self.n_hits)),
            "best_streak": int(self.best_streak),
            "last_label": self.last_label,
            "ring_distribution": self.ring_distribution(),
            "sector_distribution": self.sector_distribution(),
        }

    # --- exports ---
    def export_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, ensure_ascii=False, indent=2)

    def export_csv_dists(self, ring_csv: str, sector_csv: str, matrix_csv: str) -> None:
        os.makedirs(os.path.dirname(ring_csv) or ".", exist_ok=True)
        # ring
        with open(ring_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["ring","count"])
            for k,v in self.ring_distribution().items(): w.writerow([k, v])
        # sector
        with open(sector_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["sector","count"])
            for k,v in self.sector_distribution().items(): w.writerow([k, v])
        # matrix (7x20)
        full = self.ring_sector_matrix()
        with open(matrix_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row\\col"] + [str(i+1) for i in range(20)])
            rows = ["S_in","T","S_out","D","OB","IB","M"]
            for i, name in enumerate(rows):
                w.writerow([name] + full[i,:].tolist())

```
