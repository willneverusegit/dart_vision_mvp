# Module: `src\board\config_models.py`
Hash: `eb431be759ca` · LOC: 1 · Main guard: false

## Imports
—

## From-Imports
- `from __future__ import annotations`\n- `from pydantic import BaseModel, Field, validator`\n- `from typing import List`

## Classes
- `Angles` (L6)\n- `Radii` (L11)\n- `Sectors` (L56)\n- `RuntimeOpts` (L72)\n- `BoardConfig` (L76)

## Functions
- `in_unit_interval()` (L21)\n- `bull_monotonic()` (L27)\n- `triple_inner_after_bull()` (L33)\n- `triple_outer_after_inner()` (L39)\n- `double_inner_after_triple()` (L45)\n- `double_outer_is_one()` (L51)\n- `unique_20_entries()` (L61)\n- `width_is_18()` (L67)

## Intra-module calls (heuristic)
Angles, Field, Radii, RuntimeOpts, Sectors, ValueError, abs, len, list, range, sorted, validator

## Code
```python
# Pydantic models for board mapping config.
from __future__ import annotations
from pydantic import BaseModel, Field, validator
from typing import List

class Angles(BaseModel):
    theta0_deg: float = 90.0
    clockwise: bool = True
    tolerance_deg: float = 2.0

class Radii(BaseModel):
    r_bull_inner: float = 0.02
    r_bull_outer: float = 0.06
    r_triple_inner: float = 0.4
    r_triple_outer: float = 0.45
    r_double_inner: float = 0.93
    r_double_outer: float = 1.01
    radial_tolerance: float = 0.01

    @validator("r_bull_inner", "r_bull_outer", "r_triple_inner", "r_triple_outer", "r_double_inner", "r_double_outer")
    def in_unit_interval(cls, v):
        if not (0.0 < v <= 1.0):
            raise ValueError("Radii must be in (0,1].")
        return v

    @validator("r_bull_outer")
    def bull_monotonic(cls, v, values):
        if "r_bull_inner" in values and not (values["r_bull_inner"] < v):
            raise ValueError("r_bull_outer must be > r_bull_inner")
        return v

    @validator("r_triple_inner")
    def triple_inner_after_bull(cls, v, values):
        if "r_bull_outer" in values and not (values["r_bull_outer"] < v):
            raise ValueError("r_triple_inner must be > r_bull_outer")
        return v

    @validator("r_triple_outer")
    def triple_outer_after_inner(cls, v, values):
        if "r_triple_inner" in values and not (values["r_triple_inner"] < v):
            raise ValueError("r_triple_outer must be > r_triple_inner")
        return v

    @validator("r_double_inner")
    def double_inner_after_triple(cls, v, values):
        if "r_triple_outer" in values and not (values["r_triple_outer"] < v):
            raise ValueError("r_double_inner must be > r_triple_outer")
        return v

    @validator("r_double_outer")
    def double_outer_is_one(cls, v):
        if abs(v - 1.0) > 1e-6:
            raise ValueError("r_double_outer must equal 1.0 (normalized).")
        return v

class Sectors(BaseModel):
    order: List[int] = Field(default_factory=lambda: [20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5])
    width_deg: float = 18.0

    @validator("order")
    def unique_20_entries(cls, v):
        if len(v) != 20 or sorted(v) != list(range(1,21)):
            raise ValueError("Sector order must be a permutation of 1..20.")
        return v

    @validator("width_deg")
    def width_is_18(cls, v):
        if abs(v - 18.0) > 1e-6:
            raise ValueError("Sector width must be 18 degrees.")
        return v

class RuntimeOpts(BaseModel):
    use_hough_as_fallback: bool = True
    verify_calibration: bool = True

class BoardConfig(BaseModel):
    version: int = 1
    angles: Angles = Angles()
    radii: Radii = Radii()
    sectors: Sectors = Sectors()
    runtime: RuntimeOpts = RuntimeOpts()

```
