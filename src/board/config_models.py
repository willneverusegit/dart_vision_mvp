# Pydantic models for board mapping config.
from __future__ import annotations
from pydantic import BaseModel, Field, validator
from typing import List, Tuple

# --- Unified calibration schema additions ---
class Homography(BaseModel):
    """3x3 homography matrix in row-major order"""
    H: List[List[float]] = Field(..., min_items=3, max_items=3)

    @validator("H")
    def _shape(cls, v):
        if any(len(row) != 3 for row in v):
            raise ValueError("Homography.H must be 3x3")
        return v


class Metrics(BaseModel):
    """Derived/calibration metrics"""
    center_px: Tuple[float, float]  # (cx, cy) in ROI/canvas pixels
    roi_board_radius: float         # r_outer_double_px in the ROI frame


class OverlayAdjust(BaseModel):
    """Overlay adjustments applied after base calibration"""
    rotation_deg: float = 0.0
    r_outer_double_px: float
    center_dx_px: float = 0.0
    center_dy_px: float = 0.0


class ROIAdjust(BaseModel):
    """ROI affine adjustment prior to overlay/mapper (acting on the homography output)"""
    tx_px: float = 0.0
    ty_px: float = 0.0
    scale: float = 1.0
    rot_deg: float = 0.0


class UnifiedCalibration(BaseModel):
    """Single source of truth for calibration persistence."""
    homography: Homography
    metrics: Metrics
    overlay_adjust: OverlayAdjust
    roi_adjust: ROIAdjust
# --- end additions ---


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
