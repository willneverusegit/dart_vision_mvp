# stdlib
from pathlib import Path
from typing import Optional, Dict, Any
import math

# third-party
import numpy as np
import yaml

# project
from src.board.config_models import (
    UnifiedCalibration, Homography, Metrics, OverlayAdjust, ROIAdjust, BoardConfig
)
# Achtung: Calibration kommt aus board_mapping (nicht aus config_models)
from src.board.board_mapping import Calibration

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
# --- Unified calibration I/O and transforms -----------------------------------

def _roi_adjust_matrix(roi_adj: ROIAdjust) -> np.ndarray:
    """Build 3x3 matrix from ROIAdjust (rot -> scale -> translation)."""
    a = math.radians(roi_adj.rot_deg or 0.0)
    ca, sa = math.cos(a), math.sin(a)
    S = float(roi_adj.scale or 1.0)
    R = np.array([[ca, -sa, 0.0],
                  [sa,  ca, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    M = np.array([[S, 0.0, float(roi_adj.tx_px or 0.0)],
                  [0.0, S, float(roi_adj.ty_px or 0.0)],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return M @ R


def load_unified_calibration(path: Path) -> Optional[UnifiedCalibration]:
    """
    Load unified calibration (Single Source of Truth).
    Returns None if the file doesn't exist or doesn't contain the required blocks.
    """
    if not path.exists():
        return None
    data = load_yaml(path)
    if not data:
        return None

    # accept both: nested under "calibration" or flat root
    root = data.get("calibration", data)
    required = ("homography", "metrics", "overlay_adjust", "roi_adjust")
    if not all(k in root for k in required):
        return None

    uc = UnifiedCalibration(
        homography=Homography(**root["homography"]),
        metrics=Metrics(**root["metrics"]),
        overlay_adjust=OverlayAdjust(**root["overlay_adjust"]),
        roi_adjust=ROIAdjust(**root["roi_adjust"]),
    )
    return uc


def save_unified_calibration(path: Path, uc: UnifiedCalibration) -> None:
    """
    Persist unified calibration. Always writes homography/metrics and both adjust-blocks.
    """
    payload = {
        "calibration": {
            "homography": uc.homography.model_dump(),
            "metrics": uc.metrics.model_dump(),
            "overlay_adjust": uc.overlay_adjust.model_dump(),
            "roi_adjust": uc.roi_adjust.model_dump(),
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def compute_effective_H(uc: UnifiedCalibration) -> np.ndarray:
    """
    Compute H_eff = ROI_adjust * H.
    This H acts in the same pixel frame that the pipeline uses after ROI adjustments.
    """
    H = np.asarray(uc.homography.H, dtype=np.float64)
    R = _roi_adjust_matrix(uc.roi_adjust)
    return R @ H


def mapper_calibration_from_unified(uc: UnifiedCalibration) -> Calibration:
    """
    Translate UnifiedCalibration into the mapper's Calibration struct.
    - center_dx/dy are kept in overlay_adjust (delta versus metrics.center_px)
    - r_outer_double_px & rotation_deg live in overlay_adjust
    """
    cx0, cy0 = uc.metrics.center_px
    return Calibration(
        cx=float(cx0 + uc.overlay_adjust.center_dx_px),
        cy=float(cy0 + uc.overlay_adjust.center_dy_px),
        r_outer_double_px=float(uc.overlay_adjust.r_outer_double_px),
        rotation_deg=float(uc.overlay_adjust.rotation_deg),
    )

# --- end unified calibration block --------------------------------------------



def _deep_to_serializable(x: Any):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int32, np.int64, np.integer)):
        return int(x)
    if isinstance(x, dict):
        return {k: _deep_to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_deep_to_serializable(v) for v in x]
    return x

def save_calibration_yaml(path: str, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable = _deep_to_serializable(data)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(serializable, f, sort_keys=False, allow_unicode=True)

def load_calibration_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data
