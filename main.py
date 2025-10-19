"""
Dart Vision MVP — Main (UnifiedCalibrator only)
CPU-optimized darts detection with unified ChArUco/AruCo/manual calibration.

Usage examples:
  python main.py --webcam 0
  python main.py --video test_videos/dart_throw_1.mp4
  python main.py --calibrate --webcam 0
"""

import cv2
import argparse
import logging
import sys
import time
import yaml
import os
import math
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from collections import deque
from src.game.game import DemoGame, GameMode
from src.vision.dart_impact_detector import apply_detector_preset


# Board mapping/overlays/config
from src.board import BoardConfig, BoardMapper, Calibration, draw_ring_circles, draw_sector_labels
from src.overlay.heatmap import HeatmapAccumulator
from src.analytics.polar_heatmap import PolarHeatmap
from src.analytics.stats_accumulator import StatsAccumulator


# --- Project modules (kept) ---
from src.capture import ThreadedCamera, CameraConfig, FPSCounter
from src.calibration.roi_processor import ROIProcessor, ROIConfig
from src.calibration.unified_calibrator import UnifiedCalibrator, CalibrationMethod
from src.vision import (
    MotionDetector, MotionConfig,
    DartImpactDetector, DartDetectorConfig,
    FieldMapper, FieldMapperConfig
)
from src.utils.performance_profiler import PerformanceProfiler
from src.calibration.aruco_quad_calibrator import ArucoQuadCalibrator
from src.calibration.calib_io import save_calibration_yaml
from src.calibration.calib_io import load_calibration_yaml
# Unified calibration I/O
from src.calibration.calib_io import (
    load_yaml,
    load_unified_calibration,
    save_unified_calibration,
    compute_effective_H,
    mapper_calibration_from_unified,
)

# Pydantic models for unified calibration
from src.board.config_models import UnifiedCalibration, Homography, Metrics, OverlayAdjust, ROIAdjust



# ---------- Logging ----------
def setup_logging(debug: bool = True):
    """Configure dual logging to console + file."""
    log_level = logging.DEBUG if debug else logging.INFO

    # Formatter (kurz für Konsole, lang für File)
    fmt_console = logging.Formatter("%(levelname)s - %(message)s")
    fmt_file = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # FileHandler (alles in Log-Datei)
    fh = logging.FileHandler("dart_vision.log", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(fmt_file)

    # StreamHandler (direkt in Konsole)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(fmt_console)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)

    # Optional: Bestätigung ausgeben
    root.info(f"Logger initialized at level {logging.getLevelName(log_level)}")


# Aufruf direkt danach:
setup_logging(debug=True)
logger = logging.getLogger("main")


# ---------- Small helpers ----------
# Overlay-Modi
OVERLAY_MIN   = 0  # nur Treffer & Game-HUD (präsentationstauglich)
OVERLAY_RINGS = 1  # Ringe + ROI-Kreis
OVERLAY_FULL  = 2  # Voll: Ringe + Sektoren + technische HUDs
OVERLAY_ALIGN = 3   # NEU: Ausrichtmodus

# Extended key codes (work with cv2.waitKeyEx on Windows)
VK_LEFT  = 0x250000
VK_UP    = 0x260000
VK_RIGHT = 0x270000
VK_DOWN  = 0x280000

# Legacy OpenCV codes (falls Backend diese liefert)
OCV_LEFT  = 2424832
OCV_UP    = 2490368
OCV_RIGHT = 2555904
OCV_DOWN  = 2621440

# ROI-Feintuning (Shift+Pfeile bewegen, +/- skalieren, ,/. rotieren, 0 reset)
ROI_SIZE = (400, 400)
ROI_CENTER = (ROI_SIZE[0] // 2, ROI_SIZE[1] // 2)
STEP_T = 1.0   # px
STEP_S = 1.005 # ~0.5%
STEP_R = 0.25  # Grad
ARUCO_DICT_MAP = {
    "4X4_50":  cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50":  cv2.aruco.DICT_5X5_50,
    "6X6_250": cv2.aruco.DICT_6X6_250,
}
CALIB_YAML = Path("calibration_unified.yaml")


# ---------- Main App ----------
class DartVisionApp:
    def __init__(self, args):
        self.args = args
        self.logger = logger  # <— benutze den globalen logger auch als Instanz-Logger

        # Components
        self.camera: Optional[ThreadedCamera] = None
        self.roi: Optional[ROIProcessor] = None
        self.motion: Optional[MotionDetector] = None
        self.dart: Optional[DartImpactDetector] = None
        self.mapper: Optional[FieldMapper] = None
        self.fps: Optional[FPSCounter] = None
        self.profiler = PerformanceProfiler()

        # Calibration state
        self.cal = UnifiedCalibrator(squares_x=5, squares_y=7, square_length_m=0.04, marker_length_m=0.03)
        self.homography: Optional[np.ndarray] = None
        self.homography_eff: Optional[np.ndarray] = None  # Heff = ROI_adjust * H
        self.uc: Optional[UnifiedCalibration] = None
        self.calib_path: Path = CALIB_YAML  # ← konsistent zur Modulkonstante

        # ROI/Board Baselines
        self.mm_per_px: float = 1.0
        self.roi_board_radius: float = 160.0

        # ROI center als Instanz-Attribut (Float!)
        self.ROI_CENTER = (float(ROI_SIZE[0]) * 0.5, float(ROI_SIZE[1]) * 0.5)
        self.center_px: Tuple[float, float] = (self.ROI_CENTER[0], self.ROI_CENTER[1])

        # UI state
        self.running = False
        self.paused = False
        self.show_debug = True
        self.show_motion = False
        self.frame_count = 0
        self.session_start = time.time()

        # HUD buffers (Glättung)
        self._hud_b = deque(maxlen=15)  # Brightness
        self._hud_f = deque(maxlen=15)  # Focus (Laplacian Var)
        self._hud_e = deque(maxlen=15)  # Edge density %

        # Mapping/Scoring
        self.board_cfg: Optional[BoardConfig] = None
        self.board_mapper: Optional[BoardMapper] = None

        # Overlay-Mode (0=min, 1=rings, 2=full)
        self.overlay_mode = OVERLAY_ALIGN
        # Overlay fine-tune (applied only to BoardMapper projection)
        self.overlay_rot_deg: float = 0.0  # add to Calibration.rotation_deg
        self.overlay_scale: float = 1.0  # multiplies roi_board_radius
        self.overlay_center_dx: float = 0.0
        self.overlay_center_dy: float = 0.0
        self.align_auto: bool = False  # NEU: Hough-Autoausrichtung an/aus
        self.show_help = False  # Hilfe-Overlay an/aus

        # ROI Fine-tune (wirken auf die Warphomographie)
        self.roi_tx = 0.0  # px (Ausgaberaum)
        self.roi_ty = 0.0  # px (Ausgaberaum)
        self.roi_scale = 1.0  # 1.0 = unverändert
        self.roi_rot_deg = 0.0  # °, um ROI_CENTER

        # ROI-Referenz (fixe Basis für relative Skalierung)
        self.roi_base_radius: Optional[float] = None
        self._roi_adjust_dirty = False

        # Mini-Game
        self.game = DemoGame(mode=self.args.game if hasattr(self.args, "game") else GameMode.ATC)
        self.last_msg = ""  # HUD-Zeile für letzten Wurf

        # --- Heatmap / Polar-Heatmap state ---
        self.hm = None  # HeatmapAccumulator
        self.ph = None  # PolarHeatmap
        self.heatmap_enabled = False
        self.polar_enabled = False
        self.session_id = str(int(self.session_start))
        self.total_darts = 0
        self.stats = StatsAccumulator()

        self.src_is_video = False
        self.src_fps = 0.0
        self._next_vsync = None
        self._last_pos_msec = None

    def _update_uc_from_calibrator(self, H_base: np.ndarray,
                                   center_px: tuple[float, float],
                                   r_outer_double_px: float,
                                   roi_adjust: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.0),
                                   rotation_deg: float = 0.0):
        """
        Push calibration result from calibrator into UnifiedCalibration:
          - H_base: 3x3 homography (base, not effective)
          - center_px: (cx, cy) in ROI/canvas pixels (baseline center)
          - r_outer_double_px: absolute outer-double radius (px)
          - roi_adjust: (tx_px, ty_px, scale, rot_deg) – optional
          - rotation_deg: fine sector rotation (overlay)
        """
        tx, ty, scale, rot = roi_adjust
        Hb = np.asarray(H_base, dtype=np.float64)
        cx0, cy0 = float(center_px[0]), float(center_px[1])

        if getattr(self, "uc", None) is None:
            self.uc = UnifiedCalibration(
                homography=Homography(H=Hb.tolist()),
                metrics=Metrics(center_px=(cx0, cy0), roi_board_radius=float(r_outer_double_px)),
                overlay_adjust=OverlayAdjust(
                    rotation_deg=float(rotation_deg),
                    r_outer_double_px=float(r_outer_double_px),
                    center_dx_px=0.0,
                    center_dy_px=0.0,
                ),
                roi_adjust=ROIAdjust(
                    tx_px=float(tx), ty_px=float(ty), scale=float(scale), rot_deg=float(rot)
                ),
            )
        else:
            self.uc.homography.H = Hb.tolist()
            self.uc.metrics.center_px = (cx0, cy0)
            self.uc.metrics.roi_board_radius = float(r_outer_double_px)
            self.uc.overlay_adjust.rotation_deg = float(rotation_deg)
            self.uc.overlay_adjust.r_outer_double_px = float(r_outer_double_px)
            # center_dx/dy bleiben 0.0 direkt nach der Kalibrierung (Feinjustage kommt später)
            self.uc.overlay_adjust.center_dx_px = 0.0
            self.uc.overlay_adjust.center_dy_px = 0.0
            self.uc.roi_adjust.tx_px = float(tx)
            self.uc.roi_adjust.ty_px = float(ty)
            self.uc.roi_adjust.scale = float(scale)
            self.uc.roi_adjust.rot_deg = float(rot)

        # Heff neu & Mapper syncen
        self.homography = Hb
        self.homography_eff = compute_effective_H(self.uc)
        if hasattr(self, "_sync_mapper_from_unified"):
            self._sync_mapper_from_unified()

    def _save_calibration_unified(self, calib_path: Path | None = None):
        """
        Persist calibration in unified format:
          - homography.H (base, not effective)
          - metrics.center_px / metrics.roi_board_radius
          - overlay_adjust.{rotation_deg, r_outer_double_px, center_dx_px, center_dy_px}
          - roi_adjust.{tx_px, ty_px, scale, rot_deg}
        """
        path = Path(calib_path) if calib_path is not None else getattr(self, "calib_path",
                                                                       Path("calibration_unified.yaml"))

        # Baselines aus aktuellem State
        cx0, cy0 = getattr(self, "ROI_CENTER", (0.0, 0.0))
        r_od = float(getattr(self, "roi_board_radius", 400.0))
        rot = float(getattr(self, "overlay_rot_deg", 0.0))
        dx = float(getattr(self, "overlay_center_dx", 0.0))
        dy = float(getattr(self, "overlay_center_dy", 0.0))
        scale = float(getattr(self, "overlay_scale", 1.0))
        Hb = np.asarray(self.homography if self.homography is not None else np.eye(3), dtype=np.float64)

        # ROI-Adjust (falls vorhanden; sonst neutral)
        roi_tx = float(getattr(self, "roi_tx", 0.0))
        roi_ty = float(getattr(self, "roi_ty", 0.0))
        roi_scale = float(getattr(self, "roi_scale", 1.0))
        roi_rot = float(getattr(self, "roi_rot_deg", 0.0))

        # Unified-Objekt aktualisieren/aufbauen
        if getattr(self, "uc", None) is None:
            self.uc = UnifiedCalibration(
                homography=Homography(H=Hb.tolist()),
                metrics=Metrics(center_px=(float(cx0), float(cy0)), roi_board_radius=float(r_od)),
                overlay_adjust=OverlayAdjust(
                    rotation_deg=float(rot),
                    r_outer_double_px=float(r_od * scale),
                    center_dx_px=float(dx),
                    center_dy_px=float(dy),
                ),
                roi_adjust=ROIAdjust(
                    tx_px=roi_tx, ty_px=roi_ty, scale=roi_scale, rot_deg=roi_rot
                ),
            )
        else:
            self.uc.homography.H = Hb.tolist()
            self.uc.metrics.center_px = (float(cx0), float(cy0))
            self.uc.metrics.roi_board_radius = float(r_od)
            self.uc.overlay_adjust.rotation_deg = float(rot)
            self.uc.overlay_adjust.center_dx_px = float(dx)
            self.uc.overlay_adjust.center_dy_px = float(dy)
            self.uc.overlay_adjust.r_outer_double_px = float(r_od * scale)
            self.uc.roi_adjust.tx_px = roi_tx
            self.uc.roi_adjust.ty_px = roi_ty
            self.uc.roi_adjust.scale = roi_scale
            self.uc.roi_adjust.rot_deg = roi_rot

        # Heff aktualisieren + Mapper syncen (sofort konsistent)
        self.homography_eff = compute_effective_H(self.uc)
        if hasattr(self, "_sync_mapper_from_unified"):
            self._sync_mapper_from_unified()

        # Schreiben
        save_unified_calibration(path, self.uc)

        # Feedback
        try:
            self.toast(f"Calibration saved → {path}")
        except Exception:
            print(f"[INFO] Calibration saved → {path}")

    def _ratio_consistent(self, circles, r_out: float) -> bool:
        """Prüft, ob innere Ringe im erwarteten Verhältnis zu r_out auftauchen."""
        if self.board_cfg is None or r_out <= 0:
            return True
        target = {
            "D_in": self.board_cfg.radii.r_double_inner,  # ~0.93
            "T_out": self.board_cfg.radii.r_triple_outer,  # ~0.62
            "T_in": self.board_cfg.radii.r_triple_inner,  # ~0.55
        }
        rs = np.array([float(c[2]) for c in circles if float(c[2]) < r_out], dtype=float)
        if rs.size == 0:
            return False
        ratios = rs / r_out
        ok = 0
        for r in target.values():
            if np.min(np.abs(ratios - float(r))) < 0.07:  # 3% Toleranz
                ok += 1
        return ok >= 2  # mindestens zwei „Treffer“

    def _hough_refine_center(self, roi_bgr) -> bool:
        """
        Findet den äußeren Doppelring als Kreis und passt Overlay-Center & Scale an.
        Verwendet aktuelles Overlay-Center/-Radius als Referenz, sanfte Gains, Robustheit.
        """
        if roi_bgr is None:
            return False

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        if getattr(self.args, "clahe", False):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        gray = cv2.medianBlur(gray, 5)

        # Erwartung aus aktuellem Overlay-Zustand
        cx_cur = float(ROI_CENTER[0] + self.overlay_center_dx)
        cy_cur = float(ROI_CENTER[1] + self.overlay_center_dy)
        r_cur = float(self.roi_board_radius) * float(self.overlay_scale)

        # Hough-Radiusfenster um den aktuellen Sollradius
        r_min = int(max(10, 0.88 * r_cur))
        r_max = int(min(ROI_SIZE[0], 1.12 * r_cur))

        # Hough
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=100,
            param1=120, param2=40,
            minRadius=r_min, maxRadius=r_max
        )
        if circles is None or len(circles) == 0:
            self.last_msg = "Hough: no circle"
            return False

        cs = np.round(circles[0, :]).astype(int)

        # Nimm den Kreis, der (a) am nächsten am r_cur liegt und (b) dem erwarteten Center nahe ist
        def score(c):
            cx, cy, r = c
            dr = abs(r - r_cur)
            dc = abs(cx - cx_cur) + abs(cy - cy_cur)
            return (dr, dc)

        cs = sorted(cs, key=score)
        cx, cy, r = [int(v) for v in cs[0]]

        # Abweichungen
        dx_meas = float(cx) - cx_cur
        dy_meas = float(cy) - cy_cur
        sr_meas = float(r) / max(r_cur, 1e-6)

        # Sanfte Gains (keine Sprünge)
        k_center = 0.35  # 0..1  (Center-Anteil je Update)
        k_scale = 0.30  # 0..1  (Scale-Anteil je Update)

        # Nur anwenden, wenn plausibel
        # (Kleinere Schwellwerte = empfindlicher; größere = stabiler)
        if (abs(dx_meas) > ROI_SIZE[0] * 0.4) or (abs(dy_meas) > ROI_SIZE[1] * 0.4):
            self.last_msg = "Hough: center jump too large → ignore"
            return False
        if not (0.85 <= sr_meas <= 1.15):
            self.last_msg = "Hough: radius jump too large → ignore"
            return False

        # Update (gegen aktuelles SOLL, nicht ROI_CENTER!)
        self.overlay_center_dx += k_center * dx_meas
        self.overlay_center_dy += k_center * dy_meas
        self.overlay_scale *= (1.0 + k_scale * (sr_meas - 1.0))

        # Werte direkt in Mapper schieben, wenn vorhanden
        if self.board_mapper is not None:
            self.board_mapper.calib.cx = float(ROI_CENTER[0] + self.overlay_center_dx)
            self.board_mapper.calib.cy = float(ROI_CENTER[1] + self.overlay_center_dy)
            self.board_mapper.calib.r_outer_double_px = float(self.roi_board_radius) * float(self.overlay_scale)

        self.last_msg = f"Hough OK: d=({dx_meas:+.1f},{dy_meas:+.1f}) r={r} → cx={self.board_mapper.calib.cx:.1f}, cy={self.board_mapper.calib.cy:.1f}, s={self.overlay_scale:.4f}"
        return True

    def _hough_refine_rings(self, roi_bgr: np.ndarray) -> tuple[float, float, float] | None:
        """
        Findet konzentrische Ringe via HoughCircles (im ROI-Panel).
        Liefert (cx, cy, r_double_outer_px) oder None.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return None

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)

        # Erwartungsbereich um aktuellen Wert (±15%)
        if self.board_mapper is not None:
            r0 = float(self.board_mapper.calib.r_outer_double_px)
        elif self.uc is not None:
            r0 = float(self.uc.overlay_adjust.r_outer_double_px)
        else:
            r0 = float(self.roi_board_radius) * float(self.overlay_scale)

        rmin = max(10, int(r0 * 0.85))
        rmax = int(r0 * 1.15)

        # Hough-Parameter: ggf. bei kontrastarmen Bildern param2 kleiner (30–34) wählen
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.15, minDist=18,
            param1=110, param2=32, minRadius=int(rmin), maxRadius=int(rmax)
        )

        if circles is None:
            return None

        c = np.uint16(np.around(circles[0]))  # (N,3): x,y,r
        # --- Ratio-Check: wenn inkonsistent, abbrechen ---
        r_out_guess = float(max(c, key=lambda k: k[2])[2])
        if not self._ratio_consistent(c, r_out_guess):
            return None

        # jetzt den größten Kreis nehmen (ggf. danach noch mitteln)
        xyr = max(c, key=lambda k: k[2])
        cx, cy, r_out = float(xyr[0]), float(xyr[1]), float(xyr[2])
        # (optional: close-Mittelung wie zuvor)

        # Optionales Mitteln über nahe Kreise (stabilisiert)
        close = [k for k in c if abs(float(k[2]) - r_out) < r0 * 0.03]
        if len(close) >= 2:
            cx = float(np.mean([k[0] for k in close]))
            cy = float(np.mean([k[1] for k in close]))
            r_out = float(np.mean([k[2] for k in close]))

        # (Optional) Debug-Overlay
        if getattr(self, "show_debug", False):
            dbg = roi_bgr.copy()
            cv2.circle(dbg, (int(round(cx)), int(round(cy))), int(round(r_out)), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Hough-Rings (debug)", dbg)

        return cx, cy, r_out

    def _auto_sector_rotation_from_edges(self, roi_bgr: np.ndarray, apply: bool = True) -> tuple[float, float] | None:
        """
        Schätzt den Rotations-Offset (in Grad) für die Sektor-Ausrichtung anhand eines
        Winkel-Histogramms aus Kantenpunkten. Nutzt die 20er-Periodizität der Segmente.
        Rückgabe: (delta_deg, kappa) oder None bei schwachem Signal.
          - delta_deg: Vorschlag, den auf calib.rotation_deg addiert werden sollte
          - kappa: Norm der N-fachen Phasensumme (0..1) als Qualitätsmaß
        """
        if roi_bgr is None or roi_bgr.size == 0 or self.board_cfg is None:
            return None

        # 1) Kanten im ROI
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 70, 200)  # etwas strenger als fürs Motion-Gate

        ys, xs = np.nonzero(edges)
        if xs.size < 200:  # zu wenig Punkte → kein verlässliches Signal
            return None

        # 2) Winkel relativ zur aktuellen Kalibrierung (wie im Mapper)
        #    Screen-Winkel: 0° an +x, CCW; y-Achse ist nach unten → dy negieren.
        cal = self._overlay_calib()  # Single Source: cx, cy, r_out, rot
        dx = xs.astype(np.float32) - float(cal.cx)
        dy = ys.astype(np.float32) - float(cal.cy)
        theta_screen = (np.degrees(np.arctan2(-dy, dx)) + 360.0) % 360.0

        # Effektiven theta0 (Grenze oder Mitte) aus Konfig
        theta0 = self.board_cfg.angles.theta0_effective(self.board_cfg.sectors)
        # Rebase auf relative Winkelskala des Mappers
        theta_rel = (theta_screen - float(cal.rotation_deg) - theta0) % 360.0
        if self.board_cfg.angles.clockwise:
            theta_rel = (360.0 - theta_rel) % 360.0

        # Optional: Ring-Annulus maskieren (Singles-Bereich liefert i.d.R. die stärksten Sektor-Kanten)
        r = np.hypot(dx, dy) / max(float(cal.r_outer_double_px), 1e-6)
        annulus = (r > 0.30) & (r < 0.92)  # zwischen Single-Innen und Double-Innen
        theta_rel = theta_rel[annulus]
        if theta_rel.size < 200:
            return None

        # 3) 20-fache Periodizität aggregieren: Z = mean(exp(i*N*theta))
        N = 20
        theta_rad = np.deg2rad(theta_rel)
        Z = np.exp(1j * N * theta_rad).mean()
        kappa = float(np.abs(Z))  # 0..1 Signalstärke

        # schwaches Signal? abbrechen
        if not np.isfinite(kappa) or kappa < 0.03:
            return None

        # Phasenlage → geschätzter Offset relativ zur aktuellen 0-Linie
        # Wenn Peaks bei theta = theta0_est + k*(360/N) liegen, dann angle(Z) ≈ N * theta0_est
        phi = float(np.angle(Z))  # rad, (-π, π]
        theta0_est_deg = (phi * 180.0 / math.pi) / N  # in Grad, typ. in (-9, 9]
        delta_deg = -theta0_est_deg  # Korrektur: wir wollen, dass Peak bei 0° liegt

        # Auf sinnvollen Bereich clampen (verhindert „Sprünge“)
        width = float(self.board_cfg.sectors.width_deg)  # meist 18
        half = width * 0.5
        while delta_deg <= -half:
            delta_deg += width
        while delta_deg > half:
            delta_deg -= width

        # Optional: kleine Updates sanft begrenzen (z. B. max ±5°)
        delta_deg = float(np.clip(delta_deg, -5.0, 5.0))

        # Anwenden?
        if apply and self.uc is not None:
            self.uc.overlay_adjust.rotation_deg = float(self.uc.overlay_adjust.rotation_deg + delta_deg)
            # Refresh + persist
            self.homography_eff = compute_effective_H(self.uc)
            if hasattr(self, "_sync_mapper_from_unified"):
                self._sync_mapper_from_unified()
            save_unified_calibration(self.calib_path, self.uc)

        # Debug-Plot (optional)
        if getattr(self, "show_debug", False):
            # kleines Histogramm zum Check
            bins = 120
            hist, edges_deg = np.histogram(theta_rel, bins=bins, range=(0.0, 360.0))
            h = np.zeros((120, 360, 3), dtype=np.uint8)
            hh = (hist / max(hist.max(), 1)) * (h.shape[0] - 1)
            for i in range(360):
                b = int((i / 360.0) * bins)
                val = int(hh[b])
                if val > 0:
                    cv2.line(h, (i, h.shape[0] - 1), (i, h.shape[0] - 1 - val), (180, 255, 180), 1)
            # Markiere erwartete 20er-Gitter nach Korrektur
            for k in range(N):
                ang = (k * width) % 360.0
                x = int(round(ang))
                cv2.line(h, (x, 0), (x, h.shape[0] - 1), (100, 200, 255), 1)
            cv2.putText(h, f"dth:{delta_deg:+.2f} deg  kappa:{kappa:.2f}", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Angle-Histogram (20x)", h)

        return float(delta_deg), kappa

    def _hud_compute(self, gray: np.ndarray):
        # Helligkeit, Fokus-Proxy (Laplacian-Var), Kantenanteil in %
        b = float(np.mean(gray))
        f = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges = cv2.Canny(gray, 80, 160)
        e = 100.0 * float(np.mean(edges > 0))
        # Glättung
        self._hud_b.append(b);
        self._hud_f.append(f);
        self._hud_e.append(e)
        b_ = float(np.mean(self._hud_b));
        f_ = float(np.mean(self._hud_f));
        e_ = float(np.mean(self._hud_e))
        return b_, f_, e_

    def _hud_color(self, b: float, f: float, e: float, charuco_corners: int = 0):
        # Zielkorridore (kannst du später anpassen)
        ok_b = 120.0 <= b <= 170.0
        ok_f = f >= 800.0
        ok_e = 3.5 <= e <= 15.0
        ok_c = (charuco_corners >= 8)  # nur in Kalibrier-UI relevant
        ok = ok_b and ok_f and ok_e and ok_c
        return (0, 255, 0) if ok else (0, 200, 200)

    def _hud_draw(self, img, lines, color=(0, 255, 0), org=(12, 24)):
        x, y = org
        for ln in lines:
            cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            y += 22

    def _overlay_center_radius(self):
        """Gibt (cx, cy, r_base) zurück: Center inkl. Offsets + skalierten Doppel-Außenradius."""
        cx = int(ROI_CENTER[0] + getattr(self, "overlay_center_dx", 0.0))
        cy = int(ROI_CENTER[1] + getattr(self, "overlay_center_dy", 0.0))
        r_base = int(float(self.roi_board_radius) * float(getattr(self, "overlay_scale", 1.0)))
        return cx, cy, r_base

    def _points_from_mapping(self, ring: str, sector: int) -> int:
        """Rechnet BoardMapper-Ergebnis in Punkte um."""
        if ring == "bull_inner":
            return 50
        if ring == "bull_outer":
            return 25
        if ring == "double":
            return 2 * int(sector)
        if ring == "triple":
            return 3 * int(sector)
        if ring.startswith("single"):
            return int(sector)
        # Fallback
        try:
            return int(sector)
        except Exception:
            return 0

    def _draw_traffic_light(self, img, b, f, e, org=(12, 105)):
        def st_b(v): return 'G' if 130 <= v <= 160 else ('Y' if 120 <= v <= 170 else 'R')

        def st_f(v): return 'G' if v >= 1500 else ('Y' if v >= 800 else 'R')

        def st_e(v): return 'G' if 5.0 <= v <= 10.0 else ('Y' if 3.5 <= v <= 15.0 else 'R')

        col = {'R': (36, 36, 255), 'Y': (0, 255, 255), 'G': (0, 200, 0)}
        states = [("B", st_b(b)), ("F", st_f(f)), ("E", st_e(e))]
        x, y = org;
        w, h, pad = 18, 18, 8
        cv2.putText(img, "Status:", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
        for i, (label, st) in enumerate(states):
            p1 = (x + i * (w + pad), y);
            p2 = (p1[0] + w, p1[1] + h)
            cv2.rectangle(img, p1, p2, col[st], -1);
            cv2.rectangle(img, p1, p2, (20, 20, 20), 1)
            cv2.putText(img, label, (p1[0], p2[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        st_vals = [s for _, s in states]
        s_all = 'G' if all(s == 'G' for s in st_vals) else ('R' if 'R' in st_vals else 'Y')
        X = x + 3 * (w + pad) + 20
        cv2.putText(img, "ALL", (X, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.rectangle(img, (X, y), (X + w, y + h), col[s_all], -1)
        cv2.rectangle(img, (X, y), (X + w, y + h), (20, 20, 20), 1)

    def _sync_mapper(self):
        if self.board_mapper:
            self.board_mapper.calib.cx = float(ROI_CENTER[0] + self.overlay_center_dx)
            self.board_mapper.calib.cy = float(ROI_CENTER[1] + self.overlay_center_dy)
            self.board_mapper.calib.r_outer_double_px = float(self.roi_board_radius) * float(self.overlay_scale)
            self.board_mapper.calib.rotation_deg = float(self.overlay_rot_deg)

    def _sync_mapper_from_unified(self):
        """Single source of truth → mapper Calibration and internal H_eff."""
        if self.board_cfg is None or self.uc is None:
            return
        calib = mapper_calibration_from_unified(self.uc)
        self.board_mapper = BoardMapper(self.board_cfg, calib)
        if self.homography is None or self.homography_eff is None:
            self.homography_eff = compute_effective_H(self.uc)

    def build_roi_adjust_matrix(
            self,
            center_xy: Tuple[float, float],
            tx_px: float = 0.0,
            ty_px: float = 0.0,
            scale: float = 1.0,
            rot_deg: float = 0.0,
    ) -> np.ndarray:
        """
        Baut eine 3x3 Homographie, die um das ROI-Zentrum herum skaliert/rotiert und
        anschließend in Pixeln verschiebt (tx, ty).

        Reihenfolge: Uncenter -> Scale -> Rotate -> Translate -> Recenter
        H = T_center @ T_translate @ R @ S @ T_uncenter
        """
        cx, cy = float(center_xy[0]), float(center_xy[1])
        cos_t = float(np.cos(np.deg2rad(rot_deg)))
        sin_t = float(np.sin(np.deg2rad(rot_deg)))

        # 3x3 Matrizen
        T_uncenter = np.array([[1, 0, -cx],
                               [0, 1, -cy],
                               [0, 0, 1]], dtype=np.float64)

        S = np.array([[scale, 0, 0],
                      [0, scale, 0],
                      [0, 0, 1]], dtype=np.float64)

        R = np.array([[cos_t, -sin_t, 0],
                      [sin_t, cos_t, 0],
                      [0, 0, 1]], dtype=np.float64)

        T_translate = np.array([[1, 0, tx_px],
                                [0, 1, ty_px],
                                [0, 0, 1]], dtype=np.float64)

        T_center = np.array([[1, 0, cx],
                             [0, 1, cy],
                             [0, 0, 1]], dtype=np.float64)

        H = T_center @ T_translate @ R @ S @ T_uncenter
        return H

    def _effective_H(self):
        if self.homography is None:
            return None
        A = self.build_roi_adjust_matrix(ROI_CENTER, self.roi_tx, self.roi_ty, self.roi_scale, self.roi_rot_deg)
        return A @ self.homography

    def _apply_effective_H_if_dirty(self):
        if not getattr(self, "_roi_adjust_dirty", False):
            return
        if self.homography is None:
            self._roi_adjust_dirty = False
            return
        Heff = self._effective_H()
        self.roi.set_homography_from_matrix(Heff)
        self._roi_adjust_dirty = False

    def _draw_help_overlay(self, img):
        """
        Zeichnet ein kompaktes Hilfe-Overlay unten rechts ins ROI-Bild (img).
        Nutzt lokales ROI-Blending – sicher & unabhängig vom Backend.
        """
        pad = 10
        lines = [
            "Help / Controls",
            "1/2/3: Preset Agg/Bal/Stable",
            "o: Overlay (MIN/RINGS/FULL/ALIGN)",
            "t: Hough once   z: Auto-Hough",
            "Arrows: rot/scale overlay",
            "X: Save overlay offsets",
            "c: Recalibrate   s: Screenshot",
            "g: Game reset    h: Switch game",
            "?: Toggle help",
        ]

        # Kastenmaße
        w = 310
        h = 22 * len(lines) + 2 * pad
        H, W = img.shape[:2]
        x = max(0, W - w - pad)
        y = max(0, H - h - pad)

        # ROI ausschneiden
        roi = img[y:y + h, x:x + w]
        if roi.size == 0:
            return  # falls zu knapp

        # halbtransparenten Hintergrund vorbereiten
        bg = roi.copy()
        cv2.rectangle(bg, (0, 0), (w, h), (20, 20, 20), -1)

        # Alpha-Blending NUR in diesem Kasten (kein Inplace-Parameter)
        blended = cv2.addWeighted(bg, 0.6, roi, 0.4, 0.0)
        roi[:] = blended  # zurückschreiben

        # Text auf das (nun) eingefärbte ROI zeichnen
        ytxt = pad + 18
        cv2.putText(roi, lines[0], (pad, ytxt),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        ytxt += 10
        for ln in lines[1:]:
            ytxt += 18
            cv2.putText(roi, ln, (pad, ytxt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        # DEBUG: sichtbar machen, falls Text unsichtbar wäre
        # cv2.rectangle(roi, (0,0), (w-1,h-1), (0,0,255), 1)
    # ----- Setup -----
    def setup(self) -> bool:
        logger.info("=" * 60)
        logger.info("DART VISION MVP — init (UnifiedCalibrator only)")
        logger.info("=" * 60)
        logger.info(
            "Controls: q=Quit, p=Pause, d=Debug, m=Motion overlay, r=Reset darts, s=Screenshot, c=Recalibrate, o=Overlay mode, X=Save (unified), H=Heatmap, P=Polar")

        # 1) Explizit per CLI geladen?
        if self.args.load_yaml:
            try:
                data = load_calibration_yaml(self.args.load_yaml)
                self._apply_loaded_yaml(data)  # neuer Loader (typed)
                # ⬇️ Zusatz-Log: welches Schema ist aktiv?
                if getattr(self, "uc", None) is not None:
                    logger.info("[CALIB] Unified calibration aktiv (homography+metrics+overlay_adjust+roi_adjust).")
                else:
                    logger.info("[CALIB] Typed/Legacy calibration aktiv.")
            except Exception as e:
                logger.error(f"[LOAD] Failed to load YAML from {self.args.load_yaml}: {e}")

        # 2) Legacy-Default nur laden, wenn vorhanden und kein --load-yaml gesetzt
        elif os.path.exists(CALIB_YAML):
            try:
                cfg = load_calibration_yaml(CALIB_YAML)
                if cfg:
                    self._apply_loaded_calibration(cfg)  # verträgt beide Schemata
                    logger.info(f"[CALIB] Loaded {CALIB_YAML}")
                    # ⬇️ Zusatz-Log: welches Schema ist aktiv?
                    if getattr(self, "uc", None) is not None:
                        logger.info("[CALIB] Unified calibration aktiv (homography+metrics+overlay_adjust+roi_adjust).")
                    else:
                        logger.info("[CALIB] Typed/Legacy calibration aktiv.")
            except Exception as e:
                logger.warning(f"[CALIB] Could not load {CALIB_YAML}: {e}")
        else:
            if self.args.calibrate:
                ok = self._calibration_ui()
                if not ok:
                    return False
            else:
                logger.warning("[CALIB] No calibration present; run with --calibrate for best accuracy.")

        # ROI
        self.roi = ROIProcessor(ROIConfig(roi_size=ROI_SIZE, polar_enabled=False))
        if self.homography is not None or getattr(self, "homography_eff", None) is not None:
            H_eff = self.homography_eff if getattr(self, "homography_eff", None) is not None else self.homography
            self.roi.set_homography_from_matrix(H_eff)

            # Falls noch keine Basis gesetzt ist und wir bereits einen ROI-Radius kennen:
            if (self.roi_base_radius is None) and (self.roi_board_radius and self.roi_board_radius > 0):
                self.roi_base_radius = float(self.roi_board_radius)
                logger.debug(f"[ROI] base radius set -> {self.roi_base_radius:.2f}px")

        # --- Board mapping config laden ---
        board_path = Path(self.args.board_yaml).expanduser().resolve()
        if not board_path.exists():
            logger.warning(f"[BOARD] {board_path} nicht gefunden – nutze Defaults.")
            self.board_cfg = BoardConfig()  # Defaults
        else:
            try:
                with open(board_path, "r", encoding="utf-8") as f:
                    cfg_dict = yaml.safe_load(f) or {}
                self.board_cfg = BoardConfig(**cfg_dict)
                logger.info(f"[BOARD] geladen: {board_path}")
            except Exception as e:
                logger.warning(f"[BOARD] Fehler beim Laden ({e}) – nutze Defaults.")
                self.board_cfg = BoardConfig()

        # Mapper instanziieren (nur wenn Homography/ROI-Radius vorhanden)
        # Mapper instanziieren (Unified bevorzugt)
        if self.board_cfg is not None and (
                self.homography is not None or getattr(self, "homography_eff", None) is not None):
            if getattr(self, "uc", None) is not None:
                # Single Source of Truth
                self._sync_mapper_from_unified()
                logger.info("[BOARD] Mapper init (unified) | rOD=%.1f px, rot=%.2f°",
                            self.board_mapper.calib.r_outer_double_px, self.board_mapper.calib.rotation_deg)
            else:
                # Legacy-Fallback (nur wenn wirklich keine UC vorhanden ist)
                if self.roi_board_radius and self.roi_board_radius > 0:
                    self.board_mapper = BoardMapper(
                        self.board_cfg,
                        Calibration(
                            cx=float(ROI_CENTER[0] + self.overlay_center_dx),
                            cy=float(ROI_CENTER[1] + self.overlay_center_dy),
                            r_outer_double_px=float(self.roi_board_radius) * float(getattr(self, "overlay_scale", 1.0)),
                            rotation_deg=float(getattr(self, "overlay_rot_deg", 0.0)),
                        ),
                    )
                    logger.info("[BOARD] Mapper init (legacy) | rot=%.2f°, scale=%.4f, roiR=%.1f px",
                                float(getattr(self, "overlay_rot_deg", 0.0)),
                                float(getattr(self, "overlay_scale", 1.0)),
                                float(self.roi_board_radius))
                else:
                    self.board_mapper = None
                    logger.warning("[BOARD] Mapper nicht initialisiert (fehlende Homography/Radius).")
        else:
            self.board_mapper = None
            logger.warning("[BOARD] Mapper nicht initialisiert (fehlende Homography/Config).")

        # Vision modules
        self.motion = MotionDetector(MotionConfig(
            var_threshold=self.args.motion_threshold,
            motion_pixel_threshold=self.args.motion_pixels,
            detect_shadows=True
        ))
        base_cfg = DartDetectorConfig(
            confirmation_frames=self.args.confirmation_frames,
            position_tolerance_px=20,
            min_area=10, max_area=1000,  # werden vom Preset überschrieben
        )
        det_cfg = apply_detector_preset(base_cfg, self.args.detector_preset)
        self.dart = DartImpactDetector(det_cfg)
        self.current_preset = self.args.detector_preset  # für HUD
        self.mapper = FieldMapper(FieldMapperConfig())
        self.fps = FPSCounter(window_size=30)

        # Camera
        cam_src = self.args.video if self.args.video else self.args.webcam
        is_video_file = isinstance(cam_src, str)

        cam_cfg = CameraConfig(
            src=cam_src,
            max_queue_size=5,
            buffer_size=1,
            width=self.args.width,
            height=self.args.height,
            fps=getattr(self.args, "fps", None),
            video_sync=(self.args.video_sync if is_video_file else "off"),
            playback=(self.args.playback if is_video_file else 1.0),
        )
        self.camera = ThreadedCamera(cam_cfg)
        if not self.camera.start():
            logger.error("Camera start failed.")
            return False

        # Optional: nur fürs Log (die echte Steuerung macht ThreadedCamera)
        if is_video_file:
            fps = float(self.camera.capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0
            logger.info(f"[VIDEO] nominal FPS={fps:.3f}, sync={self.args.video_sync}, speed={self.args.playback:.2f}x")
        # --- Heatmap config (optional) ---

        try:
            overlay_cfg_path = Path("src/overlay/overlay.yaml")
            overlay_cfg = yaml.safe_load(open(overlay_cfg_path, "r", encoding="utf-8")) or {}
            hcfg = (overlay_cfg or {}).get("heatmap", {}) or {}
        except Exception:
            hcfg = {}
        self.heatmap_enabled = bool(hcfg.get("enabled", True))
        self.polar_enabled = bool((hcfg.get("polar", {}) or {}).get("enabled", True))

        # --- Heatmap init (ROI-size based, so no conversions needed) ---
        if self.heatmap_enabled:
            self.hm = HeatmapAccumulator(frame_size = (ROI_SIZE[0], ROI_SIZE[1]),
                scale = float(hcfg.get("scale", 0.25)),
                alpha = float(hcfg.get("alpha", 0.35)),
                stamp_radius_px = int(hcfg.get("stamp_radius_px", 6)),
                decay_half_life_s = hcfg.get("decay_half_life_s", 120),
            )
        if self.polar_enabled:
            cell = int((hcfg.get("polar", {}) or {}).get("cell_px", 14))
            self.ph = PolarHeatmap(cell_size=(cell, cell))
        return True

    # ----- Calibration UI (only UnifiedCalibrator) -----
    def _calibration_ui(self) -> bool:


        aruco_rect_mm = None
        if self.args.aruco_size_mm:
            try:
                wmm, hmm = self.args.aruco_size_mm.lower().split("x")
                aruco_rect_mm = (float(wmm), float(hmm))
            except Exception:
                logger.warning("Could not parse --aruco-size-mm; expected format 'WxH', e.g., '600x600'")

        aruco_dict = ARUCO_DICT_MAP.get(self.args.aruco_dict.upper(), cv2.aruco.DICT_4X4_50)
        self.aruco_quad = ArucoQuadCalibrator(
            dict_name=aruco_dict,
            roi_size=400,
            expected_ids=self.args.aruco_ids,
            debug=False
        ) if self.args.aruco_quad else None
        self._aruco_rect_mm = aruco_rect_mm
        cam_src = self.args.video if self.args.video else self.args.webcam
        cam_cfg = CameraConfig(
            src=cam_src,
            width=self.args.width,
            height=self.args.height,
            max_queue_size=2,
            buffer_size=1,
            apply_charuco_tune=self.args.charuco_tune,
            on_first_frame=(lambda w, h: self.cal.set_detector_params(
                self.cal.tune_params_for_resolution(w, h)
            ))
        )
        temp = ThreadedCamera(cam_cfg)
        if not temp.start():
            logger.error("Cannot open source for calibration preview.")
            return False

        # Try to set requested resolution if provided via CLI
        if self.args.width and self.args.height:
            temp.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            temp.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)

        # Grab one frame to know actual resolution
        ok0, frame0 = temp.read()
        if ok0 and self.args.charuco_tune:
            h0, w0 = frame0.shape[:2]
            tuned = self.cal.tune_params_for_resolution(w0, h0)
            self.cal.set_detector_params(tuned)
            logger.info(f"[TUNE] Charuco/Aruco params applied for {w0}x{h0}.")

        time.sleep(0.3)
        logger.info("Calibration UI:")
        logger.info("  c = collect ChArUco sample")
        logger.info("  k = calibrate from collected samples")
        logger.info("  m = manual 4-corner homography (click TL,TR,BR,BL)")
        logger.info("  s = save calibration")
        logger.info("  q = quit")

        clicked_pts: List[Tuple[int,int]] = []
        captured_for_manual: Optional[np.ndarray] = None

        def _metrics_for_hud(gray):
            import numpy as np, cv2
            mean = float(np.mean(gray))  # brightness 0..255
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # focus proxy
            edges = cv2.Canny(gray, 80, 160)
            edge_pct = 100.0 * float(np.mean(edges > 0))
            return mean, lap_var, edge_pct

        def on_mouse(event, x, y, flags, param):
            nonlocal clicked_pts
            if event == cv2.EVENT_LBUTTONDOWN and captured_for_manual is not None and len(clicked_pts) < 4:
                clicked_pts.append((x, y))
                logger.info(f"[MANUAL] Corner {len(clicked_pts)}/4: {(x,y)}")

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", on_mouse)

        while True:
            ok, frame = temp.read()
            if not ok:
                continue
            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.args.clahe:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

            # 1) Detection (wie bisher)
            mk_c, mk_ids, ch_c, ch_ids = self.cal.detect_charuco(frame)
            mk_n = 0 if mk_ids is None else len(mk_ids)
            ch_n = 0 if ch_ids is None else len(ch_ids)

            # 2) Optional: Marker/Corners zeichnen (wie bisher)
            if mk_ids is not None and len(mk_ids) > 0:
                cv2.aruco.drawDetectedMarkers(display, mk_c, mk_ids)
            if ch_c is not None and len(ch_c) > 0:
                for pt in ch_c:
                    p = tuple(pt.astype(int).ravel())
                    cv2.circle(display, p, 3, (0, 255, 0), -1)

            # HUD + Ampel (geglättet)
            b_mean, f_var, e_pct = self._hud_compute(gray)
            hud_color = self._hud_color(b_mean, f_var, e_pct)
            y = 30
            for line in [
                f"B:{b_mean:.0f}  F:{int(f_var)}  E:{e_pct:.1f}%   (targets B~135–150, F>800, E~4–12%)",
                "Keys: [m] manual-4  [a] aruco-quad  [s] save (unified)  [q] quit",
            ]:
                cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2, cv2.LINE_AA);
                y += 22
            self._draw_traffic_light(display, b_mean, f_var, e_pct, org=(12, 105))

            # 4) Zeigen
            cv2.imshow("Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                temp.stop(); cv2.destroyWindow("Calibration")
                return self.homography is not None


            elif key == ord('a') and self.aruco_quad is not None:
                # One-shot ArUco-Quad calibration from current frame
                ok, frame = temp.read()
                if not ok or frame is None:
                    logger.warning("[ArucoQuad] no frame")
                    continue
                okH, H, mmpp, info = self.aruco_quad.calibrate_from_frame(
                    frame,
                    rect_width_mm=self._aruco_rect_mm[0] if self._aruco_rect_mm else None,
                    rect_height_mm=self._aruco_rect_mm[1] if self._aruco_rect_mm else None
                )
                if not okH:
                    logger.warning(
                        f"[ArucoQuad] failed: {info.get('reason', 'unknown')}, markers={info.get('markers', 0)}")
                    continue
                # keep last homography in app (for draw/preview/save)
                self.cal.H = H  # store in UnifiedCalibrator for consistency
                self.cal.last_image_size = frame.shape[1], frame.shape[0]

                # --- Unified: sofort in UC übernehmen ---
                # center_px: zentrale ROI/Canvas-Referenz; hier ok: Bildmitte der Preview
                h_now, w_now = frame.shape[:2]
                center_px = (w_now * 0.5, h_now * 0.5)

                # r_outer_double_px aus mm/px + bekannter Board-Geometrie ableiten, falls möglich
                r_od_px = None
                try:
                    if mmpp and getattr(self.cal, "board_diameter_mm", None):
                        r_od_px = float(self.cal.board_diameter_mm) * 0.5 / float(mmpp)
                except Exception:
                    pass
                if r_od_px is None:
                    # Fallback: falls du bereits einen ROI-Radius im State führst
                    r_od_px = float(getattr(self, "roi_board_radius", 420.0))

                self._update_uc_from_calibrator(
                    H_base=H,
                    center_px=center_px,
                    r_outer_double_px=float(r_od_px),
                    roi_adjust=(0.0, 0.0, 1.0, 0.0),  # keine ROI-Adjusts gesetzt
                    rotation_deg=0.0,  # Feine Sektorausrichtung später im Align
                )

                # Visual feedback: draw edges of ROI projection
                disp = self.aruco_quad.draw_debug(frame, [], None, None, H)
                cv2.imshow("ArucoQuad preview", disp)
                logger.info(f"[ArucoQuad] H OK | ids={info.get('ids')} | mm/px={mmpp}")

                # Auto-save if user requested immediate save
                if self.args.save_yaml:
                    data = self.aruco_quad.to_yaml_dict(H=H, mm_per_px=mmpp, rect_size_mm=self._aruco_rect_mm,
                                                        used_ids=info.get("ids"))
                    save_calibration_yaml(self.args.save_yaml, data)
                    logger.info(f"[ArucoQuad] saved YAML → {self.args.save_yaml}")


            elif key == ord('m'):
                captured_for_manual = frame.copy()
                clicked_pts.clear()
                logger.info("[Manual] frame captured; click 4 corners (TL,TR,BR,BL).")



            elif key == ord('s'):
                # 1) Falls manuelle 4 Punkte geklickt wurden und noch keine H vorhanden ist: berechnen
                if self.homography is None and captured_for_manual is not None and len(clicked_pts) == 4:
                    H, center, roi_r, mmpp = UnifiedCalibrator._homography_and_metrics(
                        np.float32(clicked_pts),
                        roi_size=ROI_SIZE[0],
                        board_diameter_mm=self.cal.board_diameter_mm
                    )
                    self.homography = H
                    self.center_px = center
                    self.roi_board_radius = roi_r
                    self.mm_per_px = mmpp
                    # Unified-Struktur aktualisieren (center/roi_r kommen hier direkt aus der Berechnung)
                    self._update_uc_from_calibrator(
                        H_base=H,
                        center_px=center,
                        r_outer_double_px=float(roi_r),
                        roi_adjust=(0.0, 0.0, 1.0, 0.0),
                        rotation_deg=0.0,
                    )

                # 2) Ohne Homography können wir nichts speichern
                if self.homography is None:
                    logger.warning("No homography yet. Use manual (m) or ArUco-Quad (a) first.")
                    continue

                # 3) Unified speichern (Einzeiler)
                self._save_calibration_unified()

                # 4) UI schließen
                temp.stop()
                cv2.destroyWindow("Calibration")
                return True

            # If calibrated intrinsics exist, also try pose each frame (informational)
            if self.cal.K is not None and self.cal.D is not None:
                okp, rvec, tvec = self.cal.estimate_pose_charuco(frame)
                if okp:
                    cv2.drawFrameAxes(disp, self.cal.K, self.cal.D, rvec, tvec, 0.08)

    def _apply_loaded_yaml(self, data: dict):
        t = (data or {}).get("type")
        if t == "charuco":
            cam = data.get("camera", {})
            K = cam.get("matrix");
            D = cam.get("dist_coeffs")
            if K is not None:
                self.cal.K = np.array(K, dtype=np.float64)
            if D is not None:
                self.cal.D = np.array(D, dtype=np.float64).reshape(-1, 1)
            self.cal._rms = float(cam.get("rms_px", 0.0))
            self.cal.last_image_size = tuple(cam.get("image_size", (0, 0)))
            H = (data.get("homography") or {}).get("H")
            if H is not None:
                self.homography = np.array(H, dtype=np.float64)
            self.logger.info("[LOAD] Applied ChArUco intrinsics from YAML.")
            # Optional: overlay_adjust
            ov = (data or {}).get("overlay_adjust") or {}
            if "rotation_deg" in ov: self.overlay_rot_deg = float(ov["rotation_deg"])
            if "scale" in ov:        self.overlay_scale = float(ov["scale"])
            if "center_dx_px" in ov: self.overlay_center_dx = float(ov["center_dx_px"])  # NEU
            if "center_dy_px" in ov: self.overlay_center_dy = float(ov["center_dy_px"])  # NEU
            # Optional: absolut gespeicherter Radius -> Scale neu berechnen
            abs_r = ov.get("r_outer_double_px")
            if abs_r is not None and self.roi_board_radius and self.roi_board_radius > 0:
                self.overlay_scale = float(abs_r) / float(self.roi_board_radius)
            # nach den existierenden overlay_adjust-Feldern
            roi_adj = (data or {}).get("roi_adjust") or {}
            self.roi_rot_deg = float(roi_adj.get("rot_deg", self.roi_rot_deg))
            self.roi_scale = float(roi_adj.get("scale", self.roi_scale))
            self.roi_tx = float(roi_adj.get("tx_px", self.roi_tx))
            self.roi_ty = float(roi_adj.get("ty_px", self.roi_ty))
            self._roi_adjust_dirty = True


        elif t == "aruco_quad":
            H = (data.get("homography") or {}).get("H")
            if H is not None:
                self.homography = np.array(H, dtype=np.float64)
            scale = data.get("scale") or {}
            self.mm_per_px = scale.get("mm_per_px")
            self.logger.info("[LOAD] Applied ArUco-Quad homography from YAML.")
            # Optional: overlay_adjust
            ov = (data or {}).get("overlay_adjust") or {}
            if "rotation_deg" in ov: self.overlay_rot_deg = float(ov["rotation_deg"])
            if "scale" in ov:        self.overlay_scale = float(ov["scale"])
            if "center_dx_px" in ov: self.overlay_center_dx = float(ov["center_dx_px"])  # NEU
            if "center_dy_px" in ov: self.overlay_center_dy = float(ov["center_dy_px"])  # NEU
            # Optional: absolut gespeicherter Radius -> Scale neu berechnen
            abs_r = ov.get("r_outer_double_px")
            if abs_r is not None and self.roi_board_radius and self.roi_board_radius > 0:
                self.overlay_scale = float(abs_r) / float(self.roi_board_radius)
            # nach den existierenden overlay_adjust-Feldern
            roi_adj = (data or {}).get("roi_adjust") or {}
            self.roi_rot_deg = float(roi_adj.get("rot_deg", self.roi_rot_deg))
            self.roi_scale = float(roi_adj.get("scale", self.roi_scale))
            self.roi_tx = float(roi_adj.get("tx_px", self.roi_tx))
            self.roi_ty = float(roi_adj.get("ty_px", self.roi_ty))
            self._roi_adjust_dirty = True


        elif t == "homography_only":
            H = (data.get("homography") or {}).get("H")
            if H is not None:
                self.homography = np.array(H, dtype=np.float64)
            metrics = data.get("metrics") or {}
            self.mm_per_px = metrics.get("mm_per_px")
            self.logger.info("[LOAD] Applied Homography-only from YAML.")
            # Optional: overlay_adjust
            ov = (data or {}).get("overlay_adjust") or {}
            if "rotation_deg" in ov: self.overlay_rot_deg = float(ov["rotation_deg"])
            if "scale" in ov:        self.overlay_scale = float(ov["scale"])
            if "center_dx_px" in ov: self.overlay_center_dx = float(ov["center_dx_px"])  # NEU
            if "center_dy_px" in ov: self.overlay_center_dy = float(ov["center_dy_px"])  # NEU
            # Optional: absolut gespeicherter Radius -> Scale neu berechnen
            abs_r = ov.get("r_outer_double_px")
            if abs_r is not None and self.roi_board_radius and self.roi_board_radius > 0:
                self.overlay_scale = float(abs_r) / float(self.roi_board_radius)
            # nach den existierenden overlay_adjust-Feldern
            roi_adj = (data or {}).get("roi_adjust") or {}
            self.roi_rot_deg = float(roi_adj.get("rot_deg", self.roi_rot_deg))
            self.roi_scale = float(roi_adj.get("scale", self.roi_scale))
            self.roi_tx = float(roi_adj.get("tx_px", self.roi_tx))
            self.roi_ty = float(roi_adj.get("ty_px", self.roi_ty))
            self._roi_adjust_dirty = True

        else:
            self.logger.warning("[LOAD] Unknown or missing type in YAML.")

    def _apply_loaded_calibration(self, cfg: dict):
        """
        Backward/forward compatible loader:
          - Unified schema (preferred): calibration.{homography, metrics, overlay_adjust, roi_adjust}
          - New typed schema (your existing): type: charuco | aruco_quad | homography_only  -> _apply_loaded_yaml()
          - Legacy flat schema: method, homography (list|{H:list}), mm_per_px, camera_matrix, dist_coeffs, ...
        """
        if not cfg:
            return

        # --- 0) Unified schema? (Single Source of Truth) --------------------------
        # Accept both nested ("calibration": {...}) and flat root.
        root = cfg.get("calibration", cfg)
        unified_keys = ("homography", "metrics", "overlay_adjust", "roi_adjust")
        if all(k in root for k in unified_keys):
            try:
                # Build UC model
                self.uc = UnifiedCalibration(
                    homography=Homography(**root["homography"]),
                    metrics=Metrics(**root["metrics"]),
                    overlay_adjust=OverlayAdjust(**root["overlay_adjust"]),
                    roi_adjust=ROIAdjust(**root["roi_adjust"]),
                )
                # Base & effective homography
                self.homography = np.asarray(self.uc.homography.H, dtype=np.float64)
                self.homography_eff = compute_effective_H(self.uc)

                # Mapper sofort synchronisieren (Unified Pfad)
                self._sync_mapper_from_unified()
                return
            except Exception as e:
                # Falls Unified vorhanden aber invalide -> sauber auf Legacy/New typed zurückfallen
                try:
                    self.logger.warning(f"Unified calibration present but failed to parse/apply: {e}")
                except Exception:
                    pass  # logger evtl. nicht initialisiert

        # --- 1) New typed schema? delegate to your existing path -------------------
        if "type" in cfg:
            self._apply_loaded_yaml(cfg)
            return

        # --- 2) Legacy schema handling (dein bestehender Code + minimal ergänzt) ---
        # homography can be a list or {"H": list}
        H_node = cfg.get("homography")
        if isinstance(H_node, dict):
            H_list = H_node.get("H")
        else:
            H_list = H_node

        self.homography = np.array(H_list, dtype=np.float64) if H_list is not None else None
        self.mm_per_px = float(cfg.get("mm_per_px", 1.0))
        self.center_px = tuple(cfg.get("center_px", [0, 0]))
        self.roi_board_radius = float(cfg.get("roi_board_radius", 160.0))

        # camera intrinsics (optional in legacy)
        if cfg.get("camera_matrix") is not None:
            self.cal.K = np.array(cfg["camera_matrix"], dtype=np.float64)
        if cfg.get("dist_coeffs") is not None:
            self.cal.D = np.array(cfg["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

        # Overlay-Adjust (legacy-style); bei Unified NICHT hier setzen
        ov = (cfg or {}).get("overlay_adjust") or {}
        if "rotation_deg" in ov: self.overlay_rot_deg = float(ov["rotation_deg"])
        if "scale" in ov:        self.overlay_scale = float(ov["scale"])
        if "center_dx_px" in ov: self.overlay_center_dx = float(ov["center_dx_px"])
        if "center_dy_px" in ov: self.overlay_center_dy = float(ov["center_dy_px"])
        abs_r = ov.get("r_outer_double_px")
        if abs_r is not None and self.roi_board_radius and self.roi_board_radius > 0:
            self.overlay_scale = float(abs_r) / float(self.roi_board_radius)

        # OPTIONAL: Aus Legacy Feldern eine flüchtige Unified-Struktur bauen,
        # damit restliche Pipeline einheitlich auf self.uc/self.homography_eff arbeitet.
        try:
            if self.homography is not None:
                H = self.homography
            else:
                H = np.eye(3, dtype=np.float64)
            # ROI center baseline, falls bei dir als Konstante hinterlegt:
            cx0 = float(getattr(self, "ROI_CENTER", (0.0, 0.0))[0])
            cy0 = float(getattr(self, "ROI_CENTER", (0.0, 0.0))[1])
            r_od = float(self.roi_board_radius) if hasattr(self, "roi_board_radius") else 160.0
            rot = float(getattr(self, "overlay_rot_deg", 0.0))
            dx = float(getattr(self, "overlay_center_dx", 0.0))
            dy = float(getattr(self, "overlay_center_dy", 0.0))

            self.uc = UnifiedCalibration(
                homography=Homography(H=H.tolist()),
                metrics=Metrics(center_px=(cx0, cy0), roi_board_radius=r_od),
                overlay_adjust=OverlayAdjust(
                    rotation_deg=rot,
                    r_outer_double_px=float(r_od * float(getattr(self, "overlay_scale", 1.0))),
                    center_dx_px=dx,
                    center_dy_px=dy,
                ),
                roi_adjust=ROIAdjust(),  # neutral (unbekannt)
            )
            self.homography_eff = compute_effective_H(self.uc)
            self._sync_mapper_from_unified()
        except Exception:
            # Wenn das fehlschlägt, läuft Legacy ohne Unified weiter.
            pass

    # ----- Pipeline -----
    def process_frame(self, frame):
        self.frame_count += 1
        timestamp = time.time() - self.session_start

        self._apply_effective_H_if_dirty()
        roi_frame = self.roi.warp_roi(frame)
        # ROI
        roi_frame = self.roi.warp_roi(frame)
        # Motion
        motion_detected, motion_event, fg_mask = self.motion.detect_motion(roi_frame, self.frame_count, timestamp)

        # Dart detection
        impact = None
        if motion_detected:
            impact = self.dart.detect_dart(roi_frame, fg_mask, self.frame_count, timestamp)
            if impact:
                self.total_darts += 1
                if impact and self.board_mapper is not None:
                    ring, sector, label = self.board_mapper.score_from_hit(
                        float(impact.position[0]), float(impact.position[1])
                    )
                    pts = self._points_from_mapping(ring, sector)
                    impact.score_label = label  # z.B. "D20", "T5", "25", "50"
                    # Game-Update
                    self.last_msg = self.game.apply_points(pts, label)
                    if self.show_debug:
                        logger.info(f"[SCORE] {label} -> {pts} | {self.last_msg}")
                    # stats update (ROI coordinates optional)
                    self.stats.add(ring=ring, sector=sector, points=pts,cx = float(impact.position[0]), cy = float(
                        impact.position[1]))
                if self.show_debug:
                    logger.info(f"[DART #{self.total_darts}] pos={impact.position} conf={impact.confidence:.2f}")
                    # --- Heatmaps update (ROI coordinates) ---
                try:
                    cx, cy = int(impact.position[0]), int(impact.position[1])
                    if self.hm is not None:
                        self.hm.add_hit(cx, cy, weight=1.0)
                    if (self.ph is not None) and (self.board_mapper is not None):
                        # ring/sector were computed above (label/pts)
                        self.ph.add(ring, sector)
                except Exception as _e:
                    if self.show_debug:
                        logger.debug(f"[HM] update skipped: {_e}")
        # 🟡 Auto-Align (alle 15 Frames, nur im ALIGN-Modus)
        if self.overlay_mode == OVERLAY_ALIGN and self.align_auto and (self.frame_count % 15 == 0):
            # 1) Ringe/Hough zuerst → Center/Radius stabilisieren
            res = self._hough_refine_rings(roi_frame)
            if res is None:
                logger.debug("[AutoAlign] Hough: no circle / rejected (ratio/jump)")
            else:
                cx, cy, r_out = res
                if self.uc is None:
                    logger.warning("[AutoAlign] Unified calib missing; skip Hough update")
                else:
                    base_cx, base_cy = self.uc.metrics.center_px
                    self.uc.overlay_adjust.center_dx_px = float(cx) - float(base_cx)
                    self.uc.overlay_adjust.center_dy_px = float(cy) - float(base_cy)
                    self.uc.overlay_adjust.r_outer_double_px = float(r_out)
                    self.homography_eff = compute_effective_H(self.uc)
                    self._sync_mapper_from_unified()
                    # optional: nicht jedes Mal speichern – hier ok, damit’s persistent ist
                    save_unified_calibration(self.calib_path, self.uc)

            # 2) Danach Winkel-Feinjustage (Sektoren)
            res2 = self._auto_sector_rotation_from_edges(roi_frame, apply=True)
            if res2 is not None:
                dth, kappa = res2
                logger.info(f"[AutoRot] Δθ={dth:+.2f}°, κ={kappa:.2f}")

        return roi_frame, motion_detected, fg_mask, impact
    def _overlay_calib(self) -> Calibration:
        """Aktuelle Overlay-Geometrie (Single Source)."""
        if self.board_mapper is not None:
            return self.board_mapper.calib
        # Fallback, falls Mapper noch nicht steht, aber UC vorhanden ist
        if self.uc is not None:
            return mapper_calibration_from_unified(self.uc)
        # letzter Fallback: Legacy (nur damit nix crasht)
        return Calibration(
            cx=float(ROI_CENTER[0] + self.overlay_center_dx),
            cy=float(ROI_CENTER[1] + self.overlay_center_dy),
            r_outer_double_px=float(self.roi_board_radius) * float(self.overlay_scale),
            rotation_deg=float(self.overlay_rot_deg),
        )

    def create_visualization(self, frame, roi_frame, motion_detected, fg_mask, impact):
        disp_main = cv2.resize(frame, (800, 600))
        disp_roi = cv2.resize(roi_frame, ROI_SIZE)

        # HUD im Laufbetrieb (nur Anzeige)

        if self.overlay_mode == OVERLAY_FULL:
            gray_main = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if getattr(self.args, "clahe", False):
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_main = clahe.apply(gray_main)
            b_mean, f_var, e_pct = self._hud_compute(gray_main)
            hud_col = self._hud_color(b_mean, f_var, e_pct)
            cv2.putText(disp_main, f"B:{b_mean:.0f} F:{int(f_var)} E:{e_pct:.1f}%", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_col, 2, cv2.LINE_AA)
            self._draw_traffic_light(disp_main, b_mean, f_var, e_pct, org=(10, 50))

        # Motion overlay
        if self.show_motion and motion_detected:
            fg_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            fg_color = cv2.resize(fg_color, ROI_SIZE)
            disp_roi = cv2.addWeighted(disp_roi, 0.9, fg_color, 0.6, 0.3)

        # Board rings (ROI)
        # ROI-Overlays nach Modus
        if self.homography is not None or self.homography_eff is not None:
            # 1) Motion zuerst (falls aktiv)
            # (Hast du schon oben gemacht – gut so)

            # 2) Einfache Referenzringe im RINGS/FULL Modus
            if self.overlay_mode >= OVERLAY_RINGS:
                cal = self._overlay_calib()
                cx = int(round(cal.cx))
                cy = int(round(cal.cy))
                r_base = int(round(cal.r_outer_double_px))  # absoluter Doppelaußenradius

                # äußerer Doppelring (grün)
                cv2.circle(disp_roi, (cx, cy), max(1, r_base), (0, 255, 0), 2, cv2.LINE_AA)
                # weitere Referenzringe (gelb) relativ zu r_base (Board-Config)
                for f in (self.board_cfg.radii.r_triple_outer,
                          self.board_cfg.radii.r_double_inner,
                          self.board_cfg.radii.r_bull_outer,
                          self.board_cfg.radii.r_bull_inner):
                    cv2.circle(disp_roi, (cx, cy), max(1, int(round(r_base * float(f)))), (255, 255, 0), 1, cv2.LINE_AA)

            # 3) Präzises Mapping nur im FULL-Modus
            if self.overlay_mode == OVERLAY_FULL and self.board_mapper is not None:
                disp_roi[:] = draw_ring_circles(disp_roi, self.board_mapper)
                disp_roi[:] = draw_sector_labels(disp_roi, self.board_mapper)
                cal = self.board_mapper.calib
                cv2.putText(disp_roi,
                            f"cx:{cal.cx:.1f} cy:{cal.cy:.1f}",
                            (ROI_SIZE[0] - 180, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 220), 1, cv2.LINE_AA)
            # --- ALIGN-Modus: dicke Doppel-/Triple-Kreise + optional Auto-Hough ---
            if self.overlay_mode == OVERLAY_ALIGN:
                # --- Heatmap overlay (ROI panel) ---
                if self.hm is not None and self.heatmap_enabled:
                    disp_roi = self.hm.render_overlay(disp_roi, roi_mask=None)
                # --- Polar mini-panel (top-left of ROI) -
                if self.ph is not None and self.polar_enabled:
                    disp_roi = self.ph.overlay_panel(disp_roi, pos=(10, 110))
                if self.board_cfg is None:
                    cv2.putText(disp_roi, "ALIGN (no board cfg)", (ROI_SIZE[0] - 240, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    # Zur Not nur den ROI-Kreis zeichnen:
                    r = int(self.roi_board_radius * self.overlay_scale)
                    cv2.circle(disp_roi, (ROI_CENTER[0], ROI_CENTER[1]), r, (0, 255, 0), 2, cv2.LINE_AA)
                # Guideline-Kreise auf ROI
                cal = self._overlay_calib()
                cx = int(round(cal.cx))
                cy = int(round(cal.cy))
                r_base = int(round(cal.r_outer_double_px))

                r_double_outer = int(r_base)
                r_double_inner = int(round(r_base * self.board_cfg.radii.r_double_inner))
                r_triple_outer = int(round(r_base * self.board_cfg.radii.r_triple_outer))
                r_triple_inner = int(round(r_base * self.board_cfg.radii.r_triple_inner))
                r_outer_bull = int(round(r_base * self.board_cfg.radii.r_bull_outer))
                r_inner_bull = int(round(r_base * self.board_cfg.radii.r_bull_inner))

                # dicke Linien für Ausrichtung
                for rr, col, th in (
                        (r_double_outer, (0, 255, 0), 1.5),
                        (r_double_inner, (0, 255, 0), 1),
                        (r_triple_outer, (0, 200, 255), 0.75),
                        (r_triple_inner, (0, 200, 255), 0.75),
                        (r_outer_bull, (255, 200, 0), 0.5),
                        (r_inner_bull, (255, 200, 0), 0.25),
                ):
                    cv2.circle(disp_roi, (int(cx), int(cy)), int(max(rr, 1)), col, int(th), cv2.LINE_AA)


                # HUD rechts oben
                cv2.putText(disp_roi, "ALIGN", (ROI_SIZE[0] - 180, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(disp_roi, f"auto:{'ON' if self.align_auto else 'OFF'}",
                            (ROI_SIZE[0] - 180, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
                cv2.putText(disp_roi, f"cx:{cx} cy:{cy} rpx:{r_base}",
                            (ROI_SIZE[0] - 180, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(disp_roi, "Keys: t=Hough once, z=Auto Hough",
                            (ROI_SIZE[0] - 290, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # --- Ein zentraler Overlay/HUD-Block oben rechts (kein Duplikat) ---
        modes = {OVERLAY_MIN: "MIN", OVERLAY_RINGS: "RINGS", OVERLAY_FULL: "FULL", OVERLAY_ALIGN:  "ALIGN "}
        cv2.putText(disp_roi, f"Overlay: {modes[self.overlay_mode]}",
                    (ROI_SIZE[0] - 180, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

        # Zeige rot/scale nur, wenn BoardMapper aktiv ist
        if self.board_mapper is not None:
            cal = self.board_mapper.calib
            if self.uc is not None:
                scale = float(cal.r_outer_double_px) / float(self.uc.metrics.roi_board_radius)
            else:
                scale = float(self.overlay_scale)  # Fallback
            cv2.putText(disp_roi, f"rot:{cal.rotation_deg:.1f}  scale:{scale:.3f}",
                        (ROI_SIZE[0] - 180, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
           ## ROI Fine-tuning Status
           #cv2.putText(
           #    disp_roi,
           #    f"ROI SRT  r:{self.roi_rot_deg:+.2f}°  s:{self.roi_scale:.4f}  tx:{self.roi_tx:+.1f} ty:{self.roi_ty:+.1f}",
           #    (ROI_SIZE[0] - 330, 86),
           #    cv2.FONT_HERSHEY_SIMPLEX,
           #    0.5,
           #    (220, 220, 220),
           #    1,
           #    cv2.LINE_AA
           #)
        # Optional: zuletzt empfangenen Extended-Keycode (falls gesetzt)
        if getattr(self, "last_key_dbg", "") and self.overlay_mode == OVERLAY_FULL:
            cv2.putText(disp_roi, f"key:{self.last_key_dbg}",
                        (ROI_SIZE[0] - 180, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Impact markers
        for imp in self.dart.get_confirmed_impacts():
            cv2.circle(disp_roi, imp.position, 5, (0, 255, 255), 2)
            cv2.circle(disp_roi, imp.position, 2, (0, 255, 255), -1)
            if hasattr(imp, "score_label") and imp.score_label:
                cv2.putText(disp_roi, imp.score_label, (imp.position[0] + 10, imp.position[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Game-HUD unten links im ROI
        y0 = ROI_SIZE[1] - 60
        mode_txt = "ATC" if self.game.mode == GameMode.ATC else "301"
        cv2.putText(disp_roi, f"Game: {mode_txt}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        y0 += 28
        if self.game.mode == GameMode.ATC:
            status_txt = "FINISH" if self.game.done else f"Target: {self.game.target}"
        else:
            status_txt = "FINISH" if self.game.done else f"Score: {self.game.score}"
        cv2.putText(disp_roi, status_txt, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


        # Letzter Wurf (rechts unten)
        if self.last_msg:
            cv2.putText(disp_roi, self.last_msg, (10, ROI_SIZE[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2, cv2.LINE_AA)


        # Debug HUD
        if self.show_debug and self.fps is not None:
            stats = self.fps.get_stats()
            cv2.putText(disp_roi, f"FPS: {stats.fps_median:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(disp_roi, f"Time: {stats.frame_time_ms:.1f}ms", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(disp_roi, f"Darts: {self.total_darts}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        if not self.show_help:
            cv2.putText(disp_roi, f"Preset: {getattr(self, 'current_preset', 'balanced')}",
                        (ROI_SIZE[0] - 180, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        if getattr(self, "show_help", False):
            self._draw_help_overlay(disp_roi)

        canvas = np.zeros((600, 1200, 3), dtype=np.uint8)
        canvas[0:600, 0:800] = disp_main
        canvas[(600-ROI_SIZE[1])//2:(600-ROI_SIZE[1])//2+ROI_SIZE[1], 800:800+ROI_SIZE[0]] = disp_roi
        return canvas

    # ----- Run loop -----
    def run(self):
        if not self.setup():
            logger.error("Setup failed.")
            return
        diag = self.cal.selftest()
        if not diag["ok"]:
            logger.warning(f"[SelfTest] Hints: {diag['messages']}")

        self.running = True
        logger.info("Main loop started.")
        try:
            while self.running:
                ok, frame = self.camera.read(timeout=0.1)
                if not ok:
                    continue

                self.fps.update()

                if not self.paused:
                    roi_frame, motion, fg_mask, impact = self.process_frame(frame)
                else:
                    roi_frame = self.roi.warp_roi(frame)
                    motion = False
                    fg_mask = np.zeros(ROI_SIZE[::-1], dtype=np.uint8)
                    impact = None

                disp = self.create_visualization(frame, roi_frame, motion, fg_mask, impact)

                if self.paused:
                    cv2.putText(disp, "PAUSED", (500, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,165,255), 3)


                # jetzt die Tasten abfragen (bei Video gerne ein paar ms Blockzeit nutzen)
                wait_delay = 1
                if self.src_is_video and self.args.video_sync == "off":
                    # wenn "off", trotzdem minimal blocken damit die GUI reagiert
                    wait_delay = 1
                raw_key = cv2.waitKeyEx(wait_delay)
                key = raw_key & 0xFF

                cv2.imshow("Dart Vision MVP", disp)


                # 2) Debug: zuletzt empfangenen Key anzeigen
                if raw_key != -1:
                    logger.debug(f"raw_key={raw_key} (0x{raw_key:08X}) masked={key}")
                    self.last_key_dbg = f"{raw_key} (0x{raw_key:08X})"
                # 3) Pfeiltasten (immer auf raw_key prüfen!)
                if raw_key in (VK_LEFT, OCV_LEFT):
                    self.overlay_rot_deg -= 0.5
                    self._sync_mapper()
                    if self.board_mapper:
                        self.board_mapper.calib.rotation_deg = float(self.overlay_rot_deg)
                    logger.info(f"[OVERLAY] rot={self.overlay_rot_deg:.2f} deg")
                elif raw_key in (VK_RIGHT, OCV_RIGHT):
                    self.overlay_rot_deg += 0.5
                    self._sync_mapper()
                    if self.board_mapper:
                        self.board_mapper.calib.rotation_deg = float(self.overlay_rot_deg)
                    logger.info(f"[OVERLAY] rot={self.overlay_rot_deg:.2f} deg")
                elif raw_key in (VK_UP, OCV_UP):
                    self.overlay_scale = min(1.20, self.overlay_scale * 1.01)  # +1%
                    self._sync_mapper()
                    if self.board_mapper:
                        self.board_mapper.calib.r_outer_double_px = float(self.roi_board_radius) * float(
                            self.overlay_scale)
                    logger.info(f"[OVERLAY] scale={self.overlay_scale:.4f}")
                elif raw_key in (VK_DOWN, OCV_DOWN):
                    self.overlay_scale = max(0.80, self.overlay_scale / 1.01)  # -1%
                    self._sync_mapper()
                    if self.board_mapper:
                        self.board_mapper.calib.r_outer_double_px = float(self.roi_board_radius) * float(
                            self.overlay_scale)
                    logger.info(f"[OVERLAY] scale={self.overlay_scale:.4f}")

                # 4) ASCII-Tasten (auf 'key' prüfen)
                if key == ord('q'):
                    self.running = False
                elif key == ord('1'):
                    self.dart.config = apply_detector_preset(self.dart.config, "aggressive")
                    self.current_preset = "aggressive"
                    logger.info("[PRESET] detector -> aggressive")
                elif key == ord('2'):
                    self.dart.config = apply_detector_preset(self.dart.config, "balanced")
                    self.current_preset = "balanced"
                    logger.info("[PRESET] detector -> balanced")
                elif key == ord('3'):
                    self.dart.config = apply_detector_preset(self.dart.config, "stable")
                    self.current_preset = "stable"
                    logger.info("[PRESET] detector -> stable")
                elif key == ord('p'):
                    self.paused = not self.paused
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                elif key == ord('m'):
                    self.show_motion = not self.show_motion
                elif key == ord('r'):
                    self.dart.clear_impacts();
                    self.total_darts = 0
                elif key == ord('s'):
                    # Screenshot (s bleibt Screenshot)
                    fn = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(fn, disp)
                    logger.info(f"Saved {fn}")
                elif key == ord('c'):
                    self._recalibrate_and_apply()
                elif key == ord('g'):
                    self.game.reset();
                    self.last_msg = ""
                    logger.info(f"[GAME] Reset {self.game.mode}")
                elif key == ord('h'):
                    new_mode = GameMode._301 if self.game.mode == GameMode.ATC else GameMode.ATC
                    self.game.switch_mode(new_mode);
                    self.last_msg = ""
                    logger.info(f"[GAME] Switch to {self.game.mode}")
                elif key == ord('H'):
                    self.heatmap_enabled = not self.heatmap_enabled
                    logger.info(f"[HEATMAP] image-space overlay -> {'ON' if self.heatmap_enabled else 'OFF'}")
                elif key == ord('P'):
                    self.polar_enabled = not self.polar_enabled
                    logger.info(f"[HEATMAP] polar panel -> {'ON' if self.polar_enabled else 'OFF'}")
                elif key ==ord('?'):  # Shift+/, oder H als Alternative
                    self.show_help = not self.show_help
                    logger.info(f"[HELP] overlay -> {'ON' if self.show_help else 'OFF'}")

                elif key == ord('R'):
                    self.overlay_center_dx = 0.0
                    self.overlay_center_dy = 0.0
                    self.overlay_scale = 1.0
                    if self.board_mapper:
                        self.board_mapper.calib.cx = float(ROI_CENTER[0])
                        self.board_mapper.calib.cy = float(ROI_CENTER[1])
                        self.board_mapper.calib.r_outer_double_px = float(self.roi_board_radius)
                    logger.info("[OVERLAY] reset center/scale")
                elif key in (ord('t'), ord('T')):
                    res = self._hough_refine_rings(roi_frame)
                    if res is None:
                        logger.info("[HoughRings] no circles detected in current ROI")
                    else:
                        cx, cy, r_out = res
                        if self.uc is None:
                            logger.warning("[HoughRings] Unified calib not present in memory.")
                        else:
                            # center_dx/dy sind Offsets zur Basismitte (metrics.center_px)
                            base_cx, base_cy = self.uc.metrics.center_px
                            self.uc.overlay_adjust.center_dx_px = float(cx) - float(base_cx)
                            self.uc.overlay_adjust.center_dy_px = float(cy) - float(base_cy)
                            self.uc.overlay_adjust.r_outer_double_px = float(r_out)

                            # Eff. H aktualisieren + Mapper syncen + speichern
                            self.homography_eff = compute_effective_H(self.uc)
                            if hasattr(self, "_sync_mapper_from_unified"):
                                self._sync_mapper_from_unified()
                        save_unified_calibration(self.calib_path, self.uc)

                        logger.info("[HoughRings] cx=%.1f cy=%.1f rOD=%.1f", cx, cy, r_out)
                elif key == ord('z'):
                    self.align_auto = not self.align_auto
                    logger.info(f"[ALIGN] auto={'ON' if self.align_auto else 'OFF'} (mode must be ALIGN to run)")
                elif key == ord('o'):
                    self.overlay_mode = (self.overlay_mode + 1) % 4  # jetzt 0..3
                    modes = {OVERLAY_MIN: "MIN", OVERLAY_RINGS: "RINGS", OVERLAY_FULL: "FULL", OVERLAY_ALIGN: "ALIGN"}
                    logger.info(f"[OVERLAY] mode -> {modes[self.overlay_mode]}")
                elif key == ord('j'):  # left
                    self.overlay_center_dx -= 1.0;
                    self._sync_mapper()
                elif key == ord('l'):  # right
                    self.overlay_center_dx += 1.0;
                    self._sync_mapper()
                elif key == ord('i'):  # up
                    self.overlay_center_dy -= 1.0;
                    self._sync_mapper()
                elif key == ord('k'):  # down (nur wenn du Screenshot auf 's' lassen willst -> nimm z.B. ';' statt 's')
                    self.overlay_center_dy += 1.0;
                    self._sync_mapper()
                elif key in (ord('x'), ord('X')):
                    self._save_calibration_unified()  # <— EINZEILER
                elif key in (ord('y'), ord('Y')):
                    res = self._auto_sector_rotation_from_edges(roi_frame, apply=True)
                    if res is None:
                        logger.info("[AutoRot] no angular structure detected")
                    else:
                        dth, kappa = res
                        logger.info(f"[AutoRot] applied delta={dth:+.2f}°, kappa={kappa:.2f}")

                if raw_key == 0xA0000 or raw_key == 0xA30000:
                    pass  # (nur als Beispiel: SHIFT-Flags, optional)
                if key == ord('0'):
                    self.roi_tx = self.roi_ty = 0.0
                    self.roi_scale = 1.0
                    self.roi_rot_deg = 0.0
                    self._roi_adjust_dirty = True
                elif raw_key in (VK_LEFT, OCV_LEFT) and (cv2.getWindowProperty("Dart Vision MVP", 0) == 0 or True):
                    self.roi_tx -= STEP_T;
                    self._roi_adjust_dirty = True
                elif raw_key in (VK_RIGHT, OCV_RIGHT):
                    self.roi_tx += STEP_T;
                    self._roi_adjust_dirty = True
                elif raw_key in (VK_UP, OCV_UP):
                    self.roi_ty -= STEP_T;
                    self._roi_adjust_dirty = True
                elif raw_key in (VK_DOWN, OCV_DOWN):
                    self.roi_ty += STEP_T;
                    self._roi_adjust_dirty = True
                elif key == ord('+') or key == ord('='):
                    self.roi_scale *= STEP_S;
                    self._roi_adjust_dirty = True
                elif key == ord('-') or key == ord('_'):
                    self.roi_scale /= STEP_S;
                    self._roi_adjust_dirty = True
                elif key == ord(','):
                    self.roi_rot_deg -= STEP_R;
                    self._roi_adjust_dirty = True
                elif key == ord('.'):
                    self.roi_rot_deg += STEP_R;
                    self._roi_adjust_dirty = True


        except KeyboardInterrupt:
            logger.info("Interrupted.")
        finally:
            self.cleanup()

    def _recalibrate_and_apply(self):
        cv2.destroyAllWindows()
        if self.camera:
            self.camera.stop()

        ok = self._calibration_ui()

        if not ok:
            logger.warning("[RECAL] UI aborted/no homography. Keeping previous calibration.")
        else:
            # 1) Homographie ins ROI übernehmen
            if self.homography is not None:
                self.roi.set_homography_from_matrix(self.homography)
            else:
                logger.warning("[RECAL] No homography returned from UI.")

            # 2) Overlay-Offsets evtl. aus YAML nachladen (falls im UI mit 's' gespeichert)
            try:
                cfg = load_calibration_yaml(CALIB_YAML) or {}
                ov = (cfg or {}).get("overlay_adjust") or {}
                if "rotation_deg" in ov:
                    self.overlay_rot_deg = float(ov["rotation_deg"])
                if "scale" in ov:
                    self.overlay_scale = float(ov["scale"])
            except Exception as e:
                logger.debug(f"[RECAL] No overlay_adjust in YAML: {e}")
        # 2a) Basisradius (nur einmal pro Kalibrierung) festlegen oder zurücksetzen

        if (self.roi_board_radius and self.roi_board_radius > 0):
            self.roi_base_radius = float(self.roi_board_radius)
            logger.debug(f"[RECAL] base radius reset -> {self.roi_base_radius:.2f}px")
        else:
            logger.debug("[RECAL] base radius not set (roi_board_radius missing)")

        # 3) Board mapping config laden (IMMER versuchen – nicht im except!)
        from pathlib import Path
        board_path = Path(getattr(self.args, "board_yaml", "board.yaml")).expanduser().resolve()
        if not board_path.exists():
            logger.warning(f"[BOARD] {board_path} nicht gefunden – nutze Defaults.")
            self.board_cfg = BoardConfig()  # Defaults
        else:
            try:
                with open(board_path, "r", encoding="utf-8") as f:
                    cfg_dict = yaml.safe_load(f) or {}
                self.board_cfg = BoardConfig(**cfg_dict)
                logger.info(f"[BOARD] geladen: {board_path}")
            except Exception as e:
                logger.warning(f"[BOARD] Fehler beim Laden ({e}) – nutze Defaults.")
                self.board_cfg = BoardConfig()

        # 4) Mapper neu aufbauen, wenn möglich (nach Homography/Radius/Offsets)
        if self.homography is not None and self.board_cfg is not None and self.roi_board_radius and self.roi_board_radius > 0:
            self.board_mapper = BoardMapper(
                self.board_cfg,
                Calibration(
                    cx=float(ROI_CENTER[0] + self.overlay_center_dx),
                    cy=float(ROI_CENTER[1] + self.overlay_center_dy),
                    r_outer_double_px=float(self.roi_board_radius) * float(self.overlay_scale),
                    rotation_deg=float(self.overlay_rot_deg),
                ),
            )
            logger.info(f"[RECAL] Mapper updated | rot={self.overlay_rot_deg:.2f}°, "
                        f"scale={self.overlay_scale:.4f}, roiR={self.roi_board_radius:.1f}px")
        else:
            self.board_mapper = None
            logger.warning("[RECAL] Mapper not updated (missing homography/board_cfg/roi_board_radius)")

        # 5) Kamera neu starten (IMMER)
        cam_src = self.args.video if self.args.video else self.args.webcam
        self.camera = ThreadedCamera(CameraConfig(
            src=cam_src, max_queue_size=5, buffer_size=1,
            width=self.args.width, height=self.args.height
        ))
        if not self.camera.start():
            logger.error("[RECAL] camera.start() failed")
        else:
            logger.info("[RECAL] camera restarted")


    def cleanup(self):
        logger.info("Cleaning up...")
        if self.camera:
            self.camera.stop()
        cv2.destroyAllWindows()
        # --- Stats & heatmap exports ---
        try:
            os.makedirs("reports", exist_ok=True)
            # session stats JSON + CSVs
            self.stats.export_json(f"reports/session_{self.session_id}_summary.json")
            self.stats.export_csv_dists(
                ring_csv=f"reports/session_{self.session_id}_ring_dist.csv",
                sector_csv=f"reports/session_{self.session_id}_sector_dist.csv",
                matrix_csv=f"reports/session_{self.session_id}_ring_sector_matrix.csv",
            )
            # If your PolarHeatmap is active, export CSV/PNG too
            if getattr(self, "ph", None) is not None:
                self.ph.export_csv(f"reports/polar_heatmap_{self.session_id}.csv")
                self.ph.export_png(f"reports/polar_heatmap_{self.session_id}.png")
            if getattr(self, "hm", None) is not None:
                self.hm.export_png(f"reports/heatmap_{self.session_id}.png")
        except Exception as e:
            logger.debug(f"[REPORT] export skipped: {e}")


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Dart Vision MVP — UnifiedCalibrator only")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--video", "-v", type=str, help="Video file")
    grp.add_argument("--webcam", "-w", type=int, default=0, help="Webcam index")

    p.add_argument("--calibrate", "-c", action="store_true", help="Run calibration UI first")
    p.add_argument("--width", type=int, default=1920, help="Camera width")
    p.add_argument("--height", type=int, default=1080, help="Camera height")
    p.add_argument("--charuco-tune", action="store_true",
                   help="Auto-tune Charuco/Aruco detector params during calibration UI")

    p.add_argument("--game", type=str, choices=[GameMode.ATC, GameMode._301], default=GameMode.ATC,
                   help="Mini-Spielmodus: 'atc' (Around the Clock) oder '301'")

    p.add_argument("--save-yaml", type=str, default="out/calibration.yaml",
                   help="Path to write calibration YAML (both charuco or aruco-quad)")
    p.add_argument("--aruco-quad", action="store_true",
                   help="Enable ArUco-Quad mode in calibration UI (hotkey 'a')")
    p.add_argument("--aruco-dict", type=str, default="4X4_50",
                   help="ArUco dictionary (e.g., 4X4_50, 6X6_250)")
    p.add_argument("--aruco-ids", type=int, nargs="*", default=None,
                   help="Expected IDs for the 4 markers (optional)")
    p.add_argument("--aruco-size-mm", type=str, default=None,
                   help="Physical rectangle size as WxH in mm, e.g. '600x600' or '800x600'")
    p.add_argument("--board-yaml", type=str, default="board.yaml",
                   help="Pfad zur Board-Geometrie (normierte Radien/Sektoren)")

    p.add_argument("--motion-threshold", type=int, default=50, help="MOG2 variance threshold")
    p.add_argument("--motion-pixels", type=int, default=500, help="Min motion pixels")
    p.add_argument("--confirmation-frames", type=int, default=3, help="Frames to confirm dart")
    p.add_argument("--load-yaml", type=str, default=None, help="Load calibration YAML on startup")
    p.add_argument("--clahe", action="store_true",
                   help="Enable CLAHE on grayscale for HUD/detection")
    p.add_argument("--detector-preset", type=str, choices=["aggressive", "balanced", "stable"],
                   default="balanced", help="Dart detector preset")
    p.add_argument("--video-sync", choices=["off", "fps", "msec"], default="fps",
                   help="Playback-Synchronisation für Datei-Input: 'off' (so schnell wie möglich), 'fps' (per CAP_PROP_FPS), 'msec' (per Frame-Timestamp)")
    p.add_argument("--playback", type=float, default=1.0,
                   help="Abspielgeschwindigkeit für Datei-Input (1.0 = Echtzeit, 0.5 = halb so schnell, 2.0 = doppelt so schnell)")
    p.add_argument("--fps", type=float, default=None)

    args = p.parse_args()
    DartVisionApp(args).run()

if __name__ == "__main__":
    main()
