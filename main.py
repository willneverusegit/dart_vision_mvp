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
from src.board.hough_circle import HoughCircleAligner
from src.overlay.heatmap import HeatmapAccumulator
from src.analytics.polar_heatmap import PolarHeatmap
from src.analytics.stats_accumulator import StatsAccumulator

# UI Components
from src.ui import HUDRenderer, OverlayRenderer
from src.ui.overlay_renderer import OVERLAY_MIN, OVERLAY_RINGS, OVERLAY_FULL, OVERLAY_ALIGN


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
        self.show_mask = False  # per Hotkey toggeln
        self.show_mask_debug = False
        self.game_mode = True  # Game mode on/off (when off, no overlay)

        # UI Components (refactored)
        self.hud_renderer = HUDRenderer(smoothing_window=15)
        self.overlay_renderer = OverlayRenderer(
            roi_size=ROI_SIZE,
            main_size=(800, 600),
            canvas_size=(1420, 600),
            metrics_sidebar_width=220,
        )
        self.hough_aligner = None  # HoughAlignerManager, initialized after board_cfg

        # Hotkey handler (initialized after app is ready)
        self.hotkeys = None
        self._last_disp = None  # For screenshot functionality
        self._last_roi_frame = None  # For Hough functionality

        # Mapping/Scoring
        self.board_cfg: Optional[BoardConfig] = None
        self.board_mapper: Optional[BoardMapper] = None
        self._roi_annulus_mask = None  # uint8 mask same size as ROI_SIZE, 0/255

        # Overlay-Mode (0=min, 1=rings, 2=full)
        self.overlay_mode = OVERLAY_ALIGN
        # Overlay fine-tune (applied only to BoardMapper projection)
        self.overlay_rot_deg: float = 0.0  # add to Calibration.rotation_deg
        self.overlay_scale: float = 1.0  # multiplies roi_board_radius
        self.overlay_center_dx: float = 0.0
        self.overlay_center_dy: float = 0.0
        self.align_auto: bool = False  # NEU: Hough-Autoausrichtung an/aus
        self.show_help = False  # Hilfe-Overlay an/aus
        self.board_calibration_mode = False  # Board overlay calibration mode (colored dartboard)

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
        self.current_preset = "balanced"  # Track current detector preset

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

    def _ensure_roi_annulus_mask(self):
        """Build a static annulus mask in ROI coords (0..255) to ignore Netz/Hintergrund."""
        if self._roi_annulus_mask is not None or self.board_mapper is None or self.board_cfg is None:
            return
        import numpy as np, cv2
        h, w = ROI_SIZE[1], ROI_SIZE[0]
        m = np.zeros((h, w), np.uint8)
        cal = self.board_mapper.calib
        r_out = int(round(cal.r_outer_double_px * 1.02))  # leicht großzügig
        r_in = int(round(cal.r_outer_double_px * max(0.28, self.board_cfg.radii.r_bull_outer * 0.8)))
        cx, cy = int(round(cal.cx)), int(round(cal.cy))
        cv2.circle(m, (cx, cy), max(1, r_out), 255, -1)
        cv2.circle(m, (cx, cy), max(1, r_in), 0, -1)
        self._roi_annulus_mask = m

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
            self._roi_annulus_mask = None
            self._ensure_roi_annulus_mask()
            if self.dart is not None and self.board_mapper is not None:
                self.dart.config.cal_cx = float(self.board_mapper.calib.cx)
                self.dart.config.cal_cy = float(self.board_mapper.calib.cy)

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

    # _ratio_consistent removed - now in HoughCircleAligner

    # _hough_refine_center removed - now in HoughCircleAligner
    # _hough_refine_rings removed - now in HoughAlignerManager
    # _auto_sector_rotation_from_edges removed - now in HoughAlignerManager
    # HUD methods removed - now handled by HUDRenderer

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

    # _draw_traffic_light removed - now handled by HUDRenderer

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

    # _draw_help_overlay removed - now handled by HUDRenderer
    # ----- Setup -----
    def setup(self) -> bool:
        """
        Initialize app components.

        Delegates to AppInitializer for clean, maintainable setup.
        """
        from src.core import AppInitializer

        initializer = AppInitializer(self, self.args)
        return initializer.initialize()

    # ----- Calibration UI (only UnifiedCalibrator) -----
    def _calibration_ui(self) -> bool:
        """
        Run interactive calibration UI.
        
        Delegates to CalibrationUIManager for clean, maintainable calibration.
        """
        from src.calibration.calibration_ui_manager import CalibrationUIManager
        from src.calibration.aruco_quad_calibrator import ArucoQuadCalibrator
        from src.capture import ThreadedCamera, CameraConfig
        import time

        # Parse ArUco rectangle size
        aruco_rect_mm = None
        if self.args.aruco_size_mm:
            try:
                wmm, hmm = self.args.aruco_size_mm.lower().split("x")
                aruco_rect_mm = (float(wmm), float(hmm))
            except Exception:
                logger.warning("Could not parse --aruco-size-mm; expected format 'WxH'")

        # Setup ArUco-Quad calibrator if requested
        aruco_dict = ARUCO_DICT_MAP.get(self.args.aruco_dict.upper(), cv2.aruco.DICT_4X4_50)
        aruco_quad = ArucoQuadCalibrator(
            dict_name=aruco_dict,
            roi_size=400,
            expected_ids=self.args.aruco_ids,
            debug=False
        ) if self.args.aruco_quad else None

        # Setup temporary camera for calibration
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
        
        temp_camera = ThreadedCamera(cam_cfg)
        if not temp_camera.start():
            logger.error("Cannot open source for calibration preview.")
            return False

        # Apply resolution if specified
        if self.args.width and self.args.height:
            temp_camera.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            temp_camera.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)

        # Tune for actual resolution
        ok0, frame0 = temp_camera.read()
        if ok0 and self.args.charuco_tune:
            h0, w0 = frame0.shape[:2]
            tuned = self.cal.tune_params_for_resolution(w0, h0)
            self.cal.set_detector_params(tuned)
            logger.info(f"[TUNE] ChArUco/AruCo params applied for {w0}x{h0}.")

        time.sleep(0.3)

        # Create UI manager
        ui_manager = CalibrationUIManager(
            calibrator=self.cal,
            camera=temp_camera,
            roi_size=ROI_SIZE,
            aruco_quad_calibrator=aruco_quad,
            aruco_rect_mm=aruco_rect_mm,
            use_clahe=getattr(self.args, "clahe", False),
            hud_renderer=self.hud_renderer
        )

        # Callback to update app state when calibration completes
        def on_calibration_complete(H, center_px, r_outer_double_px, mm_per_px):
            self._update_uc_from_calibrator(
                H_base=H,
                center_px=center_px,
                r_outer_double_px=float(r_outer_double_px),
                roi_adjust=(0.0, 0.0, 1.0, 0.0),
                rotation_deg=0.0
            )
            self.homography = H
            self.center_px = center_px
            self.roi_board_radius = r_outer_double_px
            self.mm_per_px = mm_per_px

        ui_manager.on_calibration_complete = on_calibration_complete

        # Run interactive UI
        success = ui_manager.run_interactive_ui()

        # Cleanup
        temp_camera.stop()

        # Save if successful
        if success:
            self._save_calibration_unified()

        return success

    def _apply_loaded_yaml(self, data: dict):
        """
        Apply calibration from YAML data (typed schema).

        Delegates to CalibLoader for clean, maintainable loading.
        """
        from src.calibration.calib_loader import CalibLoader

        loader = CalibLoader(self)
        loader.apply_yaml_data(data)

    def _apply_loaded_calibration(self, cfg: dict):
        """
        Apply calibration from any schema format.

        Delegates to CalibLoader for clean, maintainable loading.
        Supports unified, typed, and legacy schemas.
        """
        from src.calibration.calib_loader import CalibLoader

        loader = CalibLoader(self)
        loader.apply_calibration(cfg)

    # ----- Pipeline -----
    def process_frame(self, frame):
        """
        Process a single frame through the pipeline.

        Delegates to FrameProcessor for clean, maintainable processing.
        """
        if not hasattr(self, 'frame_processor'):
            from src.pipeline import FrameProcessor
            self.frame_processor = FrameProcessor(self)

        return self.frame_processor.process_frame(frame)
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
        """
        Create visualization using modular renderers.
        
        Delegates to visualization_helper for clean, maintainable rendering.
        """
        from src.ui.visualization_helper import create_visualization_refactored
        
        return create_visualization_refactored(
            # Renderers
            hud_renderer=self.hud_renderer,
            overlay_renderer=self.overlay_renderer,
            
            # Input frames
            frame=frame,
            roi_frame=roi_frame,
            motion_detected=motion_detected,
            fg_mask=fg_mask,
            
            # State
            overlay_mode=self.overlay_mode,
            show_debug=self.show_debug,
            show_motion=self.show_motion,
            show_mask=self.show_mask,
            show_help=self.show_help,
            annulus_mask=self._roi_annulus_mask,
            
            # Components
            dart_detector=self.dart,
            board_mapper=self.board_mapper,
            board_cfg=self.board_cfg,
            calibration=self._overlay_calib(),
            game=self.game,
            
            # Optional components
            heatmap_accumulator=self.hm,
            polar_heatmap=self.ph,
            heatmap_enabled=self.heatmap_enabled,
            polar_enabled=self.polar_enabled,
            
            # HUD data
            fps_counter=self.fps,
            total_darts=self.total_darts,
            last_msg=self.last_msg,
            current_preset=self.current_preset,
            align_auto=self.align_auto,
            unified_calibration=self.uc,

            # System state
            paused=self.paused,

            # Board calibration mode
            board_calibration_mode=self.board_calibration_mode,

            # Game mode
            game_mode=self.game_mode,

            # CLAHE flag
            use_clahe=getattr(self.args, "clahe", False)
        )

    def toggle_hud_cards(self) -> None:
        """Toggle visibility of all HUD cards (sidebar and ROI)."""

        if self.overlay_renderer is None:
            return
        enabled = self.overlay_renderer.toggle_cards_enabled()
        state = "ON" if enabled else "OFF"
        logger.info(f"[HUD] cards {state}")
        self.last_msg = f"HUD cards {state}"

    # ----- Run loop -----
    def run(self):
        if not self.setup():
            logger.error("Setup failed.")
            return
        diag = self.cal.selftest()
        if not diag["ok"]:
            logger.warning(f"[SelfTest] Hints: {diag['messages']}")

        # Initialize hotkey handler
        from src.input.dart_vision_hotkeys import DartVisionHotkeys
        self.hotkeys = DartVisionHotkeys(self)
        logger.info("Hotkey system initialized")

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

                # Store for hotkey callbacks
                self._last_disp = disp
                self._last_roi_frame = roi_frame

                cv2.imshow("Dart Vision MVP", disp)

                # Handle keyboard input
                wait_delay = 1
                if self.src_is_video and self.args.video_sync == "off":
                    wait_delay = 1
                raw_key = cv2.waitKeyEx(wait_delay)


                # Handle keyboard input with HotkeyHandler
                if raw_key != -1:
                    logger.debug(f"raw_key={raw_key} (0x{raw_key:08X})")
                    self.last_key_dbg = f"{raw_key} (0x{raw_key:08X})"
                    # Try both raw_key and masked key
                    handled = self.hotkeys.handle_key(raw_key)
                    if not handled:
                        key = raw_key & 0xFF
                        self.hotkeys.handle_key(key)


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
