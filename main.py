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
import json
import yaml
import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from collections import deque
from src.game.game import DemoGame, GameMode


# Board mapping/overlays/config
from src.board import BoardConfig, BoardMapper, Calibration, draw_ring_circles, draw_sector_labels


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
CALIB_YAML = Path("config/calibration_unified.yaml")


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
        self.mm_per_px: float = 1.0
        self.roi_board_radius: float = 160.0
        self.center_px: Tuple[int, int] = (0, 0)

        # UI state
        self.running = False
        self.paused = False
        self.show_debug = True
        self.show_motion = False
        self.frame_count = 0
        self.session_start = time.time()
        self.total_darts = 0

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

    def _hough_refine_center(self, roi_bgr) -> bool:
        """
        Findet den äußeren Doppelring als Kreis und passt Overlay-Center & Scale an.
        Return True, wenn aktualisiert wurde.
        """
        if roi_bgr is None:
            return False

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Kontrast optional anheben
        if getattr(self.args, "clahe", False):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # leicht glätten + Kanten
        gray = cv2.medianBlur(gray, 5)
        edges_hi = 120  # du kannst das feintunen
        edges_lo = int(edges_hi * 0.5)

        # Hough-Parameter um r_guess herum
        # Nutze die Basis, falls vorhanden, sonst den aktuellen ROI-Radius
        r_guess = int(max(10, self.roi_base_radius if self.roi_base_radius else self.roi_board_radius))
        r_guess = int(max(10, self.roi_board_radius))
        minR = int(0.85 * r_guess)
        maxR = int(1.15 * r_guess)

        # Hough
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=edges_hi, param2=40,
            minRadius=minR, maxRadius=maxR
        )

        if circles is None or len(circles) == 0:
            self.last_msg = "Hough: no circle"
            return False

        circles = np.round(circles[0, :]).astype(int)

        # Nimm den Kreis, der dem ROI_CENTER am nächsten liegt (oder den größten)
        cx0, cy0 = ROI_CENTER
        circles = sorted(circles, key=lambda c: (c[2], -abs(c[0] - cx0) - abs(c[1] - cy0)), reverse=True)
        c_best = circles[0]
        cx, cy, r = int(c_best[0]), int(c_best[1]), int(c_best[2])

        # Delta zur aktuellen Overlay-Mitte
        dx = float(cx - cx0)
        dy = float(cy - cy0)

        # Sanity: kleine Offsets und sinnvolle Radien erzwingen
        if abs(dx) > ROI_SIZE[0] * 0.2 or abs(dy) > ROI_SIZE[1] * 0.2:
            self.last_msg = "Hough: center drift too large, ignored"
            return False
        if r < 0.6 * r_guess or r > 1.4 * r_guess:
            self.last_msg = "Hough: radius out of range, ignored"
            return False

        # Anwenden: Center-Offsets inkrementell addieren
        self.overlay_center_dx += dx
        self.overlay_center_dy += dy
        # Skalierung RELATIV zur festen Basis (nicht kumulativ!)
        base = float(self.roi_base_radius if self.roi_base_radius else self.roi_board_radius)
        if base <= 0:
             self.last_msg = "Hough: invalid base radius"
        return False
        self.overlay_scale = float(r) / base
        # Schutzkorridor gegen Ausreißer
        self.overlay_scale = max(0.80, min(1.20, self.overlay_scale))
        self._roi_adjust_dirty = True

        if self.board_mapper is not None:
            self.board_mapper.calib.cy = float(ROI_CENTER[1] + self.overlay_center_dy)
            self.board_mapper.calib.r_outer_double_px = float(self.roi_board_radius) * float(
            self.overlay_scale)

        self.last_msg = f"Hough OK: dx={dx:+.1f}, dy={dy:+.1f}, r={r}"
        return True

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

    # ----- Setup -----
    def setup(self) -> bool:
        logger.info("=" * 60)
        logger.info("DART VISION MVP — init (UnifiedCalibrator only)")
        logger.info("=" * 60)

        # 1) Explizit per CLI geladen?
        if self.args.load_yaml:
            try:
                data = load_calibration_yaml(self.args.load_yaml)
                self._apply_loaded_yaml(data)  # neuer Loader (typed)
            except Exception as e:
                logger.error(f"[LOAD] Failed to load YAML from {self.args.load_yaml}: {e}")

        # 2) Legacy-Default nur laden, wenn vorhanden und kein --load-yaml gesetzt
        elif os.path.exists(CALIB_YAML):
            try:
                cfg = load_calibration_yaml(CALIB_YAML)
                if cfg:
                    self._apply_loaded_calibration(cfg)  # verträgt beide Schemata
                    logger.info(f"[CALIB] Loaded {CALIB_YAML}")
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
        if self.homography is not None:
            self.roi.set_homography_from_matrix(self.homography)
            # Falls noch keine Basis gesetzt ist und wir bereits einen ROI-Radius kennen:
            if (self.roi_base_radius is None) and (self.roi_board_radius and self.roi_board_radius > 0):
                self.roi_base_radius = float(self.roi_board_radius)
                logger.debug(f"[ROI] base radius set -> {self.roi_base_radius:.2f}px")

        # --- Board mapping config laden ---
        from pathlib import Path
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
        if self.homography is not None and self.board_cfg is not None and self.roi_board_radius > 0:
            self.board_mapper = BoardMapper(
                self.board_cfg,
                Calibration(
                    cx=float(ROI_CENTER[0] + self.overlay_center_dx),
                    cy = float(ROI_CENTER[1] + self.overlay_center_dy),
                    r_outer_double_px = float(self.roi_board_radius) * float(self.overlay_scale),
                    rotation_deg=float(self.overlay_rot_deg),
                ),
            )

            logger.info(f"[BOARD] Mapper init | rot={getattr(self, 'overlay_rot_deg', 0.0):.2f}°, "
                        f"scale={getattr(self, 'overlay_scale', 1.0):.4f}, roiR={self.roi_board_radius:.1f}px")
        else:
            self.board_mapper = None
            logger.warning("[BOARD] Mapper nicht initialisiert (fehlende Homography/Radius).")

        # Vision modules
        self.motion = MotionDetector(MotionConfig(
            var_threshold=self.args.motion_threshold,
            motion_pixel_threshold=self.args.motion_pixels,
            detect_shadows=True
        ))
        self.dart = DartImpactDetector(DartDetectorConfig(
            confirmation_frames=self.args.confirmation_frames,
            position_tolerance_px=10,
            min_area=10,
            max_area=1000
        ))
        self.mapper = FieldMapper(FieldMapperConfig())
        self.fps = FPSCounter(window_size=30)

        # Camera
        cam_src = self.args.video if self.args.video else self.args.webcam
        cam_cfg = CameraConfig(src=cam_src, max_queue_size=5, buffer_size=1,
                               width=self.args.width, height=self.args.height)
        self.camera = ThreadedCamera(cam_cfg)
        if not self.camera.start():
            logger.error("Camera start failed.")
            return False

        logger.info("Controls: q=Quit, p=Pause, d=Debug, m=Motion overlay, r=Reset darts, s=Screenshot, c=Recalibrate, o=Overlay mode")
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
                "Keys: [m] manual-4  [a] aruco-quad  [s] save  [q] quit",
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
                # 1) Falls manuell 4 Punkte geklickt wurden und noch keine H vorhanden ist: berechnen
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

                # 2) Ohne Homography können wir nichts speichern
                if self.homography is None:
                    logger.warning("No homography yet. Use manual (m) or ArUco-Quad (a) first.")
                    continue

                # 3) Aktuelle Bildgröße für Metadaten (falls möglich)
                img_w = img_h = None
                ok_frame, frame_now = temp.read()  # temp ist deine Preview-Kamera in der UI
                if ok_frame and frame_now is not None:
                    img_h, img_w = frame_now.shape[:2]

                # 4) Fall A: ChArUco-Kalibrierung vorhanden → charuco YAML
                if getattr(self.cal, "K", None) is not None:
                    data = {
                        "type": "charuco",
                        "board": {
                            "squares_x": self.cal.squares_x,
                            "squares_y": self.cal.squares_y,
                            "square_length_m": float(self.cal.square_length_m),
                            "marker_length_m": float(self.cal.marker_length_m),
                            "dictionary": int(self.cal.dict_type),
                        },
                        "camera": {
                            "matrix": self.cal.K.tolist(),
                            "dist_coeffs": None if getattr(self.cal, "D", None) is None else self.cal.D.tolist(),
                            "rms_px": float(getattr(self.cal, "_rms", 0.0)),
                            "image_size": [int(img_w or 0), int(img_h or 0)],

                        },

                        # optional: falls du parallel eine Homography (manuell oder ArUco) gesetzt hast
                        "homography": {"H": self.homography.tolist()},
                        "metrics": {
                            "mm_per_px": float(self.mm_per_px) if self.mm_per_px is not None else None,
                            "center_px": [int(self.center_px[0]), int(self.center_px[1])] if getattr(self, "center_px",
                                                                                                     None) is not None else None,
                            "roi_board_radius": float(self.roi_board_radius) if getattr(self, "roi_board_radius",
                                                                                        None) is not None else None,
                        },

                    }
                    save_calibration_yaml(CALIB_YAML, data)
                    logger.info(f"[SAVE] ChArUco YAML → {CALIB_YAML}")
                    temp.stop();
                    cv2.destroyWindow("Calibration")
                    return True

                # 5) Fall B: ArUco-Quad verwendet → aruco_quad YAML
                #    Hinweis: Lege im 'a'-Hotkey (Aruco-Quad) idealerweise self.aruco_last_info = info an.
                if hasattr(self, "aruco_quad") and self.aruco_quad is not None:
                    # Versuche Zusatzinfos zu ziehen (IDs, Rechteckgröße in mm)
                    used_ids = None
                    rect_mm = None
                    mmpp = float(self.mm_per_px) if self.mm_per_px is not None else None
                    if hasattr(self, "aruco_last_info") and isinstance(self.aruco_last_info, dict):
                        used_ids = self.aruco_last_info.get("ids")
                    if hasattr(self, "_aruco_rect_mm") and self._aruco_rect_mm:
                        rect_mm = [float(self._aruco_rect_mm[0]), float(self._aruco_rect_mm[1])]
                    data = {
                        "type": "aruco_quad",
                        "aruco": {
                            "dictionary": int(self.aruco_quad.aruco_dict.bytesList.shape[0]),  # informativ
                            "expected_ids": used_ids,
                        },
                        "roi": {"size_px": int(self.aruco_quad.roi_size)},
                        "homography": {"H": self.homography.tolist()},
                        "scale": {
                            "mm_per_px": mmpp,
                            "rect_width_mm": rect_mm[0] if rect_mm else None,
                            "rect_height_mm": rect_mm[1] if rect_mm else None,
                        },
                        "image_size": [int(img_w or 0), int(img_h or 0)],
                    }
                    save_calibration_yaml(CALIB_YAML, data)
                    logger.info(f"[SAVE] ArUco-Quad YAML → {CALIB_YAML}")
                    temp.stop();
                    cv2.destroyWindow("Calibration")
                    return True

                # 6) Fall C: Nur manuelle 4-Punkt-Homographie → homography_only YAML
                data = {
                    "type": "homography_only",
                    "homography": {"H": self.homography.tolist()},
                    "metrics": {
                        "mm_per_px": float(self.mm_per_px) if self.mm_per_px is not None else None,
                        "center_px": [int(self.center_px[0]), int(self.center_px[1])] if getattr(self, "center_px",
                                                                                                    None) is not None else None,
                        "roi_board_radius": float(self.roi_board_radius) if getattr(self, "roi_board_radius",
                                                                                    None) is not None else None,
                    },
                    "image_size": [int(img_w or 0), int(img_h or 0)],
                }
                save_calibration_yaml(CALIB_YAML, data)
                logger.info(f"[SAVE] Homography YAML → {CALIB_YAML}")
                temp.stop();
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
          - New typed schema: type: charuco | aruco_quad | homography_only
          - Legacy flat schema: method, homography (list), mm_per_px, camera_matrix, dist_coeffs, ...
        """
        if not cfg:
            return

        # --- New typed schema? delegate to _apply_loaded_yaml ---
        if "type" in cfg:
            self._apply_loaded_yaml(cfg)
            return

        # --- Legacy schema handling ---
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
        ov = (cfg or {}).get("overlay_adjust") or {}
        if "rotation_deg" in ov: self.overlay_rot_deg = float(ov["rotation_deg"])
        if "scale" in ov:        self.overlay_scale = float(ov["scale"])
        if "center_dx_px" in ov: self.overlay_center_dx = float(ov["center_dx_px"])
        if "center_dy_px" in ov: self.overlay_center_dy = float(ov["center_dy_px"])
        abs_r = ov.get("r_outer_double_px")
        if abs_r is not None and self.roi_board_radius and self.roi_board_radius > 0:
            self.overlay_scale = float(abs_r) / float(self.roi_board_radius)


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
                if self.show_debug:
                    logger.info(f"[DART #{self.total_darts}] pos={impact.position} conf={impact.confidence:.2f}")

        return roi_frame, motion_detected, fg_mask, impact

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
        if self.homography is not None:
            # 1) Motion zuerst (falls aktiv)
            # (Hast du schon oben gemacht – gut so)

            # 2) Einfache Referenzringe im RINGS/FULL Modus
            if self.overlay_mode >= OVERLAY_RINGS:
                cx, cy, r_base = self._overlay_center_radius()
                # äußerer Doppelring (grün)
                cv2.circle(disp_roi, (cx, cy), r_base, (0, 255, 0), 2, cv2.LINE_AA)
                # weitere Referenzringe (gelb) skaliert um r_base
                for f in (0.05, 0.095, self.board_cfg.radii.r_triple_outer,
                          self.board_cfg.radii.r_double_inner):
                    cv2.circle(disp_roi, (cx, cy), max(1, int(r_base * float(f))), (255, 255, 0), 1, cv2.LINE_AA)

            # 3) Präzises Mapping nur im FULL-Modus
            if self.overlay_mode == OVERLAY_FULL and self.board_mapper is not None:
                disp_roi[:] = draw_ring_circles(disp_roi, self.board_mapper)
                disp_roi[:] = draw_sector_labels(disp_roi, self.board_mapper)
                cv2.putText(disp_roi,
                            f"cx:{ROI_CENTER[0] + self.overlay_center_dx:.1f} cy:{ROI_CENTER[1] + self.overlay_center_dy:.1f}",
                            (ROI_SIZE[0] - 180, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 220), 1, cv2.LINE_AA)
            # --- ALIGN-Modus: dicke Doppel-/Triple-Kreise + optional Auto-Hough ---
            if self.overlay_mode == OVERLAY_ALIGN:
                if self.board_cfg is None:
                    cv2.putText(disp_roi, "ALIGN (no board cfg)", (ROI_SIZE[0] - 240, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    # Zur Not nur den ROI-Kreis zeichnen:
                    r = int(self.roi_board_radius * self.overlay_scale)
                    cv2.circle(disp_roi, (ROI_CENTER[0], ROI_CENTER[1]), r, (0, 255, 0), 2, cv2.LINE_AA)
                # Guideline-Kreise auf ROI
                r_base = int(self.roi_board_radius * self.overlay_scale)
                r_double_outer = int(r_base)
                r_double_inner = int(r_base * self.board_cfg.radii.r_double_inner)
                r_triple_outer = int(r_base * self.board_cfg.radii.r_triple_outer)
                r_triple_inner = int(r_base * self.board_cfg.radii.r_triple_inner)
                r_outer_bull = int(r_base * self.board_cfg.radii.r_bull_outer)
                r_inner_bull = int(r_base * self.board_cfg.radii.r_bull_inner)
                cx = int(ROI_CENTER[0] + self.overlay_center_dx)
                cy = int(ROI_CENTER[1] + self.overlay_center_dy)
                # dicke Linien für Ausrichtung
                for rr, col, th in (
                        (r_double_outer, (0, 255, 0), 3),
                        (r_double_inner, (0, 255, 0), 3),
                        (r_triple_outer, (0, 200, 255), 2),
                        (r_triple_inner, (0, 200, 255), 2),
                        (r_outer_bull, (255, 200, 0), 2),
                        (r_inner_bull, (255, 200, 0), 2),
                ):
                    cv2.circle(disp_roi, (int(cx), int(cy)), int(max(rr, 1)), col, int(th), cv2.LINE_AA)

                # Auto-Hough im ALIGN-Modus (z. B. alle 10 Frames)
                if self.align_auto and (self.frame_count % 10 == 0):
                    _ = self._hough_refine_center(roi_frame)
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
            cv2.putText(disp_roi, f"rot:{self.overlay_rot_deg:.1f}  scale:{self.overlay_scale:.3f}",
                        (ROI_SIZE[0] - 180, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
            # ROI Fine-tuning Status
            cv2.putText(
                disp_roi,
                f"ROI SRT  r:{self.roi_rot_deg:+.2f}°  s:{self.roi_scale:.4f}  tx:{self.roi_tx:+.1f} ty:{self.roi_ty:+.1f}",
                (ROI_SIZE[0] - 330, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )
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

                cv2.imshow("Dart Vision MVP", disp)

                # 1) Key holen (Extended Keys für Pfeile!)
                raw_key = cv2.waitKeyEx(1)
                key = raw_key & 0xFF  # nur für 'normale' Tasten wie 'q','s',...
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

                elif key == ord('t'):
                    # einmalige automatische Ausrichtung via Hough
                    changed = self._hough_refine_center(roi_frame)
                    self._sync_mapper()
                    if changed:
                        logger.info("[HOUGH] overlay refined: "
                                    f"cx={ROI_CENTER[0] + self.overlay_center_dx:.1f}, "
                                    f"cy={ROI_CENTER[1] + self.overlay_center_dy:.1f}, "
                                    f"scale={self.overlay_scale:.4f}")
                    else:
                        logger.info("[HOUGH] no update")

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
                elif key == ord('x'):
                    try:
                        cfg = load_calibration_yaml(CALIB_YAML) or {}
                    except Exception:
                        cfg = {}
                    cfg.setdefault("overlay_adjust", {})
                    cfg["overlay_adjust"]["rotation_deg"] = float(self.overlay_rot_deg)
                    cfg["overlay_adjust"]["scale"] = float(self.overlay_scale)
                    cfg["overlay_adjust"]["center_dx_px"] = float(self.overlay_center_dx)  # NEU
                    cfg["overlay_adjust"]["center_dy_px"] = float(self.overlay_center_dy)  # NEU
                    # Zusätzlich absoluter, bereits skalierter Doppelaußenradius:
                    cfg["overlay_adjust"]["r_outer_double_px"] = float(self.roi_board_radius) * float(
                        self.overlay_scale)  # NEU
                    save_calibration_yaml(CALIB_YAML, cfg)
                    logger.info(f"[OVERLAY] saved to {CALIB_YAML} (rot={self.overlay_rot_deg:.2f}, "
                                f"scale={self.overlay_scale:.4f}, dx={self.overlay_center_dx:.1f}, "
                                f"dy={self.overlay_center_dy:.1f}, rpx={cfg['overlay_adjust']['r_outer_double_px']:.1f})")

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
        if self.profiler and len(self.profiler.timings) > 0:
            logger.info(self.profiler.get_report())
        dur = time.time() - self.session_start
        logger.info(f"Duration: {dur:.1f}s | Frames: {self.frame_count}")


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

    args = p.parse_args()
    DartVisionApp(args).run()

if __name__ == "__main__":
    main()
