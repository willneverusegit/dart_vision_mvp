"""
App Initializer - Modular setup for DartVisionApp.

Provides clean separation of initialization concerns:
- Calibration loading
- Camera setup
- Vision modules initialization
- Board config & mapping
- Heatmap configuration
"""

import os
import logging
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import time

import cv2

from src.capture import ThreadedCamera, CameraConfig, FPSCounter
from src.calibration.roi_processor import ROIProcessor, ROIConfig
from src.vision import (
    MotionDetector,
    DartImpactDetector,
    FieldMapper, FieldMapperConfig
)
from src.vision.detector_config_manager import DetectorConfigManager
from src.vision.environment_optimizer import EnvironmentOptimizer
from src.board import BoardConfig, BoardMapper, Calibration
from src.overlay.heatmap import HeatmapAccumulator
from src.analytics.polar_heatmap import PolarHeatmap
from src.calibration.calib_io import load_calibration_yaml

logger = logging.getLogger(__name__)


class AppInitializer:
    """
    Handles initialization of DartVisionApp components.

    Separates setup concerns into logical stages:
    1. Calibration loading
    2. ROI processor setup
    3. Board config & mapper
    4. Vision modules (motion, dart, field mapper)
    5. Camera initialization
    6. Heatmap configuration
    """

    def __init__(self, app, args):
        """
        Initialize app initializer.

        Args:
            app: DartVisionApp instance
            args: Command-line arguments
        """
        self.app = app
        self.args = args
        self.logger = logging.getLogger("AppInitializer")
        self.detector_manager = DetectorConfigManager(Path(getattr(args, "detector_config", "config/detectors.yaml")))
        self.environment_optimizer = EnvironmentOptimizer()

    def initialize(self) -> bool:
        """
        Run full initialization sequence.

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("=" * 60)
        self.logger.info("DART VISION MVP — init (UnifiedCalibrator only)")
        self.logger.info("=" * 60)
        self.logger.info(
            "Controls: q=Quit, p=Pause, d=Debug, m=Motion overlay, r=Reset darts, "
            "s=Screenshot, c=Recalibrate, o=Overlay mode, X=Save (unified), H=Heatmap, P=Polar"
        )

        # Stage 1: Calibration
        if not self._init_calibration():
            return False

        # Stage 2: ROI Processor
        self._init_roi_processor()

        # Stage 3: Board Config & Mapper
        self._init_board_config()
        self._init_board_mapper()

        # Stage 4: Vision Modules
        self._init_vision_modules()

        # Stage 5: Camera
        if not self._init_camera():
            return False

        # Stage 6: Heatmap
        self._init_heatmap()

        return True

    def _init_calibration(self) -> bool:
        """
        Load calibration from YAML or run calibration UI.

        Returns:
            True if successful, False if calibration UI was aborted
        """
        from src.calibration.calib_io import load_calibration_yaml

        CALIB_YAML = Path("calibration_unified.yaml")

        # 1) Explicitly loaded via CLI?
        if self.args.load_yaml:
            try:
                data = load_calibration_yaml(self.args.load_yaml)
                self.app._apply_loaded_yaml(data)

                if getattr(self.app, "uc", None) is not None:
                    self.logger.info(
                        "[CALIB] Unified calibration aktiv (homography+metrics+overlay_adjust+roi_adjust)."
                    )
                else:
                    self.logger.info("[CALIB] Typed/Legacy calibration aktiv.")
            except Exception as e:
                self.logger.error(f"[LOAD] Failed to load YAML from {self.args.load_yaml}: {e}")

        # 2) Legacy default only if exists and no --load-yaml
        elif os.path.exists(CALIB_YAML):
            try:
                cfg = load_calibration_yaml(CALIB_YAML)
                if cfg:
                    self.app._apply_loaded_calibration(cfg)
                    self.logger.info(f"[CALIB] Loaded {CALIB_YAML}")

                    if getattr(self.app, "uc", None) is not None:
                        self.logger.info(
                            "[CALIB] Unified calibration aktiv (homography+metrics+overlay_adjust+roi_adjust)."
                        )
                    else:
                        self.logger.info("[CALIB] Typed/Legacy calibration aktiv.")
            except Exception as e:
                self.logger.warning(f"[CALIB] Could not load {CALIB_YAML}: {e}")
        else:
            if self.args.calibrate:
                ok = self.app._calibration_ui()
                if not ok:
                    return False
            else:
                self.logger.warning(
                    "[CALIB] No calibration present; run with --calibrate for best accuracy."
                )

        return True

    def _init_roi_processor(self):
        """Initialize ROI processor with homography if available."""
        ROI_SIZE = self.app.roi.config.roi_size if self.app.roi else (400, 400)

        self.app.roi = ROIProcessor(ROIConfig(roi_size=ROI_SIZE, polar_enabled=False))

        if self.app.homography is not None or getattr(self.app, "homography_eff", None) is not None:
            H_eff = (
                self.app.homography_eff
                if getattr(self.app, "homography_eff", None) is not None
                else self.app.homography
            )
            self.app.roi.set_homography_from_matrix(H_eff)

            # Set base radius if available
            if (self.app.roi_base_radius is None) and (
                self.app.roi_board_radius and self.app.roi_board_radius > 0
            ):
                self.app.roi_base_radius = float(self.app.roi_board_radius)
                self.logger.debug(f"[ROI] base radius set -> {self.app.roi_base_radius:.2f}px")

    def _init_board_config(self):
        """Load board configuration from YAML and initialize Hough aligner."""
        board_path = Path(self.args.board_yaml).expanduser().resolve()

        if not board_path.exists():
            self.logger.warning(f"[BOARD] {board_path} nicht gefunden – nutze Defaults.")
            self.app.board_cfg = BoardConfig()
        else:
            try:
                with open(board_path, "r", encoding="utf-8") as f:
                    cfg_dict = yaml.safe_load(f) or {}
                self.app.board_cfg = BoardConfig(**cfg_dict)
                self.logger.info(f"[BOARD] geladen: {board_path}")
            except Exception as e:
                self.logger.warning(f"[BOARD] Fehler beim Laden ({e}) – nutze Defaults.")
                self.app.board_cfg = BoardConfig()

        # Initialize Hough aligner manager after board config is loaded
        from src.board.hough_aligner_manager import HoughAlignerManager
        self.app.hough_aligner = HoughAlignerManager(self.app, self.app.board_cfg)
        self.logger.debug("[HOUGH] HoughAlignerManager initialized")

    def _init_board_mapper(self):
        """Initialize board mapper (unified or legacy mode)."""
        if self.app.board_cfg is None:
            self.app.board_mapper = None
            self.logger.warning("[BOARD] Mapper nicht initialisiert (fehlende Config).")
            return

        if self.app.homography is None and getattr(self.app, "homography_eff", None) is None:
            self.app.board_mapper = None
            self.logger.warning("[BOARD] Mapper nicht initialisiert (fehlende Homography).")
            return

        # Unified mode (preferred)
        if getattr(self.app, "uc", None) is not None:
            self.app._sync_mapper_from_unified()
            self.app._ensure_roi_annulus_mask()

            self.logger.info(
                "[BOARD] Mapper init (unified) | rOD=%.1f px, rot=%.2f°",
                self.app.board_mapper.calib.r_outer_double_px,
                self.app.board_mapper.calib.rotation_deg
            )

            # Propagate board center to dart detector
            if hasattr(self.app, "dart") and self.app.dart is not None and self.app.board_mapper is not None:
                self.app.dart.config.cal_cx = float(self.app.board_mapper.calib.cx)
                self.app.dart.config.cal_cy = float(self.app.board_mapper.calib.cy)
                self.logger.debug(
                    f"[DART] Cal center propagated: "
                    f"({self.app.dart.config.cal_cx:.1f}, {self.app.dart.config.cal_cy:.1f})"
                )

        # Legacy fallback
        else:
            if self.app.roi_board_radius and self.app.roi_board_radius > 0:
                ROI_CENTER = (200.0, 200.0)  # Default
                self.app.board_mapper = BoardMapper(
                    self.app.board_cfg,
                    Calibration(
                        cx=float(ROI_CENTER[0] + getattr(self.app, "overlay_center_dx", 0.0)),
                        cy=float(ROI_CENTER[1] + getattr(self.app, "overlay_center_dy", 0.0)),
                        r_outer_double_px=float(self.app.roi_board_radius) * float(
                            getattr(self.app, "overlay_scale", 1.0)
                        ),
                        rotation_deg=float(getattr(self.app, "overlay_rot_deg", 0.0)),
                    ),
                )
                self.logger.info(
                    "[BOARD] Mapper init (legacy) | rot=%.2f°, scale=%.4f, roiR=%.1f px",
                    float(getattr(self.app, "overlay_rot_deg", 0.0)),
                    float(getattr(self.app, "overlay_scale", 1.0)),
                    float(self.app.roi_board_radius)
                )
            else:
                self.app.board_mapper = None
                self.logger.warning("[BOARD] Mapper nicht initialisiert (fehlende Radius).")

    def _init_vision_modules(self):
        """Initialize motion detector, dart detector, field mapper, and FPS counter."""
        motion_cfg, dart_cfg = self.detector_manager.get_configs()

        overrides_applied = False
        if getattr(self.args, "motion_threshold", None) is not None:
            motion_cfg.var_threshold = int(self.args.motion_threshold)
            overrides_applied = True
        if getattr(self.args, "motion_pixels", None) is not None:
            motion_cfg.motion_pixel_threshold = int(self.args.motion_pixels)
            overrides_applied = True
        if getattr(self.args, "confirmation_frames", None) is not None:
            dart_cfg.confirmation_frames = int(self.args.confirmation_frames)
            overrides_applied = True

        dart_cfg = self.detector_manager.apply_preset(
            dart_cfg, getattr(self.args, "detector_preset", None)
        )

        self.app.motion = MotionDetector(motion_cfg)
        self.app.dart = DartImpactDetector(dart_cfg)
        self.app.current_preset = getattr(self.args, "detector_preset", None)

        if overrides_applied:
            self.logger.info("[CONFIG] CLI overrides active (not persisted).")

        self.app.detector_manager = self.detector_manager

        # Record active profile if already optimized elsewhere
        self.app.environment_profile = None

        # Field mapper
        self.app.mapper = FieldMapper(FieldMapperConfig())

        # FPS counter
        self.app.fps = FPSCounter(window_size=30)

    def _init_camera(self) -> bool:
        """
        Initialize camera (webcam or video file).

        Returns:
            True if successful, False otherwise
        """
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

        self.app.camera = ThreadedCamera(cam_cfg)
        if not self.app.camera.start():
            self.logger.error("Camera start failed.")
            return False

        # Log video info
        if is_video_file:
            fps = float(self.app.camera.capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0
            self.logger.info(
                f"[VIDEO] nominal FPS={fps:.3f}, sync={self.args.video_sync}, "
                f"speed={self.args.playback:.2f}x"
            )

        if getattr(self.args, "auto_optimize", False):
            self._auto_optimize_detectors()

        return True

    def _auto_optimize_detectors(self) -> None:
        """Collect sample frames and adapt detector parameters."""
        self.logger.info("[CONFIG] Auto-optimizing detector parameters...")
        frames = []
        target = int(getattr(self.args, "optimize_samples", 90) or 90)
        deadline = time.time() + max(5.0, target / 15.0)

        while len(frames) < target and time.time() < deadline:
            ok, frame = self.app.camera.read(timeout=0.5)
            if ok and frame is not None:
                frames.append(frame.copy())

        if not frames:
            self.logger.warning("[CONFIG] Auto-optimization skipped (no frames captured)")
            return

        motion_cfg, dart_cfg, profile = self.environment_optimizer.optimize(
            frames, self.app.motion.config, self.app.dart.config
        )

        self.app.motion = MotionDetector(motion_cfg)
        self.app.dart = DartImpactDetector(dart_cfg)
        self.app.environment_profile = profile
        self.app.current_preset = profile.recommended_preset

        if getattr(self.args, "persist_optimized", False):
            self.detector_manager.save(motion_cfg, dart_cfg)
            self.logger.info("[CONFIG] Optimized parameters persisted to YAML")
        else:
            self.logger.info("[CONFIG] Optimized parameters active for this session")

    def _init_heatmap(self):
        """Initialize heatmap accumulators if enabled."""
        try:
            overlay_cfg_path = Path("src/overlay/overlay.yaml")
            overlay_cfg = yaml.safe_load(open(overlay_cfg_path, "r", encoding="utf-8")) or {}
            hcfg = (overlay_cfg or {}).get("heatmap", {}) or {}
        except Exception:
            hcfg = {}

        self.app.heatmap_enabled = bool(hcfg.get("enabled", True))
        self.app.polar_enabled = bool((hcfg.get("polar", {}) or {}).get("enabled", True))

        # ROI-based heatmap
        if self.app.heatmap_enabled:
            ROI_SIZE = (400, 400)  # From app config
            self.app.hm = HeatmapAccumulator(
                frame_size=(ROI_SIZE[0], ROI_SIZE[1]),
                scale=float(hcfg.get("scale", 0.25)),
                alpha=float(hcfg.get("alpha", 0.35)),
                stamp_radius_px=int(hcfg.get("stamp_radius_px", 6)),
                decay_half_life_s=hcfg.get("decay_half_life_s", 120),
            )

        # Polar heatmap
        if self.app.polar_enabled:
            cell = int((hcfg.get("polar", {}) or {}).get("cell_px", 14))
            self.app.ph = PolarHeatmap(cell_size=(cell, cell))
