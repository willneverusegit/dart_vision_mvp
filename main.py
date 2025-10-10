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
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

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


# ---------- Logging ----------
def setup_logging():
    fh = logging.FileHandler('dart_vision.log', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)

setup_logging()
logger = logging.getLogger("main")


# ---------- Small helpers ----------
CALIB_YAML = Path("config/calibration_unified.yaml")
ROI_SIZE = (400, 400)
ROI_CENTER = (ROI_SIZE[0] // 2, ROI_SIZE[1] // 2)
ARUCO_DICT_MAP = {
    "4X4_50":  cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50":  cv2.aruco.DICT_5X5_50,
    "6X6_250": cv2.aruco.DICT_6X6_250,
}


def save_calibration_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    logger.info(f"[CALIB] Saved → {path}")

def load_calibration_yaml(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------- Main App ----------
class DartVisionApp:
    def __init__(self, args):
        self.args = args

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

    # ----- Setup -----
    def setup(self) -> bool:
        logger.info("=" * 60)
        logger.info("DART VISION MVP — init (UnifiedCalibrator only)")
        logger.info("=" * 60)

        # Load previous calibration if present
        cfg = load_calibration_yaml(CALIB_YAML)
        if cfg is not None:
            self._apply_loaded_calibration(cfg)
            logger.info(f"[CALIB] Loaded {CALIB_YAML} | method={cfg.get('method')} | RMS={cfg.get('rms', 0):.4f}")
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

        logger.info("Controls: q=Quit, p=Pause, d=Debug, m=Motion overlay, r=Reset darts, s=Screenshot, c=Recalibrate")
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

            # Detection status (for display)
            mk_c, mk_ids, ch_c, ch_ids = self.cal.detect_charuco(frame)
            disp = frame.copy()
            if mk_ids is not None and len(mk_ids) > 0:
                cv2.aruco.drawDetectedMarkers(disp, mk_c, mk_ids)
            if ch_ids is not None and len(ch_ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(disp, ch_c, ch_ids, (0,255,0))

            # Manual markers
            if captured_for_manual is not None:
                for i, pt in enumerate(clicked_pts):
                    cv2.circle(disp, pt, 6, (0,255,255), -1)
                    cv2.putText(disp, str(i+1), (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # HUD
            y=30
            cv2.putText(disp, f"ChArUco corners: {0 if ch_ids is None else len(ch_ids)}", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,220,50), 2); y+=30
            cv2.putText(disp, f"samples: {len(self.cal._samples_corners)}  (c=collect, k=calibrate, m=manual, s=save, q=quit)", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2); y+=25

            cv2.imshow("Calibration", disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                temp.stop(); cv2.destroyWindow("Calibration")
                return self.homography is not None

            elif key == ord('c'):
                if self.cal.add_charuco_sample(frame):
                    logger.info("[Charuco] sample collected.")
                else:
                    logger.info("[Charuco] not enough corners; try different angle/distance.")

            elif key == ord('k'):
                # Run full charuco calibration
                try:
                    rms = self.cal.calibrate_charuco()
                    logger.info(f"[Charuco] calibrated, RMS={rms:.4f}")
                except Exception as e:
                    logger.error(f"[Charuco] calibration failed: {e}")

            elif key == ord('a') and self.aruco_quad is not None:
                # One-shot ArUco-Quad calibration from current frame
                ok, frame = self.temp_cam.read()
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
                            "square_length_m": float(self.cal.square_length),
                            "marker_length_m": float(self.cal.marker_length),
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

        # unreachable

    def _apply_loaded_calibration(self, cfg: dict):
        self.homography = np.array(cfg["homography"], dtype=np.float32) if cfg.get("homography") is not None else None
        self.mm_per_px = float(cfg.get("mm_per_px", 1.0))
        self.center_px = tuple(cfg.get("center_px", [0,0]))
        self.roi_board_radius = float(cfg.get("roi_board_radius", 160.0))
        if cfg.get("camera_matrix") is not None:
            self.cal.K = np.array(cfg["camera_matrix"], dtype=np.float64)
        if cfg.get("dist_coeffs") is not None:
            self.cal.D = np.array(cfg["dist_coeffs"], dtype=np.float64)

    # ----- Pipeline -----
    def process_frame(self, frame):
        self.frame_count += 1
        timestamp = time.time() - self.session_start

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
                if self.show_debug:
                    logger.info(f"[DART #{self.total_darts}] pos={impact.position} conf={impact.confidence:.2f}")

        return roi_frame, motion_detected, fg_mask, impact

    def create_visualization(self, frame, roi_frame, motion_detected, fg_mask, impact):
        disp_main = cv2.resize(frame, (800, 600))
        disp_roi = cv2.resize(roi_frame, ROI_SIZE)

        # Motion overlay
        if self.show_motion and motion_detected:
            fg_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            fg_color = cv2.resize(fg_color, ROI_SIZE)
            disp_roi = cv2.addWeighted(disp_roi, 0.7, fg_color, 0.3, 0)

        # Board rings (ROI)
        if self.homography is not None and self.show_debug:
            r = int(self.roi_board_radius)
            cv2.circle(disp_roi, ROI_CENTER, r, (0, 255, 0), 2)
            for f in (0.05, 0.095, 0.53, 0.58, 0.94):
                cv2.circle(disp_roi, ROI_CENTER, int(r * f), (255, 255, 0), 1)

        # Impact markers
        for imp in self.dart.get_confirmed_impacts():
            cv2.circle(disp_roi, imp.position, 12, (0, 255, 255), 2)
            cv2.circle(disp_roi, imp.position, 3, (0, 255, 255), -1)

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
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    self.running = False
                elif key == ord('p'):
                    self.paused = not self.paused
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                elif key == ord('m'):
                    self.show_motion = not self.show_motion
                elif key == ord('r'):
                    self.dart.clear_impacts(); self.total_darts = 0
                elif key == ord('s'):
                    fn = f"screenshot_{int(time.time())}.jpg"; cv2.imwrite(fn, disp); logger.info(f"Saved {fn}")
                elif key == ord('c'):
                    # Recalibrate: run UI, then re-apply homography
                    self._recalibrate_and_apply()

        except KeyboardInterrupt:
            logger.info("Interrupted.")
        finally:
            self.cleanup()

    def _recalibrate_and_apply(self):
        cv2.destroyAllWindows()
        self.camera.stop()
        ok = self._calibration_ui()
        if ok:
            # apply homography to ROI
            if self.homography is not None:
                self.roi.set_homography_from_matrix(self.homography)
        # restart camera
        cam_src = self.args.video if self.args.video else self.args.webcam
        self.camera = ThreadedCamera(CameraConfig(src=cam_src, max_queue_size=5, buffer_size=1,
                                                  width=self.args.width, height=self.args.height))
        self.camera.start()

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

    # argparse setup …
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

    p.add_argument("--motion-threshold", type=int, default=50, help="MOG2 variance threshold")
    p.add_argument("--motion-pixels", type=int, default=500, help="Min motion pixels")
    p.add_argument("--confirmation-frames", type=int, default=3, help="Frames to confirm dart")

    args = p.parse_args()
    DartVisionApp(args).run()

if __name__ == "__main__":
    main()
