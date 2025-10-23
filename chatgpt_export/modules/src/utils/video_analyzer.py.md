# Module: `src\utils\video_analyzer.py`
Hash: `83b17ba84f1c` · LOC: 11 · Main guard: false

## Imports
- `cv2`\n- `json`\n- `numpy`\n- `time`

## From-Imports
- `from pathlib import Path`\n- `from dataclasses import dataclass, asdict`\n- `from typing import List, Dict, Optional, Callable`\n- `from collections import defaultdict`\n- `from src.capture.fps_counter import FPSCounter`

## Classes
- `VideoInfo` (L19): Video file metadata\n- `AnalysisResult` (L49): Results from video analysis run\n- `VideoAnalyzer` (L72): Video analysis framework for parameter tuning.

## Functions
- `from_video()` (L29): Extract info from video file\n- `to_dict()` (L68)\n- `__init__()` (L83)\n- `analyze_video()` (L90): Analyze video with given processing callback.\n- `_annotate_frame()` (L203): Add overlay with FPS and detection info\n- `_save_detections()` (L245): Save detections to JSON\n- `_print_result()` (L254): Print analysis summary\n- `compare_configs()` (L267): Generate comparison report of all analyzed configs\n- `batch_analyze()` (L314): Analyze multiple videos with multiple configs.

## Intra-module calls (heuristic)
AnalysisResult, FPSCounter, Path, VideoCapture, _annotate_frame, _print_result, _save_detections, analyze_video, append, asdict, circle, cls, compare_configs, copy, destroyAllWindows, dump, from_video, get, get_stats, imshow, imwrite, int, items, len, mkdir, open, ord, perf_counter, print, processing_callback, putText, read, release, sorted, str, to_dict, update, waitKey

## Code
```python
"""
Video Analysis Tool for Parameter Tuning
Enables reproducible testing with recorded footage.
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Callable
from collections import defaultdict

from src.capture.fps_counter import FPSCounter


@dataclass
class VideoInfo:
    """Video file metadata"""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float

    @classmethod
    def from_video(cls, video_path: str) -> 'VideoInfo':
        """Extract info from video file"""
        cap = cv2.VideoCapture(video_path)

        info = cls(
            path=video_path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration_sec=0
        )

        info.duration_sec = info.frame_count / info.fps if info.fps > 0 else 0
        cap.release()

        return info


@dataclass
class AnalysisResult:
    """Results from video analysis run"""
    config_name: str
    video_name: str

    # Performance metrics
    fps_median: float
    fps_p95: float
    frame_time_ms: float
    frames_processed: int

    # Detection metrics (to be filled by detection pipeline)
    detections_count: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Resource usage
    processing_time_sec: float = 0

    def to_dict(self) -> dict:
        return asdict(self)


class VideoAnalyzer:
    """
    Video analysis framework for parameter tuning.

    Features:
    - Batch processing multiple videos
    - Side-by-side config comparison
    - Automatic best-config selection
    - Frame export for manual annotation
    """

    def __init__(self, output_dir: str = "analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)  # ✅ FIX

        # Results storage
        self.results: List[AnalysisResult] = []

    def analyze_video(
            self,
            video_path: str,
            config_name: str,
            processing_callback: Callable[[cv2.Mat, int], Optional[dict]],
            max_frames: Optional[int] = None,
            display: bool = False,
            save_frames: bool = False
    ) -> AnalysisResult:
        """
        Analyze video with given processing callback.

        Args:
            video_path: Path to video file
            config_name: Name for this config (for comparison)
            processing_callback: Function(frame, frame_idx) -> detection_dict
            max_frames: Limit processing (None = full video)
            display: Show real-time visualization
            save_frames: Save annotated frames to disk

        Returns:
            AnalysisResult with metrics
        """
        print(f"\n🎬 Analyzing: {Path(video_path).name}")
        print(f"📊 Config: {config_name}")

        # Video info
        video_info = VideoInfo.from_video(video_path)
        print(f"📹 Resolution: {video_info.width}x{video_info.height}")
        print(f"⏱️  Duration: {video_info.duration_sec:.1f}s ({video_info.frame_count} frames)")

        # Setup
        cap = cv2.VideoCapture(video_path)
        fps_counter = FPSCounter(window_size=30)

        frame_idx = 0
        detections = []
        start_time = time.perf_counter()

        # Frame output directory
        if save_frames:
            frame_dir = self.output_dir / config_name / Path(video_path).stem
            frame_dir.mkdir(parents=True, exist_ok=True)  # ✅ FIX

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if max_frames and frame_idx >= max_frames:
                    break

                # Process frame
                fps_counter.update()
                detection = processing_callback(frame, frame_idx)

                if detection:
                    detections.append({
                        'frame': frame_idx,
                        'timestamp': frame_idx / video_info.fps,
                        **detection
                    })

                # Display (optional)
                if display:
                    display_frame = self._annotate_frame(frame, detection, fps_counter)
                    cv2.imshow(f'Analysis: {config_name}', display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):  # Pause
                        cv2.waitKey(0)

                # Save frame (optional)
                if save_frames and (detection or frame_idx % 30 == 0):
                    out_path = frame_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(out_path), frame)

                frame_idx += 1

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

        # Calculate metrics
        processing_time = time.perf_counter() - start_time
        fps_stats = fps_counter.get_stats()

        result = AnalysisResult(
            config_name=config_name,
            video_name=Path(video_path).name,
            fps_median=fps_stats.fps_median,
            fps_p95=fps_stats.fps_p95,
            frame_time_ms=fps_stats.frame_time_ms,
            frames_processed=frame_idx,
            detections_count=len(detections),
            processing_time_sec=processing_time
        )

        self.results.append(result)

        # Save detections
        self._save_detections(config_name, Path(video_path).stem, detections)

        # Print summary
        self._print_result(result)

        return result

    def _annotate_frame(
            self,
            frame: cv2.Mat,
            detection: Optional[dict],
            fps_counter: FPSCounter
    ) -> cv2.Mat:
        """Add overlay with FPS and detection info"""
        annotated = frame.copy()

        # FPS overlay
        fps_stats = fps_counter.get_stats()
        cv2.putText(
            annotated,
            f"FPS: {fps_stats.fps_median:.1f} (P95: {fps_stats.fps_p95:.1f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Detection overlay
        if detection:
            if 'position' in detection:
                x, y = detection['position']
                cv2.circle(annotated, (x, y), 10, (0, 255, 255), 2)
                cv2.circle(annotated, (x, y), 3, (0, 255, 255), -1)

            if 'confidence' in detection:
                conf_text = f"Conf: {detection['confidence']:.2f}"
                cv2.putText(
                    annotated,
                    conf_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

        return annotated

    def _save_detections(self, config_name: str, video_name: str, detections: List[dict]):
        """Save detections to JSON"""
        output_file = self.output_dir / f"{config_name}_{video_name}_detections.json"

        with open(output_file, 'w') as f:
            json.dump(detections, f, indent=2)

        print(f"💾 Detections saved: {output_file}")

    def _print_result(self, result: AnalysisResult):
        """Print analysis summary"""
        print(f"\n{'=' * 50}")
        print(f"📊 Analysis Results: {result.config_name}")
        print(f"{'=' * 50}")
        print(f"Median FPS:      {result.fps_median:>8.1f}")
        print(f"P95 FPS:         {result.fps_p95:>8.1f}")
        print(f"Frame Time:      {result.frame_time_ms:>8.1f} ms")
        print(f"Frames Processed:{result.frames_processed:>8d}")
        print(f"Detections:      {result.detections_count:>8d}")
        print(f"Processing Time: {result.processing_time_sec:>8.1f} s")
        print(f"{'=' * 50}\n")

    def compare_configs(self) -> None:
        """Generate comparison report of all analyzed configs"""
        if not self.results:
            print("⚠️  No results to compare")
            return

        print(f"\n{'=' * 70}")
        print(f"📊 CONFIG COMPARISON REPORT")
        print(f"{'=' * 70}")

        # Sort by median FPS (descending)
        sorted_results = sorted(self.results, key=lambda r: r.fps_median, reverse=True)

        # Header
        print(f"{'Config':<20} {'FPS (Med)':<12} {'FPS (P95)':<12} {'Frame Time':<12} {'Detections':<12}")
        print(f"{'-' * 70}")

        # Results
        for result in sorted_results:
            print(
                f"{result.config_name:<20} "
                f"{result.fps_median:<12.1f} "
                f"{result.fps_p95:<12.1f} "
                f"{result.frame_time_ms:<12.1f} "
                f"{result.detections_count:<12d}"
            )

        print(f"{'-' * 70}")

        # Best config recommendation
        best = sorted_results[0]
        print(f"\n✅ BEST CONFIG: {best.config_name}")
        print(f"   - Median FPS: {best.fps_median:.1f}")
        print(f"   - P95 FPS: {best.fps_p95:.1f}")
        print(f"   - Detections: {best.detections_count}")

        # Save report
        report_file = self.output_dir / "comparison_report.json"
        with open(report_file, 'w') as f:
            json.dump(
                [r.to_dict() for r in sorted_results],
                f,
                indent=2
            )

        print(f"\n💾 Full report saved: {report_file}\n")

    def batch_analyze(
            self,
            video_paths: List[str],
            configs: Dict[str, Callable],
            max_frames: Optional[int] = None
    ):
        """
        Analyze multiple videos with multiple configs.

        Args:
            video_paths: List of video file paths
            configs: Dict of {config_name: processing_callback}
            max_frames: Frame limit per video
        """
        total_runs = len(video_paths) * len(configs)
        current_run = 0

        print(f"\n🚀 Starting batch analysis: {total_runs} runs")
        print(f"Videos: {len(video_paths)}, Configs: {len(configs)}\n")

        for video_path in video_paths:
            for config_name, callback in configs.items():
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Processing...")

                self.analyze_video(
                    video_path=video_path,
                    config_name=config_name,
                    processing_callback=callback,
                    max_frames=max_frames,
                    display=False
                )

        # Generate comparison
        self.compare_configs()
```
