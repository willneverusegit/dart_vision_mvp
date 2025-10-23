"""
Tuning Visualizer - Debug overlays and metrics visualization.

Provides comprehensive visual feedback for parameter tuning:
- Side-by-side frame comparison (original, processed mask, detections)
- Contour visualization with bounding boxes
- Shape metrics overlay (text annotations)
- Convex hull visualization
- Real-time metrics dashboard
- Detection confidence indicators
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
from collections import deque

from src.vision.detection import (
    MotionEvent,
    DartImpact,
    DartCandidate,
    DartDetectorConfig,
)


class TuningVisualizer:
    """
    Visualizer for parameter tuning with comprehensive debug overlays.

    Features:
    - Multi-panel layout (original, masks, overlays)
    - Contour and bounding box rendering
    - Shape metrics text annotations
    - Confidence indicators
    - Real-time statistics dashboard
    """

    def __init__(self):
        """Initialize visualizer"""
        # Colors
        self.color_motion = (0, 255, 255)  # Yellow
        self.color_candidate = (0, 165, 255)  # Orange
        self.color_confirmed = (0, 255, 0)  # Green
        self.color_rejected = (0, 0, 255)  # Red
        self.color_convex_hull = (255, 0, 255)  # Magenta

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.4
        self.font_thickness = 1

    def create_visualization(
        self,
        frame: np.ndarray,
        fg_mask: Optional[np.ndarray],
        processed_mask: Optional[np.ndarray],
        motion_detected: bool,
        motion_event: Optional[MotionEvent],
        dart_impact: Optional[DartImpact],
        dart_candidate: Optional[DartCandidate],
        show_debug: bool,
        show_metrics: bool,
        stats: Dict[str, Any],
        detector_config: DartDetectorConfig,
    ) -> np.ndarray:
        """
        Create comprehensive visualization for tuning.

        Args:
            frame: Original BGR frame
            fg_mask: Foreground mask from motion detection
            processed_mask: Preprocessed mask from dart detector
            motion_detected: Whether motion was detected
            motion_event: Motion event details
            dart_impact: Confirmed dart impact
            dart_candidate: Current dart candidate
            show_debug: Show debug overlays
            show_metrics: Show metrics dashboard
            stats: Statistics dictionary
            detector_config: Current dart detector configuration

        Returns:
            Visualization frame
        """
        h, w = frame.shape[:2]

        # Create panels
        panels = []

        # Panel 1: Original frame with overlays
        frame_overlay = frame.copy()
        if show_debug:
            frame_overlay = self._draw_motion_overlay(frame_overlay, motion_event)
            frame_overlay = self._draw_dart_overlay(
                frame_overlay, dart_candidate, dart_impact, detector_config
            )
        panels.append(frame_overlay)

        # Panel 2: Foreground mask (motion detection)
        if fg_mask is not None:
            fg_mask_vis = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            if motion_detected:
                # Add green tint to show motion detected
                fg_mask_vis = cv2.addWeighted(
                    fg_mask_vis, 0.7, np.full_like(fg_mask_vis, (0, 50, 0)), 0.3, 0
                )
            self._draw_text_overlay(fg_mask_vis, "Motion Mask", (10, 30))
            panels.append(fg_mask_vis)
        else:
            panels.append(np.zeros_like(frame))

        # Panel 3: Processed mask (dart detection preprocessing)
        if processed_mask is not None and show_debug:
            processed_vis = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)
            self._draw_text_overlay(processed_vis, "Processed Mask", (10, 30))

            # Draw contours on processed mask
            contours, _ = cv2.findContours(
                processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(processed_vis, contours, -1, (0, 255, 0), 1)

            panels.append(processed_vis)
        else:
            panels.append(np.zeros_like(frame))

        # Arrange panels horizontally
        if len(panels) == 3:
            # Resize panels to fit
            panel_width = w // 3
            panel_height = h
            resized_panels = [
                cv2.resize(p, (panel_width, panel_height)) for p in panels
            ]
            vis_frame = np.hstack(resized_panels)
        else:
            vis_frame = frame_overlay

        # Add metrics dashboard
        if show_metrics:
            vis_frame = self._draw_metrics_dashboard(
                vis_frame, stats, motion_detected, dart_candidate, dart_impact
            )

        return vis_frame

    def _draw_motion_overlay(
        self, frame: np.ndarray, motion_event: Optional[MotionEvent]
    ) -> np.ndarray:
        """Draw motion detection overlay"""
        if motion_event is None:
            return frame

        # Draw bounding box
        x, y, w_box, h_box = motion_event.bounding_box
        cv2.rectangle(
            frame, (x, y), (x + w_box, y + h_box), self.color_motion, 2
        )

        # Draw center
        cv2.circle(frame, motion_event.center, 5, self.color_motion, -1)

        # Draw text
        text = f"Motion: Area={motion_event.area:.0f}"
        if motion_event.brightness is not None:
            text += f" Bright={motion_event.brightness:.0f}"
        self._draw_text_overlay(frame, text, (x, y - 10))

        return frame

    def _draw_dart_overlay(
        self,
        frame: np.ndarray,
        candidate: Optional[DartCandidate],
        impact: Optional[DartImpact],
        config: DartDetectorConfig,
    ) -> np.ndarray:
        """Draw dart detection overlay"""
        # Draw confirmed impact (green)
        if impact is not None:
            pos = impact.position
            cv2.circle(frame, pos, 15, self.color_confirmed, 3)
            cv2.circle(frame, pos, 3, self.color_confirmed, -1)

            text = f"DART! Conf={impact.confidence:.2f}"
            self._draw_text_overlay(frame, text, (pos[0] + 20, pos[1] - 20), self.color_confirmed)

        # Draw current candidate (orange)
        elif candidate is not None:
            pos = candidate.position
            cv2.circle(frame, pos, 10, self.color_candidate, 2)

            # Draw shape metrics
            metrics_text = [
                f"Pos: {pos}",
                f"Conf: {candidate.confidence:.2f}",
                f"Area: {candidate.area:.0f}",
                f"AR: {candidate.aspect_ratio:.2f}",
                f"Sol: {candidate.solidity:.2f}",
                f"Ext: {candidate.extent:.2f}",
                f"Edge: {candidate.edge_density:.2f}",
            ]
            if candidate.convexity is not None:
                metrics_text.append(f"Conv: {candidate.convexity:.2f}")

            # Draw metrics near candidate
            y_offset = 20
            for line in metrics_text:
                self._draw_text_overlay(
                    frame, line, (pos[0] + 15, pos[1] + y_offset), self.color_candidate
                )
                y_offset += 15

        return frame

    def _draw_metrics_dashboard(
        self,
        frame: np.ndarray,
        stats: Dict[str, Any],
        motion_detected: bool,
        candidate: Optional[DartCandidate],
        impact: Optional[DartImpact],
    ) -> np.ndarray:
        """Draw real-time metrics dashboard"""
        h, w = frame.shape[:2]

        # Create semi-transparent overlay
        overlay = frame.copy()
        dashboard_height = 180
        cv2.rectangle(
            overlay, (0, h - dashboard_height), (w, h), (0, 0, 0), -1
        )
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Metrics text
        y_pos = h - dashboard_height + 20
        x_left = 10
        x_mid = w // 3
        x_right = 2 * w // 3

        # Column 1: General stats
        self._draw_dashboard_text(frame, "=== Statistics ===", x_left, y_pos)
        y_pos += 20
        self._draw_dashboard_text(
            frame, f"Frames: {stats.get('frames_processed', 0)}", x_left, y_pos
        )
        y_pos += 20
        self._draw_dashboard_text(
            frame, f"Motion: {stats.get('motion_detected', 0)}", x_left, y_pos
        )
        y_pos += 20
        self._draw_dashboard_text(
            frame, f"Darts: {stats.get('darts_detected', 0)}", x_left, y_pos
        )
        y_pos += 20

        # Detection rate
        frames = stats.get('frames_processed', 1)
        motion_rate = (stats.get('motion_detected', 0) / frames) * 100
        dart_rate = (stats.get('darts_detected', 0) / frames) * 100
        self._draw_dashboard_text(
            frame, f"Motion Rate: {motion_rate:.1f}%", x_left, y_pos
        )
        y_pos += 20
        self._draw_dashboard_text(
            frame, f"Dart Rate: {dart_rate:.1f}%", x_left, y_pos
        )

        # Column 2: Current status
        y_pos = h - dashboard_height + 20
        self._draw_dashboard_text(frame, "=== Current Status ===", x_mid, y_pos)
        y_pos += 20

        status_color = self.color_motion if motion_detected else (100, 100, 100)
        self._draw_dashboard_text(
            frame,
            f"Motion: {'YES' if motion_detected else 'NO'}",
            x_mid,
            y_pos,
            status_color,
        )
        y_pos += 20

        if candidate:
            self._draw_dashboard_text(
                frame,
                f"Candidate: YES",
                x_mid,
                y_pos,
                self.color_candidate,
            )
            y_pos += 20
            self._draw_dashboard_text(
                frame, f"  Conf: {candidate.confidence:.2f}", x_mid, y_pos
            )
            y_pos += 20
            if candidate.convexity:
                self._draw_dashboard_text(
                    frame, f"  Conv: {candidate.convexity:.2f}", x_mid, y_pos
                )
                y_pos += 20
        else:
            self._draw_dashboard_text(
                frame, f"Candidate: NO", x_mid, y_pos, (100, 100, 100)
            )
            y_pos += 20

        if impact:
            self._draw_dashboard_text(
                frame,
                f"IMPACT DETECTED!",
                x_mid,
                y_pos,
                self.color_confirmed,
            )

        # Column 3: Average metrics
        y_pos = h - dashboard_height + 20
        self._draw_dashboard_text(frame, "=== Averages ===", x_right, y_pos)
        y_pos += 20

        avg_conf_vals = list(stats.get('avg_confidence', []))
        if avg_conf_vals:
            avg_conf = np.mean(avg_conf_vals)
            self._draw_dashboard_text(
                frame, f"Avg Confidence: {avg_conf:.2f}", x_right, y_pos
            )
        else:
            self._draw_dashboard_text(
                frame, f"Avg Confidence: N/A", x_right, y_pos
            )

        return frame

    def _draw_text_overlay(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 255, 255),
    ):
        """Draw text with background for better visibility"""
        x, y = position

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x - 2, y - text_h - 2),
            (x + text_w + 2, y + baseline + 2),
            (0, 0, 0),
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            text,
            (x, y),
            self.font,
            self.font_scale,
            color,
            self.font_thickness,
            cv2.LINE_AA,
        )

    def _draw_dashboard_text(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int] = (255, 255, 255),
    ):
        """Draw dashboard text (no background)"""
        cv2.putText(
            frame,
            text,
            (x, y),
            self.font,
            self.font_scale,
            color,
            self.font_thickness,
            cv2.LINE_AA,
        )
