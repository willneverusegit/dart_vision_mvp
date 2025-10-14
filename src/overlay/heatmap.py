# Event-driven image-space heatmap with cached colorized texture
from __future__ import annotations
import time
from typing import Optional, Tuple
import numpy as np
import cv2

class HeatmapAccumulator:
    """
    - Low-res grid (scale factor) for cheap updates
    - Small Gaussian stamp per hit (O(1) neighborhood)
    - Colorized BGR texture cached; only re-render on data change
    - Optional exponential decay to keep it 'alive'
    """
    def __init__(
        self,
        frame_size: Tuple[int, int],      # (width, height) full-res
        scale: float = 0.25,              # internal grid scale
        stamp_radius_px: int = 6,
        stamp_sigma_px: float | None = None,
        colormap: int = cv2.COLORMAP_TURBO,
        alpha: float = 0.35,
        decay_half_life_s: float | None = None
    ):
        self.W, self.H = frame_size
        self.scale = float(np.clip(scale, 0.1, 1.0))
        self.gw, self.gh = max(4, int(self.W * self.scale)), max(4, int(self.H * self.scale))
        self.grid = np.zeros((self.gh, self.gw), dtype=np.float32)

        r_full = max(2, int(stamp_radius_px))
        sigma_full = stamp_sigma_px or max(1.0, r_full / 2.0)
        r = max(1, int(round(r_full * self.scale)))
        sig = max(0.5, sigma_full * self.scale)
        k = np.arange(-r, r + 1, dtype=np.int32)
        X, Y = np.meshgrid(k, k)
        G = np.exp(-(X**2 + Y**2) / (2.0 * sig * sig)).astype(np.float32)
        G /= max(G.max(), 1.0)
        self.stamp = G

        self.colormap = colormap
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._cached_bgr: Optional[np.ndarray] = None
        self._dirty = True
        self._version = 0

        self.decay_half_life_s = decay_half_life_s
        self._last_decay_t = time.time()

    def _maybe_decay(self):
        if not self.decay_half_life_s:
            return
        now = time.time()
        dt = now - self._last_decay_t
        if dt < 0.25:
            return
        factor = 0.5 ** (dt / self.decay_half_life_s)
        if factor < 0.999:
            self.grid *= factor
            self._dirty = True
            self._version += 1
            self._last_decay_t = now

    def add_hit(self, x_px: float, y_px: float, weight: float = 1.0):
        """Add hit in full-res coordinates."""
        self._maybe_decay()
        gx = int(round(x_px * self.scale))
        gy = int(round(y_px * self.scale))
        if gx < 0 or gy < 0 or gx >= self.gw or gy >= self.gh:
            return
        h, w = self.stamp.shape
        r = h // 2
        x0, x1 = max(0, gx - r), min(self.gw, gx + r + 1)
        y0, y1 = max(0, gy - r), min(self.gh, gy + r + 1)
        sx0 = r - (gx - x0); sx1 = sx0 + (x1 - x0)
        sy0 = r - (gy - y0); sy1 = sy0 + (y1 - y0)
        self.grid[y0:y1, x0:x1] += self.stamp[sy0:sy1, sx0:sx1] * float(max(0.0, weight))
        self._dirty = True
        self._version += 1

    def render_overlay(self, frame_bgr: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        self._maybe_decay()
        if self._dirty or self._cached_bgr is None:
            g = self.grid
            if g.size == 0 or np.all(g <= 0):
                color = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            else:
                p95 = float(np.percentile(g, 95.0))
                denom = p95 if p95 > 1e-6 else float(max(g.max(), 1.0))
                norm = np.clip((g / denom) * 255.0, 0, 255).astype(np.uint8)
                up = cv2.resize(norm, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                color = cv2.applyColorMap(up, self.colormap)
            self._cached_bgr = color
            self._dirty = False
        hm = self._cached_bgr
        if roi_mask is not None:
            if roi_mask.ndim == 2:
                m3 = cv2.merge([roi_mask, roi_mask, roi_mask])
            else:
                m3 = roi_mask
            m3 = (m3 > 0).astype(np.uint8)
            out = frame_bgr.copy()
            roi = m3[:, :, 0] > 0
            out[roi] = cv2.addWeighted(frame_bgr[roi], 1.0 - self.alpha, hm[roi], self.alpha, 0)
            return out
        return cv2.addWeighted(frame_bgr, 1.0 - self.alpha, hm, self.alpha, 0)

    def export_png(self, path: str) -> None:
        """Save current heatmap (colorized) as PNG without blending."""
        if self._dirty or self._cached_bgr is None:
            _ = self.render_overlay(np.zeros((self.H, self.W, 3), dtype=np.uint8))
        cv2.imwrite(path, self._cached_bgr)
