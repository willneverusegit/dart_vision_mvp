# Lightweight polar heatmap (ring x sector) for darts: 7 rings x 20 sectors
from __future__ import annotations
import numpy as np
import cv2
from typing import Optional, Tuple

RINGS = ["single_inner","triple","single_outer","double","outer_bull","inner_bull","miss"]
RING_DISPLAY = ["S_in","T","S_out","D","OB","IB","M"]
RING_ORDER = {name:i for i,name in enumerate(RINGS)}  # row index

class PolarHeatmap:
    """
    Counts hits per (ring, sector). Bulls/miss use sector=None.
    Efficient O(1) increments; renders to a small color panel.
    """
    def __init__(self, cell_size: Tuple[int,int]=(14,14), colormap: int=cv2.COLORMAP_TURBO):
        self.cm = colormap
        self.cell_w, self.cell_h = cell_size
        self.grid = np.zeros((len(RINGS), 20), dtype=np.float32)  # sector 0..19 (1..20)
        self.bulls = np.zeros((2,), dtype=np.float32)             # OB, IB
        self.miss = 0.0
        self._dirty = True
        self._panel: Optional[np.ndarray] = None

    def add(self, ring: str, sector: Optional[int]):
        if ring == "outer_bull":
            self.bulls[0] += 1; self._dirty = True; return
        if ring == "inner_bull":
            self.bulls[1] += 1; self._dirty = True; return
        if ring == "miss" or sector is None:
            self.miss += 1; self._dirty = True; return
        r = RING_ORDER.get(ring, None)
        if r is None:
            return
        s = (int(sector) - 1) % 20
        self.grid[r, s] += 1.0
        self._dirty = True

    def _render_core(self) -> np.ndarray:
        # stack full 7x20 matrix
        full = np.zeros((len(RINGS), 20), dtype=np.float32)
        full[:4, :] = self.grid[:4, :]
        # bulls & miss encoded in dedicated rows
        full[4, :] = self.bulls[0]   # OB replicated for viz
        full[5, :] = self.bulls[1]   # IB
        full[6, :] = self.miss       # Miss
        # normalize robustly
        p95 = np.percentile(full[full>0], 95) if np.any(full>0) else 1.0
        denom = p95 if p95 > 1e-6 else float(full.max() if full.max()>0 else 1.0)
        img = np.clip((full/denom)*255.0, 0, 255).astype(np.uint8)
        # resize to panel
        H = img.shape[0]*self.cell_h
        W = img.shape[1]*self.cell_w
        up = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        color = cv2.applyColorMap(up, self.cm)
        # draw grid lines
        for r in range(len(RINGS)+1):
            y = r*self.cell_h
            cv2.line(color, (0,y), (W,y), (50,50,50), 1, cv2.LINE_AA)
        for c in range(21):
            x = c*self.cell_w
            cv2.line(color, (x,0), (x,H), (50,50,50), 1, cv2.LINE_AA)
        # labels (compact)
        for r,name in enumerate(RING_DISPLAY):
            y = r*self.cell_h + int(self.cell_h*0.7)
            cv2.putText(color, name, (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
        return color

    def panel(self) -> np.ndarray:
        if self._dirty or self._panel is None:
            self._panel = self._render_core()
            self._dirty = False
        return self._panel

    def overlay_panel(self, frame_bgr: np.ndarray, pos: Tuple[int,int]=(10,10)) -> np.ndarray:
        panel = self.panel()
        h, w = panel.shape[:2]
        x, y = pos
        H, W = frame_bgr.shape[:2]
        x = min(max(0, x), W-w); y = min(max(0, y), H-h)
        out = frame_bgr.copy()
        roi = out[y:y+h, x:x+w]
        cv2.addWeighted(roi, 0.6, panel, 0.4, 0, dst=roi)
        # small border
        cv2.rectangle(out, (x,y), (x+w, y+h), (220,220,220), 1, cv2.LINE_AA)
        return out

    def export_png(self, path: str) -> None:
        cv2.imwrite(path, self.panel())

    # add to class PolarHeatmap:
    def export_csv(self, path: str) -> None:
        import csv, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        full = np.zeros((len(RINGS), 20), dtype=np.float32)
        full[:4, :] = self.grid[:4, :]
        full[4, :] = self.bulls[0]
        full[5, :] = self.bulls[1]
        full[6, :] = self.miss
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row\\col"] + [str(i + 1) for i in range(20)])
            rows = RING_DISPLAY
            for i, name in enumerate(rows):
                w.writerow([name] + full[i, :].astype(int).tolist())