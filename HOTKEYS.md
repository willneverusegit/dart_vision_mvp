# Dart Vision MVP - Hotkey Reference

Complete keyboard shortcut reference for Dart Vision MVP.

---

## 🎮 SYSTEM

| Key | Action | Description |
|-----|--------|-------------|
| `q` | **Quit** | Exit application |
| `s` | **Screenshot** | Save current display to file |
| `?` | **Help** | Toggle help overlay |
| `p` | **Pause** | Pause/unpause video processing |

---

## 🎯 OVERLAY MODES

| Key | Action | Description |
|-----|--------|-------------|
| `o` | **Cycle Mode** | Cycle through overlay modes: MIN → RINGS → FULL → ALIGN |

### Overlay Modes Explained:
- **MIN**: Minimal overlay (only dart impacts + game HUD)
- **RINGS**: Simple board rings visible
- **FULL**: Complete overlay with colored dartboard + sector numbers
- **ALIGN**: Alignment mode with thick calibration circles

---

## 🔧 CALIBRATION

### ROI/Image Calibration (ALIGN Mode)

| Key | Action | Description |
|-----|--------|-------------|
| `c` | **Recalibrate ROI** | Re-run ChArUco/ArUco calibration for image warp |
| `t` / `T` | **Hough Once** | Run Hough circle detection once to refine board position |
| `z` | **Auto-Align** | Toggle automatic Hough alignment (runs continuously in ALIGN mode) |

### Board Overlay Calibration (FULL Mode)

| Key | Action | Description |
|-----|--------|-------------|
| **`C`** | **🎨 Board Calibration** | **Toggle colored dartboard overlay calibration mode** |
|  |  | *Activates visual calibration with colored fields (Red/Green/Black/Cream)* |
|  |  | *Allows 1:1 alignment of overlay with physical dartboard* |

**When Board Calibration is ON (`C`):**
- Dartboard displayed with realistic colors:
  - Triple/Double: Alternating **Red/Green**
  - Singles: Alternating **Black/Cream**
  - Bullseye: Inner **Red** (50pts), Outer **Green** (25pts)
- Sector numbers centered in single fields
- Crosshair at board center
- Use adjustment keys below to align perfectly

---

## 🎛️ OVERLAY ADJUSTMENTS

### Fine-Tune Overlay Position

| Key | Action | Delta | Description |
|-----|--------|-------|-------------|
| `j` | Move Left | -1px | Shift overlay center left |
| `l` | Move Right | +1px | Shift overlay center right |
| `i` | Move Up | -1px | Shift overlay center up |
| `k` | Move Down | +1px | Shift overlay center down |

### Overlay Rotation & Scale

| Key | Action | Delta | Description |
|-----|--------|-------|-------------|
| `←` (Left Arrow) | Rotate CCW | -0.5° | Rotate overlay counter-clockwise |
| `→` (Right Arrow) | Rotate CW | +0.5° | Rotate overlay clockwise |
| `↑` (Up Arrow) | Scale Up | +1% | Increase overlay scale |
| `↓` (Down Arrow) | Scale Down | -1% | Decrease overlay scale |

### Save/Reset

| Key | Action | Description |
|-----|--------|-------------|
| `X` | **Save Overlay** | Save current center/rotation/scale to calibration file |
| `R` | **Reset Overlay** | Reset overlay adjustments to defaults |

---

## 🎲 GAME CONTROLS

| Key | Action | Description |
|-----|--------|-------------|
| `g` | **Reset Game** | Reset current game (Around the Clock or 301) |
| `h` | **Switch Mode** | Switch between game modes (ATC ↔ 301) |
| `r` | **Clear Darts** | Clear all detected dart impacts |

---

## 🐛 DEBUG & VISUALIZATION

| Key | Action | Description |
|-----|--------|-------------|
| `d` | **Debug Info** | Toggle debug overlay (FPS, frame time, detector stats) |
| `m` | **Motion Overlay** | Toggle motion detection overlay (red tint on motion) |
| `M` | **Mask Overlay** | Toggle annulus mask overlay |
| `V` | **Mask Debug Window** | Toggle separate window showing processed motion mask |

---

## 🔊 MOTION TUNING (Live Adjustment)

### Otsu Threshold Bias

| Key | Action | Delta | Description |
|-----|--------|-------|-------------|
| `b` | Decrease Bias | -1 | Make motion detection more sensitive |
| `B` | Increase Bias | +1 | Make motion detection less sensitive |

### Morphology Filters

| Key | Action | Delta | Description |
|-----|--------|-------|-------------|
| `f` | Decrease Open | -2 | Reduce opening (more noise, more detail) |
| `F` | Increase Open | +2 | Increase opening (less noise, cleaner) |
| `n` | Decrease Close | -2 | Reduce closing (preserve gaps) |
| `N` | Increase Close | +2 | Increase closing (fill gaps) |

### Min White Fraction

| Key | Action | Delta | Description |
|-----|--------|-------|-------------|
| `w` | Decrease Fraction | -0.2% | Lower threshold for valid motion mask |
| `W` | Increase Fraction | +0.2% | Higher threshold for valid motion mask |

---

## 🎚️ DETECTOR PRESETS

| Key | Preset | Description |
|-----|--------|-------------|
| `1` | **Aggressive** | More sensitive detection (more false positives) |
| `2` | **Balanced** | Default balanced preset (recommended) |
| `3` | **Stable** | Conservative detection (fewer false positives) |

---

## 📊 HEATMAP VISUALIZATION

| Key | Action | Description |
|-----|--------|-------------|
| `H` | **Cartesian Heatmap** | Toggle image-space heatmap overlay (currently disabled by default) |
| `P` | **Polar Heatmap** | Toggle polar coordinate heatmap panel |

---

## 🎯 WORKFLOW: Board Calibration

To calibrate the dartboard overlay for perfect 1:1 scoring alignment:

1. **Press `o`** repeatedly until in **FULL** mode
2. **Press `C`** to enable **Board Calibration Mode**
   - ✅ Colored dartboard overlay appears
   - ✅ See Triple (Red/Green), Singles (Black/Cream), Bullseye
3. **Adjust overlay to match physical board:**
   - Use **Arrow Keys** to rotate/scale
   - Use **`j`/`k`/`l`/`i`** to shift center
   - Fine-tune until colored segments align perfectly with real board
4. **Press `X`** to **save calibration**
5. **Press `C`** again to exit calibration mode
6. ✅ **Overlay is now calibrated for scoring!**

---

## 📝 NOTES

### Board Calibration vs. ROI Calibration

- **`c` (lowercase)**: ROI/Image calibration (ChArUco/ArUco markers, image warp)
  - For: Camera perspective, image distortion
  - Changes: Homography matrix, warp transformation

- **`C` (uppercase)**: Board overlay calibration (dartboard alignment)
  - For: Fine-tuning overlay position/rotation/scale
  - Changes: Overlay center offset, rotation, scale factor
  - **Use this for precise scoring zone alignment!**

### Tips

- **Start with ROI calibration (`c`)** to get rough alignment
- **Refine with Hough (`t`)** if board moved
- **Fine-tune with Board Calibration (`C`)** for perfect 1:1 match
- **Save frequently (`X`)** during calibration
- **Use colored overlay** to visually verify sector boundaries
- **Numbers should be centered** in single fields between Triple and Double rings

---

## 🚀 QUICK REFERENCE CARD

```
┌─────────────────────────────────────────────────┐
│ MOST USED HOTKEYS                               │
├─────────────────────────────────────────────────┤
│ C  = Board Calibration (colored dartboard) 🎨   │
│ o  = Cycle Overlay Mode                         │
│ X  = Save Calibration                           │
│ ←→ = Rotate Overlay                             │
│ ↑↓ = Scale Overlay                              │
│ jkli = Move Overlay Center                      │
│ g  = Reset Game                                 │
│ r  = Clear Darts                                │
│ s  = Screenshot                                 │
│ q  = Quit                                       │
└─────────────────────────────────────────────────┘
```

---

**Generated with Claude Code** 🤖
