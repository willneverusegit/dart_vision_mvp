# Dart Detection Parameter Tuning Tool 🎯

Interactive tool for finding optimal detection parameters with real-time visual feedback.

## Features

✅ **Live Preview** - Side-by-side comparison (original, masks, overlays)
✅ **80+ Parameters** - All detection parameters adjustable via trackbars
✅ **Real-time Metrics** - Dashboard with detection stats
✅ **Debug Overlays** - Contours, bounding boxes, shape metrics
✅ **Preset System** - Save/load/compare configurations
✅ **Frame Control** - Pause, step forward/backward, loop
✅ **Statistics** - Track detection rate, confidence, false positives

## Quick Start

### Run with Video File
```bash
python -m src.vision.tuning --video path/to/dart_video.mp4
```

### Run with Webcam
```bash
python -m src.vision.tuning --webcam 0
```

### Run with Options
```bash
python -m src.vision.tuning \
    --video test_videos/dart_throw_1.mp4 \
    --start-frame 100 \
    --preset-dir my_presets/ \
    --debug
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **SPACE** | Pause/Unpause playback |
| **.** | Step forward one frame |
| **,** | Step backward one frame |
| **s** | Save current parameters as preset |
| **l** | Load preset |
| **r** | Reset statistics |
| **h** | Show help |
| **q** | Quit |

## UI Layout

```
┌─────────────────┬─────────────────┬─────────────────┐
│  Original +     │  Motion Mask    │  Processed      │
│  Overlays       │  (MOG2)         │  Mask + Contours│
└─────────────────┴─────────────────┴─────────────────┘
│                  Metrics Dashboard                   │
└──────────────────────────────────────────────────────┘
```

## Parameter Groups

### Motion Detection
- **Var Threshold**: MOG2 variance threshold (lower = more sensitive)
- **Min Pixels**: Minimum pixels for motion event
- **Morph Kernel**: Morphological operation kernel size

### Dart Detection - Shape Constraints
- **Min/Max Area**: Valid dart blob area range (pixels)
- **Min/Max Aspect Ratio**: Shape elongation (0.3-3.0 typical)
- **Min/Max Solidity**: Filled vs. outline ratio
- **Min/Max Extent**: Bounding box fill ratio

### Dart Detection - Advanced
- **Edge Density**: Ratio of edge pixels to total area
- **Convexity**: Convex hull ratio (higher = more convex)
- **Convex Gate**: Enable/disable convexity filtering

### Dart Detection - Temporal
- **Confirm Frames**: Frames needed to confirm dart (higher = more stable)
- **Pos Tolerance**: Max pixel movement between frames
- **Cooldown Frames**: Prevent re-detection in same area

### Dart Detection - Preprocessing
- **Adaptive**: Enable adaptive binarization
- **Otsu Bias**: Threshold bias for Otsu method
- **Morph Open/Close**: Morphological cleanup

## Workflow for Finding Optimal Parameters

1. **Start with a Preset**
   - Use Preset trackbar: 0=Aggressive, 1=Balanced, 2=Stable
   - Balanced is usually a good starting point

2. **Adjust Motion Detection**
   - Lower Var Threshold if missing motion
   - Increase Min Pixels to reduce false positives
   - Adjust Morph Kernel for cleanup

3. **Fine-tune Dart Detection**
   - Watch the "Processed Mask" panel
   - Ensure dart blob is visible and clean
   - Adjust shape constraints to match your setup

4. **Test Convexity Filter**
   - Enable/disable Convex Gate to see effect
   - Adjust Convexity threshold
   - Should filter out hands/shadows

5. **Optimize Temporal Parameters**
   - Increase Confirm Frames for stability
   - Adjust Pos Tolerance based on dart movement
   - Set Cooldown to prevent double-detections

6. **Save Your Configuration**
   - Press **'s'** to save current parameters
   - Creates timestamped preset file
   - Load later with **'l'**

## Tips for Different Scenarios

### Low Light Conditions
```
- Lower Motion Var Threshold (30-40)
- Increase Dart Otsu Bias (12-15)
- Decrease Edge Density Min (0.015)
```

### High Speed Darts
```
- Reduce Confirm Frames (2)
- Increase Pos Tolerance (25-30)
- Lower Cooldown Frames (20-25)
```

### Noisy Background
```
- Increase Motion Min Pixels (600-800)
- Increase Morph Open size (7-9)
- Enable Convex Gate
- Increase Convexity threshold (0.75-0.80)
```

### Slow/Sticky Darts
```
- Increase Confirm Frames (4-5)
- Decrease Pos Tolerance (15)
- Increase Cooldown Frames (40-50)
```

## Debug Overlays

### Original Frame Panel
- **Yellow Box**: Motion detection bounding box
- **Orange Circle**: Current dart candidate
- **Green Circle**: Confirmed dart impact
- **Text Annotations**: Shape metrics (area, AR, solidity, etc.)

### Motion Mask Panel
- **White Pixels**: Detected motion
- **Green Tint**: Motion event active

### Processed Mask Panel
- **White Blobs**: After adaptive binarization + morphology
- **Green Contours**: Detected contours

## Metrics Dashboard

**Statistics Column:**
- Frames processed
- Motion detections
- Dart detections
- Detection rates

**Current Status Column:**
- Motion: YES/NO
- Candidate: Details if present
- Impact: Shown when dart confirmed

**Averages Column:**
- Average confidence over last 30 detections

## Preset Management

### Default Presets Location
```
presets/detection/
├── aggressive.yaml
├── balanced.yaml
├── stable.yaml
└── custom_*.yaml
```

### Preset File Format
```yaml
name: my_custom_preset
created: 2025-10-23T12:34:56
version: "1.0"
motion:
  var_threshold: 50
  motion_pixel_threshold: 500
  morph_kernel_size: 5
dart:
  min_area: 10
  max_area: 1100
  min_aspect_ratio: 0.3
  # ... all other parameters
```

### Comparing Presets
```python
from src.vision.tuning import PresetManager

pm = PresetManager("presets/detection")
diff = pm.compare_presets("balanced", "custom_123")
print(diff)
```

## Troubleshooting

### No Motion Detected
- Lower Motion Var Threshold
- Check camera placement/lighting
- Verify video source is working

### Too Many False Positives
- Increase Motion Min Pixels
- Enable Convex Gate
- Increase Confirm Frames
- Adjust shape constraints (Area, AR, Solidity)

### Missing Dart Impacts
- Lower shape constraint thresholds
- Decrease Confirm Frames
- Check Processed Mask - dart should be visible
- Try different preset (Aggressive)

### Detector Too Sensitive
- Increase Confirm Frames (4-5)
- Tighten shape constraints
- Increase Convexity threshold
- Higher Cooldown Frames

## Integration with Main App

After finding good parameters, export preset and load in main app:

```python
from src.vision.detection import apply_detector_preset, DartDetectorConfig

# Load your custom preset
cfg = DartDetectorConfig()
cfg = apply_detector_preset(cfg, "my_custom_preset")

# Or load from file
from src.vision.tuning import PresetManager
pm = PresetManager("presets/detection")
preset = pm.load_preset("my_custom_preset")
# Apply preset['dart'] values to cfg
```

## Advanced Usage

### Batch Testing
Test multiple presets on same video:
```bash
for preset in aggressive balanced stable; do
    python -m src.vision.tuning \
        --video test.mp4 \
        --preset $preset \
        --start-frame 0
done
```

### Frame-by-Frame Analysis
1. Pause with SPACE
2. Use . and , to step frame-by-frame
3. Adjust parameters while paused
4. Resume to see effect over time

### Statistics Tracking
- Press 'r' to reset stats at any point
- Useful for A/B testing different configs
- Watch detection rates in dashboard

## Dependencies

- OpenCV (cv2)
- NumPy
- PyYAML
- Python 3.8+

## Contributing

Found a bug or have a feature request? Please open an issue!

## License

Part of the Dart Vision MVP project.
