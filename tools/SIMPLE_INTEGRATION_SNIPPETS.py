# SIMPLE INTEGRATION SNIPPETS FÜR MAIN.PY
# Copy-paste diese Code-Blöcke an die richtigen Stellen

# ============================================================================
# SNIPPET 1: Multi-Frame Circle Detection (OPTIONAL)
# ============================================================================
# Füge dies ein in main.py __init__ (nach self.board_mapper = ...)

"""
# Multi-Frame Circle Detection (optional, für stabileres Hough)
from collections import deque
self.circle_buffer_centers = deque(maxlen=30)
self.circle_buffer_radii = deque(maxlen=30)

def smooth_circle_detection(self, detected_circle):
    '''Glättet Hough Circle Detection über mehrere Frames'''
    if detected_circle is None:
        return None
    
    cx, cy, r = detected_circle
    
    # Add to buffers
    self.circle_buffer_centers.append((cx, cy))
    self.circle_buffer_radii.append(r)
    
    # Need enough frames
    if len(self.circle_buffer_centers) < 10:
        return (cx, cy, r)
    
    # Calculate median (robust to outliers)
    centers = np.array(list(self.circle_buffer_centers))
    radii = np.array(list(self.circle_buffer_radii))
    
    median_center = np.median(centers, axis=0)
    median_radius = np.median(radii)
    
    # Filter outliers
    valid_centers = []
    valid_radii = []
    
    for (cx_i, cy_i), r_i in zip(self.circle_buffer_centers, self.circle_buffer_radii):
        center_dist = np.sqrt((cx_i - median_center[0])**2 + (cy_i - median_center[1])**2)
        radius_ratio = abs(r_i - median_radius) / median_radius
        
        if center_dist < 15 and radius_ratio < 0.05:
            valid_centers.append((cx_i, cy_i))
            valid_radii.append(r_i)
    
    if len(valid_centers) < 10:
        return (cx, cy, r)
    
    # Return smoothed values
    avg_cx = int(np.mean([c[0] for c in valid_centers]))
    avg_cy = int(np.mean([c[1] for c in valid_centers]))
    avg_r = float(np.mean(valid_radii))
    
    return (avg_cx, avg_cy, avg_r)
"""

# USAGE in _hough_refine_rings() oder wo auch immer du Hough Circle nutzt:
# Vorher:
#   detected_circle = (cx, cy, radius)
# 
# Nachher:
#   detected_circle = self.smooth_circle_detection((cx, cy, radius))


# ============================================================================
# SNIPPET 2: Auto-Center via Dart Clustering (OPTIONAL)
# ============================================================================
# Füge dies ein in main.py __init__

"""
# Auto-Center Correction via Dart Clustering
self.dart_impacts_for_centering = []  # List of (x, y) tuples

def auto_correct_center_from_darts(self):
    '''Korrigiert Center basierend auf Dart-Clustering'''
    if len(self.dart_impacts_for_centering) < 15:
        return False  # Need more darts
    
    # Calculate center of mass
    impacts = np.array(self.dart_impacts_for_centering)
    new_center = np.mean(impacts, axis=0)
    
    # Calculate offset
    if self.board_mapper:
        old_cx = self.board_mapper.calib.cx
        old_cy = self.board_mapper.calib.cy
        
        offset_dx = new_center[0] - old_cx
        offset_dy = new_center[1] - old_cy
        
        # Apply if offset is significant (>3px)
        if abs(offset_dx) > 3 or abs(offset_dy) > 3:
            self.board_mapper.calib.cx += offset_dx
            self.board_mapper.calib.cy += offset_dy
            
            logger.info(f"[AUTO-CENTER] Corrected by ({offset_dx:.1f}, {offset_dy:.1f})px")
            logger.info(f"[AUTO-CENTER] New center: ({self.board_mapper.calib.cx:.1f}, {self.board_mapper.calib.cy:.1f})")
            
            # Clear buffer
            self.dart_impacts_for_centering.clear()
            return True
    
    return False
"""

# USAGE nach Dart-Detection:
# In process_frame(), nach if impact:
"""
if impact:
    # ... existing code ...
    
    # Add to centering buffer
    self.dart_impacts_for_centering.append(impact.position)
    
    # Auto-correct every 20 darts
    if len(self.dart_impacts_for_centering) >= 20:
        self.auto_correct_center_from_darts()
"""


# ============================================================================
# SNIPPET 3: Runtime Alpha-Tuning (EASY!)
# ============================================================================
# Füge dies ein in main.py __init__

"""
# Runtime-adjustable overlay alpha
self.overlay_alpha = 0.3  # Default
"""

# Füge dies ein in keyboard handler (wo du 'q', 'o', etc. behandelst):
"""
elif key == ord('+') or key == ord('='):  # Plus key
    self.overlay_alpha = min(1.0, self.overlay_alpha + 0.1)
    logger.info(f"Overlay alpha: {self.overlay_alpha:.1f}")

elif key == ord('-') or key == ord('_'):  # Minus key
    self.overlay_alpha = max(0.0, self.overlay_alpha - 0.1)
    logger.info(f"Overlay alpha: {self.overlay_alpha:.1f}")
"""

# Dann in draw_precise_dartboard() call:
"""
disp_roi = draw_precise_dartboard(
    disp_roi, self.board_mapper,
    alpha=self.overlay_alpha,  # <-- Use dynamic value
    show_numbers=True,
    show_wires=True
)
"""


# ============================================================================
# WELCHE SNIPPETS BRAUCHST DU?
# ============================================================================

"""
EMPFEHLUNG:

1. Nutze KEIN Snippet für Multi-Frame Detection
   → Zu kompliziert, marginaler Gewinn
   → Nutze stattdessen stabilere Beleuchtung

2. Nutze KEIN Snippet für Auto-Center
   → Nutze stattdessen das Standalone-Tool: correct_center.py
   → Einmal ausführen bei Bedarf, fertig!

3. Nutze NUR Snippet 3: Runtime Alpha-Tuning
   → Super einfach (3 Zeilen in __init__, 8 Zeilen in keyboard handler)
   → Mega praktisch zum Live-Anpassen


MINIMALE INTEGRATION (nur Alpha-Tuning):

# In __init__ (Zeile ~150, nach self.board_mapper = ...):
self.overlay_alpha = 0.3

# In keyboard handler (wo du if key == ord('q'): hast):
elif key == ord('+'):
    self.overlay_alpha = min(1.0, self.overlay_alpha + 0.1)
    print(f"Alpha: {self.overlay_alpha:.1f}")
elif key == ord('-'):
    self.overlay_alpha = max(0.0, self.overlay_alpha - 0.1)
    print(f"Alpha: {self.overlay_alpha:.1f}")

# In draw_precise_dartboard():
... alpha=self.overlay_alpha ...

FERTIG! Nur 12 Zeilen Code.
"""
