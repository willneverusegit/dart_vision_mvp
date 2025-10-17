# 🎯 Smart-DARTS Vision MVP
**Automatische Wurferkennung & Scoring-System für elektronische Darts**

## 🧭 Projektüberblick
Smart-DARTS ist ein modulares Computer-Vision-System zur Erkennung und Bewertung von Dartwürfen.
Das Programm läuft in Echtzeit, analysiert Videoframes aus Webcam oder Datei und berechnet Trefferposition, Segment und Punkte.

**Technologien:** Python 3.10 +, OpenCV 4.10 +, NumPy, YAML, Pydantic  
**Architektur:** Modular (Capture · Calibration · Vision · Board · Overlay · Game)

## ⚙️ Modul-Kapselung
| Modul | Zweck | Hauptklassen |
|:------|:------|:-------------|
| src/capture | Videoeinzug & FPS-Messung | ThreadedCamera, CameraConfig, FPSCounter |
| src/calibration | Kalibrierung & ROI | UnifiedCalibrator, ROIProcessor, ArucoQuadCalibrator |
| src/vision | Bewegung & Treffererkennung | MotionDetector, DartImpactDetector |
| src/board | Geometrie & Scoring | BoardMapper, BoardConfig, Calibration |
| src/overlay | Visuelle Overlays | draw_ring_circles, draw_sector_labels, Heatmaps |
| src/game | Spiel-Logik | DemoGame, GameMode |
| src/utils | Performance & Tools | PerformanceProfiler, StatsAccumulator |

## 🧩 Programmablauf
Ablauf: Capture → ROI → Motion → Impact → Mapping → Scoring → Overlay.

## 🎮 Steuerung & Hotkeys
q=Quit | p=Pause | d=Debug | m=Motion | r=Reset | s=Screenshot | c=Recalibrate | o=Overlay | 1/2/3=Presets | t=Hough | z=Auto-Hough | Pfeile=Move | ,/.=Rotate | 0=Reset | X=Save | g=Game reset | h=Switch game | ?=Help

## 🧪 Startbefehle
python main.py --webcam 0  
python main.py --video test_videos/dart_throw.mp4  
python main.py --calibrate --webcam 0  
python main.py --load-yaml config/calibration_unified.yaml

## 🧠 Kalibrierungs-Workflow
1. Board zeigen → c drücken  
2. a (Aruco-Quad) oder m (manuell)  
3. s → Speichern → YAML  
4. Overlay mit Pfeilen justieren  
5. X → speichern

## 📈 Presets
| Name | Beschreibung | Verwendung |
|:-----|:--------------|:------------|
| aggressive | erkennt früh, evtl. mehr False Positives | schnelle Tests |
| balanced | Standard, robust | Alltag |
| stable | streng, präzise | Demo/Wettkampf |

## 🧩 Scoring-System
Bull=50, Outer=25, Double=2×n, Triple=3×n, Single=n

## 🔍 Spielmodi
ATC (Around the Clock) und 301 mit Punktabzug bis 0.

## 💡 Hinweise
Echtzeitbetrieb ~30 FPS bei 1080p CPU-only, logging in dart_vision.log
