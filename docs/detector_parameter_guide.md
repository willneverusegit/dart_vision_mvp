# Leitfaden für Bewegungs- und Darterkennungs-Parameter

Dein Ziel bleibt unverändert: **zuverlässige Erkennung der Darttreffer und korrekte Punkteberechnung** – und zwar unter wechselnden Licht- und Umgebungsbedingungen. Das folgende Playbook zeigt dir, wie die neuen Module zusammenarbeiten und wie du die Parameter schnell anpassen kannst, ohne das System aus dem Blick auf dieses Hauptziel zu verlieren.

## 1. Gemeinsame Konfigurationsdatei (`config/detectors.yaml`)
- Sowohl Bewegungs- als auch Darterkennung lesen und schreiben ihre Einstellungen über den `DetectorConfigManager`.
- Alle Updates passieren atomar (temporäre Datei → rename). Stromausfälle oder Abstürze zerstören die Datei damit nicht.
- Du kannst die Datei per Editor anpassen oder komfortabler über die Werkzeuge unten verändern. Schema-Validierung verhindert Tippfehler.

## 2. Automatische Anpassung an die Umgebung
Nutze die Heuristiken des `EnvironmentOptimizer`, wenn du schnell zur „funktioniert jetzt“-Konfiguration kommen willst:

```bash
python main.py --video <quelle.mp4> --auto-optimize --optimize-samples 120 --persist-optimized
```

1. Die App sammelt `--optimize-samples` Frames, misst Helligkeit, Bewegungsrauschen und empfiehlt ein Preset (`balanced`, `aggressive`, `stable`).
2. Die Parameter werden live auf Motion- und Dart-Detector angewendet.
3. Mit `--persist-optimized` landen die Werte dauerhaft in `config/detectors.yaml`.
4. Lass den Optimierer nach Änderungen in der Aufnahmesituation kurz laufen, statt direkt manuell an allen Stellschrauben zu drehen.

💡 **Performance-Hinweis:** Je sauberer die Kamera ausgeleuchtet ist, desto kleiner muss der Optimierer anpassen und desto stabiler laufen die Filter. Eine konstante Beleuchtung spart dir später Feinarbeit.

## 3. Geführtes manuelles Tuning (`parameter_tuner.py`)
Wenn du gezielt Feineinstellungen brauchst:

```bash
python -m src.vision.tuning.parameter_tuner --video <quelle.mp4> --optimize-samples 150
```

- `O` sammelt Beispiel-Frames und führt den selben Environment-Optimierungsschritt aus wie die Haupt-App.
- Pfeiltasten wechseln zwischen Parametergruppen (Bewegung, Vorfilter, Trefferbestätigung, usw.).
- Die Drehregler sind logisch gruppiert; rechts im Dashboard siehst du, welche Kennzahlen (z.B. Motion-Score, Treffer-Fortschritt) sich verändern.
- `S` speichert Änderungen sofort über den `DetectorConfigManager` (wieder atomar), `R` lädt die Datei neu.

Halte bei manuellen Änderungen immer das Zielbild im Kopf: **hohe Trefferquote ohne Fehlalarme**. Wenn du den Motion-Threshold senkst, prüfe z. B. im Gegenzug, ob der Dart-Detector noch sauber filtert, damit du keine falschen Punkte berechnest.

## 4. Suchmodus-Logs (Option)
Der Bewegungsmelder besitzt einen Suchmodus, der nach längerer Ruhephase empfindlicher wird. Die bisherigen Debug-Logs erzeugten viel Lärm. Standardmäßig bleiben sie jetzt aus.

- Setze `log_search_mode_transitions: true` im Motion-Block der YAML, wenn du die Ein- und Ausschaltmomente gezielt untersuchen willst.
- In der Praxis genügt der Statistikzähler (`motion_detector.get_stats()`), um zu prüfen, wie oft der Suchmodus greift.

## 5. Workflow-Empfehlung
1. **Kameraposition & Licht stabilisieren.** Das bringt die größte Gewinnzone für zuverlässige Treffererkennung.
2. **Auto-Optimierung laufen lassen.** So erhältst du einen guten Startwert ohne manuelle Arbeit.
3. **Gezielt Feintunen.** Nutze den Parameter-Tuner nur dort, wo das System noch daneben liegt (z. B. Treffernähe verbessern oder Fehlalarme entfernen).
4. **Ergebnisse testen.** Starte einen kurzen Spiel-Durchlauf und vergleiche Treffer/Score mit der Realität.

> Bleib kritisch: Wenn dir die Performance nicht gefällt, frag dich zuerst, ob das Eingangssignal (Kamera, Licht, Hintergrund) sauber ist. Gute Sensor-Daten sind oft effizienter als noch mehr Post-Processing.

Mit diesem Vorgehen hältst du die App fokussiert auf ihr Kernziel und reagierst dennoch flexibel auf wechselnde Bedingungen.
