# ⚠️ IMPORTANT FIX: Test-Imports korrigieren

## 🐛 Problem

Die Tests verwenden **relative Imports** und laden daher die alten Versionen aus dem aktuellen Verzeichnis statt aus `src/vision/`.

**Symptom:**
```
Extra inputs are not permitted [type=extra_forbidden, input_value=True, input_type=bool]
```

---

## ✅ Lösung

Die **neuen Test-Dateien** aus `/outputs/` verwenden bereits die **korrekten Imports**:

```python
# ✅ RICHTIG (neue Version in outputs/)
from src.vision.config_schema import validate_config_file
from src.vision.dart_impact_detector import DartImpactDetector
from src.vision.motion_detector import MotionDetector

# ❌ FALSCH (alte Version)
from config_schema import validate_config_file
from dart_impact_detector_enhanced import DartImpactDetector
```

---

## 🚀 Fix in 2 Schritten

### **Schritt 1: Kopiere die NEUEN Test-Dateien**
```bash
# Die Test-Dateien im Root-Verzeichnis überschreiben
cp test_convexity_gate.py test_convexity_gate_OLD.py  # Backup
cp test_adaptive_motion.py test_adaptive_motion_OLD.py  # Backup

# Aus outputs/ die korrigierten Versionen kopieren
# (Diese haben bereits die richtigen src.vision.* Imports)
```

### **Schritt 2: Tests ausführen**
```bash
python test_convexity_gate.py
# Erwartung: 🎉 5/5 PASSED

python test_adaptive_motion.py
# Erwartung: 🎉 5/5 PASSED
```

---

## 📋 Warum passiert das?

Python-Import-Reihenfolge:
1. Aktuelles Verzeichnis (`./`)
2. `sys.path` (inkl. `src/`)

Wenn `config_schema.py` im aktuellen Verzeichnis liegt, wird diese **statt** `src/vision/config_schema.py` geladen.

---

## ✅ Verification

Nach dem Fix solltest du sehen:

```bash
python test_adaptive_motion.py

============================================================
TEST 4: Motion Config Schema Validation
============================================================
✅ Valid config loaded
   Adaptive Otsu: True        # ← Das bedeutet: neue Felder erkannt!
   Dual threshold: False
   Search mode: True
✅ Correctly rejected invalid brightness threshold (> 255)

✅ TEST 4 PASSED: Motion config schema works correctly
```

---

## 🔍 Debug-Check

Falls Tests immer noch fehlschlagen:

```bash
# Prüfe welche config_schema geladen wird
python -c "import sys; sys.path.insert(0, '.'); from src.vision import config_schema; print(config_schema.__file__)"
# Erwartung: .../src/vision/config_schema.py
```

---

## 📦 Alternative: Tests aus outputs/ direkt ausführen

Falls du die Dateien nicht überschreiben willst:

```bash
# Von outputs/ aus ausführen (empfohlen für erste Tests)
cd outputs/
python test_convexity_gate.py    # Verwendet bereits src.vision.*
python test_adaptive_motion.py   # Verwendet bereits src.vision.*
```

Aber **Achtung:** Dann wird `config/detectors.yaml` nicht gefunden (relativer Pfad).

**Besser:** Kopiere die korrigierten Tests ins Root-Verzeichnis (siehe Schritt 1).

---

## ✅ Status nach Fix

- ✅ `src/vision/config_schema.py` hat alle neuen Motion-Felder
- ✅ `config/detectors.yaml` hat alle neuen Motion-Felder
- ✅ Tests importieren von `src.vision.*` (nicht mehr direkt)
- ✅ Alle 10 Tests sollten jetzt durchlaufen

---

**TL;DR:** Nutze die Test-Dateien aus `/mnt/user-data/outputs/` – die haben bereits die korrekten Imports! 🚀
