# Patch-Inbox & `git apply` – Anleitung (Smart‑DARTS)

**Ziel**: Patches sicher anwenden – lokal, automatisiert per Python (`PatchManager`) sowie sandboxed mit `git worktree`.  
**Kontext**: Python ≥3.10, Git installiert, Repository-Root als Arbeitsverzeichnis.

---

## 1) Schnellstart mit Here-Doc (dein Beispiel)

**Linux/macOS (Bash, Git Bash unter Windows ebenfalls möglich):**
```bash
(cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF'
diff --git a/src/vision/dart_impact_detector.py b/src/vision/dart_impact_detector.py
index 456b919d0f1a527502547b2cd632724d28285102..1a9cc309e07f17f3e86aab79f003c690fb07afce 100644
--- a/src/vision/dart_impact_detector.py
+++ b/src/vision/dart_impact_detector.py
@@ -153,58 +153,70 @@ class DartImpactDetector:
             distance = np.sqrt(dx * dx + dy * dy)
 
             if distance < self.config.cooldown_radius_px:
                 return True
 
         return False
 
     # ... rest bleibt gleich
 
     def _find_dart_shapes(
             self,
             frame: np.ndarray,
             motion_mask: np.ndarray,
             frame_index: int,
             timestamp: float
     ) -> List[DartCandidate]:
         """Find dart-like objects in motion mask"""
 
         # Find contours
         contours, _ = cv2.findContours(
             motion_mask,
             cv2.RETR_EXTERNAL,
             cv2.CHAIN_APPROX_SIMPLE
         )
 
-        candidates = []
+        if not contours:
+            return []
 
-        for contour in contours:
-            area = cv2.contourArea(contour)
+        areas = np.fromiter(
+            (cv2.contourArea(contour) for contour in contours),
+            dtype=np.float32,
+            count=len(contours)
+        )
+        valid_indices = np.nonzero(
+            (areas >= self.config.min_area) & (areas <= self.config.max_area)
+        )[0]
 
-            # Filter by area
-            if not (self.config.min_area <= area <= self.config.max_area):
-                continue
+        if valid_indices.size == 0:
+            return []
+
+        candidates = []
+
+        for idx in valid_indices:
+            contour = contours[idx]
+            area = float(areas[idx])
 
             # Get bounding box
             x, y, w, h = cv2.boundingRect(contour)
 
             # Calculate aspect ratio
             aspect_ratio = float(w) / h if h > 0 else 0
 
             # Filter by aspect ratio (darts are elongated)
             if not (self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio):
                 continue
 
             # Calculate center
             M = cv2.moments(contour)
             if M['m00'] == 0:
                 continue
 
             cx = int(M['m10'] / M['m00'])
             cy = int(M['m01'] / M['m00'])
 
             # Calculate confidence based on shape quality
             # Darts have low circularity (elongated shape)
             perimeter = cv2.arcLength(contour, True)
             circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
 
             # Lower circularity = higher confidence for dart 
EOF
)
```

**Hinweise**:
- `--3way` hilft bei Kontextverschiebungen (robuster gegenüber leichten Abweichungen im Ziel-File).
- Falls die Arbeitskopie nicht clean ist, vorher committen oder stagen.
- Windows PowerShell nutzt ein anderes Here-String-Syntax; nutze dort **Git Bash** (empfohlen).

---

## 2) Nutzung via Python (`tools/patch_manager.py`)

1. Datei `tools/patch_manager.py` ins Repo legen (siehe unten „Artifacts“ oder Download-Link).
2. Patch per Datei anwenden (Dry-Run → Apply → Commit):  
   ```bash
   python tools/patch_manager.py --patch patches/impact_detector_vectorized.patch --dry-run
   python tools/patch_manager.py --patch patches/impact_detector_vectorized.patch --branch --commit --message "ImpactDetector: vectorized contour filtering"
   ```
3. MBox/E-Mail-Patch (aus `git format-patch`) mit `git am`:  
   ```bash
   python tools/patch_manager.py --patch patches/0001-some-change.mbox --am
   ```
4. Rollback (Reverse):  
   ```bash
   python tools/patch_manager.py --patch patches/impact_detector_vectorized.patch -R --branch --commit
   ```

---

## 3) Sandbox-Gating mit `git worktree` (Zero-Risk-Apply)

**Zweck**: Patch in **separater Arbeitskopie** anwenden, **Tests laufen lassen**, nur bei Erfolg übernehmen.

```bash
python tools/patch_manager.py --patch patches/impact_detector_vectorized.patch     --sandbox --tests "pytest -q" --keep-on-success --commit
```

**Was passiert intern**:
- Neuer Branch `patch/<timestamp>` und Worktree in `.worktrees/patch_<timestamp>/`.
- Patch wird dort angewendet und (optional) committed.
- Tests werden ausgeführt (`--tests` Befehl).
- Bei Erfolg: Worktree bleibt erhalten (oder per `--push` direkt zum Remote).  
  Bei Fehler: Worktree wird entfernt, Branch gelöscht → Haupt-Repo bleibt unverändert.

**Beispiele**:
```bash
# 1) Nur verifizieren, kein Commit
python tools/patch_manager.py --patch patches/impact_detector_vectorized.patch --sandbox --tests "pytest -q"

# 2) Verifizieren + Commit + Push
python tools/patch_manager.py --patch patches/impact_detector_vectorized.patch --sandbox --tests "pytest -q" --commit --push
```

---

## 4) CI-Integration (GitHub Actions)

Workflow: **Manuelles Triggern mit Patch-Inhalt** oder **Dateipfad**. Führt Dry-Run + Tests aus, lädt Log als Artefakt hoch.

`.github/workflows/patch-apply.yml`:
```yaml
name: Patch Verify & Apply
on:
  workflow_dispatch:
    inputs:
      patch_text:
        description: 'Raw patch text (optional if patch_path provided)'
        required: false
      patch_path:
        description: 'Repo-relative patch path (e.g., patches/impact_detector_vectorized.patch)'
        required: false

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install deps
        run: |
          python -m pip install -U pip pytest
      - name: Prepare patch
        id: prep
        run: |
          if [ -n "${{ github.event.inputs.patch_text }}" ]; then
            mkdir -p patches
            printf "%s" "${{ github.event.inputs.patch_text }}" > patches/ci_input.patch
            echo "patchfile=patches/ci_input.patch" >> $GITHUB_OUTPUT
          elif [ -n "${{ github.event.inputs.patch_path }}" ]; then
            echo "patchfile=${{ github.event.inputs.patch_path }}" >> $GITHUB_OUTPUT
          else
            echo "No patch provided"; exit 1
          fi
      - name: Dry-run
        run: |
          python tools/patch_manager.py --patch "${{ steps.prep.outputs.patchfile }}" --dry-run
      - name: Sandbox verify with tests
        run: |
          python tools/patch_manager.py --patch "${{ steps.prep.outputs.patchfile }}" --sandbox --tests "pytest -q"
      - name: Archive logs
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: patch-logs
          path: patches/applied.log
```

---

## 5) Troubleshooting

- **CRLF/Whitespace**: `PatchManager` nutzt `--whitespace=fix`. Bei Problemen: `git config core.autocrlf true` (Windows).
- **Kontextfehler**: Mit `--3way` versuchen. Wenn weiter fehlend: Patch aus dem gleichen Commit-Stand erzeugen.
- **Dirty working tree**: Tool blockiert standardmäßig. Mit `--allow-dirty` überschreibbar (nicht empfohlen).
- **PowerShell-Here-Strings**: Verwende **Git Bash** oder speichere den Patch in eine Datei und nutze `--patch <file>`.

---

## 6) Appendix – Patch als Datei speichern

`patches/impact_detector_vectorized.patch` (siehe Download) enthält exakt die Änderungen aus deinem Beispiel-Here-Doc.  
Anwenden:
```bash
python tools/patch_manager.py --patch patches/impact_detector_vectorized.patch --branch --commit
```

---

© 2025-10-14 · Smart‑DARTS Patch‑Inbox Guide
