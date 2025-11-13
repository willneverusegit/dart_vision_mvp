# Datenverzeichnis: Raw Data

## Platzieren Sie hier Ihre 3MA-X8 Rohdaten!

### Erwarteter Dateiname

```
3ma_x8_features.csv
```

### Erwartetes Format

Ihre CSV-Datei sollte folgende Struktur haben:

| sample_id | class | feature_1 | feature_2 | ... | feature_261 |
|-----------|-------|-----------|-----------|-----|-------------|
| P001      | A     | 0.123     | 0.456     | ... | 0.789       |
| P001      | A     | 0.124     | 0.457     | ... | 0.790       |
| P002      | B     | 0.234     | 0.567     | ... | 0.891       |
| ...       | ...   | ...       | ...       | ... | ...         |

### Wichtige Spalten

1. **`sample_id`** (oder ähnlich):
   - Eindeutige Proben-ID
   - Wird für GroupKFold verwendet
   - Beispiel: `P001`, `Sample_01`, `Probe_A`

2. **`class`** (oder ähnlich):
   - Zielvariable (Material-/Zustandsklasse)
   - Beispiel: `A`, `B`, `C` oder `Material_1`, `Zustand_normal`

3. **`feature_1` ... `feature_261`**:
   - Ihre 3MA-X8 Features
   - Numerische Werte
   - Fehlende Werte (NaN) sind erlaubt (werden imputiert)

### Falls Ihre Spaltennamen anders sind

Kein Problem! Passen Sie in jedem Notebook diese Zeilen an:

```python
# In allen 4 Notebooks:
TARGET_COL = 'class'       # <-- Ihre Zielvariablen-Spalte
GROUP_COL = 'sample_id'    # <-- Ihre Proben-ID-Spalte
```

### Datenschutz

**WICHTIG:** Diese Datei wird NICHT in Git committed (siehe `.gitignore`)!

Ihre Rohdaten bleiben auf Ihrem lokalen System/Server.

---

**Nach dem Hochladen Ihrer Daten:**

Führen Sie Notebook 1 aus, um die Pipeline zu starten.
