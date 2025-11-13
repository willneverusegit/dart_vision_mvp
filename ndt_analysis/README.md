# NDT Feature Selection Pipeline

**Masterarbeit:** Zerstörungsfreie Werkstoffprüfung mittels 3MA-X8-Mikromagnetik
**Methodik:** Vierstufige Feature-Selektions-Pipeline für LDA/QDA-Klassifikation

---

## Übersicht

Diese Pipeline reduziert einen initialen 261-dimensionalen Feature-Space auf ein robustes, methodenagnostisches Subset zur Material- und Zustands-Klassifikation.

### Pipeline-Architektur

```
Phase 1: Qualitätsfilterung & Korrelations-Prepruning
         261 Features → ~84 Features
         ├─ Missing Values Filter (>15%)
         ├─ Near-Zero Variance Filter
         ├─ OvR-Signal Berechnung
         └─ Hierarchisches Clustering (|ρ| ≥ 0.90)

Phase 2: Multi-Methoden Feature-Ranking
         8 unabhängige Ranking-Methoden (Fold-Aware)
         ├─ ANOVA F-Test
         ├─ Mutual Information
         ├─ mRMR
         ├─ ReliefF
         ├─ L1-Lasso
         ├─ Random Forest
         ├─ Permutation Importance
         └─ PCA-Importance

Phase 3: Iterative Reduktions-Evaluierung
         LDA/QDA Benchmarking (10 Stufen × 8 Rankings)
         └─ 5-Fold GroupKFold CV mit 95% CI

Phase 4: Konsensus-Analyse
         Methodenagnostisches Core-Set
         └─ Rang-Normalisierung + Mittelung
```

---

## Installation

### Voraussetzungen

- Python 3.9+
- JupyterHub oder lokale Jupyter-Installation

### Setup auf JupyterHub

1. **Repository hochladen:**
   ```bash
   # Falls Git verfügbar:
   git clone <your-repo-url>
   cd ndt_analysis

   # Oder: Dateien manuell hochladen
   ```

2. **Dependencies installieren:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Datenstruktur vorbereiten:**
   ```
   ndt_analysis/
   ├── data/
   │   └── raw/
   │       └── 3ma_x8_features.csv  # <-- IHRE DATEN HIER!
   ```

---

## Datenformat

### Erwartete CSV-Struktur

Ihre Datei `3ma_x8_features.csv` sollte folgendes Format haben:

| sample_id | class | feature_1 | feature_2 | ... | feature_261 |
|-----------|-------|-----------|-----------|-----|-------------|
| P001      | A     | 0.123     | 0.456     | ... | 0.789       |
| P001      | A     | 0.124     | 0.457     | ... | 0.790       |
| P002      | B     | 0.234     | 0.567     | ... | 0.891       |

**Kritische Spalten:**

- `sample_id`: Proben-ID für GroupKFold (verhindert Data Leakage)
- `class`: Zielvariable (Material-/Zustandsklasse)
- `feature_1` ... `feature_261`: Ihre 3MA-X8 Features

### Anpassung an Ihre Daten

In jedem Notebook müssen Sie **zwei Zeilen** anpassen:

```python
# ANPASSEN: Dateipfad
DATA_PATH = '../data/raw/3ma_x8_features.csv'  # <-- Ihr Dateipfad

# ANPASSEN: Spaltennamen
TARGET_COL = 'class'       # <-- Name Ihrer Zielvariablen-Spalte
GROUP_COL = 'sample_id'    # <-- Name Ihrer Proben-ID-Spalte
```

---

## Verwendung

### 1. Notebooks nacheinander ausführen

Die Notebooks müssen **in Reihenfolge** ausgeführt werden:

```
01_Phase1_Filtering_Prepruning.ipynb
  ↓ (erzeugt: features_after_phase1.csv)

02_Phase2_Multi_Method_Ranking.ipynb
  ↓ (erzeugt: 8 Ranking-CSVs)

03_Phase3_Evaluation_Benchmarking.ipynb
  ↓ (erzeugt: Pareto-Kurven, Performance-Tabellen)

04_Phase4_Consensus_Analysis.ipynb
  ↓ (erzeugt: Finales Konsensus-Ranking)
```

### 2. Workflow pro Notebook

Jedes Notebook ist in Sektionen unterteilt:

1. **Daten laden** → Passen Sie Dateipfade an
2. **Verarbeitung** → Führen Sie alle Zellen aus
3. **Visualisierung** → Plots werden inline angezeigt
4. **Ergebnisse speichern** → CSV/PNG werden automatisch gespeichert

### 3. Ausführungszeit

**Geschätzte Laufzeit (Intel i5, 16GB RAM):**

- Notebook 1: ~2-5 Minuten
- Notebook 2: ~10-20 Minuten ⚠️ (8 Methoden × 5 Folds)
- Notebook 3: ~30-60 Minuten ⚠️ (800 CV-Trainings!)
- Notebook 4: ~10-15 Minuten

**TIPP:** Notebook 3 ist rechenintensiv. Führen Sie es idealerweise über Nacht aus oder reduzieren Sie `n_splits=5` → `n_splits=3`.

---

## Outputs

### Verzeichnisstruktur nach Ausführung

```
ndt_analysis/
├── data/
│   ├── raw/
│   │   └── 3ma_x8_features.csv         # Input
│   └── processed/
│       └── features_after_phase1.csv   # Nach Phase 1
├── results/
│   ├── rankings/
│   │   ├── phase1_feature_info.csv
│   │   ├── phase2_ranking_ANOVA.csv
│   │   ├── phase2_ranking_MutualInfo.csv
│   │   ├── ... (8 Rankings)
│   │   ├── phase4_consensus_ranking_full.csv      # ★ FINALES RANKING
│   │   └── phase4_optimal_features.csv            # ★ EMPFOHLENES SET
│   ├── evaluations/
│   │   ├── phase3_evaluation_master.csv           # Alle Benchmarks
│   │   ├── phase4_consensus_evaluation.csv
│   │   └── phase4_method_comparison.csv
│   └── plots/
│       ├── phase1_correlation_heatmap.png
│       ├── phase2_ranking_comparison.png
│       ├── phase3_pareto_lda.png
│       ├── phase3_pareto_qda.png
│       └── phase4_consensus_pareto_lda.png
```

### Wichtigste Dateien für Ihre Arbeit

| Datei | Beschreibung | Verwendung |
|-------|--------------|------------|
| `phase4_consensus_ranking_full.csv` | Komplettes Konsensus-Ranking | Methodenagnostisches Ranking |
| `phase4_optimal_features.csv` | Empfohlenes Feature-Set (Elbow-Point) | Verwenden Sie diese Features! |
| `phase3_evaluation_master.csv` | Alle Performance-Benchmarks | Methodenvergleich, Tabellen |
| `phase3_pareto_lda.png` | Pareto-Kurven LDA | Visualisierung für Paper |

---

## Methodische Details

### Kritische Aspekte

#### 1. GroupKFold Cross-Validation

**Warum?** Mehrfachmessungen derselben Probe dürfen nicht auf Train/Test aufgeteilt werden!

```python
# RICHTIG:
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=sample_ids):
    ...

# FALSCH (Data Leakage!):
kf = KFold(n_splits=5)
for train_idx, test_idx in kf.split(X, y):
    ...
```

#### 2. Preprocessing innerhalb CV-Folds

**Warum?** Imputation/Skalierung auf gesamten Daten führt zu Overfitting-Bias!

```python
# RICHTIG (Pipeline):
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LinearDiscriminantAnalysis())
])
pipeline.fit(X_train, y_train)  # Fit nur auf Train!

# FALSCH:
scaler.fit(X)  # Fit auf gesamten Daten!
X_scaled = scaler.transform(X)
# Dann CV...
```

#### 3. Konfidenzintervalle mit t-Verteilung

Bei 5 Folds haben wir nur 5 Datenpunkte → t-Verteilung (df=4) statt z-Verteilung!

```python
from scipy import stats
t_critical = stats.t.ppf(0.975, df=4)  # 95% CI, 2-seitig
```

### Parameter-Tuning

Falls Sie Parameter anpassen möchten:

#### Phase 1: Korrelationsschwellwert

```python
# Im Notebook ändern:
CORR_THRESHOLD = 0.90  # Standard
# Erhöhen → weniger Features eliminiert (konservativer)
# Senken → mehr Features eliminiert (aggressiver)
```

#### Phase 2: Random Forest Hyperparameter

```python
rf = RandomForestClassifier(
    n_estimators=100,      # Mehr → stabiler, aber langsamer
    max_depth=10,          # Kleiner bei Overfitting
    min_samples_split=5,   # Größer bei Overfitting
    random_state=42
)
```

#### Phase 3: QDA Regularisierung

```python
qda = QuadraticDiscriminantAnalysis(
    reg_param=0.1  # Erhöhen bei Singularitäts-Fehlern!
)
```

---

## Troubleshooting

### Problem 1: "Singular matrix" (QDA)

**Ursache:** Zu wenige Samples pro Klasse für QDA-Kovarianzschätzung.

**Lösung:**
```python
# Erhöhen Sie reg_param:
qda = QuadraticDiscriminantAnalysis(reg_param=0.5)  # statt 0.1
```

### Problem 2: Notebook 2 dauert sehr lange

**Lösung:**
```python
# Reduzieren Sie CV-Folds:
n_splits = 3  # statt 5

# Oder: Reduzieren Sie Random Forest Estimators:
n_estimators = 50  # statt 100
```

### Problem 3: "Feature not found in DataFrame"

**Ursache:** Spaltennamen passen nicht.

**Lösung:** Überprüfen Sie:
```python
print(df.columns.tolist())  # Alle Spaltennamen anzeigen
```

### Problem 4: Memory Error bei Notebook 3

**Lösung:**
```python
# Reduzieren Sie Reduktionsstufen:
REDUCTION_PERCENTAGES = [0.80, 0.60, 0.40, 0.20, 0.10]  # statt 10 Stufen
```

---

## Interpretation der Ergebnisse

### Pareto-Kurven

**Was zeigen sie?**
Trade-off zwischen Feature-Anzahl (x-Achse) und Performance (y-Achse).

**Wie interpretieren?**
- **Elbow-Point:** Stelle, an der weitere Features kaum noch Performance bringen
- **Steile Anstiege:** Diese Features sind kritisch
- **Flache Bereiche:** Redundante Features

**Beispiel:**
```
Performance
    │
0.9 │         ╭────────  (Plateau: Redundanz)
    │        ╱
0.8 │       ╱ ← Elbow (optimal!)
    │      ╱
0.7 │     ╱
    │____╱________________
        10   20   30   40  Features
```

### Konsensus-Score

**Was bedeutet er?**
Mittelwert der normalisierten Ränge über alle 8 Methoden.

- **Score ≈ 1.0:** Feature wird von ALLEN Methoden als wichtig eingestuft → sehr robust
- **Score ≈ 0.5:** Mittelmäßige Wichtigkeit
- **Score ≈ 0.0:** Feature wird von den meisten Methoden als unwichtig eingestuft

### Rang-Varianz

**Was bedeutet sie?**
Wie stark schwanken die Ränge eines Features über die Methoden?

- **Niedrige Varianz:** Konsens zwischen Methoden → robust
- **Hohe Varianz:** Uneinigkeit → methodenabhängig, vorsichtig verwenden

---

## Referenzen

### Implementierte Methoden

1. **ANOVA F-Test:**
   Fisher, R.A. (1925). Statistical Methods for Research Workers.

2. **Mutual Information:**
   Cover, T.M., Thomas, J.A. (2006). Elements of Information Theory.

3. **mRMR:**
   Peng, H., et al. (2005). Feature selection based on mutual information.

4. **ReliefF:**
   Kononenko, I. (1994). Estimating attributes: Analysis and extensions of RELIEF.

5. **L1-Lasso:**
   Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.

6. **Random Forest:**
   Breiman, L. (2001). Random Forests.

7. **Permutation Importance:**
   Breiman, L. (2001). Statistical modeling: The two cultures.

8. **PCA:**
   Pearson, K. (1901). On lines and planes of closest fit to systems of points in space.

### Validierungsstrategien

- **GroupKFold:**
  Sklearn Documentation - GroupKFold

- **Konfidenzintervalle:**
  Student's t-distribution (Gosset, W.S., 1908)

---

## Kontakt & Support

Bei Fragen zur Implementierung oder Methodik:

1. Überprüfen Sie die **Markdown-Zellen** in den Notebooks (enthalten methodische Erklärungen)
2. Konsultieren Sie die **Spezifikation** am Anfang dieses Projekts
3. Prüfen Sie die **Utility-Module** (`ndt_analysis/utils/`) für technische Details

---

## Lizenz

Dieses Projekt wurde für akademische Forschung entwickelt (Masterarbeit NDT/3MA-X8).
Verwendung für eigene Forschungsprojekte ausdrücklich erlaubt.

---

**Viel Erfolg mit Ihrer Masterarbeit!** 🎓
