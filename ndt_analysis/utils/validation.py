"""
Validation Utilities für NDT Feature Selection Pipeline
Enthält GroupKFold CV und Konfidenzintervall-Berechnung
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
from typing import Tuple, List, Dict


def create_group_kfold_splits(n_splits: int = 5) -> GroupKFold:
    """
    Erstellt GroupKFold Splitter für Cross-Validation.

    KRITISCH: Gruppierung nach Proben-ID verhindert Data Leakage,
    da mehrere Messungen derselben Probe nicht auf Train/Test verteilt werden.

    Parameters
    ----------
    n_splits : int
        Anzahl der Folds (Standard: 5)

    Returns
    -------
    GroupKFold
        Konfigurierter GroupKFold Splitter
    """
    return GroupKFold(n_splits=n_splits)


def calculate_confidence_intervals(
    scores: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Berechnet Mittelwert und Konfidenzintervall mittels t-Verteilung.

    METHODISCH KRITISCH: Bei k=5 Folds haben wir df=4 Freiheitsgrade.
    Die t-Verteilung ist für kleine Stichproben robuster als z-Verteilung.

    Parameters
    ----------
    scores : np.ndarray
        Array von Scores aus k CV-Folds (Shape: [n_folds])
    confidence : float
        Konfidenzniveau (Standard: 0.95 für 95% CI)

    Returns
    -------
    mean : float
        Mittelwert der Scores
    ci_lower : float
        Untere Grenze des Konfidenzintervalls
    ci_upper : float
        Obere Grenze des Konfidenzintervalls

    Example
    -------
    >>> scores = np.array([0.85, 0.87, 0.84, 0.88, 0.86])
    >>> mean, lower, upper = calculate_confidence_intervals(scores)
    >>> print(f"{mean:.3f} [{lower:.3f}, {upper:.3f}]")
    """
    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)  # ddof=1 für Stichprobenstandardabweichung
    se = std / np.sqrt(n)

    # t-Verteilung mit n-1 Freiheitsgraden
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)

    margin = t_critical * se
    ci_lower = mean - margin
    ci_upper = mean + margin

    return mean, ci_lower, ci_upper


def validate_data_structure(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    expected_features: int = 261
) -> Dict[str, any]:
    """
    Validiert die Datenstruktur gemäß Spezifikation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature-Matrix (n_samples × n_features)
    y : pd.Series
        Zielvariable (Klassen)
    groups : pd.Series
        Proben-IDs für GroupKFold
    expected_features : int
        Erwartete Anzahl Features (Standard: 261)

    Returns
    -------
    dict
        Validierungsergebnisse mit Warnungen
    """
    results = {
        'valid': True,
        'warnings': [],
        'info': {}
    }

    # Check: Feature-Anzahl
    n_features = X.shape[1]
    results['info']['n_features'] = n_features
    if n_features != expected_features:
        results['warnings'].append(
            f"Feature-Anzahl: {n_features} (erwartet: {expected_features})"
        )

    # Check: Probenanzahl
    n_samples = X.shape[0]
    results['info']['n_samples'] = n_samples
    if n_samples < 30:
        results['warnings'].append(
            f"WARNUNG: Nur {n_samples} Samples - sehr kleine Stichprobe!"
        )

    # Check: Gruppen
    n_groups = groups.nunique()
    results['info']['n_groups'] = n_groups
    if n_groups != 36:
        results['warnings'].append(
            f"Gruppen: {n_groups} (erwartet: 36 Proben)"
        )

    # Check: Klassen
    n_classes = y.nunique()
    results['info']['n_classes'] = n_classes
    samples_per_class = y.value_counts().to_dict()
    results['info']['samples_per_class'] = samples_per_class

    if n_classes > 10 and n_samples < 100:
        results['warnings'].append(
            f"KRITISCH: {n_classes} Klassen bei nur {n_samples} Samples - "
            f"QDA wird instabil sein!"
        )

    # Check: Missing Values
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        results['info']['missing_values'] = missing_count
        results['warnings'].append(
            f"Missing Values: {missing_count} Einträge"
        )

    return results


def print_validation_report(validation_results: Dict) -> None:
    """
    Druckt einen formatierten Validierungsbericht.

    Parameters
    ----------
    validation_results : dict
        Ausgabe von validate_data_structure()
    """
    print("=" * 70)
    print("DATENSTRUKTUR-VALIDIERUNG")
    print("=" * 70)

    info = validation_results['info']
    print(f"\n📊 Datenübersicht:")
    print(f"   Samples:    {info.get('n_samples', 'N/A')}")
    print(f"   Features:   {info.get('n_features', 'N/A')}")
    print(f"   Gruppen:    {info.get('n_groups', 'N/A')}")
    print(f"   Klassen:    {info.get('n_classes', 'N/A')}")

    if 'samples_per_class' in info:
        print(f"\n📈 Klassenverteilung:")
        for cls, count in sorted(info['samples_per_class'].items()):
            print(f"   Klasse {cls}: {count} Samples")

    if validation_results['warnings']:
        print(f"\n⚠️  Warnungen ({len(validation_results['warnings'])}):")
        for i, warning in enumerate(validation_results['warnings'], 1):
            print(f"   {i}. {warning}")
    else:
        print(f"\n✓ Keine Warnungen - Datenstruktur valide")

    print("=" * 70)
