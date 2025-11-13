"""
Metriken-Utilities für NDT Feature Selection Pipeline
Berechnung von Balanced Accuracy, F1-Score, Cohen's Kappa
"""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from typing import Dict, Tuple
import pandas as pd


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    return_all: bool = True
) -> Dict[str, float]:
    """
    Berechnet alle relevanten Klassifikationsmetriken.

    METRIK-HIERARCHIE (gemäß Spezifikation):
    1. Balanced Accuracy (primär bei Klassenimbalance)
    2. F1-Score (macro-averaged)
    3. Cohen's Kappa (κ)
    4. Standard Accuracy

    Parameters
    ----------
    y_true : np.ndarray
        Wahre Labels
    y_pred : np.ndarray
        Vorhergesagte Labels
    return_all : bool
        Wenn True, alle Metriken zurückgeben; sonst nur Primärmetriken

    Returns
    -------
    dict
        Dictionary mit allen berechneten Metriken
    """
    metrics = {}

    # Primärmetriken (immer berechnen)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    if return_all:
        # Zusätzliche F1-Varianten
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)

    return metrics


def aggregate_cv_metrics(
    cv_results: list,
    metric_names: list = None
) -> pd.DataFrame:
    """
    Aggregiert Metriken aus mehreren CV-Folds.

    Parameters
    ----------
    cv_results : list
        Liste von Dictionaries mit Metriken pro Fold
    metric_names : list, optional
        Liste der zu aggregierenden Metriken

    Returns
    -------
    pd.DataFrame
        Aggregierte Statistiken (Mean, Std, CI)
    """
    if metric_names is None:
        metric_names = ['balanced_accuracy', 'f1_macro', 'cohen_kappa', 'accuracy']

    from .validation import calculate_confidence_intervals

    results = {}
    for metric in metric_names:
        scores = np.array([fold[metric] for fold in cv_results])
        mean, ci_lower, ci_upper = calculate_confidence_intervals(scores)

        results[metric] = {
            'mean': mean,
            'std': np.std(scores, ddof=1),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'margin': (ci_upper - ci_lower) / 2
        }

    return pd.DataFrame(results).T


def format_metric_result(
    mean: float,
    ci_lower: float,
    ci_upper: float,
    precision: int = 3
) -> str:
    """
    Formatiert Metrik-Ergebnis als String: "mean [ci_lower, ci_upper]"

    Parameters
    ----------
    mean : float
        Mittelwert
    ci_lower : float
        Untere CI-Grenze
    ci_upper : float
        Obere CI-Grenze
    precision : int
        Dezimalstellen (Standard: 3)

    Returns
    -------
    str
        Formatierter String

    Example
    -------
    >>> format_metric_result(0.857, 0.834, 0.880)
    '0.857 [0.834, 0.880]'
    """
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(mean)} [{fmt.format(ci_lower)}, {fmt.format(ci_upper)}]"


def print_classification_report_extended(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None
) -> None:
    """
    Druckt erweiterten Klassifikationsbericht.

    Parameters
    ----------
    y_true : np.ndarray
        Wahre Labels
    y_pred : np.ndarray
        Vorhergesagte Labels
    class_names : list, optional
        Namen der Klassen
    """
    print("\n" + "=" * 70)
    print("KLASSIFIKATIONSBERICHT")
    print("=" * 70)

    # Primärmetriken
    metrics = compute_classification_metrics(y_true, y_pred)
    print(f"\n📊 Primärmetriken:")
    print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"   F1-Score (macro):  {metrics['f1_macro']:.4f}")
    print(f"   Cohen's Kappa:     {metrics['cohen_kappa']:.4f}")
    print(f"   Accuracy:          {metrics['accuracy']:.4f}")

    # Sklearn Classification Report
    print(f"\n📈 Detaillierter Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # Konfusionsmatrix-Statistiken
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔢 Konfusionsmatrix-Dimensionen: {cm.shape}")
    print(f"   Korrekt klassifiziert: {np.trace(cm)} / {np.sum(cm)}")

    print("=" * 70)


def compare_classifiers(
    results_dict: Dict[str, pd.DataFrame],
    metric: str = 'balanced_accuracy'
) -> pd.DataFrame:
    """
    Vergleicht Performance mehrerer Klassifikatoren.

    Parameters
    ----------
    results_dict : dict
        Dictionary mit Klassifikator-Namen als Keys und Metriken-DataFrames als Values
    metric : str
        Metrik zum Vergleich (Standard: 'balanced_accuracy')

    Returns
    -------
    pd.DataFrame
        Vergleichstabelle sortiert nach Performance
    """
    comparison = []

    for clf_name, df in results_dict.items():
        if metric in df.index:
            row = df.loc[metric]
            comparison.append({
                'Classifier': clf_name,
                'Mean': row['mean'],
                'Std': row['std'],
                'CI_Lower': row['ci_lower'],
                'CI_Upper': row['ci_upper'],
                'Formatted': format_metric_result(row['mean'], row['ci_lower'], row['ci_upper'])
            })

    comparison_df = pd.DataFrame(comparison)
    return comparison_df.sort_values('Mean', ascending=False).reset_index(drop=True)
