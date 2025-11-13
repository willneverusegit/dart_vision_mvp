"""
Visualisierungs-Utilities für NDT Feature Selection Pipeline
Pareto-Plots, Ranking-Vergleiche, Feature-Importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional


# Stil-Konfiguration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_pareto_curve(
    results_df: pd.DataFrame,
    metric: str = 'balanced_accuracy',
    classifier: str = 'LDA',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Erstellt Pareto-Kurve: Feature-Anzahl vs. Performance.

    Visualisiert den Trade-off zwischen Modellkomplexität (Feature-Anzahl)
    und Klassifikationsperformance. Der "Elbow-Point" zeigt das optimale
    Feature-Subset.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame mit Spalten: 'method', 'n_features', 'mean', 'ci_lower', 'ci_upper'
    metric : str
        Metrik zum Plotten (Standard: 'balanced_accuracy')
    classifier : str
        Klassifikator-Name für Titel
    figsize : tuple
        Größe der Figur
    save_path : str, optional
        Speicherpfad für Plot

    Returns
    -------
    plt.Figure
        Matplotlib Figure-Objekt
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Einzigartige Methoden
    methods = results_df['method'].unique()
    colors = sns.color_palette("husl", len(methods))

    for method, color in zip(methods, colors):
        method_data = results_df[results_df['method'] == method].sort_values('n_features')

        # Hauptlinie
        ax.plot(
            method_data['n_features'],
            method_data['mean'],
            marker='o',
            linewidth=2,
            label=method,
            color=color
        )

        # Konfidenzintervall
        ax.fill_between(
            method_data['n_features'],
            method_data['ci_lower'],
            method_data['ci_upper'],
            alpha=0.2,
            color=color
        )

    ax.set_xlabel('Anzahl Features', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Pareto-Kurve: Feature-Reduktion vs. {classifier} Performance\n'
        f'(95% Konfidenzintervalle, 5-Fold GroupKFold CV)',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot gespeichert: {save_path}")

    return fig


def plot_ranking_comparison(
    rankings_dict: Dict[str, pd.DataFrame],
    top_k: int = 20,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Vergleicht Feature-Rankings verschiedener Methoden.

    Parameters
    ----------
    rankings_dict : dict
        Dictionary mit Methoden-Namen als Keys und Ranking-DataFrames als Values
    top_k : int
        Anzahl der Top-Features zum Anzeigen
    figsize : tuple
        Größe der Figur
    save_path : str, optional
        Speicherpfad

    Returns
    -------
    plt.Figure
        Matplotlib Figure-Objekt
    """
    n_methods = len(rankings_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize, sharey=True)

    if n_methods == 1:
        axes = [axes]

    for idx, (method_name, ranking_df) in enumerate(rankings_dict.items()):
        ax = axes[idx]

        # Top-K Features
        top_features = ranking_df.head(top_k).copy()
        top_features = top_features.iloc[::-1]  # Umdrehen für bessere Visualisierung

        # Horizontal Bar Plot
        ax.barh(
            range(len(top_features)),
            top_features['score'] if 'score' in top_features.columns else range(len(top_features), 0, -1),
            color=sns.color_palette("viridis", len(top_features))
        )

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values if 'feature' in top_features.columns else top_features.index)
        ax.set_xlabel('Score', fontsize=10)
        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    axes[0].set_ylabel(f'Top-{top_k} Features', fontsize=12, fontweight='bold')
    fig.suptitle('Feature-Ranking Vergleich', fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot gespeichert: {save_path}")

    return fig


def plot_correlation_heatmap(
    X: pd.DataFrame,
    method: str = 'pearson',
    top_k: int = 30,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Erstellt Korrelations-Heatmap für Top-K Features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature-Matrix
    method : str
        Korrelationsmethode ('pearson' oder 'spearman')
    top_k : int
        Anzahl Features zum Anzeigen
    figsize : tuple
        Größe der Figur
    save_path : str, optional
        Speicherpfad

    Returns
    -------
    plt.Figure
        Matplotlib Figure-Objekt
    """
    # Korrelationsmatrix
    corr_matrix = X.iloc[:, :top_k].corr(method=method)

    # Heatmap
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': f'{method.capitalize()} Korrelation'},
        ax=ax
    )

    ax.set_title(
        f'Feature-Korrelation ({method.capitalize()}) - Top {top_k} Features',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot gespeichert: {save_path}")

    return fig


def plot_classifier_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'balanced_accuracy',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Vergleicht Klassifikator-Performance mit Fehlerbalken.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame mit Spalten: 'Classifier', 'Mean', 'CI_Lower', 'CI_Upper'
    metric : str
        Metrik-Name für Titel
    figsize : tuple
        Größe der Figur
    save_path : str, optional
        Speicherpfad

    Returns
    -------
    plt.Figure
        Matplotlib Figure-Objekt
    """
    fig, ax = plt.subplots(figsize=figsize)

    classifiers = comparison_df['Classifier'].values
    means = comparison_df['Mean'].values
    ci_lowers = comparison_df['CI_Lower'].values
    ci_uppers = comparison_df['CI_Upper'].values

    # Fehlerbalken berechnen
    errors = np.array([means - ci_lowers, ci_uppers - means])

    # Bar Plot mit Fehlerbalken
    x_pos = np.arange(len(classifiers))
    bars = ax.bar(
        x_pos,
        means,
        yerr=errors,
        capsize=5,
        alpha=0.8,
        color=sns.color_palette("Set2", len(classifiers)),
        edgecolor='black',
        linewidth=1.5
    )

    # Werte auf Balken anzeigen
    for i, (bar, mean_val) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + errors[1][i] + 0.01,
            f'{mean_val:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xlabel('Klassifikator', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(
        f'Klassifikator-Vergleich: {metric.replace("_", " ").title()}\n'
        f'(95% Konfidenzintervalle, 5-Fold GroupKFold CV)',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classifiers, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, min(1.0, max(ci_uppers) * 1.15))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot gespeichert: {save_path}")

    return fig


def plot_feature_reduction_timeline(
    reduction_history: List[Dict],
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualisiert den Feature-Reduktionsprozess über die Phasen.

    Parameters
    ----------
    reduction_history : list
        Liste von Dicts mit 'phase', 'n_features', 'description'
    figsize : tuple
        Größe der Figur
    save_path : str, optional
        Speicherpfad

    Returns
    -------
    plt.Figure
        Matplotlib Figure-Objekt
    """
    fig, ax = plt.subplots(figsize=figsize)

    phases = [h['phase'] for h in reduction_history]
    n_features = [h['n_features'] for h in reduction_history]
    descriptions = [h['description'] for h in reduction_history]

    # Linie mit Markern
    ax.plot(phases, n_features, marker='o', linewidth=3, markersize=12, color='#2E86AB')

    # Annotationen
    for i, (phase, n_feat, desc) in enumerate(zip(phases, n_features, descriptions)):
        ax.annotate(
            f'{n_feat} Features\n({desc})',
            xy=(phase, n_feat),
            xytext=(0, 20 if i % 2 == 0 else -40),
            textcoords='offset points',
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black')
        )

    ax.set_xlabel('Phase', fontsize=12, fontweight='bold')
    ax.set_ylabel('Anzahl Features', fontsize=12, fontweight='bold')
    ax.set_title(
        'Feature-Reduktions-Pipeline: Übersicht',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xticks(phases)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot gespeichert: {save_path}")

    return fig
