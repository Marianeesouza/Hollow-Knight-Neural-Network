"""Statistical analysis module for model performance comparison.

Implements plan.md §6 - statistical analysis of model performance
with cross-validation, significance testing, and effect size computation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_cross_model_significance(
    session_summaries: pd.DataFrame,
    metrics: List[str] = None,
    alpha: float = 0.05
) -> Dict:
    """Compute pairwise statistical significance between models.

    Args:
        session_summaries: DataFrame with per-session metrics
        metrics: List of metric columns to test
        alpha: Significance threshold

    Returns:
        Dict with pairwise test results
    """
    if metrics is None:
        metrics = ['n_frames', 'CSR', 'COR', 'M_cura', 'M_alinhamento',
                   'M_evasao', 'M_defesa', 'M_punicao', 'P_canto']

    if not HAS_SCIPY:
        return {'error': 'scipy not available'}

    results = {}
    models = session_summaries['model'].unique() if 'model' in session_summaries.columns else \
             session_summaries['Model'].unique()

    model_col = 'model' if 'model' in session_summaries.columns else 'Model'

    model_pairs = [(models[i], models[j])
                   for i in range(len(models)) for j in range(i + 1, len(models))]

    for model_a, model_b in model_pairs:
        data_a = session_summaries[session_summaries[model_col] == model_a]
        data_b = session_summaries[session_summaries[model_col] == model_b]

        results[f'{model_a}_vs_{model_b}'] = {}

        for metric in metrics:
            if metric not in data_a.columns or metric not in data_b.columns:
                continue

            vals_a = data_a[metric].dropna()
            vals_b = data_b[metric].dropna()

            if len(vals_a) < 2 or len(vals_b) < 2:
                results[f'{model_a}_vs_{model_b}'][metric] = {
                    'error': 'Insufficient data'
                }
                continue

            # Welch's t-test (unequal variance)
            t_stat, p_val = scipy_stats.ttest_ind(vals_a, vals_b, equal_var=False)

            # Cohen's d effect size
            pooled_std = np.sqrt(
                ((len(vals_a) - 1) * vals_a.std()**2 + (len(vals_b) - 1) * vals_b.std()**2) /
                (len(vals_a) + len(vals_b) - 2)
            )
            cohens_d = (vals_a.mean() - vals_b.mean()) / pooled_std if pooled_std > 0 else 0

            results[f'{model_a}_vs_{model_b}'][metric] = {
                'mean_a': float(vals_a.mean()),
                'mean_b': float(vals_b.mean()),
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'cohens_d': float(cohens_d),
                'significant': bool(p_val < alpha),
                'effect_size': _interpret_cohens_d(cohens_d)
            }

    return results


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return 'negligible'
    elif d < 0.5:
        return 'small'
    elif d < 0.8:
        return 'medium'
    else:
        return 'large'


def compute_confidence_intervals(
    session_summaries: pd.DataFrame,
    metrics: List[str] = None,
    confidence: float = 0.95
) -> Dict:
    """Compute confidence intervals for all model-metric combinations.

    Args:
        session_summaries: DataFrame with per-session metrics
        metrics: List of metric columns
        confidence: Confidence level (0-1)

    Returns:
        Dict with CI for each model-metric pair
    """
    if metrics is None:
        metrics = ['n_frames', 'CSR', 'COR', 'M_cura', 'M_alinhamento',
                   'M_evasao', 'M_defesa', 'M_punicao', 'P_canto']

    results = {}
    models = session_summaries['model'].unique() if 'model' in session_summaries.columns else \
             session_summaries['Model'].unique()

    model_col = 'model' if 'model' in session_summaries.columns else 'Model'
    alpha = 1 - confidence
    z_critical = 1.96  # For 95% CI

    if HAS_SCIPY:
        z_critical = scipy_stats.norm.ppf(1 - alpha / 2)

    for model in models:
        model_data = session_summaries[session_summaries[model_col] == model]
        results[model] = {}

        for metric in metrics:
            if metric not in model_data.columns:
                continue

            vals = model_data[metric].dropna()
            if len(vals) == 0:
                results[model][metric] = {'error': 'No data'}
                continue

            n = len(vals)
            mean = float(vals.mean())
            std = float(vals.std()) if n > 1 else 0.0
            se = std / np.sqrt(n) if n > 1 else 0.0
            margin = z_critical * se

            results[model][metric] = {
                'mean': mean,
                'std': std,
                'count': n,
                'ci_lower': mean - margin,
                'ci_upper': mean + margin,
                'confidence': confidence
            }

    return results


def compute_correlation_matrix(session_summaries: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute correlation matrix between contextual metrics.

    Args:
        session_summaries: DataFrame with per-session metrics

    Returns:
        Correlation matrix DataFrame
    """
    metric_cols = [c for c in ['CSR', 'COR', 'M_cura', 'M_alinhamento', 'M_evasao',
                               'M_defesa', 'M_punicao', 'P_canto', 'n_frames',
                               'n_threat', 'n_abertura', 'n_cura']
                   if c in session_summaries.columns]

    if len(metric_cols) < 2:
        return None

    return session_summaries[metric_cols].corr()


def bootstrap_funnel_scores(
    session_summaries: pd.DataFrame,
    n_bootstrap: int = 1000
) -> Dict:
    """Bootstrap confidence intervals for funnel scores.

    Args:
        session_summaries: DataFrame with per-session metrics
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dict with bootstrap CIs for each model fennel level
    """
    results = {}
    models = session_summaries['model'].unique() if 'model' in session_summaries.columns else \
             session_summaries['Model'].unique()

    model_col = 'model' if 'model' in session_summaries.columns else 'Model'

    funnel_metrics = {
        'navigation': 'P_canto',
        'defense': 'M_defesa',
        'offense': 'M_punicao',
        'resource_management': 'M_cura',
    }

    for model in models:
        model_data = session_summaries[session_summaries[model_col] == model]
        n = len(model_data)

        if n == 0:
            continue

        results[model] = {}

        for level, metric_col in funnel_metrics.items():
            if metric_col not in model_data.columns:
                continue

            vals = model_data[metric_col].dropna().values
            if len(vals) == 0:
                continue

            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(vals, size=len(vals), replace=True)
                bootstrap_means.append(np.mean(sample))

            lower = np.percentile(bootstrap_means, 2.5)
            upper = np.percentile(bootstrap_means, 97.5)
            mean_val = np.mean(bootstrap_means)

            # For navigation, invert (1 - P_canto). Os extremos trocam de lado:
            # usar variáveis temporárias (antes, `upper` era calculado a partir
            # do `lower` já sobrescrito e o IC ficava errado).
            if level == 'navigation':
                lower, upper = 1 - upper, 1 - lower
                mean_val = 1 - mean_val

            results[model][level] = {
                'mean': mean_val,
                'ci_lower': float(lower),
                'ci_upper': float(upper),
                'values': vals
            }

        # Victory rate
        if 'outcome' in model_data.columns:
            wins = (model_data['outcome'] == 'WIN').sum()
            n_total = len(model_data)
            victory_rate = wins / n_total

            # Bootstrap CI for proportion
            bootstrap_props = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(
                    [1 if o == 'WIN' else 0 for o in model_data['outcome']],
                    size=n_total,
                    replace=True
                )
                bootstrap_props.append(np.mean(sample))

            results[model]['victory'] = {
                'mean': victory_rate,
                'ci_lower': float(np.percentile(bootstrap_props, 2.5)),
                'ci_upper': float(np.percentile(bootstrap_props, 97.5)),
                'wins': int(wins),
                'total': n_total
            }

    return results


def compute_learning_curve(
    session_summaries: pd.DataFrame,
    metric: str = 'M_cura',
    window: int = 5
) -> Dict:
    """Compute rolling-window learning curve for each model.

    Args:
        session_summaries: DataFrame with per-session metrics (ordered by time)
        metric: Metric to compute learning curve for
        window: Rolling window size

    Returns:
        Dict with rolling means per model
    """
    results = {}
    models = session_summaries['model'].unique() if 'model' in session_summaries.columns else \
             session_summaries['Model'].unique()

    model_col = 'model' if 'model' in session_summaries.columns else 'Model'

    if metric not in session_summaries.columns:
        return {'error': f'Metric {metric} not found'}

    for model in models:
        model_data = session_summaries[session_summaries[model_col] == model].copy()
        model_data = model_data.sort_values('session_id' if 'session_id' in model_data.columns
                                            else model_data.columns[0])

        vals = model_data[metric].dropna().values

        if len(vals) < window:
            results[model] = {
                'rolling_mean': vals.tolist(),
                'rolling_std': [0.0] * len(vals),
            }
            continue

        rolling_means = np.convolve(vals, np.ones(window) / window, mode='valid')

        # Compute rolling std
        rolling_stds = []
        for i in range(len(vals) - window + 1):
            rolling_stds.append(np.std(vals[i:i + window]))

        results[model] = {
            'rolling_mean': rolling_means.tolist(),
            'rolling_std': rolling_stds,
            'original_values': vals.tolist(),
        }

    return results


def compute_eigenvalue_regression(
    session_summaries: pd.DataFrame,
    target: str = 'outcome'
) -> Dict:
    """Analyze which metrics contribute most to victory via eigenvalue decomposition.

    Args:
        session_summaries: DataFrame with per-session metrics
        target: Target column ('outcome' for win/loss)

    Returns:
        Dict with eigenvalue analysis results
    """
    if not HAS_SKLEARN:
        return {'error': 'sklearn not available'}

    metric_cols = ['CSR', 'COR', 'M_cura', 'M_alinhamento', 'M_evasao',
                   'M_defesa', 'M_punicao', 'P_canto', 'n_frames',
                   'n_threat', 'n_abertura', 'n_cura']

    available_cols = [c for c in metric_cols if c in session_summaries.columns]

    if len(available_cols) < 2:
        return {'error': 'Insufficient metric columns'}

    X = session_summaries[available_cols].fillna(0).values

    # Create binary target
    if target == 'outcome' and 'outcome' in session_summaries.columns:
        y = (session_summaries['outcome'] == 'WIN').astype(int).values
    else:
        y = session_summaries[target].fillna(0).values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=min(len(available_cols), 3))
    pca.fit(X_scaled)

    # Compute correlation of each metric with target
    correlations = {}
    for i, col in enumerate(available_cols):
        if np.std(X[:, i]) > 0:
            corr, _ = scipy_stats.pearsonr(X[:, i], y) if HAS_SCIPY else (0, 0)
            correlations[col] = float(corr)

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    return {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'components': pca.components_.tolist(),
        'feature_names': available_cols,
        'correlations_with_target': dict(sorted_corrs),
        'top_predictive_metrics': sorted_corrs[:5],
    }