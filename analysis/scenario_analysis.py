"""Scenario-based evaluation for critical game situations.

Implements Scenario Slicing from plan.md \u00a74.1 and \u00a72.3.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Add project root to sys.path for imports
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Critical distance thresholds (same units as df['dist'])
D_CRITICO = 5.0    # Distance threshold for high-speed attack threat
D_ALCANCE = 1.8    # Nail/magic reach
ALPHA_THREAT = 3.0
ALPHA_OPENING = 2.0
ALPHA_HEALTH = 1.5


def extract_critical_scenarios(
    df: pd.DataFrame,
    offensive_states: Optional[List[int]] = None,
    vulnerable_states: Optional[List[int]] = None,
) -> Dict[str, pd.DataFrame]:
    """Extract the three critical scenarios from plan.md \u00a74.1.

    Scenario A: High-speed boss attacks at close range
    Scenario B: Boss vulnerable states with player in range
    Scenario C: Low-health player scenarios
    """
    from Python.Src.Data.TelemetryDataset import add_derived_columns

    # Handle empty DataFrame early
    if df.empty or 'bossState' not in df.columns:
        return {
            'scenario_a_high_speed_attack': pd.DataFrame(),
            'scenario_b_vulnerable_window': pd.DataFrame(),
            'scenario_c_low_health': pd.DataFrame(),
        }

    if 'dist' not in df.columns:
        add_derived_columns(df)
    if 'hp_norm' not in df.columns:
        add_derived_columns(df)

    offensive_states = offensive_states or []
    vulnerable_states = vulnerable_states or []

    boss_state = df['bossState'].to_numpy()
    dist = df['dist'].to_numpy()
    hp_norm = df['hp_norm'].to_numpy()
    delta_x = df['delta_x'].to_numpy()
    
    velocity_boss_x = df.get('bvx', pd.Series([0.0] * len(df))).to_numpy()
    velocity_boss_y = df.get('bvy', pd.Series([0.0] * len(df))).to_numpy()

    dist_prev = df['dist'].shift(1).bfill()
    dist_closing = (dist_prev - dist) > 0.0

    # Scenario A: High-speed attack at close range
    scenario_a_mask = (
        np.isin(boss_state, offensive_states) &
        (dist <= D_CRITICO) &
        dist_closing
    )
    scenario_a = df[scenario_a_mask].copy()

    # Scenario B: Boss vulnerable + player in range
    scenario_b_mask = (
        np.isin(boss_state, vulnerable_states) &
        (dist <= D_ALCANCE)
    )
    scenario_b = df[scenario_b_mask].copy()

    # Scenario C: Low health player
    scenario_c_mask = hp_norm < 0.15
    scenario_c = df[scenario_c_mask].copy()

    return {
        'scenario_a_high_speed_attack': scenario_a,
        'scenario_b_vulnerable_window': scenario_b,
        'scenario_c_low_health': scenario_c,
    }


def compute_scenario_metrics(
    df: pd.DataFrame,
    scenario_name: str,
    actions: Optional[np.ndarray] = None,
    params=None,
) -> Dict:
    """Calculate contextual metrics for a specific scenario."""
    from Python.Src.Utils.ContextMetrics import MetricParams, compute_contextual_metrics

    params = params or MetricParams()

    if len(df) == 0:
        return {
            'n_frames': 0,
            'scenario': scenario_name,
            'CSR': np.nan,
            'COR': np.nan,
            'M_cura': np.nan,
            'M_alinhamento': np.nan,
            'M_evasao': np.nan,
            'M_punicao': np.nan,
            'P_canto': np.nan,
            'M_defesa': np.nan,
        }

    result = compute_contextual_metrics(df, actions=actions, params=params, source='model')

    return {
        'scenario': scenario_name,
        'n_frames': int(len(df)),
        'CSR': result['CSR'],
        'COR': result['COR'],
        'M_cura': result['M_cura'],
        'M_alinhamento': result['M_alinhamento'],
        'M_evasao': result['M_evasao'],
        'M_punicao': result['M_punicao'],
        'P_canto': result['P_canto'],
        'M_defesa': result['M_defesa'],
    }


def compute_capability_funnel(
    session_summaries: pd.DataFrame,
    all_sessions_metrics: Optional[List[Dict]] = None,
) -> Dict:
    """Compute capability progression across the 5 levels of the funnel.

    From plan.md §2.4:
    Navigation → Reactive Defense → Basic Punishment → Resource Management → Victory

    Aceita tanto as colunas `Model`/`Result` quanto `model`/`outcome`
    (convenções que coexistiam entre este módulo, o app e statistical_analysis).
    """
    session_summaries = session_summaries.rename(
        columns={'model': 'Model', 'outcome': 'Result'}
    )

    model_names = session_summaries['Model'].unique() if 'Model' in session_summaries.columns else ['Unknown']
    funnel_results = {}

    for model in model_names:
        model_sessions = session_summaries[session_summaries['Model'] == model] if 'Model' in session_summaries.columns else session_summaries
        total_sessions = len(model_sessions)
        wins = int((model_sessions['Result'] == 'WIN').sum()) if 'Result' in model_sessions.columns else 0

        avg_p_canto = model_sessions['P_canto'].mean() if 'P_canto' in model_sessions.columns else np.nan
        avg_m_defesa = model_sessions['M_defesa'].mean() if 'M_defesa' in model_sessions.columns else np.nan
        avg_m_punicao = model_sessions['M_punicao'].mean() if 'M_punicao' in model_sessions.columns else np.nan
        avg_m_cura = model_sessions['M_cura'].mean() if 'M_cura' in model_sessions.columns else np.nan

        funnel_results[model] = {
            'navigation': 1.0 - float(avg_p_canto) if not np.isnan(avg_p_canto) else np.nan,
            'defense': float(avg_m_defesa) if not np.isnan(avg_m_defesa) else np.nan,
            'offense': float(avg_m_punicao) if not np.isnan(avg_m_punicao) else np.nan,
            'resource_management': float(avg_m_cura) if not np.isnan(avg_m_cura) else np.nan,
            'victory': wins / total_sessions if total_sessions > 0 else 0.0,
            'total_sessions': total_sessions,
            'wins': wins,
        }

    return funnel_results


def compute_weighted_loss_components(
    df: pd.DataFrame,
    model_probs: np.ndarray,
    expert_actions: np.ndarray,
    actions: Optional[np.ndarray] = None,
    pos_weights: Optional[np.ndarray] = None,
    alpha_threat: float = ALPHA_THREAT,
    alpha_opening: float = ALPHA_OPENING,
    alpha_health: float = ALPHA_HEALTH,
) -> Dict:
    """Compute the contextual loss components from plan.md §300-373.

    Returns the individual terms and their contributions to the total loss.
    """
    from Python.Src.Data.TelemetryDataset import add_derived_columns

    if 'dist' not in df.columns:
        add_derived_columns(df)
    if 'hp_norm' not in df.columns:
        add_derived_columns(df)

    if actions is None:
        actions = expert_actions

    if pos_weights is None:
        n_total = len(df)
        pos_weights = np.array([
            (n_total - np.sum(expert_actions[:, j])) / max(np.sum(expert_actions[:, j]), 1)
            for j in range(expert_actions.shape[1])
        ])

    probs = model_probs  # sigmoid outputs
    labels = expert_actions
    logits = np.log(probs / (1 - probs + 1e-8) + 1e-8)

    cls_loss = np.zeros_like(probs)
    for j in range(probs.shape[1]):
        y = labels[:, j]
        p = probs[:, j]
        pw = pos_weights[j]
        cls_loss[:, j] = -(pw * y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8))

    threat_mask = _compute_threat_mask(df)
    opening_mask = _compute_opening_mask(df)
    hp_norm = df['hp_norm'].to_numpy()

    omega = np.ones(len(df))
    omega[threat_mask] += alpha_threat
    omega[opening_mask] += alpha_opening
    omega += alpha_health * (1.0 - hp_norm)

    omega_weighted_loss = omega[:, np.newaxis] * cls_loss
    total_cls_loss = omega_weighted_loss.mean()

    mira_loss = _compute_mira_regularization(logits, df, labels)

    total_loss = total_cls_loss + mira_loss.mean()

    return {
        'base_cls_loss': float(cls_loss.mean()),
        'weighted_cls_loss': float(total_cls_loss),
        'mira_regularization': float(mira_loss.mean()),
        'total_loss': float(total_loss),
        'pos_weights': pos_weights.tolist(),
        'omega_mean': float(omega.mean()),
        'threat_frames': int(threat_mask.sum()),
        'opening_frames': int(opening_mask.sum()),
        'avg_pos_weight': float(np.mean(pos_weights)),
    }


def _compute_threat_mask(df: pd.DataFrame) -> np.ndarray:
    """Compute boolean mask for threat frames (from plan.md §3.4)."""
    from Python.Src.Utils.ContextMetrics import MetricParams, load_boss_state_config

    params = MetricParams()
    config = load_boss_state_config()

    boss_state = df['bossState'].to_numpy()
    dist = df['dist'].to_numpy()
    dist_prev = df['dist'].shift(1).bfill()
    dist_closing = (dist_prev - dist) > 0.0

    attack_active = np.isin(boss_state, config['attack_active'])
    threat = (
        attack_active &
        (dist <= params.d_ameaca) &
        dist_closing
    )

    return threat


def _compute_opening_mask(df: pd.DataFrame) -> np.ndarray:
    """Compute boolean mask for opening frames (from plan.md §3.5)."""
    from Python.Src.Utils.ContextMetrics import MetricParams, load_boss_state_config

    params = MetricParams()
    config = load_boss_state_config()

    boss_state = df['bossState'].to_numpy()
    dist = df['dist'].to_numpy()

    vulnerable = np.isin(boss_state, config['vulnerable'])
    trick_state = vulnerable | np.isin(boss_state, config.get('stunned', []))
    opening = trick_state & (dist <= params.d_alcance)

    return opening


def _compute_mira_regularization(
    logits: np.ndarray,
    df: pd.DataFrame,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute aim alignment regularization term R_mira (from plan.md §345)."""
    delta_x = df['delta_x'].to_numpy()
    theta_star = np.sign(delta_x)
    # Mesmo critério de loss_function.py: Δx = 0 conta como "chefe à direita".
    theta_star = np.where(theta_star == 0, 1.0, theta_star)

    z_right = logits[:, 5]  # action 6: right
    z_left = logits[:, 4]   # action 5: left
    delta_z_horizontal = z_right - z_left

    offensive_actions = np.logical_or(labels[:, 3] > 0, labels[:, 2] > 0).astype(float)  # attack or cast

    tanh_term = np.tanh(delta_z_horizontal)
    alignment_term = 1.0 - theta_star * tanh_term
    r_mira = np.maximum(0.0, offensive_actions * alignment_term)

    return r_mira