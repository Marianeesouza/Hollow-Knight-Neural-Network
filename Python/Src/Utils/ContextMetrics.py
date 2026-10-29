"""Motor de métricas contextuais do plan.md (puro pandas/numpy, sem torch).

Implementa:
  - Mascara de revergo:  T_ameaca  (ataque ativo + distância fechando)
  - Mascara de abertura: T_abertura (chefe vulnerável + em alcance)
  - Metrica 1  M_cura:          Taxa de Segurança (CSR) x Revocação de Oportunidade (COR)
  - Metrica 2  M_alinhamento:   Coerência vetorial de mira
  - Metrica 3  M_evasao + FER:  Sensibilidade a ameaças críticas e evasão fútil
  - Metrica 4  M_punicao:       Punição efetiva em aberturas
  - Metrica 5  P_canto:         Penalidade de confinamento em extremidades
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dataclasses import dataclass

from ..Data.TelemetryDataset import (
    add_derived_columns,
    extract_expert_actions,
)

BOSS_STATE_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Data', 'boss_state_config.json',
)


# ------------------------------------------------------------------ Parâmetros


@dataclass
class MetricParams:
    """Distâncias em unidades do jogo (arena ~22,5 x ~11,6, ver DataHandler)."""

    d_seguro: float = 4.0     # distância mínima para canalizar cura
    d_ameaca: float = 5.0     # distância em que um ataque ativo vira ameaça crítica
    d_alcance: float = 1.8    # alcance do ferrão / raio efetivo da magia
    x_limiar: float = 0.85    # bordas laterais normalizadas ([-1, 1])
    h_crit: float = 0.25      # fração de vida baixa (<= 2 máscaras de 8)
    s_foco: float = 1.0 / 3.0  # alma necessária p/ conjurar (33 de 99)
    lambda_fer: float = 0.5   # peso da evasão fútil em M_defesa
    tau: float = 0.2          # limiar de decisão do bot
    aim_mode: str = 'facing'  # 'facing': facingRight do jogo (mecânica real);
                              # 'plan': stick direcional (formulação literal plan.md)

    # Loss engineering parameters (plan.md §300-373)
    pos_weight_scale: float = 1.0   # escala global do pos_weight
    alpha_threat: float = 3.0       # multiplicador Ω para frames de ameaça
    alpha_opening: float = 2.0      # multiplicador Ω para janelas de abertura
    alpha_health: float = 1.5       # multiplicador Ω para vida baixa
    lambda_alignment: float = 0.5   # peso da regularização de mira (R_mira)


def load_boss_state_config(path=BOSS_STATE_CONFIG_PATH):
    """Lê o mapping editável de bossState ids. Retorna listas vazias se ausente.

    Aceita dois formatos:
      - formato legado: {"attack_active": [...], "vulnerable": [...], "stunned": [...]}
      - formato registro (estado a estado; ver build_boss_state_registry.py):
        {"states": [{"state_id", "state_name", "is_offensive", "is_vulnerable",
                     "reviewed", "notes", "stats"}]}
    O novo formato é normalizado para as mesmas listas (is_vulnerable alimenta
    'vulnerable'; 'stunned' não existe mais no registro).
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raw = {}

    if isinstance(raw.get('states'), list):
        attack, vulner = [], []
        for st in raw['states']:
            sid = int(st['state_id'])
            if st.get('is_offensive'):
                attack.append(sid)
            if st.get('is_vulnerable'):
                vulner.append(sid)
        return {'attack_active': sorted(set(attack)),
                'vulnerable': sorted(set(vulner)),
                'stunned': [],
                'registry': raw}

    return {'attack_active': sorted(set(raw.get('attack_active', []))),
            'vulnerable': sorted(set(raw.get('vulnerable', []))),
            'stunned': sorted(set(raw.get('stunned', []))),
            'registry': None}


def load_boss_state_registry(path=BOSS_STATE_CONFIG_PATH):
    """Retorna o bloco 'states' do registro (lista), ou [] se indisponível."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return raw.get('states', [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


# ----------------------------------------------------------------- Mascaração


def _safe_map(df, key):
    """Série booleana: bossState ∈ subconjunto do mapping."""
    return df['bossState'].isin(key)


def build_masks(df, params=None, config=None):
    """Máscaras contextuais booleanas sobre a sessão (índice alinhado ao df)."""
    params = params or MetricParams()
    config = config or load_boss_state_config()

    if 'dist' not in df.columns:
        add_derived_columns(df)

    attack_active = _safe_map(df, config['attack_active'])
    vulnerable = _safe_map(df, config['vulnerable'])
    stunned = _safe_map(df, config['stunned'])
    trick_state = vulnerable | stunned  # B_atordoado do plan.md (vulnerável ou recuperação)

    threat = (
        attack_active
        & (df['dist'] <= params.d_ameaca)
        & df['dist_closing'].astype(bool)
    )

    abertura = trick_state & (df['dist'] <= params.d_alcance)

    seguro = (df['dist'] >= params.d_seguro) | trick_state

    oportuno = (
        (df['hp_norm'] <= params.h_crit)
        & (df['soul_norm'] >= params.s_foco)
        & seguro
    )

    borda_esq = df['x_arena'] <= -params.x_limiar
    borda_dir = df['x_arena'] >= params.x_limiar
    borda = borda_esq | borda_dir

    return {
        'attack_active': attack_active,
        'vulnerable': vulnerable,
        'stunned': stunned,
        'threat': threat,
        'abertura': abertura,
        'seguro': seguro,
        'oportuno': oportuno,
        'borda_esq': borda_esq,
        'borda_dir': borda_dir,
        'borda': borda,
        'neutro': ~threat,
    }


def _rate(condition):
    """Fração de quadros True; NaN quando a janela é vazia."""
    n = int(condition.sum())
    if n == 0:
        return float('nan'), 0
    return float(condition.mean()), n


def _subset_rate(window, condition):
    """Taxa dentro de uma janela: sum(window & condition) / sum(window)."""
    n = int(window.sum())
    if n == 0:
        return float('nan'), 0
    hits = int((window & condition).sum())
    return hits / n, n


def _harmonic(a, b):
    if np.isnan(a) or np.isnan(b) or (a + b) == 0:
        return float('nan')
    return 2.0 * a * b / (a + b)


# ------------------------------------------------------------------- Métricas


def compute_contextual_metrics(df, actions=None, params=None, config=None,
                               source='expert'):
    """Métricas contextuais do plan.md sobre uma sessão.

    actions: matriz binária (T, 8) na ordem ACTION_NAMES. Se None, extrai do
    schema (expert -> gp_*; model -> pred_* quando existentes).
    source: 'expert' ou 'model' (agente do bot).
    """
    params = params or MetricParams()
    config = config or load_boss_state_config()

    if 'dist' not in df.columns:
        add_derived_columns(df)

    if actions is None:
        if source == 'model':
            pred = [f'pred_{n}' for n in
                    ['dash', 'jump', 'cast', 'attack', 'left', 'right', 'up', 'down']]
            actions = df[pred].to_numpy(dtype=np.float32) if all(p in df.columns for p in pred) else None
        if actions is None:
            actions = extract_expert_actions(df)

    dash, jump, cast, attack, left, right, up, down = [actions[:, i] >= 0.5 for i in range(8)]

    evasive = (dash | jump).astype(bool)
    offensive = (attack | cast).astype(bool)

    theta_hat = right.astype(float) - left.astype(float)  # -1..+1 (modo 'plan')
    dx = df['delta_x'].to_numpy()
    sgn_dx = np.sign(dx)
    aligned_stick = pd.Series((theta_hat * sgn_dx) > 0, index=df.index)

    facing_sign = pd.Series(np.where(df['facingRight'] >= 0.5, 1.0, -1.0),
                            index=df.index)
    aligned_facing = pd.Series((facing_sign * sgn_dx) > 0, index=df.index)

    if params.aim_mode == 'plan':
        aligned = aligned_stick
    else:
        aligned = aligned_facing

    masks = build_masks(df, params, config)

    mask_s = pd.Series(masks['threat'], index=df.index)
    neut_s = pd.Series(masks['neutro'], index=df.index)
    mask_s_ab = pd.Series(masks['abertura'], index=df.index)

    off_s = pd.Series(offensive, index=df.index)

    t_cura = pd.Series((cast & (df['soul_norm'] >= params.s_foco)), index=df.index)
    csr_rate, n_cura = _subset_rate(t_cura, masks['seguro'])
    cor, n_oport = _subset_rate(pd.Series(masks['oportuno'], index=df.index), t_cura)
    m_cura = _harmonic(csr_rate, cor)

    # ---- Metrica 2: Alinhamento de mira
    m_alinh, n_ofens = _subset_rate(off_s, aligned)
    m_alinh_facing, _ = _subset_rate(off_s, aligned_facing)
    m_alinh_stick, _ = _subset_rate(off_s, aligned_stick)

    # ---- Metrica 3: Evasão
    ev_s = pd.Series(evasive, index=df.index)
    m_evas, n_amen = _subset_rate(mask_s, ev_s)
    fer, n_neut = _subset_rate(neut_s, ev_s)
    m_defesa = (m_evas - params.lambda_fer * fer) if not np.isnan(m_evas) else float('nan')

    # ---- Metrica 4: Punição (atacar na janela, direcionado ao chefe)
    m_punicao, n_abert = _subset_rate(mask_s_ab, off_s & aligned)

    # ---- Metrica 5: Confinamento em bordas (continuar empurrando
    #      direcional na direção da parede)
    push_wall = (pd.Series(masks['borda_esq'], index=df.index) & pd.Series(left, index=df.index))
    push_wall |= (pd.Series(masks['borda_dir'], index=df.index) & pd.Series(right, index=df.index))
    p_canto, n_borda = _subset_rate(pd.Series(masks['borda'], index=df.index), push_wall)

    return {
        'n_frames': int(len(df)),
        'n_threat': int(mask_s.sum()),
        'n_neutro': int(neut_s.sum()),
        'n_abertura': int(mask_s_ab.sum()),
        'n_oportuno': n_oport,
        'n_borda': int(masks['borda'].sum()),
        'n_cura': n_cura,
        'n_ofensivo': int(off_s.sum()),
        'CSR': csr_rate,
        'COR': cor,
        'M_cura': m_cura,
        'M_alinhamento': m_alinh,
        'M_alinhamento_facing': m_alinh_facing,
        'M_alinhamento_stick': m_alinh_stick,
        'M_evasao': m_evas,
        'FER': fer,
        'M_defesa': m_defesa,
        'M_punicao': m_punicao,
        'P_canto': p_canto,
        'masks': masks,
        'params': params,
        'config': config,
    }


def session_model_stats(df, tau=0.2):
    """Hesitação / predições contraditórias do bot (requer colunas pred_*)."""
    pred_cols = [f'pred_{n}' for n in
                 ['dash', 'jump', 'cast', 'attack', 'left', 'right', 'up', 'down']]
    prob_cols = [f'prob_{i}' for i in range(1, 9)]

    stats = {'n_frames': int(len(df))}
    if not all(c in df.columns for c in pred_cols):
        return {**stats, 'has_model': False}

    preds = df[pred_cols].to_numpy(dtype=np.float32)
    stats['has_model'] = True
    stats['contradictory_left_right'] = int(((preds[:, 4] > 0.5) & (preds[:, 5] > 0.5)).sum())
    stats['contradictory_up_down'] = int(((preds[:, 6] > 0.5) & (preds[:, 7] > 0.5)).sum())

    if all(c in df.columns for c in prob_cols):
        probs = df[prob_cols].to_numpy(dtype=np.float32)
        near = np.abs(probs - tau) <= 0.05
        stats['near_threshold_frac'] = float((near.any(axis=1)).mean())
        stats['near_threshold_per_action'] = near.mean(axis=0).astype(float).tolist()
    else:
        stats['near_threshold_frac'] = float('nan')
        stats['near_threshold_per_action'] = None

    return stats


def summarise_sessions(paths, params=None, config=None, source='expert'):
    """Agrega métricas de múltiplas sessões em um DataFrame comparativo."""
    from ..Data.TelemetryDataset import load_session_csv, session_summary_label

    rows = []
    for path in paths:
        df = load_session_csv(path)
        result = compute_contextual_metrics(df, params=params, config=config, source=source)
        flat = {k: v for k, v in result.items() if not isinstance(v, (pd.Series, dict, MetricParams))}
        flat['session'] = session_summary_label(path)
        final_hp = df['hp'].iloc[-1]
        flat['outcome'] = 'WIN' if final_hp > 0 else 'LOSS'
        rows.append(flat)

    cols_order = [
        'session', 'outcome', 'n_frames', 'n_threat', 'n_neutro',
        'n_abertura', 'n_oportuno', 'n_borda', 'n_cura', 'n_ofensivo',
        'CSR', 'COR', 'M_cura', 'M_alinhamento', 'M_alinhamento_facing',
        'M_alinhamento_stick', 'M_evasao', 'FER',
        'M_defesa', 'M_punicao', 'P_canto',
    ]
    out = pd.DataFrame(rows)
    out = out[[c for c in cols_order if c in out.columns]]
    return out


def aggregate_cross_model_analysis(session_summaries: pd.DataFrame) -> Dict:
    """Aggregate metrics across all sessions and model architectures.

    Returns per-model performance with statistical confidence intervals.
    """
    if session_summaries.empty:
        return {}

    results = {}
    models = session_summaries['Model'].unique()

    for model in models:
        model_data = session_summaries[session_summaries['Model'] == model]
        total_sessions = len(model_data)
        wins = int((model_data['outcome'] == 'WIN').sum())
        losses = total_sessions - wins

        # Compute aggregate metrics with confidence intervals
        numeric_cols = ['M_cura', 'M_alinhamento', 'M_evasao', 'M_defesa', 'M_punicao', 'P_canto']
        agg_metrics = {}

        for col in numeric_cols:
            if col in model_data.columns:
                values = model_data[col].dropna()
                if len(values) > 0:
                    mean_val = float(values.mean())
                    std_val = float(values.std()) if len(values) > 1 else 0.0
                    ci_95 = 1.96 * std_val / np.sqrt(len(values)) if len(values) > 1 else 0.0

                    agg_metrics[f'{col}_mean'] = mean_val
                    agg_metrics[f'{col}_std'] = std_val
                    agg_metrics[f'{col}_ci95'] = ci_95
                else:
                    agg_metrics[f'{col}_mean'] = np.nan
                    agg_metrics[f'{col}_std'] = np.nan
                    agg_metrics[f'{col}_ci95'] = np.nan

        results[model] = {
            'total_sessions': total_sessions,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / total_sessions if total_sessions > 0 else 0.0,
            'survival_time_mean': float(model_data['n_frames'].mean()) if 'n_frames' in model_data.columns else np.nan,
            **agg_metrics,
        }

    return results


def compute_statistical_significance(
    session_summaries: pd.DataFrame,
    metric: str = 'survival_time'
) -> Dict:
    """Compute statistical significance between model architectures."""
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return {}

    results = {}
    models = session_summaries['Model'].unique()
    model_pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i+1, len(models))]

    for model_a, model_b in model_pairs:
        data_a = session_summaries[session_summaries['Model'] == model_a]
        data_b = session_summaries[session_summaries['Model'] == model_b]

        if 'n_frames' in data_a.columns and len(data_a) > 1 and len(data_b) > 1:
            t_stat, p_val = scipy_stats.ttest_ind(
                data_a['n_frames'].dropna(),
                data_b['n_frames'].dropna(),
                equal_var=False
            )
            results[f'{model_a}_vs_{model_b}'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'significant': p_val < 0.05
            }

    return results


def compute_capability_funnel_from_summaries(session_summaries: pd.DataFrame) -> Dict:
    """Compute capability funnel progression from session summaries.

    Implements plan.md \u00a72.4 and \u00a75:
    Navega\u00e7\u00e3o \u2192 Defesa Reativa \u2192 Puni\u00e7\u00e3o Ofensiva \u2192 Gerenciamento \u2192 Vit\u00f3ria
    """
    if session_summaries.empty:
        return {}

    # Handle both 'Model' and 'model' column names
    model_col = 'Model' if 'Model' in session_summaries.columns else \
                ('model' if 'model' in session_summaries.columns else None)

    if model_col is None:
        return {}

    results = {}
    models = session_summaries[model_col].unique()

    for model in models:
        model_data = session_summaries[session_summaries[model_col] == model]
        total_sessions = len(model_data)
        wins = int((model_data['outcome'] == 'WIN').sum()) if 'outcome' in model_data.columns else 0

        # Funnel levels based on available metrics
        nav_score = 1.0 - float(model_data['P_canto'].mean()) if 'P_canto' in model_data.columns else np.nan
        def_score = float(model_data['M_defesa'].mean()) if 'M_defesa' in model_data.columns else np.nan
        off_score = float(model_data['M_punicao'].mean()) if 'M_punicao' in model_data.columns else np.nan
        rec_score = float(model_data['M_cura'].mean()) if 'M_cura' in model_data.columns else np.nan
        win_rate = wins / total_sessions if total_sessions > 0 else 0.0

        results[model] = {
            'navigation': nav_score,
            'defense': def_score,
            'offense': off_score,
            'resource_management': rec_score,
            'victory': win_rate,
            'total_sessions': total_sessions,
            'wins': wins,
        }

    return results
