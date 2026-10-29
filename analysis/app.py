import glob
import json
import math
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from Python.Src.Data.TelemetryDataset import (
    ACTION_NAMES,
    extract_expert_actions,
    extract_model_actions,
    extract_model_probabilities,
    final_player_hp,
    load_session_csv,
    session_summary_label,
)
from Python.Src.Utils.ContextMetrics import (
    BOSS_STATE_CONFIG_PATH,
    MetricParams,
    compute_contextual_metrics,
    load_boss_state_config,
    load_boss_state_registry,
    summarise_sessions,
)
from scenario_analysis import (
    compute_capability_funnel,
    compute_scenario_metrics,
    extract_critical_scenarios,
)

# ------------------------------------------------------------------ Página

st.set_page_config(
    page_title="Hollow Knight · Telemetria Contextual",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border-radius: 10px;
        padding: 0.65rem 0.9rem;
    }
    div[data-testid="stMetricLabel"] p {font-size: 0.85rem; opacity: 0.8;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎮 Métricas Contextuais · Behavior Cloning")
st.caption(
    "Inspeção de telemetria frame a frame, análise de erros *bottom-up* e "
    "métricas contextuais do plan.md para o combate contra a Hornet."
)

# --------------------------------------------------------------- Constantes

ACTION_LABELS_PT = {
    'dash': 'Dash (Trigger Dir.)',
    'jump': 'Pulo (Botão A)',
    'cast': 'Cura/Feitiço (Botão B)',
    'attack': 'Ataque (Botão X)',
    'left': 'Esquerda',
    'right': 'Direita',
    'up': 'Cima',
    'down': 'Baixo',
}

ERROR_TAXONOMY = [
    ('omissao_reativa', '1. Omissão Reativa (não esquivou de golpe)'),
    ('acao_inoportuna', '2. Ação Inoportuna (curou solto sob ataque)'),
    ('erro_mira', '3. Erro de Congruência Espacial (mirou oposto)'),
    ('encurralamento', '4. Encurralamento Posicional (morte no canto)'),
    ('latencia_inferencia', '5. Atraso de Inferência (resposta física lentíssima)'),
]
ERROR_LABELS = dict(ERROR_TAXONOMY)
ERROR_LABELS_AUTO = {**ERROR_LABELS, 'indefinido': 'Indefinido (sem padrão claro)'}

ERROR_TAXONOMY_EXTENDED = {
    'omissao_reativa': {
        'label': '1. Omissão Reativa',
        'subcategories': [
            ('no_dash_vs_fast_attack', '1.1 - Nenhuma esquiva durante investida rápida'),
            ('no_jump_vs_aerial', '1.2 - Nenhum pulo durante ataque aéreo'),
            ('late_dash', '1.3 - Esquiva tardia (após dano inevitável)'),
        ],
    },
    'acao_inoportuna': {
        'label': '2. Ação Inoportuna',
        'subcategories': [
            ('healing_under_attack', '2.1 - Cura durante animação ofensiva'),
            ('casting_during_recovery', '2.2 - Feitiço durante recuperação de golpe'),
            ('attack_during_vulnerable', '2.3 - Ataque sem janela de conjuração'),
        ],
    },
    'erro_mira': {
        'label': '3. Erro de Congruência Espacial',
        'subcategories': [
            ('opposite_direction_attack', '3.1 - Ataque com direção oposta ao chefe'),
            ('zero_direction_attack', '3.2 - Ataque sem direcionamento horizontal'),
            ('spell_towards_wall', '3.3 - Feitiço direcionado à parede'),
        ],
    },
    'encurralamento': {
        'label': '4. Encurralamento Posicional',
        'subcategories': [
            ('corner_stuck', '4.1 - Preso no canto inferior'),
            ('wall_pressure', '4.2 - Empurrado contra parede por movimento'),
            ('no_escape_path', '4.3 - Sem trajetória de fuga disponível'),
        ],
    },
    'latencia_inferencia': {
        'label': '5. Atraso de Inferência',
        'subcategories': [
            ('inference_delay_cpu', '5.1 - Latência em CPU (GRU bottleneck)'),
            ('prediction_hesitation', '5.2 - Hesitação perto do limiar τ'),
            ('contradictory_actions', '5.3 - Ações contraditórias (esq+dir simultâneo)'),
        ],
    },
}

FUNNEL_LEVELS = ['navigation', 'defense', 'offense', 'resource_management', 'victory']
FUNNEL_LABELS_PT = {
    'navigation': 'Navegação',
    'defense': 'Defesa reativa',
    'offense': 'Punição ofensiva',
    'resource_management': 'Gestão de recursos',
    'victory': 'Vitória',
}

SCENARIO_LABELS_PT = {
    'scenario_a_high_speed_attack': 'A · Investida em alta velocidade',
    'scenario_b_vulnerable_window': 'B · Chefe vulnerável',
    'scenario_c_low_health': 'C · Jogador com pouca vida',
}

METRIC_KEYS = ['CSR', 'COR', 'M_cura', 'M_alinhamento', 'M_evasao', 'FER',
               'M_defesa', 'M_punicao', 'P_canto']
SCENARIO_METRIC_KEYS = ['M_cura', 'M_alinhamento', 'M_evasao', 'M_defesa',
                        'M_punicao', 'P_canto']

TAIL_SECONDS = 10.0  # janela final do combate usada para sugerir a causa da derrota

TEST_MODEL_DIRS = {
    'cnn': 'CNN',
    'gru': 'GRU',
    'mlp': 'MLP',
    'mlp_temporal': 'MLP Temporal',
}

TAXONOMY_CSV = os.path.join(PROJECT_ROOT, 'analysis', 'error_taxonomy.csv')
SETTINGS_JSON = os.path.join(PROJECT_ROOT, 'analysis', 'settings.json')


# ----------------------------------------------------------------- Utilidades

def load_settings():
    try:
        with open(SETTINGS_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, OSError):
        return {}


def save_settings(upd):
    data = load_settings()
    data.update(upd)
    with open(SETTINGS_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_shots_dir_default():
    """Procura <plugins do BepInEx>/boss_state_shots (pasta da DLL do mod)."""
    plugins = (r'C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight'
               r'\BepInEx\plugins')
    if os.path.isdir(plugins):
        hits = glob.glob(os.path.join(plugins, 'boss_state_shots'))
        hits += glob.glob(os.path.join(plugins, '*', 'boss_state_shots'))
        hits += glob.glob(os.path.join(plugins, '*', '*', 'boss_state_shots'))
        if hits:
            return sorted(hits)[0]
    return ''


def style_fig(fig, height=320, title=None, unified_hover=False):
    """Layout padrão para todos os gráficos (legenda no topo, margens enxutas)."""
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0.0, font=dict(size=15)) if title else None,
        margin=dict(l=10, r=10, t=50 if title else 30, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        hovermode='x unified' if unified_hover else 'closest',
    )
    return fig


def wilson_ci(k, n, z=1.96):
    """Intervalo de Wilson para uma proporção (apropriado para n pequeno)."""
    if n <= 0:
        return float('nan'), float('nan')
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def model_from_path(path):
    try:
        key = os.path.relpath(path, test_root).split(os.sep, 1)[0]
    except ValueError:
        key = 'unknown'
    return TEST_MODEL_DIRS.get(key, key)


def display_session_label(path):
    """Inclui a arquitetura no rótulo para distinguir sessões com nomes iguais."""
    try:
        first_dir = os.path.relpath(path, test_root).split(os.sep, 1)[0]
        if first_dir in TEST_MODEL_DIRS:
            return f"{TEST_MODEL_DIRS[first_dir]} · {session_summary_label(path)}"
    except (ValueError, TypeError):
        pass
    return session_summary_label(path)


# ------------------------------------------------- Taxonomia e Pareto (topo!)
# (antes eram definidas depois de usadas -> NameError na aba Taxonomia)

def classify_loss_pattern(df, config):
    """Sugere a causa de uma derrota olhando só os últimos segundos da luta.

    É apenas uma DICA: a análise bottom-up pressupõe rótulo humano após assistir
    ao replay. Quando nenhum padrão bate devolve 'indefinido' em vez de
    atribuir tudo a 'latencia_inferencia' (que inflava o Pareto).
    """
    if df is None or df.empty:
        return 'indefinido'

    if 'timestamp' in df.columns:
        ts = pd.to_numeric(df['timestamp'], errors='coerce')
        tail = df[ts >= ts.max() - TAIL_SECONDS]
    else:
        tail = df.tail(int(TAIL_SECONDS * 60))
    if tail.empty:
        tail = df.tail(60)

    attack_active = list(config.get('attack_active') or [])

    if attack_active and 'bossState' in tail.columns:
        atk = tail[tail['bossState'].isin(attack_active)]
        if len(atk) > 0:
            if ('dist' in atk.columns and 'gp_trigger_right' in atk.columns
                    and atk['dist'].mean() < 4.0
                    and atk['gp_trigger_right'].mean() < 0.1):
                return 'omissao_reativa'
            if 'gp_btn_b' in atk.columns and atk['gp_btn_b'].mean() > 0.1:
                return 'acao_inoportuna'

    if 'delta_x' in tail.columns and 'gp_axis_left_x' in tail.columns:
        if abs(tail['delta_x'].mean()) > 2.0 and abs(tail['gp_axis_left_x'].mean()) < 0.1:
            return 'erro_mira'

    if 'x_arena' in tail.columns and abs(tail['x_arena'].iloc[-1]) > 0.8:
        return 'encurralamento'

    return 'indefinido'


def compute_pareto_analysis(error_data: pd.DataFrame) -> Dict:
    """Pareto 80/20 dos modos de falha (coluna `cause_code`)."""
    if error_data is None or error_data.empty or 'cause_code' not in error_data.columns:
        return {}

    error_counts = error_data.groupby('cause_code').size().sort_values(ascending=False)
    total = int(error_counts.sum())
    if total == 0:
        return {}

    cumulative_pct = error_counts.cumsum() / total * 100
    # inclui o primeiro tipo que cruza 80% (antes ficava vazio se o 1º já passasse de 80%)
    n_top = min(int((cumulative_pct < 80).sum()) + 1, len(error_counts))
    top_80 = error_counts.index[:n_top].tolist()

    model_breakdown = {}
    if 'Model' in error_data.columns:
        for code in error_counts.index:
            sub = error_data[error_data['cause_code'] == code]
            model_breakdown[code] = sub['Model'].value_counts().to_dict()

    return {
        'total_failures': total,
        'error_counts': error_counts.to_dict(),
        'cumulative_pct': cumulative_pct.to_dict(),
        'top_errors_contributing_80pct': top_80,
        'pareto_threshold': n_top,
        'model_breakdown': model_breakdown,
        'pareto_efficient': n_top <= max(1, math.ceil(0.2 * len(error_counts))),
    }


def render_pareto(error_df, labels, key):
    """Barras de frequência + curva acumulada + linha de 80%."""
    res = compute_pareto_analysis(error_df)
    if not res:
        st.info("Sem falhas rotuladas para montar o Pareto.")
        return
    codes = list(res['error_counts'].keys())
    names = [labels.get(c, c) for c in codes]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de falhas", res['total_failures'])
    c2.metric("Tipos de erro distintos", len(codes))
    c3.metric("Tipos que explicam ≥80%", res['pareto_threshold'])

    fig = go.Figure()
    fig.add_bar(x=names, y=[res['error_counts'][c] for c in codes], name='Frequência')
    fig.add_scatter(x=names, y=[res['cumulative_pct'][c] for c in codes],
                    name='% acumulado', yaxis='y2', mode='lines+markers',
                    line=dict(color='#e67e22'))
    fig.add_shape(type='line', xref='paper', x0=0, x1=1, yref='y2', y0=80, y1=80,
                  line=dict(dash='dash', color='#e74c3c'))
    fig.update_layout(yaxis=dict(title='Frequência'),
                      yaxis2=dict(overlaying='y', side='right', range=[0, 105],
                                  title='% acumulado', showgrid=False))
    style_fig(fig, height=360, title='Pareto das causas de derrota')
    st.plotly_chart(fig, width='stretch', key=f'pareto_{key}')

    top_names = [labels.get(c, c) for c in res['top_errors_contributing_80pct']]
    st.markdown(f"**Modos de falha que concentram ≥80%:** {', '.join(top_names)}")

    if res['model_breakdown']:
        cross = pd.crosstab(
            error_df['cause_code'].map(lambda c: labels.get(c, c)), error_df['Model'])
        st.dataframe(cross, width='stretch')


# ---------------------------------------------------------------- Carregamento

@st.cache_data(show_spinner=False)
def load_battle(path):
    return load_session_csv(path)


@st.cache_data(show_spinner=False)
def load_summary(files, source, d_seguro, d_ameaca, d_alcance, x_limiar, h_crit,
                 s_foco, lambda_fer, tau, aim_mode):
    config = load_boss_state_config()
    params = MetricParams(d_seguro=d_seguro, d_ameaca=d_ameaca,
                          d_alcance=d_alcance, x_limiar=x_limiar,
                          h_crit=h_crit, s_foco=s_foco, lambda_fer=lambda_fer,
                          tau=tau, aim_mode=aim_mode)
    return summarise_sessions(list(files), params=params, config=config, source=source)


@st.cache_data(show_spinner=False)
def load_test_run_summaries(paths):
    """Resume cada combate a partir dos CSVs frame a frame em test_runs."""
    rows = []
    root = os.path.join(PROJECT_ROOT, 'dataset', 'test_runs')

    for path in paths:
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue

        required = {'timestamp', 'hp', 'bossHp'}
        if frame.empty or not required.issubset(frame.columns):
            continue

        frame = frame.sort_values('timestamp')
        boss_hp = pd.to_numeric(frame['bossHp'], errors='coerce').dropna()
        timestamps = pd.to_numeric(frame['timestamp'], errors='coerce').dropna()
        if boss_hp.empty:
            continue

        boss_start = float(boss_hp.iloc[0])
        boss_end = float(boss_hp.iloc[-1])
        survival_time = (
            max(0.0, float(timestamps.iloc[-1] - timestamps.iloc[0]))
            if len(timestamps) >= 2 else float('nan')
        )
        model_key = os.path.relpath(path, root).split(os.sep, 1)[0]

        rows.append({
            'Model': TEST_MODEL_DIRS.get(model_key, model_key),
            'Result': 'WIN' if boss_end <= 0 else 'LOSS',
            'Survival_Time_Sec': survival_time,
            'Damage_Dealt': max(0.0, boss_start - boss_end),
            'Session': os.path.basename(path),
            'File': path,
        })

    return pd.DataFrame(rows, columns=[
        'Model', 'Result', 'Survival_Time_Sec', 'Damage_Dealt', 'Session', 'File'
    ])


@st.cache_data(show_spinner="Calculando métricas contextuais por sessão…")
def compute_test_metrics(files, **p):
    """Métricas contextuais de cada sessão de teste (cacheado)."""
    config = load_boss_state_config()
    params = MetricParams(**p)
    rows, failures = [], []
    for path in files:
        df = load_battle(path)
        if df.empty:
            continue
        try:
            res = compute_contextual_metrics(df, params=params, config=config,
                                             source='model')
        except Exception as exc:  # noqa: BLE001 - reportado na UI, não engolido
            failures.append((os.path.basename(path), repr(exc)))
            continue
        row = {'Model': model_from_path(path), 'Session': os.path.basename(path),
               'File': path, 'n_frames': len(df)}
        for k in METRIC_KEYS:
            row[k] = res.get(k, np.nan)
        rows.append(row)
    cols = ['Model', 'Session', 'File', 'n_frames'] + METRIC_KEYS
    return pd.DataFrame(rows, columns=cols), failures


@st.cache_data(show_spinner="Fatiando cenários críticos…")
def compute_scenario_table(files, offensive, vulnerable, **p):
    """Métricas por cenário crítico, em TODAS as sessões (antes: só as 40 primeiras,
    o que excluía a MLP Temporal)."""
    params = MetricParams(**p)
    rows, failures = [], []
    for path in files:
        df = load_battle(path)
        if df.empty:
            continue
        df = df.copy().reset_index(drop=True)  # índice posicional p/ fatiar as ações
        try:
            scenarios = extract_critical_scenarios(
                df, offensive_states=list(offensive), vulnerable_states=list(vulnerable))
            model_actions = extract_model_actions(df)
            for name, sdf in scenarios.items():
                if len(sdf) == 0:
                    continue
                # ações alinhadas ao recorte (antes recebiam o array da luta inteira)
                acts = None if model_actions is None else model_actions[sdf.index.to_numpy()]
                m = compute_scenario_metrics(sdf, name, actions=acts, params=params)
                row = {'Arquitetura': model_from_path(path),
                       'Cenário': SCENARIO_LABELS_PT.get(name, name),
                       'Sessão': os.path.basename(path), 'Frames': len(sdf)}
                for k in SCENARIO_METRIC_KEYS:
                    row[k] = m.get(k, np.nan)
                rows.append(row)
        except Exception as exc:  # noqa: BLE001
            failures.append((os.path.basename(path), repr(exc)))
    cols = ['Arquitetura', 'Cenário', 'Sessão', 'Frames'] + SCENARIO_METRIC_KEYS
    return pd.DataFrame(rows, columns=cols), failures


@st.cache_data(show_spinner=False)
def classify_losses(files, cfg_mtime):
    """Sugestão automática de causa para cada derrota (cacheado por config)."""
    config = load_boss_state_config()
    rows = []
    for path in files:
        df = load_battle(path)
        if df.empty:
            continue
        rows.append({'Model': model_from_path(path), 'Session': os.path.basename(path),
                     'cause_code': classify_loss_pattern(df, config)})
    return pd.DataFrame(rows, columns=['Model', 'Session', 'cause_code'])


@st.cache_data(show_spinner=False)
def load_registry_maps(_mtime=0.0):
    """Leve: {'name': {id: nome}, 'offensive': set(ids), 'vulnerable': set(ids),
    'stats': {id: dict}}. Cache quebra quando o arquivo muda (mtime)."""
    states = load_boss_state_registry()
    return {
        'name': {s['state_id']: s.get('state_name') for s in states},
        'offensive': {s['state_id'] for s in states if s.get('is_offensive')},
        'vulnerable': {s['state_id'] for s in states if s.get('is_vulnerable')},
        'reviewed': {s['state_id'] for s in states if s.get('reviewed')},
        'stats': {s['state_id']: s.get('stats') or {} for s in states},
    }


def config_mtime():
    try:
        return os.path.getmtime(BOSS_STATE_CONFIG_PATH)
    except OSError:
        return 0.0


def get_registry_maps():
    return load_registry_maps(_mtime=config_mtime())


def state_label(sid, maps=None):
    maps = maps or get_registry_maps()
    name = maps['name'].get(int(sid)) or 'desconhecido'
    return f'{name} ({int(sid)})'


def load_taxonomy_csv():
    if os.path.isfile(TAXONOMY_CSV):
        return pd.read_csv(TAXONOMY_CSV)
    return pd.DataFrame(columns=['file', 'timestamp', 'cause_code', 'cause_label', 'Model'])


def save_taxonomy_row(row):
    existing = load_taxonomy_csv()
    base = existing[existing['file'] != row['file']]
    updated = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(TAXONOMY_CSV), exist_ok=True)
    updated.to_csv(TAXONOMY_CSV, index=False)
    return updated


# ------------------------------------------------------------------- Dados

train_dir = os.path.join(PROJECT_ROOT, 'dataset', 'hornet_boss')
training_files = sorted(
    os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')
) if os.path.isdir(train_dir) else []

test_root = os.path.join(PROJECT_ROOT, 'dataset', 'test_runs')
test_files = []
for _dir in TEST_MODEL_DIRS:
    _path = os.path.join(test_root, _dir)
    if os.path.isdir(_path):
        test_files.extend(glob.glob(os.path.join(_path, '**', '*.csv'), recursive=True))
test_files = sorted(set(test_files))

# ------------------------------------------------------------------ Sidebar

st.sidebar.title("⚙️ Controles")

st.sidebar.markdown("### 📂 Dados")
origin = st.sidebar.radio(
    "Origem",
    ['Demonstrações de Treino', 'Testes do Bot'],
    help="Demonstrações: ações do especialista. Testes: mesmo schema + "
         "pred_* / prob_* do modelo.",
)
pool = training_files if origin == 'Demonstrações de Treino' else test_files
labels_by_path = {p: display_session_label(p) for p in pool}

if not pool:
    st.sidebar.warning("Nenhum arquivo nesta origem ainda.")

select_all = st.sidebar.checkbox("Selecionar todas as sessões", value=False,
                                 key=f"all_{origin}")
all_labels = list(labels_by_path.values())
selected_labels = st.sidebar.multiselect(
    "Sessões", all_labels, default=all_labels if select_all else [],
    key=f"sessions_{origin}_{select_all}",
)
selected_paths = [p for p in pool if labels_by_path[p] in selected_labels]
st.sidebar.caption(f"{len(selected_paths)} de {len(pool)} sessões selecionadas")

with st.sidebar.expander("📏 Parâmetros das métricas (plan.md)"):
    col_a, col_b = st.columns(2)
    d_seguro = col_a.number_input("d_seguro", 0.5, 15.0, 4.0, 0.5,
                                  help="Distância mínima segura para curar")
    d_ameaca = col_b.number_input("d_ameaca", 0.5, 15.0, 5.0, 0.5,
                                  help="Raio da janela de ameaça")
    d_alcance = col_a.number_input("d_alcance", 0.3, 6.0, 1.8, 0.1,
                                   help="Alcance do ferrão/magia")
    x_limiar = col_b.number_input("x_limiar", 0.5, 1.0, 0.85, 0.05,
                                  help="|x_arena| a partir do qual é 'canto'")
    h_crit = col_a.number_input("h_crit", 0.05, 0.9, 0.25, 0.05,
                                help="Vida normalizada crítica")
    s_foco = col_b.number_input("s_foco", 0.1, 1.0, 1.0 / 3.0, 1.0 / 99.0,
                                help="Alma mínima para focar")
    lambda_fer = col_a.number_input("lambda (FER)", 0.0, 2.0, 0.5, 0.1,
                                    help="Peso da evasão fútil em M_defesa")
    tau = col_b.number_input("tau (limiar bot)", 0.05, 0.9, 0.2, 0.05,
                             help="Limiar de ativação dos botões")

    aim_mode = st.radio(
        "Modo de mira (M_alinhamento)", ['facing', 'plan'],
        format_func=lambda m: {
            'facing': 'facing (mecânica real do jogo)',
            'plan': 'stick direcional (fórmula literal plan.md)',
        }[m])

    metric_source = st.radio(
        "Ações avaliadas", ['expert', 'model'],
        format_func=lambda s: {
            'expert': 'Especialista (treino)',
            'model': 'Bot (saída do modelo)',
        }[s])

PARAM_ARGS = dict(d_seguro=d_seguro, d_ameaca=d_ameaca, d_alcance=d_alcance,
                  x_limiar=x_limiar, h_crit=h_crit, s_foco=s_foco,
                  lambda_fer=lambda_fer, tau=tau, aim_mode=aim_mode)

config = load_boss_state_config()
registry_maps = get_registry_maps()

with st.sidebar.expander("🗂️ boss_state_config.json"):
    st.caption("Registro editável em `Python/Src/Data/boss_state_config.json` "
               "(gerado por `Python/build_boss_state_registry.py`).")
    if config['registry']:
        ns = config['registry'].get('states', [])
        n_rev = sum(1 for s_ in ns if s_.get('reviewed'))
        st.caption(f"Estados: {len(ns)} | revisados: {n_rev}")
        st.caption(f"attack_active: {config['attack_active'] or '(vazio)'}")
        st.caption(f"vulnerable: {config['vulnerable'] or '(vazio)'}")
        conf_ids = [s_['state_id'] for s_ in ns
                    if not s_.get('reviewed')
                    and (s_['state_id'] in config['attack_active']
                         or s_['state_id'] in config['vulnerable'])]
        if conf_ids:
            st.info(f"Hipóteses automáticas a revisar na arena: ids {conf_ids}. "
                    "Confirme/corrija no JSON e marque reviewed: true.")
    else:
        st.caption(f"attack_active: {config['attack_active'] or '(vazio)'}")
        st.caption(f"vulnerable: {config['vulnerable'] or '(vazio)'}")
    if not (config['attack_active'] or config['vulnerable']):
        st.warning("Preencha boss_state_config.json: métricas com janelas "
                   "dependentes de ameaça/abertura ficarão vazias.")

# ------------------------------------------------- Pré-cálculos compartilhados

agg_all = load_test_run_summaries(tuple(test_files)) if test_files else pd.DataFrame(
    columns=['Model', 'Result', 'Survival_Time_Sec', 'Damage_Dealt', 'Session', 'File'])

test_metrics_df, metric_failures = (pd.DataFrame(), [])
if test_files:
    test_metrics_df, metric_failures = compute_test_metrics(tuple(test_files), **PARAM_ARGS)
    if not test_metrics_df.empty and not agg_all.empty:
        # resultado REAL da luta (antes todas eram marcadas como WIN)
        test_metrics_df = test_metrics_df.merge(
            agg_all[['File', 'Result']], on='File', how='left')

tab_agg, tab_battle, tab_metrics, tab_taxonomy, tab_funnel = st.tabs([
    "🏁 Resultados", "🔬 Batalha", "📐 Métricas", "🩹 Taxonomia de Erros",
    "🧭 Funil & Cenários",
])

# ================================================================ Resultados
with tab_agg:
    st.subheader("Resultados agregados dos combates de teste")

    if agg_all.empty:
        st.info("Nenhum CSV de teste válido encontrado em "
                "dataset/test_runs/{cnn, gru, mlp, mlp_temporal}/. "
                "Verifique se os arquivos contêm timestamp, hp e bossHp.")
    else:
        models = agg_all['Model'].unique().tolist()
        pick = st.multiselect("Arquiteturas", models, default=models, key="agg_models")
        view = agg_all[agg_all['Model'].isin(pick)]

        if view.empty:
            st.warning("Selecione ao menos uma arquitetura.")
        else:
            wins = int((view['Result'] == 'WIN').sum())
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Lutas", len(view))
            k2.metric("Vitórias", f"{wins} ({wins / len(view):.0%})")
            k3.metric("Sobrevivência média", f"{view['Survival_Time_Sec'].mean():.1f} s")
            k4.metric("Dano médio", f"{view['Damage_Dealt'].mean():.0f}")

            c4, c5 = st.columns(2)
            with c4:
                fig = px.box(view, x='Model', y='Survival_Time_Sec', points='all',
                             color='Model')
                fig.update_layout(showlegend=False,
                                  yaxis_title='segundos', xaxis_title=None)
                st.plotly_chart(style_fig(fig, 340, 'Tempo de sobrevivência'),
                                width='stretch')
            with c5:
                fig = px.box(view, x='Model', y='Damage_Dealt', points='all', color='Model')
                fig.update_layout(showlegend=False, yaxis_title='dano', xaxis_title=None)
                st.plotly_chart(style_fig(fig, 340, 'Dano acumulado ao chefe'),
                                width='stretch')

            st.markdown("**Comparação entre arquiteturas**")
            rows = []
            for model, g in view.groupby('Model', sort=False):
                n = len(g)
                w = int((g['Result'] == 'WIN').sum())
                lo, hi = wilson_ci(w, n)
                rows.append({
                    'Arquitetura': model, 'Partidas': n, 'Vitórias': w,
                    'Taxa de vitória (%)': 100 * w / n,
                    'IC95% Wilson inf. (%)': 100 * lo, 'IC95% Wilson sup. (%)': 100 * hi,
                    'Sobrev. média (s)': g['Survival_Time_Sec'].mean(),
                    'Sobrev. desvio': g['Survival_Time_Sec'].std(),
                    'Dano médio': g['Damage_Dealt'].mean(),
                    'Dano desvio': g['Damage_Dealt'].std(),
                })
            st.dataframe(
                pd.DataFrame(rows), hide_index=True, width='stretch',
                column_config={
                    'Taxa de vitória (%)': st.column_config.NumberColumn(format="%.1f%%"),
                    'IC95% Wilson inf. (%)': st.column_config.NumberColumn(format="%.1f%%"),
                    'IC95% Wilson sup. (%)': st.column_config.NumberColumn(format="%.1f%%"),
                    'Sobrev. média (s)': st.column_config.NumberColumn(format="%.1f"),
                    'Sobrev. desvio': st.column_config.NumberColumn(format="%.1f"),
                    'Dano médio': st.column_config.NumberColumn(format="%.0f"),
                    'Dano desvio': st.column_config.NumberColumn(format="%.0f"),
                })

            if len(pick) > 1:
                if not HAS_SCIPY:
                    st.info("Para testes de hipótese, instale scipy: pip install scipy")
                else:
                    sig = []
                    for i in range(len(pick)):
                        for j in range(i + 1, len(pick)):
                            a = view[view['Model'] == pick[i]]
                            b = view[view['Model'] == pick[j]]
                            entry = {'Comparação': f'{pick[i]} vs {pick[j]}'}
                            for col, name in (('Survival_Time_Sec', 'Sobrevivência'),
                                              ('Damage_Dealt', 'Dano')):
                                x, y = a[col].dropna(), b[col].dropna()
                                if len(x) > 1 and len(y) > 1:
                                    entry[f'p Welch · {name}'] = float(
                                        scipy_stats.ttest_ind(x, y, equal_var=False).pvalue)
                            wa, wb = int((a['Result'] == 'WIN').sum()), int((b['Result'] == 'WIN').sum())
                            entry['p Fisher · vitórias'] = float(scipy_stats.fisher_exact(
                                [[wa, len(a) - wa], [wb, len(b) - wb]])[1])
                            sig.append(entry)
                    st.markdown("**Testes de hipótese (p-valores)**")
                    st.dataframe(pd.DataFrame(sig), hide_index=True, width='stretch',
                                 column_config={c: st.column_config.NumberColumn(format="%.4f")
                                                for c in pd.DataFrame(sig).columns
                                                if c.startswith('p ')})
                    st.caption("Vitórias são contagens binárias com n≈15 por modelo: "
                               "use Fisher/Wilson, não t-test. Diferenças pequenas "
                               "(ex.: 2/15 vs 1/15) raramente são significativas.")

            with st.expander("Tabela de combates"):
                st.dataframe(view.drop(columns=['File'], errors='ignore'),
                             hide_index=True, width='stretch')

            st.info("O funil de capacidades e os cenários críticos estão na aba "
                    "**🧭 Funil & Cenários**.")

# ================================================================== Batalha
with tab_battle:
    st.caption("Fonte: sessões selecionadas na barra lateral "
               "(demonstrações de treino ou dataset/test_runs/{cnn, gru, mlp, mlp_temporal}/).")

    if not selected_paths:
        st.info("Selecione ao menos uma sessão na barra lateral.")
    else:
        session_path = st.selectbox("Sessão para inspeção", selected_paths,
                                    format_func=display_session_label)
        df = load_battle(session_path)

        if df.empty or len(df) < 2:
            st.warning("Sessão vazia ou com menos de 2 quadros.")
        else:
            expert_actions = extract_expert_actions(df)
            model_actions = extract_model_actions(df)
            model_probs = extract_model_probabilities(df)
            has_model = model_actions is not None

            top1, top2, top3, top4 = st.columns(4)
            top1.metric("Quadros", len(df))
            if 'timestamp' in df.columns:
                span = float(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0])
                top2.metric("Duração", f"{span:.1f} s")
            top3.metric("HP final jogador", f"{df['hp'].iloc[-1]:.0f}")
            top4.metric("HP final chefe", f"{df['bossHp'].iloc[-1]:.0f}")

            opt1, opt2, opt3 = st.columns([1, 1, 3])
            show_model = opt1.checkbox("Ações do modelo (pred_*)", value=has_model,
                                       disabled=not has_model)
            show_probs = opt2.checkbox("Probabilidades (prob_*)", value=has_model,
                                       disabled=not has_model)
            max_frame = len(df) - 1
            with opt3:
                frame_range = st.slider("Intervalo de quadros", 0, max_frame,
                                        (0, min(1200, max_frame)))
            a0, a1 = frame_range[0], frame_range[1] + 1
            sub = df.iloc[a0:a1].copy()
            frames = sub.index.to_numpy()

            t_over, t_act, t_state, t_frame = st.tabs(
                ["Visão geral", "Ações & probabilidades", "bossState", "Inspeção de quadro"])

            with t_over:
                vitality_col, axis_col, position_col = st.columns(3)
                with vitality_col:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=frames, y=sub['hp'], name='HP Jogador',
                                             line=dict(color='#2ecc71')))
                    fig.add_trace(go.Scatter(x=frames, y=sub['soul'], name='Alma',
                                             line=dict(color='#1abc9c')))
                    fig.add_trace(go.Scatter(x=frames, y=sub['bossHp'], name='HP Chefe',
                                             line=dict(color='#e74c3c', dash='dot')))
                    st.plotly_chart(style_fig(fig, 340, 'HP / Alma / HP Chefe', True),
                                    width='stretch')
                with axis_col:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=frames, y=sub['dist'], name='Distância (Δp)',
                                             line=dict(color='#9b59b6')))
                    if show_model:
                        fig.add_trace(go.Scatter(
                            x=frames, y=model_actions[a0:a1, 3], name='Ataque modelo',
                            yaxis='y2', opacity=0.6,
                            line=dict(shape='hv', color='#f39c12')))
                    fig.add_trace(go.Scatter(
                        x=frames, y=expert_actions[a0:a1, 3], name='Ataque especialista',
                        yaxis='y2', opacity=0.35,
                        line=dict(shape='hv', color='#3498db')))
                    fig.update_layout(yaxis=dict(title='unidades'),
                                      yaxis2=dict(overlaying='y', side='right',
                                                  range=[-0.1, 1.4], showgrid=False))
                    st.plotly_chart(style_fig(fig, 340, 'Distância × ataque', True),
                                    width='stretch')
                with position_col:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['px'], y=df['py'], mode='lines', opacity=0.18,
                        name='Trajeto completo', line=dict(color='#95a5a6')))
                    fig.add_trace(go.Scatter(
                        x=sub['bx'], y=sub['by'], mode='lines', name='Chefe',
                        line=dict(color='#e74c3c')))
                    fig.add_trace(go.Scatter(
                        x=sub['px'], y=sub['py'], mode='markers+lines', name='Jogador (sel.)',
                        marker=dict(color=frames, colorscale='Viridis', size=4),
                        line=dict(color='#3498db')))
                    fig.update_layout(xaxis=dict(range=[15.0, 38.5]),
                                      yaxis=dict(range=[27.5, 40.5],
                                                 scaleanchor='x', scaleratio=1))
                    st.plotly_chart(style_fig(fig, 340, 'Trajetória px × py'),
                                    width='stretch')

            with t_act:
                chosen = st.multiselect(
                    "Ações a exibir", list(ACTION_NAMES),
                    default=['dash', 'jump', 'cast', 'attack', 'left', 'right'],
                    format_func=lambda a: ACTION_LABELS_PT[a])
                if chosen:
                    fig = go.Figure()
                    idx_map = [ACTION_NAMES.index(a) for a in chosen]
                    for i, a in zip(idx_map, chosen):
                        fig.add_trace(go.Scatter(
                            x=frames, y=expert_actions[a0:a1, i], name=f'{a} (esp.)',
                            line=dict(shape='hv'), opacity=0.55))
                    if show_model and has_model:
                        for i, a in zip(idx_map, chosen):
                            fig.add_trace(go.Scatter(
                                x=frames, y=model_actions[a0:a1, i], name=f'{a} (bot)',
                                line=dict(shape='hv', dash='dot'), opacity=0.75))
                    st.plotly_chart(style_fig(fig, 320, 'Timeline de ações', True),
                                    width='stretch')

                if show_probs and model_probs is not None:
                    fig = go.Figure()
                    for j, name in enumerate(ACTION_NAMES):
                        fig.add_trace(go.Scatter(x=frames, y=model_probs[a0:a1, j],
                                                 name=ACTION_LABELS_PT[name]))
                    fig.add_hline(y=tau, line=dict(color='red', dash='dash'),
                                  annotation_text=f"τ = {tau}")
                    fig.update_layout(yaxis=dict(range=[-0.02, 1.02]))
                    st.plotly_chart(style_fig(fig, 380, 'Probabilidades do modelo vs. limiar τ', True),
                                    width='stretch')

                if has_model and show_model:
                    diverg = (expert_actions[a0:a1] != model_actions[a0:a1]).any(axis=1)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=frames, y=diverg.astype(int), mode='markers',
                        marker=dict(color='#e74c3c', size=6),
                        name='Divergência (esp. ≠ bot)'))
                    fig.update_layout(yaxis=dict(range=[-0.2, 1.2]))
                    st.plotly_chart(style_fig(fig, 220, f'Divergência especialista × bot '
                                              f'({100 * diverg.mean():.1f}% dos quadros)'),
                                    width='stretch')

            with t_state:
                reg = get_registry_maps()
                state_ids_np = sub['bossState'].astype(int).to_numpy()
                off_v = np.isin(state_ids_np, list(reg['offensive']))
                vuln_v = np.isin(state_ids_np, list(reg['vulnerable']))
                hover = np.array([state_label(int(i), reg) for i in state_ids_np])
                neutral_v = ~off_v & ~vuln_v

                fig = go.Figure()
                if neutral_v.any():
                    fig.add_trace(go.Scatter(
                        x=frames[neutral_v], y=state_ids_np[neutral_v],
                        mode='markers+lines', name='neutro',
                        marker=dict(color=state_ids_np[neutral_v],
                                    colorscale='Portland', size=5),
                        line=dict(color='#bdc3c7'), customdata=hover[neutral_v],
                        hovertemplate='%{customdata}<extra></extra>'))
                for mask_v, lbl, color in (
                        (off_v, 'is_offensive (estado de ataque)', '#e74c3c'),
                        (vuln_v & ~off_v, 'is_vulnerable (janela de punição)', '#f39c12')):
                    if mask_v.any():
                        fig.add_trace(go.Scatter(
                            x=frames[mask_v], y=state_ids_np[mask_v],
                            mode='markers+lines', name=lbl,
                            marker=dict(color=color, size=7), line=dict(color=color),
                            customdata=hover[mask_v],
                            hovertemplate='%{customdata}<extra></extra>'))
                fig.update_layout(yaxis=dict(title='id estado'))
                st.plotly_chart(style_fig(fig, 280, 'bossState (animação da chefe)'),
                                width='stretch')

                with st.expander("📊 Estatísticas por estado (referência para "
                                 "preencher boss_state_config.json)"):
                    rows = []
                    for sid in sorted(set(state_ids_np.tolist()) | set(reg['stats'])):
                        stx = reg['stats'].get(int(sid), {})
                        rows.append({
                            'id': int(sid), 'estado': state_label(int(sid), reg),
                            'revisado': int(sid) in reg.get('reviewed', set()),
                            'off.': int(sid) in reg['offensive'],
                            'vuln.': int(sid) in reg['vulnerable'],
                            'frames': stx.get('frames', 0), '% uso': stx.get('pct', 0.0),
                            '|bvx| méd': stx.get('mean_abs_bvx'),
                            '|bvy| méd': stx.get('mean_abs_bvy'),
                            'closing': stx.get('closing_frac'),
                            'score of.': stx.get('offensive_score'),
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
                    st.caption("Score of. = 0.5·min(|bvx|/25,1) + 0.3·min(|bvy|/30,1) "
                               "+ 0.2·closing. Edite o JSON e recarregue (F5).")

                with st.expander("🖼️ Capturas de tela dos estados (mod C#)"):
                    st.caption("O mod fotografa a tela na 1ª ocorrência de cada bossState "
                               "por sessão (boss_state_shots/, ao lado da DLL do "
                               "DataExtractor; captures_log.txt tem id→arquivo).")
                    settings = load_settings()
                    shots_dir = st.text_input(
                        "Pasta boss_state_shots",
                        settings.get('shots_dir') or find_shots_dir_default())
                    if shots_dir and os.path.isdir(shots_dir):
                        if shots_dir != settings.get('shots_dir'):
                            save_settings({'shots_dir': shots_dir})
                        ids = []
                        for p_ in sorted(glob.glob(os.path.join(shots_dir, 'state_*.png'))):
                            base = os.path.basename(p_)
                            try:
                                ids.append(int(base[len('state_'):-len('.png')]))
                            except ValueError:
                                pass
                        if not ids:
                            st.caption("Nenhum state_*.png ainda — rode uma luta com "
                                       "o mod compilado.")
                        else:
                            pick_state = st.selectbox("Estado", ids,
                                                      format_func=lambda i: state_label(i))
                            st.image(os.path.join(shots_dir, f'state_{pick_state:02d}.png'),
                                     width=640)
                            stx = reg['stats'].get(pick_state, {})
                            st.caption(f"frames: {stx.get('frames', 0)} | % uso: "
                                       f"{stx.get('pct', 0.0)} | |bvx|: {stx.get('mean_abs_bvx')} | "
                                       f"closing: {stx.get('closing_frac')} | "
                                       f"score of.: {stx.get('offensive_score')}")
                    else:
                        st.caption("Pasta não encontrada/informada. O caminho padrão é "
                                   "detectado em BepInEx\\plugins\\ depois da primeira "
                                   "luta com o mod compilado.")

            with t_frame:
                reg = get_registry_maps()
                if frame_range[0] == frame_range[1]:
                    selected_frame = frame_range[0]
                else:
                    selected_frame = st.slider("Quadro", frame_range[0], frame_range[1],
                                               frame_range[0], key="frame_pick")
                row = df.loc[selected_frame]
                i1, i2 = st.columns([1, 1.6])
                with i1:
                    st.markdown("**Estado do jogo**")
                    st.write(f"Jogador (`px`, `py`): ({row['px']:.2f}, {row['py']:.2f})")
                    st.write(f"Chefe (`bx`, `by`): ({row['bx']:.2f}, {row['by']:.2f})")
                    st.write(f"Δ: ({row['delta_x']:.2f}, {row['delta_y']:.2f}) · "
                             f"dist: {row['dist']:.2f}")
                    st.write(f"x_arena: {row['x_arena']:.2f} · θ*: {int(np.sign(row['delta_x']))}")
                    st.write(f"bossState: `{state_label(int(row['bossState']), reg)}`"
                             f"{' ⚠️ ofensivo' if int(row['bossState']) in reg['offensive'] else ''}")
                    st.write(f"HP: `{row['hp']:.0f}` · Alma: `{row['soul']:.0f}`")
                with i2:
                    st.markdown("**Ações do quadro**")
                    tbl = []
                    for j, name in enumerate(ACTION_NAMES):
                        e = int(round(float(expert_actions[selected_frame, j])))
                        entry = {'Ação': ACTION_LABELS_PT[name],
                                 'Especialista': '🟢' if e else '⚪'}
                        if has_model:
                            b = int(model_actions[selected_frame, j])
                            entry['Bot'] = '🟢' if b else '⚪'
                            if model_probs is not None:
                                entry['p'] = float(model_probs[selected_frame, j])
                            entry['Diverge'] = '⚠️' if b != e else ''
                        tbl.append(entry)
                    st.dataframe(pd.DataFrame(tbl), hide_index=True, width='stretch',
                                 column_config={'p': st.column_config.NumberColumn(format="%.3f")})
                    if not has_model:
                        st.caption("Sem colunas pred_*/prob_* (demonstração de treino).")

# ================================================================== Métricas
with tab_metrics:
    st.subheader("Métricas contextuais por sessão (plan.md §3)")

    if not selected_paths:
        st.info("Selecione sessões na barra lateral para calcular as métricas.")
    else:
        summary = load_summary(tuple(selected_paths), metric_source, d_seguro, d_ameaca,
                               d_alcance, x_limiar, h_crit, s_foco, lambda_fer, tau,
                               aim_mode)

        with np.errstate(all='ignore'):
            def _weighted(metric_col, count_col):
                if metric_col not in summary.columns or count_col not in summary.columns:
                    return None
                total = summary[count_col].sum()
                if not total > 0:
                    return None
                return round(float(np.nansum(summary[metric_col] * summary[count_col]) / total), 4)

            csr_w = _weighted('CSR', 'n_cura')
            cor_w = _weighted('COR', 'n_oportuno')
            m_cura_w = None
            if csr_w is not None and cor_w is not None and (csr_w + cor_w) > 0:
                m_cura_w = round(2 * csr_w * cor_w / (csr_w + cor_w), 4)

            agg_row = {
                'CSR': csr_w, 'COR': cor_w, 'M_cura': m_cura_w,
                'M_alinhamento': _weighted('M_alinhamento', 'n_ofensivo'),
                'M_evasao': _weighted('M_evasao', 'n_threat'),
                # plan.md §3.4: denominador do FER é |T_neutro|, não o total
                'FER': _weighted('FER', 'n_neutro'),
                'M_punicao': _weighted('M_punicao', 'n_abertura'),
                'P_canto': _weighted('P_canto', 'n_borda'),
            }
            if agg_row['M_evasao'] is not None and agg_row['FER'] is not None:
                agg_row['M_defesa'] = round(agg_row['M_evasao'] - lambda_fer * agg_row['FER'], 4)

        shown = {k: v for k, v in agg_row.items() if v is not None}
        st.markdown("**Agregado das sessões selecionadas** (média ponderada por contagem de quadros relevantes)")
        if shown:
            items = list(shown.items())
            for start in range(0, len(items), 5):
                cols = st.columns(5)
                for col, (k, v) in zip(cols, items[start:start + 5]):
                    col.metric(k, f"{v:.3f}")
        else:
            st.warning("Nenhuma janela contextual encontrada. Preencha o mapping "
                       "`boss_state_config.json` e/ou ajuste os limiares.")

        st.markdown("**Tabela por sessão**")
        st.dataframe(summary.set_index('session'), width='stretch')

        value_cols = [c for c in ['M_cura', 'M_alinhamento', 'M_evasao', 'M_defesa',
                                  'M_punicao', 'P_canto'] if c in summary.columns]
        melted = summary.melt(id_vars=['session'], value_vars=value_cols,
                              var_name='métrica', value_name='valor').dropna()
        if not melted.empty:
            if summary['session'].nunique() <= 12:
                fig = px.bar(melted, x='métrica', y='valor', color='session', barmode='group')
            else:  # muitas sessões: distribuição em vez de uma barra por sessão
                fig = px.box(melted, x='métrica', y='valor', points='all')
            st.plotly_chart(style_fig(fig, 380, 'Distribuição das métricas entre as sessões'),
                            width='stretch')

# ================================================================ Taxonomia
with tab_taxonomy:
    st.subheader("Análise de erros bottom-up (taxonomia de falhas)")
    st.caption("Assista ao replay de cada combate perdido e escolha a causa principal "
               "(plan.md §1). Cada sessão aceita uma categoria; re-classificar substitui.")

    df_tax = load_taxonomy_csv()

    losses = []
    if selected_paths:
        losses = [p for p in selected_paths if final_player_hp(load_battle(p)) <= 0]

    if not selected_paths:
        st.info("Selecione sessões na barra lateral para rotular derrotas.")
    elif not losses:
        st.success("Nenhuma sessão selecionada terminou com o jogador morto.")
    else:
        labeled_n = sum(1 for p in losses if display_session_label(p) in set(df_tax['file']))
        st.progress(labeled_n / len(losses),
                    text=f"{labeled_n} de {len(losses)} derrotas rotuladas")

        file_path = st.selectbox("Combate perdido", losses, format_func=display_session_label)
        label = display_session_label(file_path)

        suggestion = classify_loss_pattern(load_battle(file_path), config)
        st.caption(f"💡 Heurística sugere **{ERROR_LABELS_AUTO[suggestion]}** "
                   f"(últimos {TAIL_SECONDS:.0f} s). É só uma dica — decida pelo replay.")

        form_col, info_col = st.columns([2, 1])
        with form_col:
            code = st.radio("Causa principal", [c for c, _ in ERROR_TAXONOMY],
                            format_func=lambda c: ERROR_LABELS[c])
            subcat_options = [('', 'Nenhuma')]
            if code in ERROR_TAXONOMY_EXTENDED:
                subcat_options += ERROR_TAXONOMY_EXTENDED[code]['subcategories']
            subcat_map = dict(subcat_options)
            selected_subcat = st.selectbox("Subcategoria (opcional)",
                                           [s[0] for s in subcat_options],
                                           format_func=lambda x: subcat_map.get(x, 'Nenhuma'))
            notes = st.text_input("Notas (opcional)")
        with info_col:
            existing = df_tax[df_tax['file'] == label]
            if not existing.empty:
                st.info(f"Classificação atual:\n\n**{existing['cause_label'].iloc[0]}**")
            else:
                st.caption("Ainda sem classificação.")

        if st.button("💾 Salvar classificação", type="primary"):
            save_taxonomy_row({
                'file': label,
                'Model': model_from_path(file_path),
                'timestamp': pd.Timestamp.now().isoformat(),
                'cause_code': code,
                'cause_label': ERROR_LABELS[code],
                'subcategory': selected_subcat or '',
                'subcategory_label': subcat_map.get(selected_subcat, ''),
                'notes': notes,
            })
            st.success("Taxonomia atualizada em analysis/error_taxonomy.csv")
            st.rerun()

    st.divider()
    df_tax = load_taxonomy_csv()
    if df_tax.empty:
        st.info("Nenhuma derrota rotulada ainda.")
    else:
        render_pareto(df_tax, ERROR_LABELS, key='manual')
        with st.expander("Classificações salvas"):
            st.dataframe(df_tax, hide_index=True, width='stretch')

# ============================================================ Funil & Cenários
with tab_funnel:
    st.subheader("🧭 Funil de capacidades e cenários críticos")
    st.caption("plan.md §2.4 e §5: Navegação → Defesa reativa → Punição ofensiva → "
               "Gestão de recursos → Vitória. Mostra onde cada arquitetura trava.")

    if not test_files:
        st.info("Nenhum arquivo de teste disponível para análise do funil.")
    elif test_metrics_df.empty:
        st.warning("Não foi possível calcular métricas contextuais para os testes.")
    else:
        if metric_failures:
            with st.expander(f"⚠️ {len(metric_failures)} sessão(ões) falharam no cálculo"):
                st.dataframe(pd.DataFrame(metric_failures, columns=['Sessão', 'Erro']),
                             hide_index=True, width='stretch')

        funnel_results = compute_capability_funnel(test_metrics_df)
        funnel_rows = []
        for model, levels in funnel_results.items():
            for level in FUNNEL_LEVELS:
                score = levels.get(level, np.nan)
                if score is not None and not np.isnan(score):
                    funnel_rows.append({'Arquitetura': model,
                                        'Nível': FUNNEL_LABELS_PT[level],
                                        'Pontuação': float(score)})

        if funnel_rows:
            funnel_df = pd.DataFrame(funnel_rows)
            order = [FUNNEL_LABELS_PT[l] for l in FUNNEL_LEVELS]
            f1, f2 = st.columns([3, 2])
            with f1:
                fig = px.line(funnel_df, x='Nível', y='Pontuação', color='Arquitetura',
                              markers=True, category_orders={'Nível': order})
                fig.update_layout(yaxis=dict(range=[0, 1.05]), xaxis_title=None)
                st.plotly_chart(style_fig(fig, 380, 'Progresso pelo funil'), width='stretch')
            with f2:
                pivot = funnel_df.pivot(index='Arquitetura', columns='Nível',
                                        values='Pontuação').reindex(columns=order)
                fig = px.imshow(pivot, text_auto='.2f', aspect='auto',
                                color_continuous_scale='Blues', zmin=0, zmax=1)
                fig.update_layout(coloraxis_showscale=False, xaxis_title=None,
                                  yaxis_title=None)
                st.plotly_chart(style_fig(fig, 380, 'Pontuação por nível'),
                                width='stretch', key='funnel_heat')
            st.caption("Navegação = 1 − P_canto · Vitória = taxa de vitórias real das lutas.")

        # ---- cenários críticos
        st.divider()
        st.markdown("### Cenários críticos")
        scen_df, scen_failures = compute_scenario_table(
            tuple(test_files), tuple(config['attack_active']), tuple(config['vulnerable']),
            **PARAM_ARGS)
        if scen_failures:
            st.warning(f"{len(scen_failures)} sessão(ões) falharam no fatiamento de cenários.")
        if scen_df.empty:
            st.info("Nenhum quadro caiu nos cenários críticos. Verifique o "
                    "boss_state_config.json (attack_active / vulnerable).")
        else:
            scenario_summary = scen_df.groupby(['Arquitetura', 'Cenário']).agg(
                Frames=('Frames', 'sum'),
                **{k: (k, 'mean') for k in SCENARIO_METRIC_KEYS}).reset_index()
            st.dataframe(scenario_summary, hide_index=True, width='stretch',
                         column_config={k: st.column_config.NumberColumn(format="%.3f")
                                        for k in SCENARIO_METRIC_KEYS})
            melted = scenario_summary.melt(
                id_vars=['Arquitetura', 'Cenário'],
                value_vars=['M_cura', 'M_alinhamento', 'M_evasao', 'M_punicao', 'P_canto'],
                var_name='Métrica', value_name='Valor').dropna()
            if not melted.empty:
                fig = px.bar(melted, x='Cenário', y='Valor', color='Arquitetura',
                             barmode='group', facet_col='Métrica', facet_col_wrap=3)
                fig.update_xaxes(showticklabels=False, title=None)
                fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
                st.plotly_chart(style_fig(fig, 520, 'Desempenho por cenário crítico'),
                                width='stretch')
                st.caption("Eixo X = cenários A/B/C (ver tabela acima para os rótulos).")

        # ---- Pareto automático (heurístico)
        st.divider()
        st.markdown("### Pareto automático das derrotas (heurístico)")
        loss_files = tuple(agg_all.loc[agg_all['Result'] == 'LOSS', 'File'])
        if not loss_files:
            st.info("Nenhuma derrota nos testes.")
        else:
            auto_df = classify_losses(loss_files, config_mtime())
            render_pareto(auto_df, ERROR_LABELS_AUTO, key='auto')
            st.caption("Classificação automática dos últimos segundos de cada derrota. "
                       "Use como triagem e confirme manualmente na aba Taxonomia de Erros.")