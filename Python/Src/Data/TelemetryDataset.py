"""Camada de dados compartilhada entre o treinamento, o bot de testes e o viewer.

Replica exatamente o pipeline de features de ``DataHandler.treat_data`` (104
features) de forma vetorizada, e unifica o acesso aos CSVs de demonstração
(experiente) e de testes do bot (com colunas pred_*/prob_* extras).
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- Constantes

STATE_KEYS = [
    'attackDown', 'attackDuration', 'attackForward', 'attackUp',
    'bossHp', 'bossMaxHp', 'bossScene', 'bossState', 'bossStateAmount',
    'bvx', 'bvy', 'bx', 'by',
    'canCast', 'casting', 'dashCoolDown', 'dashing', 'doubleJumping',
    'facingRight', 'falling', 'focusing', 'hp', 'invulnerable',
    'isAttacking', 'jumping', 'maxHp', 'maxSoul', 'onGround',
    'pvx', 'pvy', 'px', 'py', 'shadowDashCoolDown', 'shadowDashing', 'soul',
]

GP_KEYS = [
    'gp_axis_left_x', 'gp_axis_left_y', 'gp_trigger_right',
    'gp_btn_a', 'gp_btn_b', 'gp_btn_x',
]

# Ações na ordem formal do plan.md (J = 8):
# j1 dash | j2 pulo | j3 cura/feitiço | j4 ataque | j5 esq | j6 dir | j7 cima | j8 baixo
ACTION_NAMES = ['dash', 'jump', 'cast', 'attack', 'left', 'right', 'up', 'down']

PRED_KEYS = [f'pred_{name}' for name in ACTION_NAMES]
PROB_KEYS = [f'prob_{i}' for i in range(1, 9)]

BOOL_KEYS = [
    'attackDown', 'attackForward', 'attackUp',
    'canCast', 'casting', 'dashing', 'doubleJumping', 'facingRight',
    'falling', 'focusing', 'invulnerable', 'isAttacking', 'jumping',
    'onGround', 'shadowDashing',
]

# Limites físicos da arena / velocidades (iguais a DataHandler.py)
MIN_POSITION_X = 15.27
MAX_POSITION_X = 37.74
MIN_POSITION_Y = 28.41
MAX_POSITION_Y = 40
MAX_PLAYER_VELOCITY_X = 20
MAX_PLAYER_VELOCITY_Y = 20.95
MAX_BOSS_VELOCITY_X = 29.37
MAX_BOSS_VELOCITY_Y = 41

RANGE_X = MAX_POSITION_X - MIN_POSITION_X
RANGE_Y = MAX_POSITION_Y - MIN_POSITION_Y

NUM_FEATURES = 104
BOSS_STATE_DEFAULT_WIDTH = 77  # bossStateAmount constante nos datasets capturados


# ------------------------------------------------------------------ Leitura

def load_session_csv(path, add_features=True):
    """Carrega um CSV de sessão (treino ou teste) em um DataFrame homogêneo.

    Colunas ausentes (gp_*, pred_*, prob_*) são preenchidas com 0 para que o
    schema seja sempre completo; presença delas é sinalizada em ``attrs``.
    """
    df = pd.read_csv(path)

    original_columns = set(df.columns)
    has_model_outputs = any(f'prob_{i}' in original_columns for i in range(1, 9))

    for col in STATE_KEYS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    for col in GP_KEYS + PRED_KEYS + PROB_KEYS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    df.attrs['path'] = str(path)
    df.attrs['has_expert_actions'] = bool(df[GP_KEYS].abs().max().max() > 0)
    df.attrs['has_model_outputs'] = has_model_outputs

    if add_features:
        add_derived_columns(df)

    return df


def load_sessions(paths, add_features=True):
    """Carrega múltiplos arquivos; retorna lista de DataFrames."""
    if isinstance(paths, str):
        paths = [paths]
    return [load_session_csv(p, add_features=add_features) for p in paths]


def discover_sessions(root=None, dataset_dir=None, test_dir=None):
    """Descobre CSVs de treino e de testes do bot nas pastas padrão."""
    import glob
    root = root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_dir = dataset_dir or os.path.join(root, 'dataset', 'hornet_boss')
    test_dir = test_dir or os.path.join(root, 'dataset', 'test_runs')

    sessions = []
    if os.path.isdir(dataset_dir):
        sessions += sorted(glob.glob(os.path.join(dataset_dir, 'demo_*.csv')))
    if os.path.isdir(test_dir):
        sessions += sorted(glob.glob(os.path.join(test_dir, '**', '*.csv'), recursive=True))
    return sessions


# ------------------------------------------------------------- Derivações

def add_derived_columns(df):
    """Adiciona colunas derivadas usadas por métricas e pelo viewer.

    delta_x = bx - px ; delta_y = by - py
    dist    = |Delta p| (unidades do jogo)
    dist_closing = True quando a distância está diminuindo (chefe se aproximando)
    theta_star   = sgn(delta_x): orientação requerida em direção ao chefe
    x_arena      = px normalizado para [-1, 1] nos limites da arena
    hp_norm / soul_norm / boss_hp_norm: frações normalizadas
    """
    dx = df['bx'] - df['px']
    dy = df['by'] - df['py']
    df['delta_x'] = dx
    df['delta_y'] = dy
    df['dist'] = np.sqrt(dx ** 2 + dy ** 2)
    dist_prev = df['dist'].shift(1)
    df['dist_closing'] = (dist_prev - df['dist']) > 0.0
    df['theta_star'] = np.sign(dx)

    px = df['px']
    df['x_arena'] = (px - MIN_POSITION_X) / RANGE_X * 2.0 - 1.0

    max_hp = df['maxHp'].replace(0.0, np.nan)
    df['hp_norm'] = (df['hp'] / max_hp).fillna(0.0)

    max_soul = df['maxSoul'].replace(0.0, np.nan)
    df['soul_norm'] = (df['soul'] / max_soul).fillna(0.0)

    max_boss_hp = df['bossMaxHp'].replace(0.0, np.nan)
    df['boss_hp_norm'] = (df['bossHp'] / max_boss_hp).fillna(0.0)

    return df


# ------------------------------------------------------- Ações (especialista x modelo)

def extract_expert_actions(df):
    """Converte colunas gp_* em matriz binária (T, 8) na ordem de ACTION_NAMES."""
    actions = np.column_stack([
        (df['gp_trigger_right'] >= 0.5).to_numpy(dtype=np.float32),
        (df['gp_btn_a'] >= 0.5).to_numpy(dtype=np.float32),
        (df['gp_btn_b'] >= 0.5).to_numpy(dtype=np.float32),
        (df['gp_btn_x'] >= 0.5).to_numpy(dtype=np.float32),
        (df['gp_axis_left_x'] <= -0.5).to_numpy(dtype=np.float32),
        (df['gp_axis_left_x'] >= 0.5).to_numpy(dtype=np.float32),
        (df['gp_axis_left_y'] >= 0.5).to_numpy(dtype=np.float32),
        (df['gp_axis_left_y'] <= -0.5).to_numpy(dtype=np.float32),
    ])
    return actions


def extract_model_actions(df):
    """Matriz binária (T, 8) das predições do bot (colunas pred_*).

    Retorna None se não houver colunas pred_* preenchidas."""
    if not all(f'pred_{name}' in df.columns for name in ACTION_NAMES):
        return None
    values = df[PRED_KEYS].to_numpy(dtype=np.float32)
    if not values.any():
        return None
    return values


def extract_model_probabilities(df):
    """Probabilidades sigmoides (T, 8) do bot (colunas prob_1..prob_8)."""
    if not all(f'prob_{i}' in df.columns for i in range(1, 9)):
        return None
    values = df[PROB_KEYS].to_numpy(dtype=np.float32)
    if not values.any():
        return None
    return values


# ------------------------------------------------- Vetor de 104 features

_FIXED_TAIL_BOOL = [
    'attackDown', 'attackForward', 'attackUp',       # 3
    'canCast', 'casting', 'dashing', 'doubleJumping',
    'facingRight', 'falling', 'focusing', 'invulnerable',
    'isAttacking', 'jumping', 'onGround', 'shadowDashing',  # +12 = 15 bools
]


def build_feature_matrix(df):
    """Matriz (T, 104) equivalente linha a linha a ``DataHandler.treat_data``.

    Ordem: 2 posição relativa + 2 velocidade relativa + 5 escalares
    (attackDown/attackDuration/attackForward/attackUp/bossHp) + one-hot
    bossState + 18 escalares restantes.
    """
    n = len(df)

    # Posição relativa normalizada
    pos_dx = ((df['bx'] - df['px']) / RANGE_X).clip(-1.0, 1.0).to_numpy()
    pos_dy = ((df['by'] - df['py']) / RANGE_Y).clip(-1.0, 1.0).to_numpy()

    # Velocidade relativa normalizada
    vel_dx = ((df['bvx'] - df['pvx']) /
              (MAX_BOSS_VELOCITY_X + MAX_PLAYER_VELOCITY_X)).to_numpy()
    vel_dy = ((df['bvy'] - df['pvy']) /
              (MAX_BOSS_VELOCITY_Y + MAX_PLAYER_VELOCITY_Y)).to_numpy()

    blocks = [pos_dx, pos_dy, vel_dx, vel_dy]

    # Escalares antes do one-hot
    blocks.append(df['attackDown'].to_numpy(dtype=float))
    blocks.append(df['attackDuration'].to_numpy(dtype=float))
    blocks.append(df['attackForward'].to_numpy(dtype=float))
    blocks.append(df['attackUp'].to_numpy(dtype=float))
    blocks.append((df['bossHp'] / df['bossMaxHp'].replace(0.0, np.nan))
                  .fillna(0.0).to_numpy(dtype=float))

    # One-hot do bossState
    width = int(round(max(BOSS_STATE_DEFAULT_WIDTH, df['bossStateAmount'].max())))
    state_id = df['bossState'].to_numpy(dtype=int)
    one_hot = np.zeros((n, width), dtype=float)
    valid = (state_id >= 0) & (state_id < width)
    one_hot[valid, state_id[valid]] = 1.0
    blocks.append(one_hot)

    # Escalares/booleanos após o one-hot (ordem alfabética de CHAVES_TREINO)
    blocks += [
        df['canCast'].to_numpy(dtype=float),
        df['casting'].to_numpy(dtype=float),
        df['dashCoolDown'].to_numpy(dtype=float),
        df['dashing'].to_numpy(dtype=float),
        df['doubleJumping'].to_numpy(dtype=float),
        df['facingRight'].to_numpy(dtype=float),
        df['falling'].to_numpy(dtype=float),
        df['focusing'].to_numpy(dtype=float),
        (df['hp'] / df['maxHp'].replace(0.0, np.nan)).fillna(0.0).to_numpy(dtype=float),
        df['invulnerable'].to_numpy(dtype=float),
        df['isAttacking'].to_numpy(dtype=float),
        df['jumping'].to_numpy(dtype=float),
        df['onGround'].to_numpy(dtype=float),
        (df['pvx'] / MAX_PLAYER_VELOCITY_X).to_numpy(dtype=float),
        (df['pvy'] / MAX_PLAYER_VELOCITY_Y).to_numpy(dtype=float),
        df['shadowDashCoolDown'].to_numpy(dtype=float),
        df['shadowDashing'].to_numpy(dtype=float),
        (df['soul'] / df['maxSoul'].replace(0.0, np.nan)).fillna(0.0).to_numpy(dtype=float),
    ]

    features = np.concatenate([b.reshape(n, -1) for b in blocks], axis=1)
    assert features.shape[1] == NUM_FEATURES, \
        f"Esperado {NUM_FEATURES} features, obtido {features.shape[1]}"
    return features


def session_summary_label(path):
    """Rótulo curto para o viewer, ex.: 'treino/hornet demo_1780703999'."""
    path = os.path.normpath(str(path))
    parts = path.split(os.sep)
    origin = 'treino' if 'hornet_boss' in parts else 'teste'
    return f"[{origin}] {parts[-1]}"


def final_player_hp(df):
    """Último HP registrado da sessão (0 => derrota)."""
    return float(df['hp'].iloc[-1])
