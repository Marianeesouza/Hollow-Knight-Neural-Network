"""Verificação rápida: equivalência do vetor de 104 features + smoke de métricas."""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Python'))

from Src.Data.DataHandler import DataHandler                      # noqa
from Src.Data.TelemetryDataset import load_session_csv, build_feature_matrix
from Src.Utils.ContextMetrics import (MetricParams, compute_contextual_metrics,
                                      load_boss_state_config)

CHAVES = ['attackDown', 'attackDuration', 'attackForward', 'attackUp', 'bossHp',
          'bossMaxHp', 'bossScene', 'bossState', 'bossStateAmount', 'bvx', 'bvy',
          'bx', 'by', 'canCast', 'casting', 'dashCoolDown', 'dashing',
          'doubleJumping', 'facingRight', 'falling', 'focusing', 'hp',
          'invulnerable', 'isAttacking', 'jumping', 'maxHp', 'maxSoul',
          'onGround', 'pvx', 'pvy', 'px', 'py', 'shadowDashCoolDown',
          'shadowDashing', 'soul']

CSV = os.path.join(PROJECT_ROOT, 'dataset', 'hornet_boss',
                   'demo_1780703999_hash423158243.csv')

df = load_session_csv(CSV)
print(f"Frames: {len(df)} | colunas: {len(df.columns)}")

# ---- 1) equivalência trat_data x build_feature_matrix
max_diff = 0.0
for i in range(0, len(df), 500):
    row = df.iloc[i]
    state = {}
    for key in CHAVES:
        v = row[key]
        if key in ('attackDown', 'attackForward', 'attackUp', 'canCast',
                   'casting', 'dashing', 'doubleJumping', 'facingRight',
                   'falling', 'focusing', 'invulnerable', 'isAttacking',
                   'jumping', 'onGround', 'shadowDashing'):
            v = bool(v)
        elif key in ('bossScene', 'bossState', 'bossStateAmount', 'hp',
                     'maxHp', 'soul', 'maxSoul', 'bossHp', 'bossMaxHp'):
            v = int(v)
        else:
            v = float(v)
        state[key] = v
    ref = DataHandler.treat_data(state)
    got = build_feature_matrix(df)[i]
    max_diff = max(max_diff, float(np.abs(np.asarray(ref) - got).max()))
print(f"[1] features 104: max|ref-got| = {max_diff:.2e} -> "
      f"{'OK' if max_diff < 1e-6 else 'FALHOU'}")

# ---- 2) ações do especialista Goku
from Src.Data.TelemetryDataset import extract_expert_actions
acts = extract_expert_actions(df)
print(f"[2] ações especialistas (T,8): shape={acts.shape}, "
      f"totais por canal = {acts.sum(axis=0).astype(int).tolist()}")

# ---- 3) métricas contextuais
config = load_boss_state_config()
metrics = compute_contextual_metrics(df, actions=None,
                                     params=MetricParams(), config=config,
                                     source='expert')
print("[3] métricas (mapping vazio => janelas NaN):")
for k in ('n_frames', 'n_threat', 'n_abertura', 'n_oportuno', 'n_borda',
          'n_cura', 'n_ofensivo', 'CSR', 'COR', 'M_cura', 'M_alinhamento',
          'M_evasao', 'FER', 'M_punicao', 'P_canto'):
    print(f"    {k}: {metrics[k]}")

# ---- sanity com mapping sintético (threat = todos os estados >= 14)
configFake = {'attack_active': [7, 8, 9, 14, 15, 16], 'vulnerable': [14, 15],
              'stunned': [16]}
m2 = compute_contextual_metrics(df, actions=None, config=configFake)
print(f"[3b] com mapping sintético: n_threat={m2['n_threat']}, "
      f"n_abertura={m2['n_abertura']}, M_alinhamento={m2['M_alinhamento']:.3f}, "
      f"P_canto={m2['P_canto']:.3f}")
