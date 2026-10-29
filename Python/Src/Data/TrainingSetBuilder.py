"""Gera o export de arrays para re-treinamento BC com a perda contextual do plan.md.

Saída (NPZ + manifest):
 - X            (N, 104): features idênticas a DataHandler.treat_data
 - y            (N, 8)  : ações binárias do especialista (ordem ACTION_NAMES)
 - pos_weight   (8,)    : (N - N_j+) / N_j+
 - omega        (N,)    : pesos de criticidade do frame Omega(s_t)
 - theta_star   (N,)    : sgn(delta_x) p/ termo de regularização R_mira
 - y_offensive  (N,)    : y_4 | y_3 (máscara do R_mira)
 - threat_mask / abertura_mask / hp_norm / soul_norm
 - file_start   (S+1,)  : offsets por arquivo, p/ re-dividir treino/teste por sessão
"""

import json
import os

import numpy as np

from .TelemetryDataset import ACTION_NAMES, build_feature_matrix, extract_expert_actions, load_session_csv
from ..Utils.ContextMetrics import MetricParams, build_masks, load_boss_state_config


class RetrainParams:
    """Pesos da função de perda Omega do plan.md (seção 2)."""

    def __init__(self, alpha_ameaca=4.0, alpha_abertura=2.0, alpha_vida=1.0):
        self.alpha_ameaca = alpha_ameaca
        self.alpha_abertura = alpha_abertura
        self.alpha_vida = alpha_vida


def build_training_arrays(dataset_dir, retrain_params=None, metric_params=None,
                          config=None):
    """Concatena todas as demonstrações da pasta em arrays de treino."""
    retrain_params = retrain_params or RetrainParams()
    metric_params = metric_params or MetricParams()
    config = config or load_boss_state_config()

    files = sorted(
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith('.csv')
    )
    if not files:
        raise FileNotFoundError(f"Nenhum CSV em {dataset_dir}")

    X_list, y_list = [], []
    omega_list, theta_list, y_off_list = [], [], []
    threat_list, abertura_list = [], []
    hp_list, soul_list = [], []

    meta_files = []
    file_start = [0]
    total = 0

    for path in files:
        df = load_session_csv(path, add_features=True)

        X = build_feature_matrix(df)
        y = extract_expert_actions(df)

        masks = build_masks(df, metric_params, config)
        threat = masks['threat'].to_numpy(dtype=bool)
        abertura = masks['abertura'].to_numpy(dtype=bool)
        theta_star = np.sign((df['bx'] - df['px']).to_numpy())
        y_offensive = ((y[:, 3] > 0.5) | (y[:, 2] > 0.5)).astype(np.float32)

        hp_norm = df['hp_norm'].to_numpy()
        soul_norm = df['soul_norm'].to_numpy()

        omega = (
            1.0
            + retrain_params.alpha_ameaca * threat.astype(np.float32)
            + retrain_params.alpha_abertura * abertura.astype(np.float32)
            + retrain_params.alpha_vida * np.clip(1.0 - hp_norm, 0.0, 1.0)
        )

        X_list.append(X)
        y_list.append(y)
        omega_list.append(omega)
        theta_list.append(theta_star)
        y_off_list.append(y_offensive)
        threat_list.append(threat)
        abertura_list.append(abertura)
        hp_list.append(hp_norm)
        soul_list.append(soul_norm)
        meta_files.append(os.path.basename(path))

        total += len(df)
        file_start.append(total)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    n_pos = y.sum(axis=0)
    pos_weight = np.where(
        n_pos > 0,
        (len(y) - n_pos) / np.maximum(n_pos, 1),
        1.0,
    ).astype(np.float32)

    return {
        'X': X.astype(np.float32),
        'y': y.astype(np.float32),
        'pos_weight': pos_weight,
        'omega': np.concatenate(omega_list).astype(np.float32),
        'theta_star': np.concatenate(theta_list).astype(np.float32),
        'y_offensive': np.concatenate(y_off_list).astype(np.float32),
        'threat_mask': np.concatenate(threat_list),
        'abertura_mask': np.concatenate(abertura_list),
        'hp_norm': np.concatenate(hp_list).astype(np.float32),
        'soul_norm': np.concatenate(soul_list).astype(np.float32),
        'file_start': np.array(file_start, dtype=np.int64),
        'files': meta_files,
        'action_names': list(ACTION_NAMES),
    }


def export_training_arrays(dataset_dir, output_dir, retrain_params=None,
                           metric_params=None, config=None):
    """Constrói os arrays e grava ``bc_training_arrays.npz`` + ``manifest.json``."""
    data = build_training_arrays(dataset_dir, retrain_params=retrain_params,
                                 metric_params=metric_params, config=config)
    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(output_dir, 'bc_training_arrays.npz')
    np.savez_compressed(
        npz_path,
        X=data['X'],
        y=data['y'],
        pos_weight=data['pos_weight'],
        omega=data['omega'],
        theta_star=data['theta_star'],
        y_offensive=data['y_offensive'],
        threat_mask=data['threat_mask'],
        abertura_mask=data['abertura_mask'],
        hp_norm=data['hp_norm'],
        soul_norm=data['soul_norm'],
        file_start=data['file_start'],
    )

    manifest = {
        'npz': os.path.basename(npz_path),
        'dataset_dir': dataset_dir,
        'n_frames': int(len(data['X'])),
        'n_sessions': len(data['files']),
        'files': data['files'],
        'action_names': data['action_names'],
        'pos_weight': {name: float(w) for name, w
                       in zip(data['action_names'], data['pos_weight'])},
        'threat_frames': int(data['threat_mask'].sum()),
        'abertura_frames': int(data['abertura_mask'].sum()),
    }
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest
