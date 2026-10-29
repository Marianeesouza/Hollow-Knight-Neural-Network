"""Contextual loss functions for behavior cloning training.

Implements the unified loss from plan.md §300-373:
    L_total = (1/B) * Σ [ Ω(s_t) * (1/8) Σ ℓ_classe(z_j, y_j) + λ * R_mira(z, s) ]
"""

import numpy as np
from typing import Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ContextualBCELoss:
    """BCE loss with contextual weighting from plan.md.

    Features:
    - pos_weight for rare action classes
    - State-aware temporal weighting Ω(s_t)
    - Aim alignment regularization R_mira
    """

    def __init__(
        self,
        pos_weights: Optional[np.ndarray] = None,
        alpha_threat: float = 3.0,
        alpha_opening: float = 2.0,
        alpha_health: float = 1.5,
        lambda_alignment: float = 0.5,
    ):
        self.pos_weights = pos_weights
        self.alpha_threat = alpha_threat
        self.alpha_opening = alpha_opening
        self.alpha_health = alpha_health
        self.lambda_alignment = lambda_alignment

    def compute_pos_weights(self, labels: np.ndarray) -> np.ndarray:
        """Compute pos_weight for each action class from plan.md §309.

        p_j = (N - N_j+) / N_j+
        """
        N = labels.shape[0]
        pos_counts = labels.sum(axis=0)
        pos_weights = np.where(pos_counts > 0, (N - pos_counts) / pos_counts, 1.0)
        pos_weights = np.where(pos_counts == 0, 0.1, pos_weights)
        return pos_weights

    def compute_omega(self, states_df) -> np.ndarray:
        """Compute state criticality multiplier Ω(s_t) from plan.md §327.

        Ω(s_t) = 1 + α_threat * I(s_t ∈ T_threat) + α_opening * I(s_t ∈ T_abertura) + α_health * (1 - h_t)
        """
        from Python.Src.Utils.ContextMetrics import MetricParams, load_boss_state_config

        params = MetricParams()
        config = load_boss_state_config()

        boss_state = states_df['bossState'].to_numpy()
        dist = states_df['dist'].to_numpy()
        hp_norm = states_df['hp_norm'].to_numpy()

        # `dist_closing` não pode atravessar a fronteira entre lutas quando os
        # dados estão concatenados: se houver coluna de luta, calcula por grupo.
        group_col = next(
            (c for c in ('fight_id', 'session', 'source_file', 'file')
             if c in states_df.columns),
            None,
        )
        if group_col is not None:
            dist_prev = states_df.groupby(group_col)['dist'].shift(1)
            dist_prev = dist_prev.fillna(states_df['dist']).to_numpy()
        else:
            dist_prev = states_df['dist'].shift(1).bfill().to_numpy()
        dist_closing = (dist_prev - dist) > 0.0

        threat_mask = (
            np.isin(boss_state, config['attack_active']) &
            (dist <= params.d_ameaca) &
            dist_closing
        )

        vulnerable = np.isin(boss_state, config['vulnerable'])
        trick_state = vulnerable | np.isin(boss_state, config.get('stunned', []))
        opening_mask = trick_state & (dist <= params.d_alcance)

        omega = np.ones(len(states_df))
        omega[threat_mask] += self.alpha_threat
        omega[opening_mask] += self.alpha_opening
        omega += self.alpha_health * (1.0 - hp_norm)

        return omega

    def compute_mira_regularization(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        states_df,
    ) -> np.ndarray:
        """Compute R_mira regularization term from plan.md §345.

        R_mira = max(0, y_ofensivo * [1 - θ* * tanh(Δz_horizontal)])
        """
        delta_x = states_df['delta_x'].to_numpy()
        theta_star = np.sign(delta_x)
        theta_star = np.where(theta_star == 0, 1.0, theta_star)

        z_right = logits[:, 5]
        z_left = logits[:, 4]
        delta_z_horizontal = z_right - z_left

        # `|` bit a bit quebra com labels float; usar OR lógico explícito.
        offensive_actions = np.logical_or(
            labels[:, 3] > 0, labels[:, 2] > 0
        ).astype(float)
        tanh_term = np.tanh(delta_z_horizontal)
        alignment_term = 1.0 - theta_star * tanh_term

        r_mira = np.maximum(0.0, offensive_actions * alignment_term)
        return r_mira

    def compute_loss(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        states_df,
    ) -> dict:
        """Compute total contextual loss from plan.md §367.

        Returns dict with:
        - base_loss: unweighted BCE loss
        - weighted_loss: Ω-weighted loss
        - mira_loss: aim alignment regularization
        - total: combined loss
        """
        if self.pos_weights is None:
            self.pos_weights = self.compute_pos_weights(labels)

        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.clip(probs, 1e-7, 1 - 1e-7)

        weighted_bce = np.zeros_like(probs)
        for j in range(len(self.pos_weights)):
            pw = self.pos_weights[j]
            y = labels[:, j]
            p = probs[:, j]
            weighted_bce[:, j] = -(
                pw * y * np.log(p) + (1 - y) * np.log(1 - p)
            )

        omega = self.compute_omega(states_df)
        weighted_loss_per_frame = omega[:, np.newaxis] * weighted_bce
        weighted_loss = weighted_loss_per_frame.mean()

        mira_loss = self.compute_mira_regularization(logits, labels, states_df)
        mira_total = self.lambda_alignment * mira_loss.mean()

        total_loss = weighted_loss + mira_total

        return {
            'base_loss': float(weighted_bce.mean()),
            'weighted_loss': float(weighted_loss),
            'mira_loss': float(mira_loss.mean()),
            'omega_mean': float(omega.mean()),
            'omega_std': float(omega.std()),
            'total': float(total_loss),
            'pos_weights': self.pos_weights.tolist(),
        }

    def __call__(self, logits: np.ndarray, labels: np.ndarray, states_df):
        """PyTorch-style interface for compatibility."""
        return self.compute_loss(logits, labels, states_df)