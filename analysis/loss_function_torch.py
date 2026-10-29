"""Versão PyTorch (diferenciável) da loss contextual de plan.md §300-373.

A classe `loss_function.ContextualBCELoss` é apenas numpy e serve só para
inspecionar/depurar os componentes da loss fora de um loop de treino — ela
não pode ser usada com `loss.backward()`. Este módulo reimplementa os
mesmos três componentes (pos_weight, Ω(s_t), R_mira) em PyTorch, com
gradiente fluindo até os logits do modelo, para uso real no treino (ex.:
notebook Kaggle).

    L_total = (1/B) * Σ_t [ Ω(s_t) * (1/8) Σ_j ℓ_classe(z_j, y_j) ] + λ * R_mira

Uso típico num loop de treino:

    loss_fn = ContextualBCELossTorch(alpha_threat=3.0, alpha_opening=2.0,
                                      alpha_health=1.5, lambda_alignment=0.5)
    pos_weights = loss_fn.compute_pos_weights(train_labels_np)  # uma vez, no dataset inteiro
    loss_fn.pos_weights = torch.tensor(pos_weights, dtype=torch.float32, device=device)

    for batch in loader:
        logits = model(batch['state'])                  # (B, 8), requires_grad
        out = loss_fn(logits, batch['labels'], batch['omega_inputs'])
        out['total'].backward()

`omega_inputs` é um dict de tensores pré-computados no CPU/numpy a partir do
DataFrame de estados (bossState, dist, hp_norm, delta_x, dist_closing), já
que essas colunas não fazem parte do grafo de autograd — só os logits do
modelo precisam de gradiente. Use `prepare_omega_inputs(states_df)` para
gerar esse dict a partir de um DataFrame pandas, reaproveitando a mesma
lógica de `loss_function.py` (threat/opening/dist_closing por luta).
"""

from typing import Dict, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover - ambiente sem torch instalado
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch não está instalado. Instale com `pip install torch` "
            "para usar loss_function_torch.py (treino). Para apenas "
            "inspecionar os componentes da loss sem treinar, use "
            "loss_function.ContextualBCELoss (numpy)."
        )


def prepare_omega_inputs(states_df) -> Dict[str, np.ndarray]:
    """Pré-computa, em numpy, tudo que Ω(s_t) e R_mira precisam do estado.

    Mantém a mesma lógica de `loss_function.ContextualBCELoss.compute_omega`
    (threat/opening a partir de boss_state_config, dist_closing por luta
    quando houver coluna de agrupamento) para que a versão torch produza
    os mesmos valores que a numpy em `loss_function.py`.

    Retorna um dict de arrays numpy (float32/bool) prontos para virar
    tensores: 'threat_mask', 'opening_mask', 'hp_norm', 'delta_x'.
    """
    from Python.Src.Utils.ContextMetrics import MetricParams, load_boss_state_config

    params = MetricParams()
    config = load_boss_state_config()

    boss_state = states_df['bossState'].to_numpy()
    dist = states_df['dist'].to_numpy()
    hp_norm = states_df['hp_norm'].to_numpy()
    delta_x = states_df['delta_x'].to_numpy()

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

    return {
        'threat_mask': threat_mask.astype(np.float32),
        'opening_mask': opening_mask.astype(np.float32),
        'hp_norm': hp_norm.astype(np.float32),
        'delta_x': delta_x.astype(np.float32),
    }


if HAS_TORCH:

    class ContextualBCELossTorch(nn.Module):
        """Loss contextual diferenciável (plan.md §300-373), apta a treino.

        Parâmetros e semântica idênticos a
        `loss_function.ContextualBCELoss`, mas operando em tensores com
        autograd. `pos_weights` deve ser um tensor 1-D de tamanho 8
        (uma entrada por ação); se None, é computado sob demanda a partir
        dos próprios labels do batch (menos recomendado — prefira computar
        uma vez no dataset de treino inteiro com `compute_pos_weights`).

        Índices das ações (mesma ordem de `TelemetryDataset.ACTION_NAMES`):
        0=dash, 1=jump, 2=cast, 3=attack, 4=left, 5=right, 6=up, 7=down.
        """

        def __init__(
            self,
            pos_weights: Optional["torch.Tensor"] = None,
            alpha_threat: float = 3.0,
            alpha_opening: float = 2.0,
            alpha_health: float = 1.5,
            lambda_alignment: float = 0.5,
            normalize_omega: bool = True,
        ):
            super().__init__()
            if pos_weights is not None:
                self.register_buffer('pos_weights', pos_weights.float())
            else:
                self.pos_weights = None
            self.alpha_threat = alpha_threat
            self.alpha_opening = alpha_opening
            self.alpha_health = alpha_health
            self.lambda_alignment = lambda_alignment
            # Ω tem média > 1 por construção (1 + termos não-negativos), o
            # que infla a escala efetiva da loss/learning rate. Normalizar
            # pela média do batch deixa Ω com média ~1, preservando os
            # pesos *relativos* entre frames críticos e frames normais.
            self.normalize_omega = normalize_omega

        @staticmethod
        def compute_pos_weights(labels: np.ndarray) -> np.ndarray:
            """p_j = (N - N_j+) / N_j+ — idêntico a loss_function.py (numpy)."""
            N = labels.shape[0]
            pos_counts = labels.sum(axis=0)
            pos_weights = np.where(pos_counts > 0, (N - pos_counts) / pos_counts, 1.0)
            pos_weights = np.where(pos_counts == 0, 0.1, pos_weights)
            return pos_weights

        def compute_omega(
            self,
            threat_mask: "torch.Tensor",
            opening_mask: "torch.Tensor",
            hp_norm: "torch.Tensor",
        ) -> "torch.Tensor":
            """Ω(s_t) = 1 + α_threat*I(threat) + α_opening*I(opening) + α_health*(1-h_t)."""
            omega = (
                1.0
                + self.alpha_threat * threat_mask
                + self.alpha_opening * opening_mask
                + self.alpha_health * (1.0 - hp_norm)
            )
            if self.normalize_omega:
                omega = omega / omega.mean().clamp_min(1e-8)
            return omega

        def compute_mira_regularization(
            self,
            logits: "torch.Tensor",
            labels: "torch.Tensor",
            delta_x: "torch.Tensor",
        ) -> "torch.Tensor":
            """R_mira = max(0, y_ofensivo * [1 - θ* * tanh(Δz_horizontal)]).

            θ* = sign(Δx), com Δx=0 tratado como "chefe à direita" (+1),
            igual à versão numpy, para manter os dois caminhos consistentes.
            """
            theta_star = torch.sign(delta_x)
            theta_star = torch.where(theta_star == 0, torch.ones_like(theta_star), theta_star)

            z_right = logits[:, 5]
            z_left = logits[:, 4]
            delta_z_horizontal = z_right - z_left

            offensive_actions = ((labels[:, 3] > 0) | (labels[:, 2] > 0)).float()
            tanh_term = torch.tanh(delta_z_horizontal)
            alignment_term = 1.0 - theta_star * tanh_term

            r_mira = torch.clamp(offensive_actions * alignment_term, min=0.0)
            return r_mira

        def forward(
            self,
            logits: "torch.Tensor",
            labels: "torch.Tensor",
            omega_inputs: Dict[str, "torch.Tensor"],
        ) -> Dict[str, "torch.Tensor"]:
            """Computa a loss total e seus componentes, com gradiente em logits.

            logits: (B, 8) saída crua do modelo (sem sigmoid).
            labels: (B, 8) 0/1 (float ou long).
            omega_inputs: dict de tensores já no mesmo device de `logits`,
                com chaves 'threat_mask', 'opening_mask', 'hp_norm',
                'delta_x' (ver `prepare_omega_inputs`).
            """
            labels = labels.float()
            if self.pos_weights is None:
                pos_weights = torch.as_tensor(
                    self.compute_pos_weights(labels.detach().cpu().numpy()),
                    dtype=logits.dtype, device=logits.device,
                )
            else:
                pos_weights = self.pos_weights.to(logits.device)

            # BCEWithLogits por classe, com pos_weight por classe (estável
            # numericamente; evita sigmoid + log manual como na versão numpy).
            per_class_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weights, reduction='none',
            )  # (B, 8)

            omega = self.compute_omega(
                omega_inputs['threat_mask'].to(logits.device),
                omega_inputs['opening_mask'].to(logits.device),
                omega_inputs['hp_norm'].to(logits.device),
            )  # (B,)

            weighted_loss_per_frame = omega.unsqueeze(1) * per_class_loss  # (B, 8)
            weighted_loss = weighted_loss_per_frame.mean()

            mira_loss = self.compute_mira_regularization(
                logits, labels, omega_inputs['delta_x'].to(logits.device),
            )
            mira_total = self.lambda_alignment * mira_loss.mean()

            total_loss = weighted_loss + mira_total

            return {
                'base_loss': per_class_loss.mean(),
                'weighted_loss': weighted_loss,
                'mira_loss': mira_loss.mean(),
                'omega_mean': omega.mean().detach(),
                'omega_std': omega.std().detach(),
                'total': total_loss,
            }

        def __call__(self, logits, labels, omega_inputs):  # type: ignore[override]
            return self.forward(logits, labels, omega_inputs)

else:  # pragma: no cover

    class ContextualBCELossTorch:  # type: ignore[no-redef]
        """Stub usado quando torch não está instalado (ver `_require_torch`)."""

        def __init__(self, *args, **kwargs):
            _require_torch()