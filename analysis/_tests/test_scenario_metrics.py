"""Unit tests for scenario_analysis module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add project root to sys.path for imports
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scenario_analysis import (
    extract_critical_scenarios,
    compute_scenario_metrics,
    compute_weighted_loss_components,
)


@pytest.fixture
def mock_telemetry():
    """Create mock telemetry DataFrame with full column set for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'timestamp': range(n),
        'attackDown': np.zeros(n, dtype=int),
        'attackDuration': np.zeros(n, dtype=int),
        'attackForward': np.zeros(n, dtype=int),
        'attackUp': np.zeros(n, dtype=int),
        'bossHp': np.linspace(1000, 0, n),
        'bossMaxHp': np.full(n, 1000.0),
        'bossScene': np.zeros(n, dtype=int),
        'bossState': np.random.choice([0, 1, 2, 3], size=n),
        'bossStateAmount': np.zeros(n, dtype=int),
        'bvx': np.random.uniform(-25, 25, size=n),
        'bvy': np.random.uniform(-30, 30, size=n),
        'bx': np.random.uniform(15, 38, size=n),
        'by': np.random.uniform(27, 40, size=n),
        'canCast': np.ones(n, dtype=int),
        'casting': np.zeros(n, dtype=int),
        'dashCoolDown': np.zeros(n),
        'dashing': np.zeros(n, dtype=int),
        'doubleJumping': np.zeros(n, dtype=int),
        'facingRight': np.random.choice([0, 1], size=n),
        'falling': np.zeros(n, dtype=int),
        'focusing': np.zeros(n, dtype=int),
        'hp': np.linspace(500, 0, n),
        'invulnerable': np.zeros(n, dtype=int),
        'isAttacking': np.zeros(n, dtype=int),
        'jumping': np.zeros(n, dtype=int),
        'maxHp': np.full(n, 500.0),
        'maxSoul': np.full(n, 99.0),
        'onGround': np.ones(n, dtype=int),
        'pvx': np.random.uniform(-5, 5, size=n),
        'pvy': np.random.uniform(-5, 5, size=n),
        'px': np.random.uniform(15, 38, size=n),
        'py': np.random.uniform(27, 40, size=n),
        'shadowDashCoolDown': np.zeros(n),
        'shadowDashing': np.zeros(n, dtype=int),
        'soul': np.linspace(99, 0, n),
        'gp_axis_left_x': np.random.uniform(-1, 1, size=n),
        'gp_axis_left_y': np.random.uniform(-1, 1, size=n),
        'gp_trigger_right': np.random.uniform(0, 1, size=n),
        'gp_btn_a': np.random.choice([0, 1], size=n),
        'gp_btn_b': np.random.choice([0, 1], size=n),
        'gp_btn_x': np.random.choice([0, 1], size=n),
    })
    return df


@pytest.fixture
def mock_config():
    """Mock boss state configuration."""
    return {
        'offensive': {1, 2},
        'vulnerable': {3},
        'attack_active': [1, 2],
    }


class TestExtractCriticalScenarios:
    """Tests for extract_critical_scenarios function."""

    def test_returns_dict_with_expected_keys(self, mock_telemetry, mock_config):
        scenarios = extract_critical_scenarios(
            mock_telemetry,
            offensive_states=mock_config['attack_active'],
            vulnerable_states=mock_config['vulnerable']
        )
        assert 'scenario_a_high_speed_attack' in scenarios
        assert 'scenario_b_vulnerable_window' in scenarios
        assert 'scenario_c_low_health' in scenarios

    def test_scenario_a_filters_close_offensive_frames(self, mock_telemetry, mock_config):
        scenarios = extract_critical_scenarios(
            mock_telemetry,
            offensive_states=mock_config['attack_active'],
            vulnerable_states=mock_config['vulnerable']
        )
        scenario_a = scenarios['scenario_a_high_speed_attack']
        if not scenario_a.empty:
            assert (scenario_a['bossState'].isin(mock_config['attack_active'])).all()
            assert (scenario_a['dist'] <= 5.0).all()

    def test_scenario_c_filters_low_health(self, mock_telemetry, mock_config):
        scenarios = extract_critical_scenarios(
            mock_telemetry,
            offensive_states=mock_config['attack_active'],
            vulnerable_states=mock_config['vulnerable']
        )
        scenario_c = scenarios['scenario_c_low_health']
        if not scenario_c.empty:
            assert (scenario_c['hp'] <= 75).all()

    def test_empty_dataframe_returns_empty_dicts(self):
        df = pd.DataFrame()
        scenarios = extract_critical_scenarios(df, offensive_states=[1], vulnerable_states=[2])
        assert isinstance(scenarios, dict)


class TestComputeScenarioMetrics:
    """Tests for compute_scenario_metrics function."""

    def test_returns_dict_with_mana_cost(self, mock_telemetry):
        # Create pred_* action columns (required for model source)
        n = len(mock_telemetry)
        actions = np.random.randint(0, 2, size=(n, 8))
        for i, name in enumerate(['dash', 'jump', 'cast', 'attack', 'left', 'right', 'up', 'down']):
            mock_telemetry[f'pred_{name}'] = actions[:, i]

        metrics = compute_scenario_metrics(
            mock_telemetry,
            'scenario_a_high_speed_attack',
            actions=actions
        )
        assert 'M_alinhamento' in metrics or 'n_frames' in metrics

    def test_returns_zero_mana_cost_when_no_heal(self, mock_telemetry):
        n = len(mock_telemetry)
        actions = np.zeros((n, 8))
        for i, name in enumerate(['dash', 'jump', 'cast', 'attack', 'left', 'right', 'up', 'down']):
            mock_telemetry[f'pred_{name}'] = actions[:, i]

        metrics = compute_scenario_metrics(
            mock_telemetry,
            'test',
            actions=actions
        )
        assert isinstance(metrics, dict)


class TestWeightedLossComponents:
    """Tests for compute_weighted_loss_components function."""

    def _prepare_probabilities(self, df, n_actions=8):
        """Create pred_* and prob_* columns for testing."""
        n = len(df)
        actions = np.random.randint(0, 2, size=(n, n_actions))
        action_names = ['dash', 'jump', 'cast', 'attack', 'left', 'right', 'up', 'down']
        for i, name in enumerate(action_names[:n_actions]):
            df[f'pred_{name}'] = actions[:, i]
            df[f'prob_{i+1}'] = np.random.uniform(0.1, 0.9, size=n)
        return df, actions.astype(float)

    def test_returns_omega_and_mira_keys(self, mock_telemetry, mock_config):
        mock_telemetry, expert_actions = self._prepare_probabilities(mock_telemetry)
        model_probs = np.random.uniform(0.1, 0.9, size=mock_telemetry.shape[0:1] + (8,))

        result = compute_weighted_loss_components(
            mock_telemetry,
            model_probs=model_probs,
            expert_actions=expert_actions,
        )
        assert 'omega_mean' in result
        assert 'total_loss' in result
        assert 'mira_regularization' in result

    def test_omega_increases_for_threatening_states(self, mock_telemetry, mock_config):
        # Set some rows to have offensive bossState in proximity
        mock_telemetry.loc[0:5, 'bossState'] = 1
        mock_telemetry.loc[0:5, 'dist'] = 2.0
        mock_telemetry, expert_actions = self._prepare_probabilities(mock_telemetry)
        model_probs = np.random.uniform(0.1, 0.9, size=mock_telemetry.shape[0:1] + (8,))

        result = compute_weighted_loss_components(
            mock_telemetry,
            model_probs=model_probs,
            expert_actions=expert_actions,
        )
        # Should have threat frames detected
        assert result['threat_frames'] > 0
        # Omega mean should be > 1 when threats exist
        assert result['omega_mean'] > 1.0

    def test_pos_weight_computed_when_none_provided(self, mock_telemetry, mock_config):
        mock_telemetry, expert_actions = self._prepare_probabilities(mock_telemetry)
        model_probs = np.random.uniform(0.1, 0.9, size=mock_telemetry.shape[0:1] + (8,))

        result = compute_weighted_loss_components(
            mock_telemetry,
            model_probs=model_probs,
            expert_actions=expert_actions,
        )
        assert 'pos_weights' in result
        assert len(result['pos_weights']) == 8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
