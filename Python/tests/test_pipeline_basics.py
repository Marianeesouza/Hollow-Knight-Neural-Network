import os
from collections import deque

import numpy as np
import pytest

from Src.config import (BC_NUM_FEATURES, BC_NUM_FRAMES, CHAVES_TREINO, DATASET_DIR,
                        BC_WEIGHTS_DIR)
from Src.Data.DataHandler import DataHandler


BOOLS = {'attackDown', 'attackForward', 'attackUp', 'canCast', 'casting', 'dashing',
         'doubleJumping', 'facingRight', 'falling', 'focusing', 'invulnerable',
         'isAttacking', 'jumping', 'onGround', 'shadowDashing'}
BOSS_STATE_AMOUNT = 50


def _state(boss_state=3):
    s = {k: (False if k in BOOLS else 0.0) for k in CHAVES_TREINO}
    s.update(bossState=boss_state, bossStateAmount=BOSS_STATE_AMOUNT)
    return s


def test_chaves_treino_sem_duplicatas_e_com_35_itens():
    assert len(CHAVES_TREINO) == 35
    assert len(set(CHAVES_TREINO)) == len(CHAVES_TREINO)


def test_treat_data_vetor_finito_e_dimensao_estavel():
    a = DataHandler.treat_data(_state(boss_state=3))
    b = DataHandler.treat_data(_state(boss_state=7))
    assert a.shape == b.shape and np.isfinite(a).all()
    assert not np.array_equal(a, b)  # one-hot do bossState muda


def test_treat_data_one_hot_do_boss_state():
    base = DataHandler.treat_data(_state(boss_state=0))
    assert (base == 1.0).sum() >= 1


def test_treat_data_vazio():
    assert DataHandler.treat_data({}).size == 0


def test_treat_data_posicao_relativa_limitada():
    s = _state()
    s.update(px=0.0, bx=1000.0)
    data = DataHandler.treat_data(s)
    assert -1.0 <= data[0] <= 1.0


@pytest.fixture
def tests_module():
    torch = pytest.importorskip('torch')
    import main_tests
    return main_tests, torch


def test_actions_to_pad(tests_module):
    mt, _ = tests_module
    assert mt.actions_to_pad([0] * 8) == []
    assert mt.actions_to_pad([1, 0, 0, 1, 0, 1, 0, 0]) == [6, 7, 4]


def test_build_input_tensor_formas(tests_module):
    mt, torch = tests_module
    data = np.zeros(BC_NUM_FEATURES, dtype=np.float32)
    stack = deque(maxlen=BC_NUM_FRAMES)
    t = mt.build_input_tensor("GRU", data, stack, torch.device('cpu'))
    assert tuple(t.shape) == (1, BC_NUM_FRAMES, BC_NUM_FEATURES)
    t = mt.build_input_tensor("MLP", data, stack, torch.device('cpu'))
    assert tuple(t.shape) == (1, BC_NUM_FEATURES)


def test_build_model_invalido(tests_module):
    mt, _ = tests_module
    with pytest.raises(ValueError):
        mt.build_model("XYZ")


@pytest.mark.parametrize("tipo", ["CNN", "MLP", "MLP_TEMPORAL", "GRU"])
def test_checkpoints_carregam(tests_module, tipo):
    mt, torch = tests_module
    modelo, caminho = mt.build_model(tipo)
    if not os.path.exists(caminho):
        pytest.skip("checkpoint ausente")
    modelo.load_state_dict(torch.load(caminho, map_location='cpu', weights_only=True))
