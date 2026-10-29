"""Constantes compartilhadas entre os pipelines (RL, BC, gravação e análise)."""
import os

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PYTHON_DIR)

CHECKPOINTS_DIR = os.path.join(PYTHON_DIR, 'Checkpoints')
WEIGHTS_DIR = os.path.join(CHECKPOINTS_DIR, 'Weights')
BC_WEIGHTS_DIR = os.path.join(CHECKPOINTS_DIR, 'BC_weights')
STATISTIC_DIR = os.path.join(CHECKPOINTS_DIR, 'Statistic')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'hornet_boss')

BOSS_SCENE_HASH = 423158243  # Hornet (Hall of Gods)

# Lista oficial de chaves vinda do treino do Kaggle (ordem importa)
CHAVES_TREINO = [
    'attackDown', 'attackDuration', 'attackForward', 'attackUp', 'bossHp',
    'bossMaxHp', 'bossScene', 'bossState', 'bossStateAmount', 'bvx', 'bvy',
    'bx', 'by', 'canCast', 'casting', 'dashCoolDown', 'dashing',
    'doubleJumping', 'facingRight', 'falling', 'focusing', 'hp',
    'invulnerable', 'isAttacking', 'jumping', 'maxHp', 'maxSoul',
    'onGround', 'pvx', 'pvy', 'px', 'py', 'shadowDashCoolDown',
    'shadowDashing', 'soul',
]

BC_NUM_FEATURES = 104  # dimensão do vetor de DataHandler.treat_data
BC_NUM_ACTIONS = 8
BC_NUM_FRAMES = 10
BC_THRESHOLD = 0.2
