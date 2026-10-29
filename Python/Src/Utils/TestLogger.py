"""Logger de lados puro por frame para as rodadas de teste do bot.

Grava cada combate em um CSV com o MESMO schema das demonstrações de treino
(timestamp + telemetria + gp_*), acrescentando as colunas de saída do modelo
(pred_* binárias em tau e prob_* probabilidades). Este log puro habilita:
 - métricas contextuais equivalentes à especialista (aba Métricas do viewer);
 - futuro fine-tuning com dados do comportamento do próprio bot;
 - inspeção frame a frame hesitação / divergência no Data Viewer.
"""

import csv
import os
import time
from datetime import datetime

from ..Data.TelemetryDataset import (
    ACTION_NAMES, GP_KEYS, PROB_KEYS, PRED_KEYS, STATE_KEYS,
)

# Âncora na raiz do repositório (independe do diretório de execução):
# .../repo/Python/Src/Utils/TestLogger.py → 4 níveis acima = raiz.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

AGG_CSV = os.path.join(_REPO_ROOT, 'all_model_tests.csv')

AGG_HEADER = [
    'Timestamp', 'Model', 'Result', 'Survival_Time_Sec', 'Damage_Dealt',
    'Total_Nail_Hits', 'Nail_Accuracy', 'Final_Bot_HP', 'Hits_Taken',
    'Spells_Casted', 'Spell_Hits', 'Successful_Heals', 'Dashes_Executed',
    'Successful_Dodges', 'Complex_Behaviors',
]


class TestLogger:
    """Recorder por frame + contadores automáticos de combate.

    Ciclo de uso (em main_tests.py):
        logger.start_battle(state)
        for frame: logger.log_frame(state, acoes_binarias, probs)
        logger.end_battle(state)
    """

    def __init__(self, model_type, boss_scene_hash=423158243, threshold=0.2,
                 out_root=os.path.join(_REPO_ROOT, 'dataset', 'test_runs')):
        self.model_type = model_type
        self.boss_scene_hash = boss_scene_hash
        self.threshold = threshold
        self.out_dir = os.path.join(out_root, model_type.lower())
        os.makedirs(self.out_dir, exist_ok=True)

        self.csv_file = None
        self.file_handle = None
        self.writer = None
        self.n_frames = 0

        self.start_time = None
        self.last_hp_player = None
        self.last_hp_boss = None

        self.hits_taken = 0
        self.hits_made = 0
        self.hit_attempts = 0
        self.dashes = 0
        self.spells_casted = 0
        self.spell_hits = 0
        self.heal_success = 0

        self.was_attack_pressed = False
        self.was_dash_pressed = False
        self.post_spell_frames = 0
        self.post_heal_frames = 0
        self._last_soul = None

    # ------------------------------------------------------------ ciclo

    def start_battle(self, state):
        self._reset_counters()
        filename = f"demo_{int(time.time())}_hash{self.boss_scene_hash}.csv"
        path = os.path.join(self.out_dir, filename)
        seq = 2
        while os.path.exists(path):  # lutas no mesmo segundo não se sobrescrevem
            path = os.path.join(self.out_dir,
                                f"demo_{int(time.time())}_{seq}_hash{self.boss_scene_hash}.csv")
            seq += 1
        self.csv_file = path
        self.file_handle = open(self.csv_file, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file_handle)
        self.writer.writerow(['timestamp', *STATE_KEYS, *GP_KEYS,
                              *PRED_KEYS, *PROB_KEYS])
        self.n_frames = 0

        self.start_time = time.time()
        self.last_hp_player = state.get('hp', 0)
        self.last_hp_boss = state.get('bossHp', 0)

        print(f"[TestLogger] Gravando luta em: {self.csv_file}")

    def log_frame(self, state, binary_actions=None, probs=None):
        if self.writer is None:
            return

        gp = self.binary_to_gamepad(binary_actions)
        preds = list(binary_actions) if binary_actions is not None else [0] * 8
        prob_list = list(probs) if probs is not None else [0.0] * 8

        row = [f"{time.time():.6f}"]
        for key in STATE_KEYS:
            value = state.get(key, 0.0)
            if isinstance(value, bool):
                value = int(value)
            row.append(value)
        row += [gp.get(key, 0.0) for key in GP_KEYS]
        row += [int(v) for v in preds]
        row += [f"{float(p):.4f}" for p in prob_list]

        self.writer.writerow(row)
        self.n_frames += 1

        self._update_counters(state, binary_actions)

    def end_battle(self, state):
        """Fecha o CSV da luta e grava o resumo. Retorna 'WIN' ou 'LOSS'."""
        if self.writer is None:
            return None

        survival_time = time.time() - self.start_time
        max_boss_hp = state.get('bossMaxHp', 900)
        final_hp_player = state.get('hp', 0)
        hp_boss = state.get('bossHp', 0)

        result = 'WIN' if final_hp_player > 0 else 'LOSS'
        damage_dealt = max_boss_hp if result == 'WIN' else max(max_boss_hp - hp_boss, 0)
        nail_accuracy = (self.hits_made / self.hit_attempts * 100.0) if self.hit_attempts > 0 else 0.0

        self.file_handle.close()
        self.writer = None
        self.file_handle = None

        self._append_summary(result, survival_time, damage_dealt,
                             final_hp_player, nail_accuracy)

        print(f"[TestLogger] Luta finalizada ({result}) | {survival_time:.1f}s | "
              f"frames: {self.n_frames} | linha agregada em '{AGG_CSV}'")
        return result

    def _reset_counters(self):
        self.hits_taken = 0
        self.hits_made = 0
        self.hit_attempts = 0
        self.dashes = 0
        self.spells_casted = 0
        self.spell_hits = 0
        self.heal_success = 0
        self.was_attack_pressed = False
        self.was_dash_pressed = False
        self.post_spell_frames = 0
        self.post_heal_frames = 0
        self._last_soul = None

    @staticmethod
    def binary_to_gamepad(binary_actions):
        """Replicação determinística da convenção gp_*: eixos discretos {-1,0,1}."""
        if binary_actions is None or len(binary_actions) < 8:
            return {key: 0.0 for key in GP_KEYS}
        return {
            'gp_axis_left_x': float(binary_actions[5]) - float(binary_actions[4]),
            'gp_axis_left_y': float(binary_actions[6]) - float(binary_actions[7]),
            'gp_trigger_right': float(binary_actions[0]),
            'gp_btn_a': float(binary_actions[1]),
            'gp_btn_b': float(binary_actions[2]),
            'gp_btn_x': float(binary_actions[3]),
        }

    def _update_counters(self, state, binary_actions):
        hp = state.get('hp', 0)
        hp_boss = state.get('bossHp', 0)

        # borda de subida: ataque e dash
        attack_now = binary_actions is not None and len(binary_actions) > 3 and binary_actions[3] == 1
        if attack_now and not self.was_attack_pressed:
            self.hit_attempts += 1
        self.was_attack_pressed = attack_now

        dash_now = binary_actions is not None and len(binary_actions) > 0 and binary_actions[0] == 1
        if dash_now and not self.was_dash_pressed:
            self.dashes += 1
        self.was_dash_pressed = dash_now

        # feitiço x canalização de cura pelo consumo de alma
        soul = state.get('soul', None)
        if soul is not None:
            if self._last_soul is not None and soul < self._last_soul:
                if not state.get('focusing', False):
                    self.post_spell_frames = 30
                    self.spells_casted += 1
                else:
                    self.post_heal_frames = 35
            self._last_soul = soul

        # HP do jogador: dano sofrido / cura concluída
        if self.last_hp_player is not None:
            if hp < self.last_hp_player:
                self.hits_taken += 1
                self.post_heal_frames = 0
            elif hp > self.last_hp_player and self.post_heal_frames > 0:
                self.heal_success += 1
                self.post_heal_frames = 0
            self.last_hp_player = hp

        # dano no chefe (janela de feitiço)
        if self.last_hp_boss is not None and hp_boss < self.last_hp_boss:
            self.hits_made += 1
            if self.post_spell_frames > 0:
                self.spell_hits += 1
                self.post_spell_frames = 0
            self.last_hp_boss = hp_boss

        if self.post_spell_frames > 0:
            self.post_spell_frames -= 1
        if self.post_heal_frames > 0:
            self.post_heal_frames -= 1

    def _append_summary(self, result, survival_time, damage_dealt,
                        final_hp_player, nail_accuracy):
        file_exists = os.path.isfile(AGG_CSV)
        with open(AGG_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(AGG_HEADER)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M'),
                self.model_type,
                result,
                f"{survival_time:.2f}",
                int(damage_dealt),
                self.hits_made,
                f"{nail_accuracy:.3f}",
                final_hp_player,
                self.hits_taken,
                self.spells_casted,
                self.spell_hits,
                self.heal_success,
                self.dashes,
                '',  # Successful_Dodges: manual via Data Viewer (taxonomia)
                '',  # Complex_Behaviors: manual via Data Viewer (taxonomia)
            ])
