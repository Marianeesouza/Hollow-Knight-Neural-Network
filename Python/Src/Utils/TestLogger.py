from datetime import datetime
import time
import os
import csv

class TestLogger:
    def __init__(self, model_type, boss_scene_hash=423158243):
        self.model_type = model_type
        self.boss_scene_hash = boss_scene_hash
        self.csv_file = "all_model_tests.csv"
        
        self.is_ai_running = False
        self.start_test_time = None
        self.last_hp_player = None
        self.last_hp_boss = None
        self.last_soul = None

        self.post_spell_frames = 0
        self.post_heal_frames = 0
        
        self.hits_taken = 0
        self.hits_made = 0
        self.hit_attempts = 0
        self.dashes_counter = 0
        self.spells_casted = 0
        self.spell_hits = 0
        self.heal_success = 0

        self.was_attack_button_pressed = False
        self.was_dash_button_pressed = False

    def monitor_frame(self, state, binary_actions):
        """Monitors telemetry variations, manages counters, and handles battle lifecycle."""
        boss_scene = state.get('bossScene')
        current_hp_player = state.get('hp', 0)
        current_hp_boss = state.get('bossHp', 0)
        current_soul = state.get('soul', 0)
        is_focusing = state.get('focusing', False)

        if (boss_scene == self.boss_scene_hash and not self.is_ai_running):
            if current_hp_player <= 0 or current_hp_boss < 900:
                return False

            print(f"\n Battle started! Monitoring dynamics for {self.model_type}...")
            self.is_ai_running = True
            self.start_test_time = time.time()
            self.last_hp_player = current_hp_player
            self.last_hp_boss = current_hp_boss
            self.last_soul = current_soul
            self.post_spell_frames = 0
            self.post_heal_frames = 0

            self.hits_taken = 0
            self.hits_made = 0
            self.hit_attempts = 0
            self.dashes_counter = 0
            self.spells_casted = 0
            self.spell_hits = 0
            self.heal_success = 0

            self.was_attack_button_pressed = False
            self.was_dash_button_pressed = False

        if not self.is_ai_running:
            return False
        
        is_attack_pressed_now = (binary_actions is not None and len(binary_actions) > 3 and binary_actions[3] == 1)

        if is_attack_pressed_now:
            if not self.was_attack_button_pressed:
                self.hit_attempts += 1
                self.was_attack_button_pressed = True
        else:
            self.was_attack_button_pressed = False

        is_dash_pressed_now = (binary_actions is not None and len(binary_actions) > 0 and binary_actions[0] == 1)
        if is_dash_pressed_now:
            if not self.was_dash_button_pressed:
                self.dashes_counter += 1
                self.was_dash_button_pressed = True 
        else:
            self.was_dash_button_pressed = False
        
        ai_pressed_cast_or_focus = (binary_actions is not None and len(binary_actions) > 2 and binary_actions[2] == 1)
        
        if self.last_soul is not None and current_soul < self.last_soul:
            if not is_focusing and not (ai_pressed_cast_or_focus and state.get('onGround', False)):
                self.post_spell_frames = 30
                self.spells_casted += 1
                print(f"🔮 [SPELL CASTED] Spells casted total: {self.spells_casted}")
            else:
                self.post_heal_frames = 35 
            
        self.last_soul = current_soul

        if self.last_hp_player is not None:
            if current_hp_player < self.last_hp_player:
                self.hits_taken += 1
                self.post_heal_frames = 0 
                self.last_hp_player = current_hp_player
                
            elif current_hp_player > self.last_hp_player:
                if self.post_heal_frames > 0:
                    self.heal_success += 1
                    self.post_heal_frames = 0
                self.last_hp_player = current_hp_player

        if self.last_hp_boss is not None and current_hp_boss < self.last_hp_boss:
            self.hits_made += 1
            
            if self.post_spell_frames > 0:
                self.spell_hits += 1
                self.post_spell_frames = 0                 
                
            self.last_hp_boss = current_hp_boss

        if self.post_spell_frames > 0: self.post_spell_frames -= 1
        if self.post_heal_frames > 0: self.post_heal_frames -= 1

        if binary_actions is not None and len(binary_actions) > 0 and binary_actions[0] == 1:
            self.dashes_counter += 1

        if current_hp_player <= 0 or (boss_scene != self.boss_scene_hash):
            survival_time = time.time() - self.start_test_time
            max_boss_hp = state.get('bossMaxHp', 900)
            current_boss_hp_safe = current_hp_boss if current_hp_boss is not None else 0
            total_damage_dealt = max_boss_hp - current_boss_hp_safe
            
            if current_hp_player > 0:
                battle_result = "WIN"
                total_damage_dealt = max_boss_hp
            else:
                battle_result = "LOSS"
                total_damage_dealt = max_boss_hp - current_boss_hp_safe

            nail_accuracy = (self.hits_made / self.hit_attempts * 100) if self.hit_attempts > 0 else 0.0
            
            print(f"\n Battle ended")
            
            self.is_ai_running = False 
            self.last_hp_player = None
            self.last_hp_boss = None
            self.last_soul = None
            
            print("Manual Evaluation")
            
            try:
                successful_dodges = input("# Successful dodges").strip()
                if not successful_dodges: successful_dodges = "0"
            except Exception:
                successful_dodges = "0"
            
            try:
                complex_behaviors = input("# Complex behaviours").strip()
                if not complex_behaviors: complex_behaviors = "0"
            except Exception:
                complex_behaviors = "0"

            self._save_to_csv(survival_time, total_damage_dealt, current_hp_player, successful_dodges, complex_behaviors, nail_accuracy, battle_result)
            
            print("Awaiting arena initialization for the next round...")
            return False

        return True

    def _save_to_csv(self, survival_time, total_damage_dealt, current_hp_player, successful_dodges, complex_behaviors, nail_accuracy, battle_result):
        """Appends the consolidated run metrics into the master CSV file."""
        file_exists = os.path.isfile(self.csv_file)
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        with open(self.csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Timestamp", "Model", "Result", "Survival_Time_Sec", "Damage_Dealt", "Total_Nail_Hits", "Nail_Accuracy" , "Final_Bot_HP",
                    "Hits_Taken", "Spells_Casted", "Spell_Hits", "Successful_Heals", "Dashes_Executed", "Successful_Dodges","Complex_Behaviors"
                ])
            
            writer.writerow([
                current_timestamp,
                self.model_type,
                battle_result,
                f"{survival_time:.2f}", 
                total_damage_dealt,
                self.hits_made,
                nail_accuracy,
                current_hp_player,
                self.hits_taken, 
                self.spells_casted,
                self.spell_hits,
                self.heal_success,
                self.dashes_counter,
                successful_dodges,
                complex_behaviors
            ])
        print(f" Run metrics successfully exported to '{self.csv_file}'!")