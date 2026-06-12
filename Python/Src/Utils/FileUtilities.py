import csv
import pickle
import os
from collections import deque
import time
from ..Data.InputExtractor import normalize_trigger, discretize_axis, discretize_trigger

class FileUtilities:

    @staticmethod
    def save_file(file_name: str = 'stats.pkl', *data):

        file_path = os.path.join('Checkpoints', 'Statistic', file_name)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Stats saved to {file_path}")

    @staticmethod
    def load_file(file_name: str = 'stats.pkl'):

        file_path = os.path.join('Checkpoints', 'Statistic', file_name)

        try:
            with open(file_path, "rb") as file:
                print("Stats loaded")
                data = pickle.load(file)

                if len(data) == 1 and isinstance(data[0], tuple):
                    data = data[0]
                    return data
                elif len(data) == 4:
                    episodes_counter, reward_stack, episode_stats, mean_stats = data
                    best_mean_reward = -float("inf")
                elif len(data) == 5:
                    episodes_counter, reward_stack, episode_stats, mean_stats, best_mean_reward = data
                else:
                    raise ValueError("Invalid file format")

                return episodes_counter, reward_stack, episode_stats, mean_stats, best_mean_reward

        except FileNotFoundError:
            print(f"⚠️ File not found at {file_path}. Creating new empty file")
            return 0, deque(maxlen=100), [], [], -float("inf")

    @staticmethod
    def save_demo(file_name: str = 'demos.pkl', demonstrations=None):
        """Salva o conjunto de demonstrações em disco."""
        file_path = os.path.join('Checkpoints', 'Statistic', file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(demonstrations, file, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Demonstrações salvas em {file_path}")

    @staticmethod
    def load_demo(file_name: str = 'demos.pkl'):
        """Carrega um conjunto de demonstrações salvo."""
        file_path = os.path.join('Checkpoints', 'Statistic', file_name)

        try:
            with open(file_path, "rb") as file:
                print("Demonstrações carregadas")
                return pickle.load(file)
        except FileNotFoundError:
            print(f"Arquivo de demonstrações não encontrado em {file_path}")
            return []
    

    @staticmethod
    def write_state_inputs_to_csv(state: dict, inputs: dict, csv_path: str = 'hk_data.csv', include_timestamp: bool = True, overwrite: bool = False):
        """Escreve um único registro combinando `state` e `inputs` em `csv_path`.
        
        Garante que colunas ficassem alinhadas mesmo mudando o arquivo dinamicamente.
        """
        if state is None and inputs is None:
            return

        from ..Data.DataExtractor import KNOWN_GAMEPAD_COLUMNS

        # Prefixar chaves dos inputs
        prefixed_inputs = {f'gp_{k}': v for k, v in (inputs or {}).items()}

        # Construir linha única
        row = {}
        if include_timestamp:
            row['timestamp'] = time.time()
            
        for k in sorted(state.keys()) if state else []:
            row[k] = state.get(k)
        for k in sorted(prefixed_inputs.keys()):
            row[k] = prefixed_inputs.get(k)

        # Garantir diretório
        csv_dir = os.path.dirname(csv_path)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)

        # CORREÇÃO: write_header só deve ser verdadeiro se for overwrite OU se o arquivo físico REALMENTE não existir na assinatura atual
        write_header = overwrite or not os.path.exists(csv_path)

        # Determinar colunas fixas
        extra_input_keys = [k for k in sorted(prefixed_inputs.keys()) if k not in KNOWN_GAMEPAD_COLUMNS]
        fieldnames = [
            'timestamp',
            *([k for k in sorted(state.keys())] if state else []),
            *KNOWN_GAMEPAD_COLUMNS,
            *extra_input_keys,
        ]

        mode = 'w' if overwrite else 'a'
        with open(csv_path, mode, newline='', encoding='utf-8') as f:
            # Usamos extrasaction='ignore' para evitar travar se alguma chave sumir no frame
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            
            if write_header:
                writer.writeheader()
                
            # Converte tipos para formatos primitivos seguros
            safe_row = {
                k: (v if isinstance(v, (int, float, bool)) else '' if v is None else str(v))
                for k, v in ((key, row.get(key)) for key in fieldnames)
            }
            writer.writerow(safe_row)


    @staticmethod
    def normalize_and_discretize(dataset_path: str):
        """
        Varre a pasta de datasets, normaliza e discretiza os valores de cada coluna
        de todos os arquivos CSV, salvando as alterações de volta no disco.
        """
        for file_name in os.listdir(dataset_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(dataset_path, file_name)
                
                # 1. ETAPA DE LEITURA
                with open(file_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                # Se o arquivo CSV estiver completamente vazio, pula para o próximo
                if not rows:
                    continue

                # 2. ETAPA DE PROCESSAMENTO (Sua lógica intacta e corrigida)
                for row in rows:
                    for coluna in list(row.keys()):
                        val_cru = row[coluna]
                        
                        if val_cru is None or val_cru == '':
                            continue
                            
                        try:
                            if coluna in ["gp_axis_left_x", "gp_axis_left_y"]:
                                v = float(val_cru)
                                row[coluna] = discretize_axis(v)

                            elif coluna == "gp_trigger_right":
                                v = float(val_cru)
                                row[coluna] = discretize_trigger(normalize_trigger(v))
                                
                            # Norma 4: Booleanos vindos da telemetria do jogo
                            elif coluna in ["onGround", "dashing", "invulnerable", "isAttacking", "jumping", "facingRight", "attackDown", "attackForward", "attackUp", "canCast", "casting", "dashing", "doubleJumping", "facingRight", "falling", "focusing", "shadowDashing"]:
                                # Trata se vier como string "True"/"False" ou números binários
                                val_str = str(val_cru).strip().lower()
                                row[coluna] = 1 if val_str in ['true', '1', '1.0'] else 0

                            elif coluna in ["gp_btn_back", "gp_btn_start", "gp_trigger_left", "gp_axis_right_x", "gp_axis_right_y"]:
                                del row[coluna]

                        except (ValueError, TypeError):
                            # Proteção contra dados corrompidos
                            pass

                colunas_salvamento = list(rows[0].keys())

                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    # Garantimos fieldnames válidos e usamos extrasaction='ignore' por segurança
                    writer = csv.DictWriter(f, fieldnames=colunas_salvamento, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(rows)
                    
        print(f"✨ Processamento e normalização concluídos para a pasta: {dataset_path}")