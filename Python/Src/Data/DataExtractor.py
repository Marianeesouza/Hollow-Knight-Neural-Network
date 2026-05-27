import csv
import os
import time
from ..Utils.VirtualGamePad import VirtualGamePad
from ..Utils.FileUtilities import FileUtilities

from . import ClientPipe
from .InputExtractor import capturar_snapshot_gamepad

SAVE_INTERVAL = 90000
UPDATE_TIMESTEP = 4096
BOSS_SCENE_HASH = 423158243 # Hornet
NUM_FRAMES = 4
DATASET_DIR = "dataset/hornet_boss"

FILE_NAME = 'stats.pkl'

KNOWN_GAMEPAD_COLUMNS = [
    'gp_axis_left_x', 'gp_axis_left_y', 'gp_trigger_right',
    'gp_btn_a', 'gp_btn_b', 'gp_btn_x',
    'gp_btn_back', 'gp_btn_start',
]


def is_game_state_active(s: dict) -> bool:
    """Heurística global para verificar se o jogo está numa cena jogável ativa."""
    if not s:
        return False
    try:
        if float(s.get('maxHp', 0)) > 0: return True
    except Exception: pass
    try:
        if float(s.get('px', 0)) != 0.0: return True
    except Exception: pass
    try:
        if int(s.get('bossScene', 0)) != 0: return True
    except Exception: pass
    return False

def detectou_movimento_no_gamepad(inputs: dict) -> bool:
    """Verifica se o snapshot do comando físico contém alguma ação ativa diferente de zero."""
    if not inputs:
        return False
        
    # 1. Verifica Botões Digitais e Gatilhos (onde 1 ou maior indica pressionado)
    botoes_e_gatilhos = [
        'btn_a', 'btn_b', 'btn_x', 'btn_y', 
        'bumper_left', 'bumper_right', 
        'btn_back', 'btn_start',
        'trigger_left', 'trigger_right'
    ]

    for chave in botoes_e_gatilhos:
        try:
            if float(inputs.get(chave, 0.0)) > 0.05: # Pequena tolerância para ruído
                return True
        except (ValueError, TypeError):
            pass
            
    # 2. Verifica os Analógicos (Sticks) - Qualquer valor distante de 0.0 indica movimento
    analogicos = ['axis_left_x', 'axis_left_y', 'axis_right_x', 'axis_right_y']
    for chave in analogicos:
        try:
            if abs(float(inputs.get(chave, 0.0))) > 0.1: # Tolerância de inclinação
                return True
        except (ValueError, TypeError):
            pass
            
    return False

def extract_and_log(pipe: ClientPipe.Pipe = None, target_hash: int = 423158243, dataset_dir: str = 'dataset/hornet_boss', poll_interval: float = 0.0166):
    """
    Loop automatizado de demonstrações. 
    Sempre que entrar na sala com o target_hash e o jogo estiver ativo, aguarda o primeiro input de movimento.
    Ao iniciar o movimento, abre um novo arquivo CSV individual e grava até o jogador sair da sala/morrer.
    """
    created_pipe = False
    if pipe is None:
        pipe = ClientPipe.Pipe()
        created_pipe = True

    if created_pipe:
        pipe.connect()

    recording = False
    csv_atual_path = None
    session_samples = 0

    print(f"Automatizador de Demonstrações Iniciado.")
    print(f"Alvo: Sala Hash {target_hash} | Pasta: {dataset_dir}")

    try:
        while True:
            state = pipe.read_state()
            
            if state is None:
                time.sleep(0.005)
                continue

            actual_scene = int(state.get('bossScene', 0))
            is_active = is_game_state_active(state)

            # --- MAQUINA DE ESTADOS DA AUTOMAÇÃO ---
            if not recording:
                
                if actual_scene == target_hash and is_active:
                    
                    gp_snapshot = capturar_snapshot_gamepad()
                    if detectou_movimento_no_gamepad(gp_snapshot):
                        timestamp_id = int(time.time())
                        csv_atual_path = os.path.join(dataset_dir, f"demo_{timestamp_id}_hash{target_hash}.csv")
                        
                        recording = True
                        session_samples = 0
                        print(f"\n▶️ Movimento detectado na sala alvo! Gravando em: {csv_atual_path}")
                        
                        FileUtilities.write_state_inputs_to_csv(state, gp_snapshot, csv_path=csv_atual_path, overwrite=True)
                        session_samples += 1
            else:
                if actual_scene != target_hash or not is_active:
                    print(f"\n⏹️ Demonstração finalizada e salva! Total de frames: {session_samples}")
                    recording = False
                    csv_atual_path = None
                    session_samples = 0
                else:
                    gp_snapshot = capturar_snapshot_gamepad()
                    FileUtilities.write_state_inputs_to_csv(state, gp_snapshot, csv_path=csv_atual_path, overwrite=False)
                    session_samples += 1
                    print(f"\rFrames capturados nesta demonstração: {session_samples}", end="", flush=True)

            # Controla o ciclo de repetição (padrão 0.0166s para espelhar 60 FPS)
            if poll_interval > 0:
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n👋 Gravação automatizada interrompida pelo usuário.")
    finally:
        if created_pipe:
            pipe.disconnect()
        FileUtilities.normalize_and_discretize(dataset_dir)

def mapear_snapshot_para_virtual_actions(inputs: dict, threshold: float = 0.2) -> list:
    """
    Traduz os inputs extraídos pelo 'capturar_snapshot_gamepad' (sem o prefixo 'gp_')
    para a lista de IDs de ações aceites pela classe VirtualGamePad.
    """
    active_actions = []

    try:
        # Extrai os eixos garantindo conversão para float
        lx = float(inputs.get('axis_left_x', 0.0))
        ly = float(inputs.get('axis_left_y', 0.0))
        
        # --- 1. Movimentação Horizontal (IDs 3 e 4) ---
        if lx < -threshold:
            active_actions.append(3)  # move left
        elif lx > threshold:
            active_actions.append(4)  # move right
            
        # --- 2. Movimentação Vertical (IDs 1 e 2) ---
        if ly > threshold:
            active_actions.append(1)  # Look up
        elif ly < -threshold:
            active_actions.append(2)  # Look down
            
    except (ValueError, TypeError):
        pass

    # --- 3. Gatilho Direito / Dash (ID 6) ---
    try:
        tr = float(inputs.get('trigger_right', 0.0))
        if tr > threshold:
            active_actions.append(6)  # dash
    except (ValueError, TypeError):
        pass

    # --- 4. Botões Digitais (IDs 5, 7 e 8) ---
    map_buttons = {
        'btn_a': 5,  # jump
        'btn_x': 7,  # attack
        'btn_b': 8,  # cast / heal
    }

    for csv_key, action_id in map_buttons.items():
        val = inputs.get(csv_key, 0)
        try:
            if float(val) >= 0.5:
                active_actions.append(action_id)
        except (ValueError, TypeError):
            pass

    return active_actions


def replay_inputs_from_csv(csv_path: str = 'hk_data.csv', frame_delay: float = 0.0166):
    if not os.path.exists(csv_path):
        print(f"Erro: arquivo '{csv_path}' não encontrado.")
        return
    
    # Aguardar jogo ativo via Pipe usando a função global extraída
    print("Esperando jogo estar ativo para o replay...")
    pipe = ClientPipe.Pipe()
    pipe.connect()
    
    while True:
        state = pipe.read_state()
        if state is not None and is_game_state_active(state):
            print("Jogo ativo detectado! Iniciando replay...")
            break
        time.sleep(0.1)
    
    pipe.disconnect()
    
    v_gamepad = VirtualGamePad()
    print("Gamepad virtual inicializado via classe com sucesso!")
    
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print(f"Aviso: '{csv_path}' vazio ou sem dados.")
        return
    
    print(f"Iniciando replay de {len(rows)} frames (delay: {frame_delay}s)...")
    
    for i, row in enumerate(rows):
        inputs_limpos = {k.replace('gp_', ''): v for k, v in row.items() if k.startswith('gp_')}
        
        active_actions = mapear_snapshot_para_virtual_actions(inputs_limpos, threshold=0.2)
        v_gamepad.update_gamepad(active_actions)
        
        print(f"\rReproduzindo frame {i+1}/{len(rows)} -> Ações Ativas: {active_actions}", end="", flush=True)
        time.sleep(frame_delay)
        
    v_gamepad.update_gamepad([])
    print(f"\nReplay concluído com sucesso!")
