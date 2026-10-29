import torch
import numpy as np
import os
from collections import deque
import time

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'Checkpoints', 'BC_weights')

from Src.Data import ClientPipe
from Src.Data.DataHandler import DataHandler
from Src.Utils.CampaignManager import CampaignManager
from Src.Utils.TestLogger import TestLogger
from Src.Utils.VirtualGamePad import VirtualGamePad
from Src.Models.CNN_HollowKnight import CNN_HollowKnight
from Src.Models.MLP_HollowKnight import MLP_HollowKnight
from Src.Models.MLP_Temporal import MLP_Temporal
from Src.Models.GRU_HollowKnight import GRU_HollowKnight

# --- Campanha: roda LUTAS_POR_MODELO lutas por arquitetura (nesta ordem) e
# --- troca o modelo automaticamente, mantendo a reentrada autônoma.
# --- Lista vazia = usa TIPO_MODELO único, lutas ilimitadas (até Ctrl+C).
CAMPANHA_MODELOS = ["CNN", "MLP", "MLP_TEMPORAL", "GRU"]
LUTAS_POR_MODELO = 50
TIPO_MODELO = "GRU"  # usado apenas quando CAMPANHA_MODELOS == []

THRESHOLD = 0.2

NUM_FRAMES = 10             # Tamanho da janela temporal (conforme treino)
BOSS_SCENE_HASH = 423158243 # Código da Hornet

# --- Reentrada automática: segura D-PAD cima e pulsa A até a luta começar ---
AUTO_REENTER = True
REENTRY_PRESS_FRAMES = 6     # ~0.1s segurando A (loop ~60 fps)
REENTRY_RELEASE_FRAMES = 24  # ~0.4s solto (ciclo de 30 frames = ~2 pulsos/s)
REENTRY_STATUS_EVERY_SEC = 5.0

# Lista oficial de chaves vinda do treino do Kaggle
CHAVES_TREINO = ['attackDown', 'attackDuration', 'attackForward', 'attackUp', 'bossHp', 'bossMaxHp', 'bossScene', 'bossState', 'bossStateAmount', 'bvx', 'bvy', 'bx', 'by', 'canCast', 'casting', 'dashCoolDown', 'dashing', 'doubleJumping', 'facingRight', 'falling', 'focusing', 'hp', 'invulnerable', 'isAttacking', 'jumping', 'maxHp', 'maxSoul', 'onGround', 'pvx', 'pvy', 'px', 'py', 'shadowDashCoolDown', 'shadowDashing', 'soul']


def build_model(tipo):
    if tipo == "CNN":
        modelo = CNN_HollowKnight(num_features=104, num_acoes=8, tamanho_janela=NUM_FRAMES)
        caminho_pesos = os.path.join(CHECKPOINT_DIR, 'melhor_modelo_cnn_hollowknight.pth')
    elif tipo == "MLP":
        modelo = MLP_HollowKnight(num_features=104, num_acoes=8, taxa_dropout=0.2)
        caminho_pesos = os.path.join(CHECKPOINT_DIR, 'melhor_modelo_mlp_hollowknight.pth')
    elif tipo == "MLP_TEMPORAL":
        modelo = MLP_Temporal(num_features=104, num_acoes=8, tamanho_janela=NUM_FRAMES)
        caminho_pesos = os.path.join(CHECKPOINT_DIR, 'melhor_modelo_mlp_temporal.pth')
    elif tipo == "GRU":
        modelo = GRU_HollowKnight(num_features=104, num_acoes=8, tamanho_janela=NUM_FRAMES)
        caminho_pesos = os.path.join(CHECKPOINT_DIR, 'melhor_modelo_gru_hollowknight.pth')
    else:
        raise ValueError("Tipo inválido! Escolha 'CNN', 'MLP', 'MLP_TEMPORAL' ou 'GRU'.")
    return modelo, caminho_pesos


def load_campaign_models(tipos, device):
    """Carrega os checkpoints de todas as arquiteturas da campanha na memória.

    Modelos com checkpoint ausente/corrompido são descartados com aviso."""
    modelos = {}
    for tipo in tipos:
        try:
            modelo, caminho_pesos = build_model(tipo)
            modelo.load_state_dict(torch.load(caminho_pesos, map_location=device, weights_only=True))
            modelo.eval()
            modelos[tipo] = modelo
            print(f"✅ {tipo}: pesos carregados de {caminho_pesos}")
        except (FileNotFoundError, RuntimeError, OSError) as e:
            print(f"⚠️ {tipo} descartado da campanha (checkpoint indisponível): {e}")
    return modelos


def main():
    campanha_planejada = CAMPANHA_MODELOS if CAMPANHA_MODELOS else [TIPO_MODELO]
    campanha_limitada = bool(CAMPANHA_MODELOS)

    print(f"Iniciando o Bot - campanha: {' -> '.join(campanha_planejada)} "
          f"({LUTAS_POR_MODELO} lutas por modelo)..." if campanha_limitada
          else f"Iniciando o Bot com a arquitetura: {TIPO_MODELO}...")

    pipe = ClientPipe.Pipe()
    print("Aguardando o Hollow Knight criar o canal de comunicação...")
    while True:
        try:
            pipe.connect()
            break
        except OSError:
            time.sleep(2)

    virtual_gamepad = VirtualGamePad()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Todos os modelos carregados de uma vez: a troca é só trocar a referência.
    modelos = load_campaign_models(campanha_planejada, device)
    if not modelos:
        print("Nenhum checkpoint disponível. Encerrando.")
        return

    manager = CampaignManager([t for t in campanha_planejada if t in modelos],
                              LUTAS_POR_MODELO, modelos)
    manager.start()

    # Cada luta gera em dataset/test_runs/<modelo>/ um CSV puro frame a frame,
    # com colunas pred_* / prob_* extras além do schema de treino.
    logger = TestLogger(manager.tipo_atual, BOSS_SCENE_HASH, threshold=THRESHOLD)

    print(f"✅ {manager.tipo_atual} pronto para a luta 1"
          + (f"/{LUTAS_POR_MODELO}" if campanha_limitada else ""))

    frame_stack = deque(maxlen=NUM_FRAMES)
    battle_active = False
    acoes_binarias = None
    probs_numpy = None
    ultimo_state = None

    # Estado da reentrada automática (só ativa após o fim da 1ª luta,
    # para não conflitar com a entrada manual do jogador)
    reentry_armed = False
    reentry_pulse_frame = 0
    reentry_started_at = None
    reentry_last_status = 0.0

    try:
        while True:
            state = pipe.read_state()
            if state is None:
                print("Conexão perdida.")
                if battle_active and ultimo_state is not None:
                    logger.end_battle(ultimo_state)
                break

            ultimo_state = state
            boss_scene = state.get('bossScene')
            hp_player = state.get('hp', 0)
            hp_boss = state.get('bossHp', 0)
            boss_max_hp = state.get('bossMaxHp', 900)

            # --- DETECÇÃO DE FIM DE COMBATE (luta vigente) ---
            if battle_active and (hp_player <= 0 or boss_scene != BOSS_SCENE_HASH or hp_boss <= 0):
                result = logger.end_battle(state)
                manager.record_result(result)
                battle_active = False
                frame_stack.clear()
                virtual_gamepad.update_gamepad([])
                acoes_binarias = None
                probs_numpy = None
                # reentra no próximo ciclo (modo autônomo) sem intervenção
                reentry_armed = AUTO_REENTER
                reentry_pulse_frame = 0
                reentry_started_at = time.time()
                reentry_last_status = 0.0

                # --- TROCA AUTOMÁTICA DE MODELO AO COMPLETAR A COTA ---
                if campanha_limitada and manager.quota_reached():
                    if manager.next_model():
                        print(f"\n🔄 {LUTAS_POR_MODELO} lutas de "
                              f"{manager.queue[manager.idx - 1]} completas. "
                              f"Trocando para {manager.tipo_atual}...")
                        frame_stack.clear()
                        logger = TestLogger(manager.tipo_atual, BOSS_SCENE_HASH,
                                            threshold=THRESHOLD)
                    else:
                        print("\n" + manager.summary())
                        virtual_gamepad.update_gamepad([])
                        break
                else:
                    meta = f"/{LUTAS_POR_MODELO}" if campanha_limitada else ""
                    print(f"[{manager.tipo_atual}] {manager.lutas_do_modelo} luta(s) "
                          f"completa(s){meta}. Reentrada automática ativa.")
                continue

            if not battle_active:
                # luta nova: cena da Hornet com barras de vida originais
                if (boss_scene == BOSS_SCENE_HASH and hp_player > 0
                        and hp_boss > 0 and hp_boss >= boss_max_hp):
                    logger.start_battle(state)
                    battle_active = True
                    frame_stack.clear()
                    # solta cima/A assim que a luta começa
                    virtual_gamepad.update_gamepad([])
                    reentry_armed = False
                    reentry_started_at = None
                else:
                    if AUTO_REENTER and reentry_armed:
                        # segura D-PAD cima e pulsa A até entrar na arena
                        pulse_phase = reentry_pulse_frame % (
                            REENTRY_PRESS_FRAMES + REENTRY_RELEASE_FRAMES)
                        reentry_actions = [1]  # cima contínuo
                        if pulse_phase < REENTRY_PRESS_FRAMES:
                            reentry_actions.append(5)  # pulso do A
                        virtual_gamepad.update_gamepad(reentry_actions)
                        reentry_pulse_frame += 1

                        now = time.time()
                        if reentry_started_at is None:
                            reentry_started_at = now
                        if now - reentry_last_status >= REENTRY_STATUS_EVERY_SEC:
                            reentry_last_status = now
                            elapsed = now - reentry_started_at
                            print(f"Aguardando entrada na arena... {elapsed:.0f}s "
                                  f"(cima+A automático, modelo: {manager.tipo_atual})")
                    continue

            state_limpo = {}
            for chave in CHAVES_TREINO:
                state_limpo[chave] = state.get(chave, 0.0)

            data = DataHandler.treat_data(state_limpo)

            # --- FORMATAÇÃO DO TENSOR DE ENTRADA ---
            if manager.tipo_atual in ("CNN", "MLP_TEMPORAL", "GRU"):
                if len(frame_stack) == 0:
                    for _ in range(NUM_FRAMES):
                        frame_stack.append(data)
                else:
                    frame_stack.append(data)

                stacked_data = np.array(frame_stack, dtype=np.float32)
                state_tensor = torch.FloatTensor(stacked_data).unsqueeze(0).to(device)

            elif manager.tipo_atual == "MLP":
                # Ignora o histórico e envia apenas o frame atual em 2D -> (1, 104)
                state_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)

            # --- INFERÊNCIA DA REDE ---
            with torch.no_grad():
                logits = manager.modelo(state_tensor)
                probs = torch.sigmoid(logits)
                probs_numpy = probs.squeeze(0).cpu().numpy()
                acoes_binarias = (probs_numpy > THRESHOLD).astype(int)

            # --- REGISTRO PURO FRAME A FRAME (schema de treino + saída do modelo) ---
            logger.log_frame(state, acoes_binarias, probs_numpy)

            # --- TRADUTOR PARA O VIRTUALGAMEPAD ---
            active_actions = []

            # Botões de Ação
            if acoes_binarias[0] == 1: active_actions.append(6) # Dash
            if acoes_binarias[1] == 1: active_actions.append(5) # Pulo (A)
            if acoes_binarias[2] == 1: active_actions.append(8) # Cast (B)
            if acoes_binarias[3] == 1: active_actions.append(7) # Ataque (X)

            # Movimentação (D-PAD)
            if acoes_binarias[4] == 1: active_actions.append(3) # Esquerda
            if acoes_binarias[5] == 1: active_actions.append(4) # Direita
            if acoes_binarias[6] == 1: active_actions.append(1) # Cima
            if acoes_binarias[7] == 1: active_actions.append(2) # Baixo

            virtual_gamepad.update_gamepad(active_actions)

    except KeyboardInterrupt:
        if battle_active and ultimo_state is not None:
            logger.end_battle(ultimo_state)
        virtual_gamepad.update_gamepad([])  # solta cima/A da reentrada
        if campanha_limitada:
            print("\n" + manager.summary())
        print("Bot desligado manualmente.")
    finally:
        pipe.disconnect()


if __name__ == "__main__":
    main()
