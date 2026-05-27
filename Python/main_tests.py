import torch
import numpy as np
from collections import deque
import time

from Src.Data import ClientPipe
from Src.Data.DataHandler import DataHandler
from Src.Utils.VirtualGamePad import VirtualGamePad
from Src.Models.CNN_HollowKnight import CNN_HollowKnight
from Src.Models.MLP_HollowKnight import MLP_HollowKnight
import torch.nn.functional as F
import torch.nn as nn

# Altere para "CNN" ou "MLP" dependendo do modelo que quer testar
TIPO_MODELO = "MLP"  

NUM_FRAMES = 10             # Tamanho da janela temporal
BOSS_SCENE_HASH = 423158243 # Código da Hornet

# Lista oficial de chaves vinda do treino do Kaggle
CHAVES_TREINO = ['attackDown', 'attackDuration', 'attackForward', 'attackUp', 'bossHp', 'bossMaxHp', 'bossScene', 'bossState', 'bossStateAmount', 'bvx', 'bvy', 'bx', 'by', 'canCast', 'casting', 'dashCoolDown', 'dashing', 'doubleJumping', 'facingRight', 'falling', 'focusing', 'hp', 'invulnerable', 'isAttacking', 'jumping', 'maxHp', 'maxSoul', 'onGround', 'pvx', 'pvy', 'px', 'py', 'shadowDashCoolDown', 'shadowDashing', 'soul']


def main():
    print(f"Iniciando o Bot com a arquitetura: {TIPO_MODELO}...")
    
    pipe = ClientPipe.Pipe()
    print("Aguardando o Hollow Knight criar o canal de comunicação...")
    while True:
        try:
            pipe.connect()
            print("✅ Conectado com sucesso ao jogo!")
            break
        except OSError:
            time.sleep(2)
            
    virtual_gamepad = VirtualGamePad()
    frame_stack = deque(maxlen=NUM_FRAMES)

    # --- IF/ELSE: INICIALIZAÇÃO DO MODELO ---
    if TIPO_MODELO == "CNN":
        modelo = CNN_HollowKnight(num_features=104, num_acoes=10, tamanho_janela=NUM_FRAMES)
        caminho_pesos = r'Python\Checkpoints\BC_weights\melhor_modelo_cnn_hollowknight.pth'
    elif TIPO_MODELO == "MLP":
        modelo = MLP_HollowKnight(num_features=104, num_acoes=10, taxa_dropout=0.2)
        caminho_pesos = r'Python\Checkpoints\BC_weights\melhor_modelo_hollowknight.pth'
    else:
        raise ValueError("TIPO_MODELO inválido! Escolha 'CNN' ou 'MLP'.")

    modelo.load_state_dict(torch.load(caminho_pesos, weights_only=True))
    modelo.eval()
    print(f"Cérebro da {TIPO_MODELO} carregado com sucesso!")

    is_ai_running = False

    try:
        while True:
            state = pipe.read_state()
            if state is None:
                print("⚠️ Conexão perdida.")
                break

            boss_scene = state['bossScene']

            # Limpeza estrita de chaves para bater com as 104 colunas do treat_data
            state_limpo = {}
            for chave in CHAVES_TREINO:
                state_limpo[chave] = state.get(chave, 0.0)

            data = DataHandler.treat_data(state_limpo) 

            if (boss_scene is not None and boss_scene == BOSS_SCENE_HASH):
                is_ai_running = True

            if not is_ai_running:
                continue

            # --- IF/ELSE: FORMATAÇÃO DO TENSOR DE ENTRADA ---
            if TIPO_MODELO == "CNN":
                # Alimenta e mantém a janela temporal de 10 frames
                if len(frame_stack) == 0:
                    for _ in range(NUM_FRAMES):
                        frame_stack.append(data)
                else:
                    frame_stack.append(data)

                # Cria o formato 3D esperado pela CNN -> (1, 10, 104)
                stacked_data = np.array(frame_stack, dtype=np.float32)
                state_tensor = torch.FloatTensor(stacked_data).unsqueeze(0)
                
            elif TIPO_MODELO == "MLP":
                # Ignora o histórico e envia apenas o frame atual em 2D -> (1, 104)
                state_tensor = torch.FloatTensor(data).unsqueeze(0)

            # --- INFERÊNCIA DA REDE ---
            with torch.no_grad():
                logits = modelo(state_tensor)
                probabilidades = torch.sigmoid(logits)
                acoes_binarias = (probabilidades > 0.2).int().squeeze(0).numpy()
                
            print(f"[{TIPO_MODELO}] Probs: {np.round(probabilidades.squeeze(0).numpy(), 2)} -> Ações: {acoes_binarias}")

            # --- TRADUTOR PARA O VIRTUALGAMEPAD ---
            active_actions = []
            
            # Botões de Ação
            if acoes_binarias[0] == 1: active_actions.append(6) # Dash
            if acoes_binarias[1] == 1: active_actions.append(5) # Pulo (A)
            if acoes_binarias[2] == 1: active_actions.append(8) # Cast (B)
            if acoes_binarias[3] == 1: active_actions.append(7) # Ataque (X)
            
            # Movimentação (D-PAD)
            if acoes_binarias[6] == 1: active_actions.append(3) # Esquerda
            if acoes_binarias[7] == 1: active_actions.append(4) # Direita
            if acoes_binarias[8] == 1: active_actions.append(1) # Cima
            if acoes_binarias[9] == 1: active_actions.append(2) # Baixo

            virtual_gamepad.update_gamepad(active_actions)

            # Reinicia o bot se sair da arena
            if (is_ai_running and boss_scene is not None and boss_scene != BOSS_SCENE_HASH):
                print("⚠️ O jogador saiu da cena do chefe. Aguardando nova luta.")
                is_ai_running = False
                frame_stack.clear()

    except KeyboardInterrupt:
        print("Bot desligado manualmente.")
    finally:
        pipe.disconnect()

if __name__ == "__main__":
    main()