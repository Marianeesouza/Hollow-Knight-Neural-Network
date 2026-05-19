from Src.Data.DataExtractor import extract_and_log, replay_inputs_from_csv
from Src.Data.InputExtractor import capturar_snapshot_gamepad
from Src.Utils.FileUtilities import FileUtilities

# Opção 1: Gravar inputs
extract_and_log()

# Opção 2: Reproduzir inputs gravados (aguarda jogo ativo, 0.2s entre frames)
#replay_inputs_from_csv(csv_path='dataset\hornet_boss\demo_1779118608_hash423158243.csv', frame_delay=0.02)

#while True:
#    snapshot = capturar_snapshot_gamepad()
    
    # Se o controle não estiver conectado, o snapshot pode vir None
#    if snapshot and any(abs(v) > 0 for v in snapshot.values()):
#        print(snapshot)

#FileUtilities.normalize_and_discretize('dataset/hornet_boss')