import torch.nn as nn
import torch.nn.functional as F

class CNN_HollowKnight(nn.Module):
    """
    CNN para o Hollow Knight, projetada para processar uma janela temporal de dados.
    A arquitetura é inspirada em modelos de séries temporais, com camadas convolucionais seguidas de camadas totalmente conectadas.
    """

    def __init__(self, num_features, num_acoes, tamanho_janela=10, taxa_dropout=0.3):
        super(CNN_HollowKnight, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        tamanho_achatado = 64 * tamanho_janela 
        
        self.fc1 = nn.Linear(tamanho_achatado, 128)
        self.dropout = nn.Dropout(p=taxa_dropout)
        self.saida = nn.Linear(128, num_acoes)

    def forward(self, x):
        x = x.transpose(1, 2) 
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Achata a matriz 3D para um vetor 1D
        x = x.view(x.size(0), -1) 
        
        # Toma a decisão dos botões
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits_botoes = self.saida(x)
        
        return logits_botoes