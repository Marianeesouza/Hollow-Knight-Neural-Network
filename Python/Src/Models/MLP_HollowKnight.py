import torch.nn as nn
import torch.nn.functional as F

class MLP_HollowKnight(nn.Module):

    """
    MLP projetada para processar os dados atuais sem uma janela temporal.
    A arquitetura é composta por camadas totalmente conectadas, com dropout para evitar overfitting
    """

    def __init__(self, num_features, num_acoes, taxa_dropout=0.2):
        super(MLP_HollowKnight, self).__init__()
        
        self.fc1 = nn.Linear(num_features, 256)
        
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        
        self.dropout = nn.Dropout(p=taxa_dropout)
        
        self.saida = nn.Linear(64, num_acoes)

    def forward(self, x):        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        
        logits_botoes = self.saida(x)
        return logits_botoes