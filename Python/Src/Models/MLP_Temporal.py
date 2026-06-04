import torch.nn as nn
import torch.nn.functional as F

class MLP_Temporal(nn.Module):

    """
    MLP com janela deslizante: recebe vários frames empilhados (igual à CNN),
    achata tudo num vetor único e processa com camadas totalmente conectadas.
    Assim consegue considerar estados passados, sem usar convolução.
    """

    def __init__(self, num_features, num_acoes, tamanho_janela=10, taxa_dropout=0.2):
        super(MLP_Temporal, self).__init__()

        input_dim = num_features * tamanho_janela  # ex.: 104 * 10 = 1040

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.dropout = nn.Dropout(p=taxa_dropout)

        self.saida = nn.Linear(128, num_acoes)

    def forward(self, x):
        # x chega como (batch, janela, features) -> achata para (batch, janela*features)
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))

        logits_botoes = self.saida(x)
        return logits_botoes
