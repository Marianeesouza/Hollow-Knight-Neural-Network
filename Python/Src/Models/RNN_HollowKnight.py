import torch.nn as nn
import torch.nn.functional as F


class RNN_HollowKnight(nn.Module):
    """
    RNN para o Hollow Knight, projetada para processar uma janela temporal de dados.
    Diferente da CNN (convolução) e da MLP_Temporal (achatamento), a RNN percorre
    os frames em ordem e mantém um estado interno, considerando o passado de forma
    sequencial. Usamos a saída do ÚLTIMO frame da janela para decidir os botões.
    """

    def __init__(self, num_features, num_acoes, tamanho_janela=10,
                 tamanho_oculto=128, num_camadas=2, taxa_dropout=0.2):
        super(RNN_HollowKnight, self).__init__()

        # batch_first=True -> espera entrada (batch, janela, features), igual ao resto do pipeline.
        # dropout entre as camadas da RNN só atua quando num_camadas > 1.
        self.rnn = nn.RNN(
            input_size=num_features,
            hidden_size=tamanho_oculto,
            num_layers=num_camadas,
            batch_first=True,
            dropout=taxa_dropout if num_camadas > 1 else 0.0,
        )

        self.dropout = nn.Dropout(p=taxa_dropout)
        self.saida = nn.Linear(tamanho_oculto, num_acoes)

    def forward(self, x):
        # x chega como (batch, janela, features)
        # saidas: (batch, janela, tamanho_oculto) com a saída de cada frame
        saidas, _ = self.rnn(x)

        # pega só o estado do ÚLTIMO frame da janela (resumo de toda a sequência)
        ultimo_frame = saidas[:, -1, :]

        x = self.dropout(ultimo_frame)
        logits_botoes = self.saida(x)
        return logits_botoes
