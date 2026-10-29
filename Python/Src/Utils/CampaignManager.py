"""Rotação automática de arquiteturas na campanha de testes.

Roda ``lutas_por_modelo`` combates por arquitetura (na ordem da lista),
registrando o placar. A reentrada automática da arena continua valendo
entre as trocas — sem intervenção humana.
"""


class CampaignManager:

    def __init__(self, tipos, lutas_por_modelo, modelos):
        """tipos: ordem da campanha | modelos: dict {tipo: nn.Module} já carregados."""
        self.queue = list(tipos)
        self.lutas_por_modelo = lutas_por_modelo
        self.modelos = modelos
        self.idx = -1
        self.tipo_atual = None
        self.modelo = None
        self.lutas_do_modelo = 0
        self.placar = {}

    def start(self):
        self.idx = 0
        self.tipo_atual = self.queue[0]
        self.modelo = self.modelos[self.tipo_atual]
        self.lutas_do_modelo = 0
        self.placar = {self.tipo_atual: {'WIN': 0, 'LOSS': 0}}
        return self.modelo

    def record_result(self, result):
        """Conta uma luta finalizada ('WIN' ou 'LOSS') para o modelo corrente."""
        resultado = result if result in ('WIN', 'LOSS') else 'LOSS'
        self.placar.setdefault(self.tipo_atual, {'WIN': 0, 'LOSS': 0})[resultado] += 1
        self.lutas_do_modelo += 1

    def quota_reached(self):
        return self.lutas_do_modelo >= self.lutas_por_modelo

    def next_model(self):
        """Avança para a próxima arquitetura da campanha.

        Retorna True se trocou de modelo; False se a campanha terminou.
        """
        if self.idx + 1 >= len(self.queue):
            return False
        self.idx += 1
        self.tipo_atual = self.queue[self.idx]
        self.modelo = self.modelos[self.tipo_atual]
        self.lutas_do_modelo = 0
        self.placar[self.tipo_atual] = {'WIN': 0, 'LOSS': 0}
        return True

    def summary(self):
        lines = ["===== RESULTADO DA CAMPANHA ====="]
        for tipo in self.queue:
            p = self.placar.get(tipo, {'WIN': 0, 'LOSS': 0})
            total = p['WIN'] + p['LOSS']
            lines.append(f"  {tipo}: {total} lutas | {p['WIN']} vitórias | {p['LOSS']} derrotas")
        lines.append("CSVs puros em dataset/test_runs/<modelo>/ | agregados em all_model_tests.csv")
        return "\n".join(lines)
