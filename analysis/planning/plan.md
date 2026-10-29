# Relatório Técnico: Alinhamento Estratégico e Formulação Matemática de Avaliações Contextuais para *Behavior Cloning* em Jogos

A relação entre o guia prático de Hamel Husain (https://hamel.dev/blog/posts/field-guide/index.html) e o projeto acadêmico de *Behavior Cloning* em *Hollow Knight* (analysis\planning\Relatório_Redes_Neurais.pdf) é direta: **o próprio relatório comprovou, na prática, a principal tese de Hamel sobre a "armadilha das métricas genéricas"**. 

Hamel explica que adotar métricas padrão cria uma **falsa sensação de progresso**, pois elas medem dimensões agregadas que não refletem se o sistema realmente resolve o problema no contexto real. Isso ocorreu exatamente no seu experimento: a **MLP com Janela Temporal obteve o maior F1-score estático de teste (0,86)**, mas foi a que teve o **pior desempenho na arena de jogo**, com menor tempo de sobrevivência (55,36 s), menor dano médio (51,98%) e apenas uma vitória em 15 combates. Enquanto isso, a MLP Pura e a CNN 1-D, mesmo com F1 estático sutilmente menor, foram os agentes mais eficazes na arena (duas vitórias cada).

Com base no framework de Hamel Husain, as seguintes melhorias podem ser implementadas no projeto:

---

### 1. Substituir Métricas Estáticas Globais por Avaliações Contextuais (Domain-Specific Evals)
Em vez de avaliar apenas *precision*, *recall* e *F1-score* médios sobre as 8 ações binarizadas em milhares de quadros, a avaliação deve medir o sucesso de comportamentos atrelados às regras e dinâmicas do combate:

* **Taxa de Oportunidade e Segurança de Cura:** No relatório, foi observado que curas e esquivas correlacionam-se fortemente com a sobrevivência. No entanto, em vez de medir apenas o F1 global do Botão B (cura), o ideal é criar uma métrica contextual: *o comando de cura foi disparado apenas quando a Hornet estava em estado de repouso/atordoamento (`bossState`) ou a uma distância segura (\\(\Delta x, \Delta y\\))?* Disparar a cura na frente do chefe deve ser penalizado como erro grave, mesmo que o modelo acerte outros frames de cura.
* **Métrica de Coerência Vetorial e Alinhamento de Mira:** O relatório identificou uma falha qualitativa crítica: o modelo acertava a janela de ativação de feitiços, mas errava a orientação espacial, disparando magias para o lado oposto ao chefe. Uma métrica específica simples avaliaria: *quando o comando de magia ou ataque é acionado, a direção horizontal predita é congruente com a posição relativa (\\(\Delta x\\)) da Hornet?*.
* **Recall de Evasão por Tipo de Ataque do Chefe:** Grande parte do dataset é dominada por movimentação horizontal e pulos neutros. Uma métrica contextual deve segmentar o teste por estados da chefe (`bossState`), avaliando especificamente o *recall* de esquiva (dash ou pulo) durante os quadros que antecedem ataques perigosos (como investidas aéreas ou arremesso de agulha).

---

### 2. Implementar Análise de Erros *Bottom-Up*
Hamel destaca que a **análise de erros orientada aos dados (*bottom-up*) é a atividade de maior retorno (ROI) em IA**. No lugar de confiar apenas no resumo numérico de vitórias e perdas ao final de 60 partidas, deve-se categorizar as causas das derrotas:

* Registrar os 5 a 10 segundos finais de cada combate perdido e rotular a causa da morte em uma taxonomia de falhas, como por exemplo:
  1. *Falta de resposta física / latência na inferência* (gargalo sofrido pela arquitetura GRU em CPU);
  2. *Tentativa de cura durante animação ofensiva do chefe*;
  3. *Encurralamento em extremidades da arena* (perda de espaço por movimentação horizontal passiva);
  4. *Omissão de esquiva perante ataques específicos*.
* Como Hamel exemplifica com o caso da NurtureBoss, mapear a frequência dessas falhas revela que **poucos modos de erro costumam responder pela maioria dos fracassos**, permitindo atuar cirurgicamente nas deficiências do agente.

---

### 3. Criar um Visualizador de Dados Simples (*Data Viewer*)
Segundo o guia, o investimento mais relevante de uma equipe é **uma interface dedicada para inspecionar as decisões da IA no contexto da aplicação**. 
* No projeto de *Hollow Knight*, em vez de inspecionar apenas arquivos CSV desconectados da arena, um visualizador simples de telemetria sincronizado com a reprodução dos combates permitiria ver, quadro a quadro:
  * A posição relativa do chefe e do jogador;
  * As probabilidades (logits) de saída de cada uma das 8 ações;
  * Os botões efetivamente acionados sob o limiar ajustado (como o limiar de 0,2 utilizado).
* Isso tornaria evidente em quais quadros o modelo hesitou ou realizou previsões contraditórias (ex.: tentar correr para a esquerda e para a direita simultaneamente, ou acionar ataque sem mirar).

---

### 4. Avaliação Baseada em Cenários Críticos e Dados Sintéticos
Hamel recomenda avaliar os sistemas por **dimensões de cenários e casos de borda (*edge cases*)**.
* Em vez de calcular métricas apenas sobre a média global das 40 lutas de teste, é possível extrair ou gerar um conjunto de testes offline focado estritamente em **estados críticos**:
  * *Cenário A:* Chefe iniciando investida em alta velocidade com o jogador próximo;
  * *Cenário B:* Chefe vulnerável após ataque, com o jogador dispondo de recursos de Alma para conjuração;
  * *Cenário C:* Jogador com 1 ponto de vida e chefe distante.
* Avaliar o modelo nesses cenários específicos evita que a alta acurácia em situações rotineiras (como simplesmente andar pelo chão da arena) mascare a incapacidade de reagir a momentos decisivos da luta.

---

### 5. Estruturar o Progresso pelo "Funil de Capacidades" (*Capability Funnel*)
Hamel sugere que o avanço de um projeto de IA não deve ser medido apenas pelo resultado final ("vencer o chefe"), mas por **etapas progressivas de utilidade**. Para o agente de combate, o funil pode ser estruturado como:
1. **Navegação Básica:** Manter-se móvel sem ficar preso em cantos da arena;
2. **Defesa Reativa:** Esquivar de investidas e ataques diretos do chefe;
3. **Punição Ofensiva:** Conectar ataques básicos e magias direcionadas corretamente durante aberturas;
4. **Gerenciamento de Recursos:** Curar-se com segurança quando a vida estiver baixa;
5. **Vitória Consistente:** Derrotar o chefe com taxa de sucesso elevada.

Essa abordagem permite diagnosticar com exatidão em qual nível do funil cada arquitetura está travando (por exemplo, a GRU avançou em comportamentos complexos, mas travou na latência de resposta física).

---

## 1. O Diagnóstico: A "Armadilha das Métricas Genéricas" no Projeto

O projeto acadêmico de *Behavior Cloning* em *Hollow Knight* ilustra de forma paradigmática o que Hamel Husain define como a **armadilha das ferramentas e das métricas genéricas**. Segundo Husain, equipes frequentemente confiam em métricas agregadas padrão que criam uma **falsa sensação de medição e progresso**, otimizando dimensões que não se correlacionam com a resolução do problema no ambiente real.

No relatório da disciplina, essa dissociação manifestou-se empiricamente de forma explícita:
* A **MLP com Janela Temporal** atingiu o maior desempenho nas métricas estatísticas convencionais de classificação multirrótulo offline (F1-score micro de 0,86 e macro de 0,78).
* No entanto, ao ser colocada na arena contra a chefe Hornet Protector, essa mesma arquitetura apresentou a **pior resiliência e eficácia prática** entre todas as redes testadas: menor tempo médio de sobrevivência (55,36 s), menor dano médio acumulado (51,98%) e apenas uma vitória em 15 combates.
* Em contraste, arquiteturas com F1-score estático ligeiramente inferior (como a CNN 1-D e a MLP Pura, ambas com F1-score micro de 0,85) foram substancialmente mais eficazes em jogo, alcançando os maiores tempos de vida, maior agressividade e duas vitórias cada.

Essa contradição ocorre porque o F1-score médio calcula o acerto frame a frame sobre um conjunto desbalanceado. Uma rede pode acumular acertos triviais (como andar para a esquerda ou direita em momentos neutros) e alcançar uma métrica agregada elevada, enquanto falha catastroficamente em quadros esparsos e críticos de tomada de decisão (como esquivar de uma investida letal ou direcionar um ataque).

---

## 2. Aplicação dos Pilares de Hamel Husain ao Agente de Combate

Para transformar a avaliação desse sistema em um processo confiável de engenharia de IA, os princípios descritos por Husain no *Field Guide* devem ser transpostos para o pipeline de aprendizado por imitação:

### 2.1 Análise de Erros *Bottom-Up* (Taxonomia de Falhas)
Em vez de depender apenas do placar final de vitórias/derrotas ou da perda global de treino/teste, deve-se aplicar uma análise de erros orientada aos dados (*bottom-up*). O método consiste em examinar detalhadamente as derrotas do agente e categorizar as causas-raiz da morte em uma taxonomia estrita:
1. **Dano por Omissão Reativa:** O agente permaneceu no trajeto de um golpe sem acionar comandos evasivos.
2. **Dano por Ação Inoportuna:** O agente travou em uma animação lenta de cura enquanto a chefe executava uma investida ativa.
3. **Erro de Congruência Espacial:** O agente identificou a janela de oportunidade de feitiço, mas disparou para o lado oposto da arena.
4. **Encurralamento Posicional:** O agente recuou passivamente até o canto da arena, eliminando seus graus de liberdade.
5. **Atraso de Inferência (Gargalo Físico):** A predição correta ocorreu com latência temporal devido ao processamento concorrente em CPU, como diagnosticado na GRU.

Como Hamel demonstra no caso da NurtureBoss, mapear a frequência dessas falhas revela que poucos modos de erro costumam responder pela maioria dos fracassos.

### 2.2 Criação de um *Data Viewer* Customizado para Telemetria
Husain afirma que o investimento mais crítico para uma equipe de IA não é um painel genérico de gráficos, mas um visualizador customizado que elimine o atrito na inspeção dos dados. 
* No contexto do *Hollow Knight*, isso significa construir uma ferramenta simples que sobreponha, em sincronia frame a frame:
  * O replay visual do combate;
  * Os valores instantâneos do vetor de 104 atributos (distâncias relativas \\(\Delta x, \Delta y\\), velocidades, vida e estado da chefe);
  * Os logits contínuos \\(z_j\\) e as probabilidades sigmoides \\(\sigma(z_j)\\) para cada uma das 8 ações;
  * A sinalização do limiar \\(\tau = 0,2\\) e os comandos efetivamente despachados para o gamepad virtual.
Essa inspeção detalhada permite verificar se o modelo hesitou perto do limiar ou se tomou decisões contraditórias.

### 2.3 Avaliação Baseada em Fatiamento de Cenários Críticos (*Scenario Slicing*)
Husain recomenda substituir testes monolíticos por conjuntos de dados estruturados em torno de dimensões e cenários específicos. Em vez de avaliar a rede sobre o fluxo contínuo de 40 partidas de teste, o conjunto de validação offline deve ser fatiado em subconjuntos orientados ao contexto:
* *Cenário 1 (Defesa Crítica):* Quadros em que a chefe inicia animações de ataque de alta velocidade a curta distância.
* *Cenário 2 (Janela de Punição):* Quadros em que a chefe encontra-se em repouso após errar um golpe ou sofrer atordoamento.
* *Cenário 3 (Gestão de Vitalidade):* Quadros em que a vida do jogador está baixa (\\(HP < 1,0\\)) e há Alma disponível para conjuração (\\(Soul \ge 0,33\\)).

### 2.4 Funil de Capacidades (*Capability Funnel*) e Roteiro de Experimentos
O progresso do modelo não deve ser visto como uma entrega binária de uma funcionalidade ("o agente vence a Hornet"). Seguindo a metodologia do funil de capacidades proposta por Bryan Bischof e citada por Hamel, o desenvolvimento deve validar patamares sucessivos de utilidade:
\\[\text{Estabilidade Locomotora} \longrightarrow \text{Sobrevivência Reativa} \longrightarrow \text{Punição Básica} \longrightarrow \text{Gerenciamento de Recursos} \longrightarrow \text{Vitória Consistente}\\]
Isso permite isolar onde cada arquitetura falha no funil: por exemplo, a GRU dominou comportamentos complexos sequenciais, mas estagnou na sobrevivência reativa por conta do atraso da inferência em CPU.

---

## 3. Formulação Matemática das Métricas Específicas de Contexto

Para traduzir o comportamento qualitativo do combate em métricas quantitativas formais sem a necessidade de código, formaliza-se a dinâmica do sistema a partir do vetor de telemetria e das saídas do modelo.

### 3.1 Definições Formais e Espaço de Variáveis

Seja \\(t \in \{1, 2, \dots, T\}\\) o índice temporal discreto correspondente aos quadros capturados a cada ciclo de física do jogo.

* **Vetor de Estado \\(\mathbf{s}_t \in \mathbb{R}^{104}\\):**
  O estado normalizado é decomposto em variáveis operacionais:
  * \\(\Delta \mathbf{p}_t = (\Delta x_t, \Delta y_t) \in [-1, 1]^2\\): vetor de posição relativa entre o jogador e a chefe.
  * \\(\Delta \mathbf{v}_t = (\Delta v_{x,t}, \Delta v_{y,t})\\): vetor de velocidades relativas.
  * \\(h_t \in\\): vitalidade normalizada do jogador (\\(HP\\)).
  * \\(s_t \in\\): recurso de Alma normalizado (\\(Soul\\)).
  * \\(h_t^{\text{boss}} \in\\): vitalidade normalizada da chefe.
  * \\(\mathbf{b}_t \in \{0, 1\}^K\\): vetor *one-hot* indicando o estado de animação/ação corrente da chefe (\\(bossState\\)).

* **Vetor de Ações Preditas \\(\hat{\mathbf{a}}_t \in \{0, 1\}^8\\):**
  Dada a saída da rede \\(z_{j,t}\\) para cada ação \\(j \in \{1, \dots, 8\}\\):
  \\[\hat{a}_{j,t} = \mathbb{I}(\sigma(z_{j,t}) \ge \tau)\\]
  onde \\(\sigma(\cdot)\\) é a função sigmoide, \\(\tau \in (0, 1)\\) é o limiar de decisão (\\(\tau = 0,2\\) no ambiente de execução real) e \\(\mathbb{I}(\cdot)\\) denota a função indicadora.
  * \\(j = 1\\): Gatilho Direito (*Dash* / Esquiva)
  * \\(j = 2\\): Botão A (Pulo)
  * \\(j = 3\\): Botão B (Foco / Conjuração de Magia)
  * \\(j = 4\\): Botão X (Ataque Básico com Ferrão)
  * \\(j = 5\\): Direcional Esquerdo
  * \\(j = 6\\): Direcional Direito
  * \\(j = 7\\): Direcional Cima
  * \\(j = 8\\): Direcional Baixo

---

### 3.2 Métrica 1: Índice de Segurança e Oportunidade de Cura (\\(\mathcal{M}_{\text{cura}}\\))

O relatório empírico demonstrou que a longevidade dos agentes é fortemente correlacionada à média de curas bem-sucedidas (\\(r = 0,96\\)). No entanto, a cura exige canalização estática; acioná-la perto da chefe durante um ataque ativo resulta em dano inevitável.

Define-se o conjunto de estados seguros para canalização \\(\mathcal{S}_{\text{seguro}}\\) com base na distância euclidiana \\(\|\Delta \mathbf{p}_t\|_2\\) e nos estados da chefe:
\\[\mathcal{S}_{\text{seguro}} = \left\{ \mathbf{s}_t \;\middle|\; \|\Delta \mathbf{p}_t\|_2 \ge d_{\text{seguro}} \quad \lor \quad \mathbf{b}_t \in \mathcal{B}_{\text{atordoado}} \right\}\\]
onde \\(d_{\text{seguro}} \in (0, 1]\\) é a distância mínima de segurança e \\(\mathcal{B}_{\text{atordoado}}\\) é o subconjunto de estados em que a chefe está vulnerável ou recuperando-se de animação.

Seja \\(T_{\text{cura}} = \{t \in \{1, \dots, T\} \mid \hat{a}_{3,t} = 1 \land s_t \ge s_{\text{foco}}\}\\) o conjunto de quadros em que o agente aciona a cura com recurso de Alma suficiente.

1. **Taxa de Segurança de Cura (\\(\text{CSR}\\) - *Cure Safety Rate*):**
   Mede a proporção de ativações de cura que respeitaram o critério de segurança:
   \\[\text{CSR} = \frac{1}{|T_{\text{cura}}|} \sum_{t \in T_{\text{cura}}} \mathbb{I}(\mathbf{s}_t \in \mathcal{S}_{\text{seguro}})\\]

2. **Revocação de Oportunidade de Cura (\\(\text{COR}\\) - *Cure Opportunity Recall*):**
   Seja \\(T_{\text{oportuno}} = \{t \mid h_t \le h_{\text{crit}} \land s_t \ge s_{\text{foco}} \land \mathbf{s}_t \in \mathcal{S}_{\text{seguro}}\}\\) o conjunto de quadros em que a cura era necessária e segura:
   \\[\text{COR} = \frac{\sum_{t \in T_{\text{oportuno}}} \hat{a}_{3,t}}{|T_{\text{oportuno}}|}\\]

O índice combinado de eficácia de cura é a média harmônica ponderada entre precisão contextual e oportunidade:
\\[\mathcal{M}_{\text{cura}} = \frac{2 \cdot \text{CSR} \cdot \text{COR}}{\text{CSR} + \text{COR}}\\]

---

### 3.3 Métrica 2: Coerência Vetorial e Alinhamento de Mira (\\(\mathcal{M}_{\text{alinhamento}}\\))

O relatório identificou um erro qualitativo severo: a rede disparava feitiços ou ataques na janela temporal correta, mas com orientação invertida, desperdiçando recursos ofensivos contra o ar.

Define-se a orientação horizontal física requerida \\(\theta_t^* \in \{-1, +1\}\\) em direção à chefe como:
\\[\theta_t^* = \operatorname{sgn}(\Delta x_t)\\]
onde \\(\Delta x_t = x_{\text{chefe}} - x_{\text{jogador}}\\).

A orientação predita pelo modelo no eixo horizontal \\(\hat{\theta}_t \in \{-1, 0, +1\}\\) é dada pela combinação das saídas direcionais:
\\[\hat{\theta}_t = \hat{a}_{6,t} - \hat{a}_{5,t}\\]
(onde \\(\hat{\theta}_t = +1\\) representa direita, \\(-1\\) esquerda e \\(0\\) neutralidade/conflito).

Seja \\(T_{\text{ofensivo}} = \{t \in \{1, \dots, T\} \mid \hat{a}_{4,t} = 1 \lor \hat{a}_{3,t} = 1\}\\) o conjunto de quadros em que uma ação ofensiva (ataque físico ou magia) é executada.

A **Taxa de Congruência de Mira** (\\(\mathcal{M}_{\text{alinhamento}}\\)) avalia se o vetor de ação coincide com a posição relativa da chefe no instante do ataque:
\\[\mathcal{M}_{\text{alinhamento}} = \frac{1}{|T_{\text{ofensivo}}|} \sum_{t \in T_{\text{ofensivo}}} \mathbb{I}\left( \hat{\theta}_t \cdot \operatorname{sgn}(\Delta x_t) > 0 \right)\\]

Essa métrica penaliza formalmente disparos em que \\(\hat{\theta}_t \cdot \Delta x_t < 0\\) (disparos para o lado oposto) ou \\(\hat{\theta}_t = 0\\) (ataque neutro sem direcionamento quando \\(\Delta x_t \neq 0\\)).

---

### 3.4 Métrica 3: Sensibilidade de Evasão a Ataques Críticos (\\(\mathcal{M}_{\text{evasao}}\\))

No *Hollow Knight*, esquivas efetivas (*dash* ou pulo) correlacionam-se linearmente com o tempo de sobrevivência (\\(r = 0,92\\)). Uma métrica global de F1-score falha porque a grande maioria dos quadros não exige esquiva. 

Define-se o subconjunto de quadros sob ameaça iminente \\(T_{\text{ameaça}}\\):
\\[T_{\text{ameaça}} = \left\{ t \;\middle|\; \mathbf{b}_t \in \mathcal{B}_{\text{ataque\_ativo}} \quad \land \quad \|\Delta \mathbf{p}_t\|_2 \le d_{\text{ameaça}} \quad \land \quad \frac{d}{dt}\|\Delta \mathbf{p}_t\|_2 < 0 \right\}\\]
onde \\(\mathcal{B}_{\text{ataque\_ativo}}\\) representa os estados em que a Hornet executa investidas ou arremesso de agulha, e \\(\frac{d}{dt}\|\Delta \mathbf{p}_t\|_2 < 0\\) indica que a chefe está aproximando-se do jogador.

Define-se a resposta evasiva binária do agente no quadro \\(t\\) como:
\\[e_t = \hat{a}_{1,t} \lor \hat{a}_{2,t} = \max(\hat{a}_{1,t}, \hat{a}_{2,t})\\]
(indicando o acionamento de *dash* ou salto).

A **Sensibilidade de Evasão Contextual** (\\(\mathcal{M}_{\text{evasao}}\\)) é formulada como a revocação condicional estrita sobre a janela de ameaça:
\\[\mathcal{M}_{\text{evasao}} = \frac{1}{|T_{\text{ameaça}}|} \sum_{t \in T_{\text{ameaça}}} e_t\\]

Complementarmente, penaliza-se o gasto desnecessário de esquiva em estados de repouso (\\(T_{\text{neutro}}\\)) através da **Taxa de Evasão Fútil** (\\(\text{FER}\\) - *Futile Evasion Rate*):
\\[\text{FER} = \frac{1}{|T_{\text{neutro}}|} \sum_{t \in T_{\text{neutro}}} e_t\\]

O balanço evasivo do modelo passa a ser:
\\[\mathcal{M}_{\text{defesa}} = \mathcal{M}_{\text{evasao}} - \lambda \cdot \text{FER}\\]
onde \\(\lambda \in\\) é um fator de ponderação de custo de recursos.

---

### 3.5 Métrica 4: Índice de Punição em Aberturas Ofensivas (\\(\mathcal{M}_{\text{punicao}}\\))

A vitória contra a chefe exige converter janelas de vulnerabilidade em dano acumulado. O relatório apontou que CNN 1-D e GRU geraram maior dano médio (\\(\approx 70\%\\)) exatamente por sustentarem pressão ofensiva consistente.

Define-se o conjunto de quadros de abertura ofensiva \\(T_{\text{abertura}}\\):
\\[T_{\text{abertura}} = \left\{ t \;\middle|\; \mathbf{b}_t \in \mathcal{B}_{\text{vulneravel}} \quad \land \quad \|\Delta \mathbf{p}_t\|_2 \le d_{\text{alcance}} \right\}\\]
onde \\(d_{\text{alcance}}\\) representa a amplitude do ferrão ou raio da magia.

A ação ofensiva combinada é:
\\[u_t = \max(\hat{a}_{4,t}, \hat{a}_{3,t})\\]
(ataque com ferrão ou conjuração de feitiço).

A **Taxa de Punição Efetiva** é dada por:
\\[\mathcal{M}_{\text{punicao}} = \frac{1}{|T_{\text{abertura}}|} \sum_{t \in T_{\text{abertura}}} u_t \cdot \mathbb{I}\left( \hat{\theta}_t \cdot \operatorname{sgn}(\Delta x_t) > 0 \right)\\]
Esta formulação exige não apenas que o botão de ataque seja pressionado, mas que o golpe esteja corretamente orientado para acertar o alvo.

---

### 3.6 Métrica 5: Penalidade de Confinamento Espacial em Extremidades (\\(\mathcal{P}_{\text{canto}}\\))

No relatório, os atributos de posição absoluta na arena foram descartados para promover a invariância espacial do modelo. No entanto, agentes reativos frequentemente sofrem por acumular comandos de recuo que os prendem nas bordas da arena, onde a esquiva física torna-se impossível contra investidas da chefe.

Seja \\(x_t^{\text{arena}} \in [-1, 1]\\) a posição horizontal normalizada do jogador no espaço global da arena (dado disponível no snapshot bruto da telemetria antes da normalização).

Define-se a região de confinamento lateral \\(\mathcal{C}_{\text{borda}} = \{t \mid |x_t^{\text{arena}}| \ge x_{\text{limiar}}\}\\), onde \\(x_{\text{limiar}} \approx 0,85\\) representa a proximidade da parede.

A **Penalidade de Confinamento Espacial** mede a persistência com que o modelo continua pressionando a movimentação em direção à parede quando já está confinado:
\\[\mathcal{P}_{\text{canto}} = \frac{1}{|\mathcal{C}_{\text{borda}}|} \sum_{t \in \mathcal{C}_{\text{borda}}} \left[ \hat{a}_{5,t} \cdot \mathbb{I}(x_t^{\text{arena}} \le -x_{\text{limiar}}) + \hat{a}_{6,t} \cdot \mathbb{I}(x_t^{\text{arena}} \ge x_{\text{limiar}}) \right]\\]
Valores próximos de 0 indicam que o agente reconhece a proximidade da borda e busca reposicionar-se em direção ao centro da arena.

---

## 4. Estrutura Comparativa: Métricas Padrão vs. Métricas Contextuais

A tabela abaixo sintetiza como a substituição das métricas genéricas pelas formulações contextuais corrige as distorções apontadas por Hamel Husain:

| Mecânica de Jogo | Métrica Padrão Adotada no Projeto | Falha Diagnosticada por Hamel | Métrica Específica Proposta | Formulação Matemática Central |
| :--- | :--- | :--- | :--- | :--- |
| **Cura / Foco** | F1-Score do Botão B (\\(0,84 - 0,86\\)) | Confunde canalização segura com canalizações suicidas sob ataque. | Eficácia de Cura Contextual (\\(\mathcal{M}_{\text{cura}}\\)) | \\(\text{CSR} \times \text{COR}\\) condicionado a \\(\mathbf{s}_t \in \mathcal{S}_{\text{seguro}}\\) |
| **Uso de Feitiços** | F1-Score do Botão B com Alma | Ignora que o feitiço pode ser lançado na direção oposta ao chefe. | Coerência Vetorial (\\(\mathcal{M}_{\text{alinhamento}}\\)) | \\(\frac{1}{\|T\|} \sum \mathbb{I}(\hat{\theta}_t \cdot \operatorname{sgn}(\Delta x_t) > 0)\\) |
| **Esquiva e Salto** | F1-Score do Gatilho Dir. e Pulo (\\(0,78 - 0,90\\)) | Diluição estatística: a maioria dos quadros não apresenta ameaça. | Evasão a Ameaças Críticas (\\(\mathcal{M}_{\text{evasao}}\\)) | \\(\frac{1}{\|T_{\text{ameaca}}\|} \sum \max(\hat{a}_{1,t}, \hat{a}_{2,t})\\) |
| **Ataque com Ferrão** | Precision e Recall do Botão X | Não distingue golpes no vazio de conversão efetiva em aberturas. | Punição em Aberturas (\\(\mathcal{M}_{\text{punicao}}\\)) | \\(\frac{1}{\|T_{\text{abertura}}\|} \sum u_t \cdot \mathbb{I}(\text{alinhamento})\\) |
| **Movimentação** | F1-Score de Esquerda/Direita (\\(\approx 0,90\\)) | Alta acurácia em movimento plano mas o agente morre preso no canto. | Penalidade de Borda (\\(\mathcal{P}_{\text{canto}}\\)) | Frequência de comandos direcionados contra as paredes da arena |

---

## 5. Roteiro de Iteração Baseado em Experimentos

Hamel Husain defende que o roteiro (*roadmap*) de desenvolvimento de um produto de inteligência artificial deve mensurar o **número de experimentos rodados e aprendizados gerados**, e não um cronograma rígido de funcionalidades ou suposições teóricas.

Para dar continuidade ao projeto da UFRPE sob essa metodologia, o fluxo de melhoria iterativa deve ser estruturado em ciclos rápidos:

```
      ┌────────────────────────────────────────────────────────┐
      │  Execução do Agente na Arena (ou Validação Offline)     │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │   Inspeção via Data Viewer + Categorização de Falhas   │
      │   (Identificação do maior gargalo via Pareto)          │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │   Cálculo das Métricas Matemáticas Contextuais         │
      │   (Segurança de cura, coerência de mira, evasão)       │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │   Intervenção Focalizada (Engenharia de Dados/Perda)   │
      │   - Ajuste de pos_weight na BCE loss                   │
      │   - Filtragem de demonstrações ruidosas                │
      │   - Correção de latência de inferência via GPU         │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  └────────► Retorno ao ciclo
```

1. **Ciclo de Tempo Determinado (*Timeboxing*):** Em vez de tentar refatorar todo o pipeline de uma só vez, definir iterações semanais com foco exclusivo em uma métrica contextual específica (por exemplo, "Semana 1: zerar os feitiços disparados na direção oposta ao chefe").
2. **Engenharia da Função de Perda Orientada ao Contexto:** Como sugerido nos trabalhos futuros do relatório, o desbalanceamento de ações como "Cima" gerou F1-score crítico (0,26 a 0,35). A perda supervisionada (atualmente a *BCEWithLogitsLoss* com pesos uniformes) deve incorporar fatores de ponderação dinâmicos (\\(pos\_weight_j\\)) e multiplicadores de penalidade em quadros críticos de perigo iminente (\\(w_t > 1\\) se \\(\mathbf{s}_t \in T_{\text{ameaca}}\\)).
3. **Resolução do Gargalo Físico de Inferência:** O relatório documentou que o atraso na execução da GRU em CPU atrasou comandos essenciais. Essa falha, detectada empiricamente, ilustra que a latência é uma métrica de produto tão crítica quanto a própria capacidade de modelagem do agente.

Ao adotar essa abordagem fundamentada em análise de erros e métricas contextuais matematicamente formuladas, a equipe elimina a desconexão observada entre a modelagem teórica e o desempenho real na arena.

---
A reestruturação da função de perda visa resolver a raiz do problema apontado no projeto: **na *Binary Cross-Entropy* (BCE) padrão, todos os quadros (*frames*) e todas as ações têm exatamente o mesmo peso matemático**, tratando um frame neutro de corrida com a mesma importância de um frame crítico de esquiva letal.

No relatório original, o treinamento utilizou a média não ponderada da BCE sobre os \\(J = 8\\) botões:
\\[\mathcal{L}_{\text{BCE}}(\mathbf{z}_t, \mathbf{y}_t) = - \frac{1}{8} \sum_{j=1}^8 \Big[ y_{j,t} \log(\sigma(z_{j,t})) + (1 - y_{j,t}) \log(1 - \sigma(z_{j,t})) \Big]\\]

Para que a função de perda reflita o contexto do jogo e guie o gradiente em direção aos comportamentos que realmente determinam a vitória, a reestruturação pode ser dividida em **três camadas complementares**: balanceamento de raridade de classes, ponderação por criticidade do estado e termos de regularização contextual.

---

### 1. Ponderação por Desbalanceamento de Ações (\\(pos\_weight\\))

O relatório documentou que ações cruciais como **Cima** (F1 de 0,26 a 0,35), **Gatilho Direito** (esquiva) e **Botão B** (magia/cura) são muito esparsas no dataset em relação ao deslocamento horizontal neutro. Isso forçou a equipe a reduzir o limiar de inferência arbitrariamente para \\(\tau = 0,2\\).

Matematicamente, para cada ação \\(j \in \{1, \dots, 8\}\\), introduz-se um peso positivo \\(p_j\\):
\\[p_j = \frac{N - N_j^+}{N_j^+}\\]
onde \\(N\\) é o total de quadros no dataset e \\(N_j^+\\) é o número de quadros em que o botão \\(j\\) estava pressionado (\\(y_{j,t} = 1\\)).

A perda ponderada por classe torna-se:
\\[\ell_{\text{classe}}(z_{j,t}, y_{j,t}) = - \Big[ p_j \cdot y_{j,t} \log(\sigma(z_{j,t})) + (1 - y_{j,t}) \log(1 - \sigma(z_{j,t})) \Big]\\]

**Efeito no gradiente:**
O gradiente em relação ao logit \\(z_{j,t}\\) passa a ser:
\\[\frac{\partial \ell_{\text{classe}}}{\partial z_{j,t}} = \sigma(z_{j,t}) (1 + (p_j - 1)y_{j,t}) - p_j y_{j,t}\\]
* Quando \\(y_{j,t} = 1\\) (ação rara deveria ocorrer), o gradiente de penalidade pelo erro é multiplicado por \\(p_j\\). A rede é forçada a elevar a probabilidade predita dessas ações raras, eliminando a necessidade de limiares artificiais baixos na inferência.

---

### 2. Ponderação Temporal por Criticidade do Estado (\\(\Omega(\mathbf{s}_t)\\))

Alinhado ao conceito de *Scenario Slicing* e análise de erros de Hamel Husain, um erro cometido em um momento de perigo iminente deve gerar um gradiente muito maior do que um erro em um momento neutro.

Define-se um multiplicador de relevância do frame \\(\Omega(\mathbf{s}_t) \ge 1\\), parametrizado pelas variáveis do vetor de telemetria \\(\mathbf{s}_t\\):

\\[\Omega(\mathbf{s}_t) = 1 + \alpha_{\text{ameaça}} \cdot \mathbb{I}(\mathbf{s}_t \in T_{\text{ameaça}}) + \alpha_{\text{abertura}} \cdot \mathbb{I}(\mathbf{s}_t \in T_{\text{abertura}}) + \alpha_{\text{vida}} \cdot (1 - h_t)\\]

Onde:
* **\\(\mathbb{I}(\mathbf{s}_t \in T_{\text{ameaça}})\\):** Ativa-se quando a Hornet inicia animação de investida a uma distância euclidiana \\(\|\Delta \mathbf{p}_t\|_2 \le d_{\text{crítico}}\\). Configura-se \\(\alpha_{\text{ameaça}} > 0\\) (ex.: \\(3.0\\) a \\(5.0\\)) para que a omissão de esquiva ou pulo gere um sinal de erro massivo.
* **\\(\mathbb{I}(\mathbf{s}_t \in T_{\text{abertura}})\\):** Ativa-se quando o chefe está atordoado ou em recuperação de golpe dentro do alcance de ataque (\\(\|\Delta \mathbf{p}_t\|_2 \le d_{\text{alcance}}\\)). Penaliza severamente a hesitação ofensiva.
* **\\((1 - h_t)\\):** Fator proporcional à perda de vida do jogador. Conforme a vida \\(h_t \in\\) diminui, o custo de qualquer decisão incorreta é amplificado linearmente.

Aplicando ao frame:
\\[\mathcal{L}_{\text{estado}}(\mathbf{z}_t, \mathbf{y}_t, \mathbf{s}_t) = \Omega(\mathbf{s}_t) \cdot \left( \frac{1}{8} \sum_{j=1}^8 \ell_{\text{classe}}(z_{j,t}, y_{j,t}) \right)\\]

---

### 3. Termo de Regularização para Coerência Vetorial e Mira (\\(\mathcal{R}_{\text{mira}}\\))

Para eliminar o erro qualitativo observado no relatório — em que o agente conjurava magias ou desferia golpes na direção contrária à Hornet —, pode-se adicionar uma penalidade que acopla a previsão direcional à intenção de ataque.

Seja a orientação horizontal desejada em direção à chefe:
\\[\theta_t^* = \operatorname{sgn}(\Delta x_t) \in \{-1, +1\}\\]
E seja a diferença entre os logits preditos para os direcionais horizontal direito (\\(z_6\\)) e esquerdo (\\(z_5\\)):
\\[\Delta z_{\text{horizontal}, t} = z_{6,t} - z_{5,t}\\]

Sempre que o alvo de demonstração (ou a predição da rede) exigir um ataque básico ou feitiço (\\(y_{4,t} = 1 \lor y_{3,t} = 1\\)), penaliza-se a divergência de sinal através de uma perda de margem suave:

\\[\mathcal{R}_{\text{mira}}(\mathbf{z}_t, \mathbf{s}_t) = \max\Big(0, \,\, y_{\text{ofensivo}, t} \cdot \big[ 1 - \theta_t^* \cdot \tanh(\Delta z_{\text{horizontal}, t}) \big] \Big)\\]

* Se o chefe está à direita (\\(\theta_t^* = +1\\)), \\(\Delta z_{\text{horizontal}, t}\\) precisa ser positivo para que \\(\tanh(\cdot) \to +1\\), zerando a penalidade.
* Se a rede tentar atacar enquanto mantém o direcional apontado para a esquerda (\\(\theta_t^* \cdot \tanh(\dots) < 0\\)), o termo soma um valor positivo à perda total, empurrando o gradiente para inverter a direção do controle.

---

### 4. Formulação da Perda Total Unificada

Integrando todos os módulos, a nova função de perda para um lote de \\(B\\) instâncias durante o treinamento supervisionado torna-se:

\\[\mathcal{L}_{\text{total}} = \frac{1}{B} \sum_{t=1}^B \left[ \Omega(\mathbf{s}_t) \cdot \left( \frac{1}{8} \sum_{j=1}^8 \ell_{\text{classe}}(z_{j,t}, y_{j,t}) \right) + \lambda \cdot \mathcal{R}_{\text{mira}}(\mathbf{z}_t, \mathbf{s}_t) \right]\\]

Onde:
1. **\\(\ell_{\text{classe}}\\)** garante que botões raros recebam magnitude de gradiente proporcional à sua escassez (\\(pos\_weight\\)).
2. **\\(\Omega(\mathbf{s}_t)\\)** atua como uma atenção temporal contextual: quadros fáceis de navegação têm gradiente pequeno; quadros decisivos de perigo iminente dominam a atualização dos pesos da rede.
3. **\\(\mathcal{R}_{\text{mira}}\\)** força o alinhamento físico entre a intenção ofensiva e a geometria espacial da arena, com peso regulador \\(\lambda\\).

Essa formulação traduz diretamente o princípio de Hamel Husain: **em vez de deixar a rede otimizar cegamente uma média estatística global, a própria função de otimização passa a incorporar as regras críticas de sobrevivência e sucesso do produto.**