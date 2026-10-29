# Passo a passo — Estados do boss (registro + screenshots + revisão)

Fluxo para deixar tudo rodando sozinho e revisar os estados da Hornet depois.

## 1. Compilar o mod (uma vez)

1. Abra `C#\DataExtractor\DataExtractor.sln` no **Visual Studio** e compile (Debug).
2. Copie a `DataExtractor.dll` de `C#\DataExtractor\DataExtractor\bin\Debug\` para
   a pasta de plugins do jogo (substitua a antiga):

   ```
   ...\Hollow Knight\BepInEx\plugins\
   ```

O mod agora, automaticamente, ao detectar o boss:
- salva **`boss_states_dump.txt`** ao lado da DLL (id → nome dos 77 estados da FSM);
- tira **1 screenshot por estado** na 1ª ocorrência de cada um na sessão
  (`boss_state_shots/state_<id>.png` + `captures_log.txt` com data/hora).

> Esses arquivos vão para a **mesma pasta da DLL** (`BepInEx\plugins\`).
> Apague `boss_states_dump.txt` e/ou a pasta `boss_state_shots\` para regenerar.

## 2. Rodar a campanha de testes

Com o jogo aberto na estátua da Hornet (Hall of Gods):

```powershell
cd Python
..\venv\Scripts\python.exe main_tests.py
```

A campanha (`CAMPANHA_MODELOS` × `LUTAS_POR_MODELO`) roda sozinha, incluindo
reentrada automática nas lutas. Enquanto roda, gera:
- `dataset/test_runs/<modelo>/teste_*.csv` — telemetria frame a frame dos testes;
- linha de resumo por luta em `all_model_tests.csv`.

Pode deixar rodando e fazer outras coisas — nada do passo seguinte depende disso.

## 3. Preencher os nomes dos estados no registro

Volte para a **raiz do projeto** (`cd ..`). Depois da primeira luta (com a DLL nova):

```powershell
.\venv\Scripts\python.exe Python\build_boss_state_registry.py --names-from "C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight\BepInEx\plugins\boss_states_dump.txt"
```

Isso preenche `state_name` dos 77 estados em `Python\Src\Data\boss_state_config.json`
(sem tocar nos estados que você já marcou `reviewed: true`).

## 4. Revisar as hipóteses no viewer

Ainda na **raiz do projeto**:

```powershell
.\venv\Scripts\streamlit.exe run analysis\app.py
```

Na aba **Batalha**:
- faixa **bossState** com hover `nome (id)` e realce: vermelho = `is_offensive`,
  laranja = `is_vulnerable` (hipóteses automáticas dos demos);
- expander **Estatísticas por estado** — frames, % de uso, |bvx|/|bvy|, closing, score;
- expander **Capturas de tela dos estados** — a foto de cada estado junto das
  suas características (caminho da pasta detectado/salvo em `analysis\settings.json`).

## 5. Confirmar no registro

Abra `Python\Src\Data\boss_state_config.json` e, para cada estado conferido na arena/viewer:
1. corrija `state_name`, `is_offensive`, `is_vulnerable` (e anote em `notes` se quiser);
2. marque **`"reviewed": true`** — estados revisados nunca são sobrescritos pelo gerador.

As métricas contextuais (`M_cura`, `M_evasao`, `M_punicao`, ...) usam
`is_offensive` (janela de ameaça) e `is_vulnerable` (janela de abertura)
automaticamente — não há outra config pra editar.

## Referência rápida

| O quê                     | Onde                                                     |
|---------------------------|----------------------------------------------------------|
| Registro dos estados      | `Python\Src\Data\boss_state_config.json`                 |
| Gerador do registro       | `python Python\build_boss_state_registry.py`             |
| Teste do bot (campanha)   | `cd Python` → `python main_tests.py`                     |
| Logs por frame dos testes | `dataset\test_runs\<modelo>\teste_*.csv`                 |
| Resumo por luta           | `all_model_tests.csv`                                    |
| Dump de nomes / screenshots | `BepInEx\plugins\boss_states_dump.txt` / `boss_state_shots\` |
| Viewer                    | `streamlit run analysis\app.py`                          |
