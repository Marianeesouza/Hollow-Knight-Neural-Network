"""Valida se as métricas contextuais predizem sobrevivência/dano por sessão.

Problema que este script resolve: o relatório do projeto reporta r=0,96
(curas) e r=0,92 (esquivas) calculados sobre apenas 4 pontos — a média de
cada arquitetura. Isso não é evidência de que a métrica prediz desempenho;
com 4 pontos quase qualquer correlação alta é possível por acaso. Aqui a
correlação é recalculada por SESSÃO (n≈60, i.e. todas as lutas de teste de
todas as arquiteturas juntas), que é a unidade de observação correta.

Para cada métrica contextual (M_cura, M_alinhamento, M_evasao, M_defesa,
M_punicao, FER, CSR, COR, P_canto) o script reporta:
  - correlação de Pearson com 3 desfechos contínuos: tempo de sobrevivência,
    dano causado ao chefe, dano recebido pelo jogador;
  - IC95% por bootstrap (percentil, 2000 reamostragens) em vez de assumir
    normalidade;
  - diferença entre sessões WIN e LOSS (Mann-Whitney U, não paramétrico,
    mais apropriado que o Welch t-test para n pequeno e possível
    não-normalidade) com effect size (r = Z / sqrt(n)).

Uso:
    python validate_metrics.py --project-root /caminho/para/o/projeto
    python validate_metrics.py --project-root . --teacher-forced  # só expert
    python validate_metrics.py --project-root . --on-policy        # só modelo

Por padrão avalia os dois modos (teacher-forced com `source='expert'` e
on-policy com `source='model'`) separadamente e imprime os dois, porque uma
métrica pode prever bem num modo e mal no outro — o modelo joga diferente
do especialista, então a mesma métrica pode medir coisas distintas.

Saída: tabela no console + `validation_results.csv` em --output-dir com uma
linha por (modo, métrica, desfecho).
"""

import argparse
import glob
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

TEST_MODEL_DIRS = {
    'cnn': 'CNN',
    'gru': 'GRU',
    'mlp': 'MLP',
    'mlp_temporal': 'MLP Temporal',
}

METRIC_KEYS = ['CSR', 'COR', 'M_cura', 'M_alinhamento', 'M_evasao', 'FER',
               'M_defesa', 'M_punicao', 'P_canto']

# Métricas em que valor MENOR é melhor desempenho (sinal invertido ao
# interpretar o sentido da correlação, não no cálculo).
LOWER_IS_BETTER = {'P_canto'}

OUTCOME_COLS = ['Survival_Time_Sec', 'Damage_Dealt', 'Player_Damage_Taken']
OUTCOME_LABELS_PT = {
    'Survival_Time_Sec': 'Tempo de sobrevivência (s)',
    'Damage_Dealt': 'Dano causado ao chefe',
    'Player_Damage_Taken': 'Dano recebido pelo jogador',
}

N_BOOTSTRAP = 2000
RNG_SEED = 42


def find_test_sessions(project_root: str) -> List[Tuple[str, str]]:
    """Retorna [(caminho_csv, chave_modelo)] em dataset/test_runs/<modelo>/*.csv."""
    root = os.path.join(project_root, 'dataset', 'test_runs')
    out = []
    for model_key in TEST_MODEL_DIRS:
        for path in sorted(glob.glob(os.path.join(root, model_key, '*.csv'))):
            out.append((path, model_key))
    if not out:
        # fallback: qualquer subpasta sob test_runs, não só as 4 conhecidas
        for path in sorted(glob.glob(os.path.join(root, '*', '*.csv'))):
            model_key = os.path.basename(os.path.dirname(path))
            out.append((path, model_key))
    return out


def session_outcomes(df: pd.DataFrame) -> Optional[Dict]:
    """Replica a lógica de app.py:load_test_run_summaries para um único CSV."""
    required = {'timestamp', 'hp', 'bossHp'}
    if df.empty or not required.issubset(df.columns):
        return None

    df = df.sort_values('timestamp')
    boss_hp = pd.to_numeric(df['bossHp'], errors='coerce').dropna()
    player_hp = pd.to_numeric(df['hp'], errors='coerce').dropna()
    timestamps = pd.to_numeric(df['timestamp'], errors='coerce').dropna()
    if boss_hp.empty or player_hp.empty:
        return None

    boss_start, boss_end = float(boss_hp.iloc[0]), float(boss_hp.iloc[-1])
    player_start, player_end = float(player_hp.iloc[0]), float(player_hp.iloc[-1])
    survival_time = (
        max(0.0, float(timestamps.iloc[-1] - timestamps.iloc[0]))
        if len(timestamps) >= 2 else float('nan')
    )

    return {
        'Result': 'WIN' if boss_end <= 0 else 'LOSS',
        'Survival_Time_Sec': survival_time,
        'Damage_Dealt': max(0.0, boss_start - boss_end),
        'Player_Damage_Taken': max(0.0, player_start - player_end),
    }


def load_session_metrics(
    project_root: str, source: str,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Carrega cada sessão de teste e computa métricas contextuais + desfechos.

    source: 'model' (on-policy: ações do próprio modelo avaliado) ou
    'expert' (teacher-forced: reavalia as métricas assumindo as ações do
    especialista no mesmo estado, quando a coluna existir no CSV).
    """
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from Python.Src.Data.TelemetryDataset import load_session_csv
    from Python.Src.Utils.ContextMetrics import (
        MetricParams, compute_contextual_metrics, load_boss_state_config,
    )

    config = load_boss_state_config()
    params = MetricParams()

    rows, failures = [], []
    for path, model_key in find_test_sessions(project_root):
        try:
            df = load_session_csv(path)
        except Exception as exc:  # noqa: BLE001
            failures.append((path, f'load_session_csv: {exc!r}'))
            continue
        if df.empty:
            continue

        outcomes = session_outcomes(df)
        if outcomes is None:
            failures.append((path, 'colunas timestamp/hp/bossHp ausentes ou vazias'))
            continue

        try:
            res = compute_contextual_metrics(df, params=params, config=config, source=source)
        except Exception as exc:  # noqa: BLE001
            failures.append((path, f'compute_contextual_metrics({source}): {exc!r}'))
            continue

        row = {
            'Model': TEST_MODEL_DIRS.get(model_key, model_key),
            'Session': os.path.basename(path),
            'n_frames': len(df),
            **outcomes,
        }
        for k in METRIC_KEYS:
            row[k] = res.get(k, np.nan)
        rows.append(row)

    cols = ['Model', 'Session', 'n_frames', 'Result'] + OUTCOME_COLS + METRIC_KEYS
    return pd.DataFrame(rows, columns=cols), failures


def pearson_with_bootstrap_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED,
) -> Dict:
    """Pearson r + IC95% por bootstrap percentil + p-valor (teste t de r)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4 or np.std(x) == 0 or np.std(y) == 0:
        return {'n': n, 'r': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p': np.nan}

    r = float(np.corrcoef(x, y)[0, 1])

    if HAS_SCIPY:
        _, p = _scipy_stats.pearsonr(x, y)
    else:
        # fallback: estatística t com n-2 graus de liberdade, aproximação
        # normal para o p-valor (menos precisa que scipy, mas sem dependência)
        t_stat = r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
        p = 2 * (1 - _normal_cdf(abs(t_stat)))

    rng = np.random.default_rng(seed)
    boot_r = np.empty(n_boot)
    idx = np.arange(n)
    for i in range(n_boot):
        sample = rng.choice(idx, size=n, replace=True)
        xs, ys = x[sample], y[sample]
        if np.std(xs) == 0 or np.std(ys) == 0:
            boot_r[i] = np.nan
        else:
            boot_r[i] = np.corrcoef(xs, ys)[0, 1]
    boot_r = boot_r[~np.isnan(boot_r)]
    if len(boot_r) < n_boot * 0.5:
        ci_low, ci_high = np.nan, np.nan
    else:
        ci_low, ci_high = np.percentile(boot_r, [2.5, 97.5])

    return {'n': n, 'r': r, 'ci_low': float(ci_low), 'ci_high': float(ci_high), 'p': float(p)}


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + np.math.erf(z / np.sqrt(2)))


def mannwhitney_win_vs_loss(values: np.ndarray, result: np.ndarray) -> Dict:
    """Compara a métrica entre sessões WIN e LOSS (Mann-Whitney U)."""
    win = values[(result == 'WIN') & ~np.isnan(values)]
    loss = values[(result == 'LOSS') & ~np.isnan(values)]
    n1, n2 = len(win), len(loss)
    if n1 < 2 or n2 < 2:
        return {'n_win': n1, 'n_loss': n2, 'median_win': np.nan, 'median_loss': np.nan,
                'p': np.nan, 'effect_r': np.nan}

    median_win, median_loss = float(np.median(win)), float(np.median(loss))

    if HAS_SCIPY:
        u_stat, p = _scipy_stats.mannwhitneyu(win, loss, alternative='two-sided')
        n = n1 + n2
        mean_u = n1 * n2 / 2
        std_u = np.sqrt(n1 * n2 * (n + 1) / 12)
        z = (u_stat - mean_u) / std_u if std_u > 0 else np.nan
        effect_r = float(z / np.sqrt(n)) if std_u > 0 else np.nan
    else:
        p, effect_r = np.nan, np.nan

    return {'n_win': n1, 'n_loss': n2, 'median_win': median_win, 'median_loss': median_loss,
            'p': float(p) if p == p else np.nan, 'effect_r': effect_r}


def run_validation(session_df: pd.DataFrame, mode_label: str) -> pd.DataFrame:
    rows = []
    for metric in METRIC_KEYS:
        if metric not in session_df.columns:
            continue
        values = session_df[metric].to_numpy(dtype=float)

        for outcome in OUTCOME_COLS:
            y = session_df[outcome].to_numpy(dtype=float)
            res = pearson_with_bootstrap_ci(values, y)
            rows.append({
                'mode': mode_label, 'metric': metric, 'comparison': outcome,
                'n': res['n'], 'statistic': res['r'],
                'ci_low': res['ci_low'], 'ci_high': res['ci_high'], 'p': res['p'],
                'lower_is_better': metric in LOWER_IS_BETTER,
            })

        mw = mannwhitney_win_vs_loss(values, session_df['Result'].to_numpy())
        rows.append({
            'mode': mode_label, 'metric': metric, 'comparison': 'WIN_vs_LOSS (Mann-Whitney)',
            'n': mw['n_win'] + mw['n_loss'], 'statistic': mw['effect_r'],
            'ci_low': np.nan, 'ci_high': np.nan, 'p': mw['p'],
            'lower_is_better': metric in LOWER_IS_BETTER,
            'median_win': mw['median_win'], 'median_loss': mw['median_loss'],
        })

    return pd.DataFrame(rows)


def print_report(results: pd.DataFrame, mode_label: str, n_sessions: int):
    print(f"\n{'=' * 70}")
    print(f"Modo: {mode_label}  (n = {n_sessions} sessões)")
    print(f"{'=' * 70}")
    sub = results[results['mode'] == mode_label]
    for metric in METRIC_KEYS:
        m = sub[sub['metric'] == metric]
        if m.empty:
            continue
        flag = ' (menor = melhor)' if metric in LOWER_IS_BETTER else ''
        print(f"\n{metric}{flag}:")
        for _, row in m.iterrows():
            if 'Mann-Whitney' in row['comparison']:
                sig = '*' if (row['p'] == row['p'] and row['p'] < 0.05) else ' '
                print(f"  {sig} WIN vs LOSS: mediana WIN={row.get('median_win', float('nan')):.3f}"
                      f" | LOSS={row.get('median_loss', float('nan')):.3f}"
                      f" | effect r={row['statistic']:.3f} | p={row['p']:.3f} (n={row['n']})")
            else:
                sig = '*' if (row['p'] == row['p'] and row['p'] < 0.05) else ' '
                label = OUTCOME_LABELS_PT.get(row['comparison'], row['comparison'])
                print(f"  {sig} vs {label}: r={row['statistic']:.3f}"
                      f" [IC95% {row['ci_low']:.3f}, {row['ci_high']:.3f}]"
                      f" | p={row['p']:.3f} (n={row['n']})")
    print("\n(* = p < 0.05; IC95% calculado por bootstrap, não assume normalidade)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project-root', required=True,
                     help='Raiz do projeto (contém dataset/test_runs/ e Python/Src/...)')
    ap.add_argument('--output-dir', default='.', help='Onde salvar validation_results.csv')
    ap.add_argument('--teacher-forced', action='store_true',
                     help='Avalia só com source="expert" (ignora --on-policy)')
    ap.add_argument('--on-policy', action='store_true',
                     help='Avalia só com source="model" (ignora --teacher-forced)')
    args = ap.parse_args()

    if not HAS_SCIPY:
        warnings.warn(
            "scipy não encontrado: p-valores usam uma aproximação normal "
            "(menos precisa) e o teste de Mann-Whitney fica indisponível. "
            "Instale com `pip install scipy` para resultados completos.")

    modes = []
    if args.teacher_forced and not args.on_policy:
        modes = [('expert', 'Teacher-forced (ações do especialista)')]
    elif args.on_policy and not args.teacher_forced:
        modes = [('model', 'On-policy (ações do próprio modelo)')]
    else:
        modes = [
            ('model', 'On-policy (ações do próprio modelo)'),
            ('expert', 'Teacher-forced (ações do especialista)'),
        ]

    all_results = []
    for source, mode_label in modes:
        session_df, failures = load_session_metrics(args.project_root, source=source)
        if session_df.empty:
            print(f"\n[{mode_label}] Nenhuma sessão válida encontrada em "
                  f"{os.path.join(args.project_root, 'dataset', 'test_runs')}.")
            for path, reason in failures:
                print(f"  falha: {os.path.basename(path)} — {reason}")
            continue

        if failures:
            print(f"\n[{mode_label}] {len(failures)} sessão(ões) ignorada(s):")
            for path, reason in failures:
                print(f"  - {os.path.basename(path)}: {reason}")

        results = run_validation(session_df, mode_label)
        print_report(results, mode_label, n_sessions=len(session_df))
        all_results.append(results)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out_path = os.path.join(args.output_dir, 'validation_results.csv')
        combined.to_csv(out_path, index=False)
        print(f"\nResultados salvos em: {out_path}")
    else:
        print("\nNenhum resultado gerado — verifique --project-root.")


if __name__ == '__main__':
    main()