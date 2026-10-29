"""Gera/atualiza o registro de estados do boss a partir das demonstrações.

Uso (a partir da raiz do projeto):
    python Python/build_boss_state_registry.py
    python Python/build_boss_state_registry.py --names-from boss_states_dump.txt

O registro `Python/Src/Data/boss_state_config.json` ganha uma linha por estado
declarado na FSM "Control" da Hornet (bossStateAmount, 77 no dataset atual), com
os campos:
    state_id, state_name, is_offensive, is_vulnerable, reviewed, notes
mais um bloco `stats` (apenas referência, gerado automaticamente).

Hipóteses automáticas (para revisão na arena via viewer):
    is_offensive  <- score = 0.5*min(|bvx|/25, 1) + 0.3*min(|bvy|/30, 1)
                              + 0.2*fração de dist fechando,  limiar 0.30
    is_vulnerable <- fração da duração do estado cujo quadro anterior é um
                     estado ofensivo (janela de punição), limiar 0.35
Estados com `reviewed: true` (confirmados por você) nunca são sobrescritos;
state_name tem preenchimento único (nunca é apagado/renomeado pelo script).
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(PROJECT_ROOT, 'Python', 'Src', 'Data',
                             'boss_state_config.json')

THRESH_OFFENSIVE = 0.30
THRESH_VULNERABLE = 0.35
MIN_FRAMES = 30


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default=os.path.join('dataset', 'hornet_boss'),
                        help='Pasta com os CSVs de demonstração')
    parser.add_argument('--out', default=REGISTRY_PATH,
                        help='Caminho do JSON de registro')
    parser.add_argument('--names-from', default=None,
                        help='boss_states_dump.txt do mod C# (id<TAB>name por linha)')
    parser.add_argument('--force-recreate', action='store_true',
                        help='Descarta o registro existente (reviewed perdido!)')
    return parser.parse_args()


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_names(dump_path):
    """Lê boss_states_dump.txt: 'id\\tname' por linha."""
    names = {}
    with open(dump_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            sid, name = line.split('\t', 1)
            try:
                names[int(sid)] = name.strip()
            except ValueError:
                continue
    return names


def scan_dataset(dataset_dir):
    """Estatísticas por bossState em todos os demos (dist/shift por luta)."""
    files = sorted(glob.glob(os.path.join(dataset_dir, 'demo_*.csv')))
    if not files:
        raise SystemExit(f"Nenhum demo_*.csv em {dataset_dir}")

    frames = []
    for path in files:
        df = pd.read_csv(path, usecols=['bossState', 'bossStateAmount',
                                        'bvx', 'bvy', 'bx', 'by'])
        dist = ((df['bx'] - df['bx'].shift()) ** 2 +
                (df['by'] - df['by'].shift()) ** 2).pow(0.5)
        df['dist'] = dist
        df['dist_closing'] = (dist.shift(1) - dist) > 0
        df['_file'] = path
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    state = df['bossState'].astype(int)
    total = len(df)
    n_states = max(int(df['bossStateAmount'].max()), int(state.max()) + 1)

    rows = {}
    for sid, g in df.groupby(state):
        rows[int(sid)] = {
            'frames': int(len(g)),
            'pct': round(len(g) / total * 100, 3),
            'mean_abs_bvx': round(float(g['bvx'].abs().mean()), 3),
            'mean_abs_bvy': round(float(g['bvy'].abs().mean()), 3),
            'closing_frac': round(float(g['dist_closing'].mean()), 3),
            'mean_dist': round(float(g['dist'].mean()), 3),
        }
    return df, state, rows, total, n_states


def offensive_score(st):
    s = 0.5 * min(st['mean_abs_bvx'] / 25.0, 1.0)
    s += 0.3 * min(st['mean_abs_bvy'] / 30.0, 1.0)
    s += 0.2 * st['closing_frac']
    return round(s, 3)


def post_offensive_fracs(df, state, offensive_ids):
    """Fração das durações (runs de um mesmo estado) precedidas por uma run de
    estado ofensivo — aproxima a janela de punição pós-ataque."""
    lens, post = {}, {}
    for _, g in df.groupby('_file', sort=False):
        s = state.loc[g.index].to_numpy()
        starts = np.r_[0, np.where(s[1:] != s[:-1])[0] + 1]
        run_states = s[starts]
        ends = np.r_[starts[1:], len(s)]
        for k in range(len(starts)):
            sid = int(run_states[k])
            n = int(ends[k] - starts[k])
            lens[sid] = lens.get(sid, 0) + n
            if k > 0 and int(run_states[k - 1]) in offensive_ids:
                post[sid] = post.get(sid, 0) + n
    return {sid: (round(post.get(sid, 0) / lens[sid], 3) if lens.get(sid) else 0.0)
            for sid in lens}


def main():
    args = parse_args()
    dataset = resolve(args.dataset)
    out_path = resolve(args.out)

    df, state, stats, total, n_states = scan_dataset(dataset)
    scores = {sid: offensive_score(st) for sid, st in stats.items()}
    offensive_ids = sorted(
        sid for sid, sc in scores.items()
        if sc >= THRESH_OFFENSIVE and stats[sid]['frames'] >= MIN_FRAMES)
    print(f"Lutas analisadas | frames: {total:,} | estados declarados: "
          f"{n_states} | observados: {len(stats)}")
    print(f"Hipóteses ofensivas (score>={THRESH_OFFENSIVE}): {offensive_ids}")

    vuln_candidates = []
    if offensive_ids:
        fracs = post_offensive_fracs(df, state, offensive_ids)
        vuln_candidates = sorted(
            sid for sid, frac in fracs.items()
            if frac >= THRESH_VULNERABLE and stats.get(sid, {}).get('frames', 0) >= MIN_FRAMES)
    print(f"Hipóteses vulneráveis (janela pós-ofensiva>={THRESH_VULNERABLE}): "
          f"{vuln_candidates}")

    existing = {}
    if os.path.exists(out_path) and not args.force_recreate:
        with open(out_path, 'r', encoding='utf-8') as f:
            old = json.load(f)
        existing = {st['state_id']: st for st in old.get('states', [])}

    dump_names = load_names(args.names_from) if args.names_from else {}
    if dump_names:
        print(f"Nomes do dump C#: {len(dump_names)} linhas carregadas")

    def make_stats(sid, old_stats=None):
        st = dict(stats.get(sid, {'frames': 0, 'pct': 0.0, 'mean_abs_bvx': 0.0,
                                  'mean_abs_bvy': 0.0, 'closing_frac': 0.0,
                                  'mean_dist': None}))
        st['offensive_score'] = scores.get(sid, 0.0)
        return st

    states = []
    for sid in range(n_states):
        old = existing.get(sid, {})
        reviewed = bool(old.get('reviewed', False))

        state_name = old.get('state_name')
        if state_name in (None, ''):
            state_name = dump_names.get(sid)

        is_off = old.get('is_offensive', False)
        if not reviewed:
            is_off = sid in offensive_ids

        is_vuln = old.get('is_vulnerable', False)
        if not reviewed:
            is_vuln = sid in vuln_candidates

        states.append({
            'state_id': sid,
            'state_name': state_name,
            'is_offensive': is_off,
            'is_vulnerable': is_vuln,
            'reviewed': reviewed,
            'notes': old.get('notes', ''),
            'stats': make_stats(sid),
        })

    registry = {
        '_doc': ("Registro de estados da FSM 'Control' da Hornet (state_id = "
                 "índice em FsmStates). Preencha state_name (dump C# ou "
                 "observação), confirme/corrija is_offensive e is_vulnerable e "
                 "margue reviewed: true. is_offensive = ataque ativo (janela "
                 "T_ameaca); is_vulnerable = vulnerável/recuperando "
                 "(T_abertura). Bloco 'stats' é referência automática."),
        'boss': 'Hornet Protector',
        'states': states,
        'hypothesis_offensive': offensive_ids,
        '_thresholds': {'offensive': THRESH_OFFENSIVE,
                        'vulnerable': THRESH_VULNERABLE,
                        'min_frames': MIN_FRAMES},
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"Registro salvo: {out_path} ({n_states} estados)")


if __name__ == '__main__':
    main()
