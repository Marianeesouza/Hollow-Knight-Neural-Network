"""Exporta arrays de re-treinamento BC a partir das demonstrações capturadas.

Uso (a partir da raiz do projeto):
    python Python/build_training_set.py --dataset dataset/hornet_boss --output dataset/training_export

Opções:
    --alpha-ameaca 4.0  --alpha-abertura 2.0  --alpha-vida 1.0
= pesos Omega(s_t) da perda contextual (plan.md, seção 2).

O export é carregado no notebook Kaggle com:
    data = numpy.load('bc_training_arrays.npz')
    X, y, pos_weight, omega = data['X'], data['y'], data['pos_weight'], data['omega']
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from Python.Src.Data.TrainingSetBuilder import RetrainParams, export_training_arrays


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default=os.path.join('dataset', 'hornet_boss'),
                        help='Pasta com os CSVs de demonstração')
    parser.add_argument('--output', default=os.path.join('dataset', 'training_export'),
                        help='Pasta de saída do NPZ + manifest')
    parser.add_argument('--alpha-ameaca', type=float, default=4.0)
    parser.add_argument('--alpha-abertura', type=float, default=2.0)
    parser.add_argument('--alpha-vida', type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()

    refit = RetrainParams(alpha_ameaca=args.alpha_ameaca,
                          alpha_abertura=args.alpha_abertura,
                          alpha_vida=args.alpha_vida)

    dataset = os.path.join(PROJECT_ROOT, args.dataset) \
        if not os.path.isabs(args.dataset) else args.dataset
    output = os.path.join(PROJECT_ROOT, args.output) \
        if not os.path.isabs(args.output) else args.output

    print(f"Dataset: {dataset}\nSaída:   {output}")
    manifest = export_training_arrays(dataset, output, retrain_params=refit)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nArquivos gerados:")
    print(f"  - {os.path.join(output, manifest['npz'])}")
    print(f"  - {os.path.join(output, 'manifest.json')}")


if __name__ == '__main__':
    main()
