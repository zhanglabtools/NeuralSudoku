#!/usr/bin/env python3
"""Recompute Tables 2/5 and C.1/E.2 from the supplied per-puzzle records.

Run from the supplement directory:
    python scripts/verify_reported_results.py --root .

Requires Python 3 and NumPy. This performs statistical verification of saved
outputs; it does not retrain models or rerun inference. All rates in the CSV are
percentages; standard deviations in the JSON are sample SDs (ddof=1), in
percentage points. The zero-result hypergraph seed 1 is retained.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_A07 = {
    'graph': [(269486, 4309), (269917, 4393), (263990, 4076)],
    'hypergraph': [(271734, 3985), (0, 0), (273702, 4050)],
    'joint': [(277718, 4622), (273558, 4418), (279024, 4612)],
}
EXPECTED_A08 = [(299657, 6356), (299162, 6279), (300083, 6406)]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(root, output_dir):
    results = root / 'results'
    clean_ids = np.load(results / 'clean_evaluation/test_global_ids.npy', allow_pickle=False)
    rows, checks, sources = [], [], []

    def check(name, passed, details=None):
        record = {'check': name, 'passed': bool(passed)}
        if details is not None:
            record['details'] = details
        checks.append(record)

    check('clean_test_count', clean_ids.shape == (300702,))
    check('clean_test_unique_ids', np.unique(clean_ids).size == 300702)
    check('known_leak_excluded', not np.any(clean_ids == 240822))

    def one_run(experiment, architecture, seed, path, summary_path, expected):
        summary = read_json(summary_path)
        with np.load(path, allow_pickle=False) as saved:
            if experiment == 'A07':
                ratings = saved['ratings']
                exact = saved['exact'].astype(bool)
                valid = saved['valid'].astype(bool)
                ids = saved['source_row']
                reported = {r['bucket']: r for r in summary['metrics']}
                check(f'{experiment}/{architecture}/{seed}/saved_step',
                      summary['checkpoint']['saved_step'] == 50000)
                check(f'{experiment}/{architecture}/{seed}/test_steps', summary['steps'] == 64)
            else:
                ratings = saved['rating']
                exact = saved['selected_exact'].astype(bool)
                valid = saved['first_valid'].astype(bool)
                ids = saved['raw_cache_id']
                reported = {r['bucket']: r for r in summary['rows']}
                check(f'{experiment}/{architecture}/{seed}/completed',
                      summary['status'] == 'complete' and summary.get('smoke') is False)
        prefix = f'{experiment}/{architecture}/{seed}'
        check(prefix + '/aligned_clean_ids', np.array_equal(ids, clean_ids))
        check(prefix + '/shapes',
              exact.shape == valid.shape == ratings.shape == clean_ids.shape)
        check(prefix + '/exact_equals_valid', np.array_equal(exact, valid))
        masks = [('all', np.ones(len(exact), dtype=bool), 300702),
                 ('D4', ratings > 4, 6543)]
        for index, (bucket, mask, expected_n) in enumerate(masks):
            n = int(mask.sum())
            exact_count = int(exact[mask].sum())
            valid_count = int(valid[mask].sum())
            reported_key = 'rating_4_plus' if experiment == 'A07' and bucket == 'D4' else bucket
            reported_row = reported[reported_key]
            check(prefix + '/' + bucket + '/denominator', n == expected_n)
            check(prefix + '/' + bucket + '/paper_count',
                  exact_count == expected[index],
                  {'computed': exact_count, 'paper': expected[index]})
            check(prefix + '/' + bucket + '/stored_summary',
                  reported_row['n'] == n and reported_row['exact_count'] == exact_count
                  and reported_row['valid_count'] == valid_count)
            rows.append({
                'experiment': experiment, 'architecture': architecture, 'seed': seed,
                'bucket': bucket, 'n': n, 'exact_count': exact_count,
                'valid_count': valid_count, 'exact_pct': exact_count / n * 100,
                'valid_pct': valid_count / n * 100,
                'source_npz': path.relative_to(root).as_posix(),
            })
        sources.append({'path': path.relative_to(root).as_posix(), 'sha256': sha256(path)})

    for architecture, counts in EXPECTED_A07.items():
        for seed in range(3):
            directory = results / 'A07' / f'{architecture}_seed{seed}' / 'matched_parameters_fixed_exposure_final'
            matches = sorted(directory.glob('final_*/T64/per_puzzle.npz'))
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one formal final T64 NPZ: {directory}; found {len(matches)}')
            one_run('A07', architecture, seed, matches[0], matches[0].with_name('summary.json'), counts[seed])

    for seed in range(3):
        directory = results / 'A08' / f'seed{seed}' / 'test'
        one_run('A08', 'NeuralSudoku', seed, directory / 'details.npz',
                directory / 'summary.json', EXPECTED_A08[seed])

    aggregates = []
    for experiment, architecture in sorted({(row['experiment'], row['architecture']) for row in rows}):
        for bucket in ['all', 'D4']:
            matching = [row for row in rows if row['experiment'] == experiment
                        and row['architecture'] == architecture and row['bucket'] == bucket]
            values = np.array([row['exact_pct'] for row in matching])
            aggregates.append({'experiment': experiment, 'architecture': architecture,
                'bucket': bucket, 'independent_training_runs': len(values),
                'mean_exact_pct': float(values.mean()),
                'sample_sd_percentage_points': float(values.std(ddof=1)),
                'display_mean_plus_minus_sd': f'{values.mean():.2f} ± {values.std(ddof=1):.2f}'})

    main_summary = read_json(results / 'A08/three_seed_summary.json')
    for reference in main_summary['summary']:
        aggregate = next(row for row in aggregates if row['experiment'] == 'A08'
                         and row['bucket'] == reference['bucket'])
        check('A08/' + reference['bucket'] + '/three_seed_summary',
              np.isclose(aggregate['mean_exact_pct'], reference['mean'] * 100, atol=1e-10)
              and np.isclose(aggregate['sample_sd_percentage_points'], reference['sample_std'] * 100, atol=1e-10))

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'training_repeats.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'passed': all(c['passed'] for c in checks),
              'scope': 'Statistical recomputation of saved per-puzzle outputs; no model inference or training.',
              'training_runs': {'A07': 9, 'A08': 3},
              'percentage_units': 'Mean in percent; sample SD in percentage points; ddof=1.',
              'aggregates': aggregates, 'checks': checks, 'sources': sources}
    (output_dir / 'validation_summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'checks': len(checks),
                      'aggregates': aggregates, 'output_directory': str(output_dir)},
                     ensure_ascii=False, indent=2))
    return report['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Supplement directory containing results/.')
    parser.add_argument('--output-dir', type=Path, help='Default: ROOT/verification_outputs.')
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else root / 'verification_outputs'
    try:
        passed = verify(root, output_dir)
    except Exception as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'validation_summary.json').write_text(json.dumps(
            {'passed': False, 'error_type': type(error).__name__, 'error': str(error)},
            ensure_ascii=False, indent=2), encoding='utf-8')
        raise
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
