"""Evaluate a relocated C5 checkpoint using explicit data/backbone/reflection paths."""
import argparse
import json
from pathlib import Path
from portable_common import add_source_paths, file_path, load_symbolic_explicit, package_label, sha256


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone', required=True)
    p.add_argument('--reflection', required=True)
    p.add_argument('--cache', required=True, help='NPZ cache or extracted array directory')
    p.add_argument('--split', choices=('val', 'test'), default='test')
    p.add_argument('--global-ids', help='Global cache row IDs; required for clean test evaluation')
    p.add_argument('--output', required=True, help='New output directory')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=20260929)
    p.add_argument('--limit', type=int, default=0, help='0 = complete selected split; positive = smoke subset')
    p.add_argument('--rating-min-exclusive', type=float)
    p.add_argument('--dry-run', action='store_true')
    return p


def select_positions(raw_split_ids, allowed_ids):
    import numpy as np
    raw_split_ids = np.asarray(raw_split_ids)
    allowed_ids = np.asarray(allowed_ids)
    if allowed_ids.ndim != 1 or not np.issubdtype(allowed_ids.dtype, np.integer) or not len(allowed_ids):
        raise ValueError('Global IDs must be a nonempty one-dimensional integer array.')
    if len(np.unique(allowed_ids)) != len(allowed_ids) or not np.isin(allowed_ids, raw_split_ids).all():
        raise ValueError('Global IDs must be unique and belong to the selected split.')
    return np.flatnonzero(np.isin(raw_split_ids, allowed_ids))


def main():
    p = parser()
    args = p.parse_args()
    if args.split == 'test' and not args.global_ids:
        p.error('--global-ids is required for test evaluation; use the supplied clean IDs.')
    if min(args.batch_size, args.threads) < 1 or args.limit < 0:
        p.error('Batch size/threads must be positive and limit nonnegative.')
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return
    add_source_paths()
    import numpy as np
    import torch
    from sudoku_cache_utils import load_sudoku_cache, load_sudoku_dataset
    from sudoku_exchange_experiment import set_seed
    from a08_seed_pipeline import evaluate_indices
    backbone, reflection = file_path(args.backbone), file_path(args.reflection)
    cache = Path(args.cache).expanduser().resolve()
    if not cache.exists():
        p.error('Cache does not exist.')
    out = Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        p.error('Output directory is not empty; choose a new directory.')
    split = {'val': 1, 'test': 2}[args.split]
    arrays = load_sudoku_cache(cache)
    raw_ids = np.flatnonzero(np.asarray(arrays['splits']) == split)
    if hasattr(arrays, 'close'):
        arrays.close()
    ids_path = file_path(args.global_ids) if args.global_ids else None
    ids = np.load(ids_path, allow_pickle=False) if ids_path else raw_ids
    positions = select_positions(raw_ids, ids)
    ds = load_sudoku_dataset(cache, split=split, limit=0)
    if args.rating_min_exclusive is not None:
        positions = positions[np.asarray(ds.ratings)[positions] > args.rating_min_exclusive]
    eligible_n = len(positions)
    if args.limit:
        positions = positions[:args.limit]
    if not len(positions):
        p.error('Selection is empty.')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    set_seed(args.seed)
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, cfg, _ = load_symbolic_explicit(reflection, backbone, torch.device(args.device))
    summary, details = evaluate_indices(model, ds, positions, torch.device(args.device), args.batch_size, True)
    summary.update(split=args.split, eligible_n=eligible_n, complete_selection=len(positions) == eligible_n,
                   global_ids=package_label(ids_path) if ids_path else None,
                   global_ids_sha256=sha256(ids_path) if ids_path else None,
                   backbone=package_label(backbone), backbone_sha256=sha256(backbone),
                   reflection=package_label(reflection), reflection_sha256=sha256(reflection),
                   cache=package_label(cache), selection_rule='first valid: parent, cycle, lowest slot; target used only for Exact scoring',
                   unresolved_grid='all zeros; inspect first_valid before using selected_grid',
                   device=args.device, batch_size=args.batch_size, seed=args.seed)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'details.npz', global_ids=raw_ids[positions], split_positions=positions, **details)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
