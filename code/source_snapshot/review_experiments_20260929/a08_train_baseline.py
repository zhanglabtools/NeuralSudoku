"""Auditable M3 baseline migration, with validation-only selection and full resume.

This is a controlled adapter, not a claim to reproduce original benchmark data.
HRM preserves the official full-batch offered stream; only halted slots accept
new samples. Both offered and accepted exposures are recorded explicitly.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
for path in (BASE, BASE.parent, Path('code/optional_dependencies/python_sat'),
             Path('code/optional_dependencies/hrm'), BASE / 'external/SATNet-master',
             BASE / 'external/HRM'):
    if path.exists():
        sys.path.insert(0, str(path))
from e2_metrics import board_metrics, aggregate_metrics


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


def tree_to(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if dataclasses.is_dataclass(value):
        return type(value)(**{f.name: tree_to(getattr(value, f.name), device)
                              for f in dataclasses.fields(value)})
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(tree_to(v, device) for v in value)
    if isinstance(value, list):
        return [tree_to(v, device) for v in value]
    return value


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ShuffledStream:
    """No workers/prefetch: permutation, cursor and RNG are exact resume state."""
    def __init__(self, indices, seed):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(self.indices))
        self.cursor = 0
        self.epochs = 0

    def take(self, n):
        parts = []
        while n:
            left = min(n, len(self.order) - self.cursor)
            parts.append(self.indices[self.order[self.cursor:self.cursor + left]])
            self.cursor += left
            n -= left
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(len(self.indices))
                self.cursor = 0
                self.epochs += 1
        return np.concatenate(parts)

    def state_dict(self):
        return {'order': self.order.copy(), 'cursor': self.cursor, 'epochs': self.epochs,
                'rng': self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.order = state['order'].copy()
        self.cursor = state['cursor']
        self.epochs = state['epochs']
        self.rng.bit_generator.state = state['rng']


def load_data(cache_path, val_limit, selection_seed, train_limit=0, val_ids_path=None):
    with np.load(cache_path, allow_pickle=False) as data:
        puzzles = data['puzzles'].reshape(-1, 9, 9)
        solutions = data['solutions'].reshape(-1, 9, 9)
        splits = data['splits']
        ratings = data['ratings']
    train = np.flatnonzero(splits == 0)
    val = np.flatnonzero(splits == 1)
    if train_limit:
        train = train[:train_limit]
    if val_ids_path:
        val = np.load(val_ids_path, allow_pickle=False).astype(np.int64)
        if (val.ndim != 1 or not len(val) or len(np.unique(val)) != len(val)
                or np.any(val < 0) or np.any(val >= len(splits)) or np.any(splits[val] != 1)):
            raise ValueError('Shared validation IDs must be unique global IDs from split 1')
        if val_limit and len(val) != val_limit:
            raise ValueError('Shared validation manifest size differs from preregistered limit')
    elif val_limit and len(val) > val_limit:
        val = np.sort(np.random.default_rng(selection_seed).choice(val, val_limit, replace=False))
    if not len(train) or not len(val):
        raise ValueError('Nonempty training and validation sets required')
    assert not np.intersect1d(train, val).size
    return puzzles, solutions, ratings, train, val


def make_batch(puzzles, solutions, ids, device):
    return (torch.as_tensor(puzzles[ids].astype(np.int64), device=device),
            torch.as_tensor(solutions[ids].astype(np.int64), device=device))


def build_model(cfg, device):
    method = cfg['method']
    if method == 'satnet':
        from train_official_satnet_kaggle import SATNetCfg, OfficialSATNetSudoku
        model = OfficialSATNetSudoku(SATNetCfg(**cfg['model']))
    elif method == 'abl':
        from reproduce_satnet_abl_hrm import GNNCfg, ABLReflGNN
        model = ABLReflGNN(GNNCfg(**cfg['model']))
    elif method == 'hrm':
        from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
        from models.losses import ACTLossHead
        model = ACTLossHead(HierarchicalReasoningModel_ACTV1(cfg['model']),
                            loss_type='softmax_cross_entropy')
    else:
        raise ValueError(method)
    return model.to(device)


def build_optimizer(model, cfg):
    opt = cfg['optimizer']
    if cfg['method'] == 'hrm':
        from adam_atan2 import AdamATan2
        return AdamATan2(model.parameters(), lr=opt['lr'], betas=(.9, .95),
                         weight_decay=opt['weight_decay'])
    return torch.optim.Adam(model.parameters(), lr=opt['lr'],
                            weight_decay=opt.get('weight_decay', 0))


def hrm_batch(puzzle, solution):
    from train_hrm_official_kaggle import make_hrm_batch
    return make_hrm_batch(puzzle, solution)


def train_loss(model, cfg, puzzle, solution, carry=None):
    if cfg['method'] == 'satnet':
        from train_official_satnet_kaggle import one_hot_board
        probs = model(puzzle)
        return F.binary_cross_entropy(probs, one_hot_board(solution)), None, len(puzzle)
    if cfg['method'] == 'abl':
        from reproduce_satnet_abl_hrm import supervised_loss
        from train_abl_refl_paper_kaggle import paper_reflection_losses
        logits, probs, reflection = model(puzzle)
        supervised = supervised_loss(logits, puzzle, solution, cfg['empty_weight'])
        consistency, size, _ = paper_reflection_losses(reflection, probs, puzzle,
                                                       cfg['reflection_size_c'], baseline='none')
        return supervised + cfg['alpha'] * consistency + cfg['beta'] * size, None, len(puzzle)
    batch = hrm_batch(puzzle, solution)
    if carry is None:
        with torch.device(puzzle.device):
            carry = model.initial_carry(batch)
    accepted = int(carry.halted.sum().item())
    carry, loss, _, _, _ = model(carry=carry, batch=batch, return_keys=[])
    return loss / len(puzzle), carry, accepted


def scheduled_lr(cfg, step):
    opt = cfg['optimizer']
    if cfg['method'] != 'hrm':
        return opt['lr']
    warmup = opt['warmup_steps']
    if step <= warmup:
        return opt['lr'] * step / max(1, warmup)
    progress = (step - warmup) / max(1, cfg['max_steps'] - warmup)
    return opt['lr'] * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))


def repair_abl(pred, reflection, puzzle, threshold, timeout):
    """Bounded, single SAT call; timeout is unknown, never an invalid proof."""
    from pysat.solvers import Minisat22
    from train_abl_refl_paper_kaggle import BASE_CNF, varnum, grid_from_model
    clauses = list(BASE_CNF)
    for r in range(9):
        for c in range(9):
            digit = int(puzzle[r, c])
            if not digit and reflection[r, c] < threshold:
                digit = int(pred[r, c])
            if digit:
                clauses.append([varnum(r, c, digit - 1)])
    with Minisat22(bootstrap_with=clauses) as solver:
        timer = threading.Timer(timeout, solver.interrupt)
        timer.start()
        try:
            status = solver.solve_limited(expect_interrupt=True)
            grid = grid_from_model(solver.get_model()) if status is True else pred.copy()
        finally:
            timer.cancel()
            timer.join()
    return grid, 'solved' if status is True else 'unsat' if status is False else 'unknown'


@torch.no_grad()
def predict(model, cfg, puzzle, solution):
    method = cfg['method']
    if method == 'satnet':
        probs = model(puzzle)
        return (probs.reshape(-1, 9, 9, 9).argmax(-1) + 1), None
    if method == 'abl':
        logits, _, reflection = model(puzzle)
        return logits.argmax(-1) + 1, reflection.sigmoid()
    batch = hrm_batch(puzzle, solution)
    with torch.device(puzzle.device):
        carry = model.initial_carry(batch)
    for _ in range(cfg['model']['halt_max_steps']):
        carry, _, _, result, all_finish = model(carry=carry, batch=batch, return_keys=['logits'])
        if bool(all_finish):
            break
    else:
        raise RuntimeError('HRM did not finish at configured halt_max_steps')
    # Invalid special tokens remain 0 and fail independent validation.
    return (result['logits'].argmax(-1) - 1).clamp(0, 9).reshape(-1, 9, 9), None


def evaluate(model, cfg, puzzles, solutions, ratings, ids, device):
    saved_rng = rng_state()
    was_training = model.training
    try:
        seed_all(cfg['evaluation_seed'])
        model.eval()
        raw, final, repair_status = [], [], []
        for start in range(0, len(ids), cfg['eval_batch_size']):
            selected = ids[start:start + cfg['eval_batch_size']]
            p, s = make_batch(puzzles, solutions, selected, device)
            pred, reflection = predict(model, cfg, p, torch.zeros_like(s))
            pred = pred.cpu().numpy().astype(np.uint8)
            raw.append(pred)
            if reflection is not None:
                repaired = []
                for i, refl in enumerate(reflection.cpu().numpy()):
                    grid, status = repair_abl(pred[i], refl, puzzles[selected[i]],
                                              cfg['reflection_threshold'], cfg['repair_timeout'])
                    repaired.append(grid)
                    repair_status.append(status)
                final.append(np.asarray(repaired))
            else:
                final.append(pred)
        raw, final = np.concatenate(raw), np.concatenate(final)
        raw_m = board_metrics(raw, puzzles[ids], solutions[ids])
        final_m = board_metrics(final, puzzles[ids], solutions[ids])
        empty = puzzles[ids] == 0
        empty_correct = int(((raw == solutions[ids]) & empty).sum())
        result = {'n': len(ids), 'raw': aggregate_metrics(raw_m, ratings[ids]),
                  'final': aggregate_metrics(final_m, ratings[ids]),
                  'raw_empty_correct': empty_correct, 'empty_cells': int(empty.sum()),
                  'repair_status': {s: repair_status.count(s) for s in ('solved', 'unsat', 'unknown')}}
        # Predeclared: final reference Exact, final Valid, raw empty-cell accuracy.
        # Ties retain earliest checkpoint; no test feedback enters this tuple.
        result['selection_key'] = [int(final_m['exact'].sum()), int(final_m['valid'].sum()),
                                   empty_correct]
        return result, raw, final
    finally:
        model.train(was_training)
        restore_rng(saved_rng)


def source_hashes():
    paths = [Path(__file__), HERE / 'e2_metrics.py']
    for name in ('train_official_satnet_kaggle.py', 'train_abl_refl_paper_kaggle.py',
                 'train_hrm_official_kaggle.py', 'reproduce_satnet_abl_hrm.py',
                 'sudoku_exchange_experiment.py', 'kaggle_sudoku_experiment.py'):
        candidate = BASE / name
        if not candidate.exists():
            candidate = BASE.parent / name
        if candidate.exists():
            paths.append(candidate)
    for folder in (BASE / 'external/HRM/models', BASE / 'external/SATNet-master/satnet'):
        if folder.exists():
            paths.extend(folder.rglob('*.py'))
            paths.extend(folder.rglob('*.so'))
    return {str(p): sha(p) for p in sorted(set(paths))}


def checkpoint(path, model, optimizer, stream, carry, state, identity):
    value = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'stream': stream.state_dict(), 'carry': tree_to(carry, 'cpu'),
             'rng': rng_state(), 'state': state.copy(), 'identity': identity}
    tmp = Path(str(path) + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def restore(path, model, optimizer, stream, identity, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['identity'] != identity:
        raise ValueError('Resume refused: config, code, input, or selected IDs changed')
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    stream.load_state_dict(saved['stream'])
    carry = tree_to(saved['carry'], device)
    restore_rng(saved['rng'])
    return carry, saved['state']


def train(args):
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    out = Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f'Refusing existing run: {out}')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg['threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_all(cfg['seed'])
    device = torch.device(args.device)
    puzzles, solutions, ratings, train_ids, val_ids = load_data(
        args.cache, cfg['val_limit'], cfg['selection_seed'], cfg.get('train_limit', 0), cfg.get('val_ids'))
    model = build_model(cfg, device)
    optimizer = build_optimizer(model, cfg)
    stream = ShuffledStream(train_ids, cfg['seed'])
    identity = {'config': cfg, 'cache_sha256': sha(args.cache), 'source_sha256': source_hashes(),
                'torch_version': torch.__version__, 'numpy_version': np.__version__,
                'train_ids_sha256': hashlib.sha256(train_ids.tobytes()).hexdigest(),
                'val_ids_sha256': hashlib.sha256(val_ids.tobytes()).hexdigest()}
    state = {'step': 0, 'offered_examples': 0, 'accepted_examples': 0, 'completed_episodes': 0,
             'best_key': [-1, -1, -1],
             'best_step': 0, 'no_improvement': 0, 'wall_seconds': 0., 'stopped_early': False}
    carry = None
    if args.resume:
        carry, state = restore(out / 'latest.pt', model, optimizer, stream, identity, device)
    else:
        np.savez(out / 'selection_ids.npz', train=train_ids, validation=val_ids)
        atomic_json(out / 'metadata.json', {
            **identity, 'train_n': len(train_ids), 'validation_n': len(val_ids),
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
            'precision': cfg['model'].get('forward_dtype', 'float32'), 'tf32': False,
            'selection': 'lexicographic final reference Exact, final Valid, raw empty correct; earliest complete tie',
            'test_used': False, 'original_benchmark_reproduced': False,
            'teacher_label_use': 'supervised training and scoring only; inference receives fixed dummy labels',
            'hrm_exposure': 'full new batch offered per update, only previously halted slots accepted',
            'reproduction_level': cfg['reproduction_level']})
        # Commit the initialized run before its first update. The scheduler can
        # then append --resume even if the process stops before save_every.
        checkpoint(out / 'latest.pt', model, optimizer, stream, carry, state, identity)
    if state['stopped_early'] or state['step'] >= cfg['max_steps']:
        print('Training already complete', flush=True)
        return
    stop = {'requested': False}
    def request_stop(signum, frame):
        stop['requested'] = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    start_time = time.monotonic()
    previous_wall = state['wall_seconds']
    stop_after = min(args.stop_after or cfg['max_steps'], cfg['max_steps'])
    model.train()
    for step in range(state['step'] + 1, stop_after + 1):
        ids = stream.take(cfg['batch_size'])
        p, s = make_batch(puzzles, solutions, ids, device)
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = scheduled_lr(cfg, step)
        micro = cfg.get('micro_batch_size', cfg['batch_size'])
        chunks = math.ceil(len(ids) / micro)
        if carry is None:
            carry = [None] * chunks
        accepted = 0
        loss_value = 0.
        for chunk, offset in enumerate(range(0, len(ids), micro)):
            pc, sc = p[offset:offset + micro], s[offset:offset + micro]
            loss, carry[chunk], fresh = train_loss(model, cfg, pc, sc, carry[chunk])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Nonfinite loss at step {step}; previous safe checkpoint retained')
            weight = len(pc) / len(ids)
            (loss * weight).backward()
            loss_value += float(loss.detach()) * weight
            accepted += fresh
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        state['step'] = step
        state['offered_examples'] += len(ids)
        state['accepted_examples'] += accepted
        state['completed_episodes'] += sum(int(c.halted.sum().item()) for c in carry) if cfg['method'] == 'hrm' else len(ids)
        state['wall_seconds'] = previous_wall + time.monotonic() - start_time
        if step == 1 or step % cfg['log_every'] == 0:
            row = {'step': step, 'loss': loss_value, 'gradient_norm': float(norm),
                   'lr': optimizer.param_groups[0]['lr'], 'offered': state['offered_examples'],
                   'accepted': state['accepted_examples'], 'wall_seconds': state['wall_seconds']}
            with (out / 'training.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
        if step % cfg['eval_every'] == 0 or step == cfg['max_steps']:
            metrics, _, _ = evaluate(model, cfg, puzzles, solutions, ratings, val_ids, device)
            metrics['step'] = step
            atomic_json(out / f'validation_{step:07d}.json', metrics)
            if tuple(metrics['selection_key']) > tuple(state['best_key']):
                state['best_key'] = metrics['selection_key']
                state['best_step'] = step
                state['no_improvement'] = 0
                checkpoint(out / 'best.pt', model, optimizer, stream, carry, state, identity)
            else:
                state['no_improvement'] += 1
            if step >= cfg['min_steps'] and state['no_improvement'] >= cfg['patience']:
                state['stopped_early'] = True
            print(json.dumps({'validation': metrics}), flush=True)
        state['wall_seconds'] = previous_wall + time.monotonic() - start_time
        if step % cfg['save_every'] == 0 or step == stop_after or stop['requested'] or state['stopped_early']:
            checkpoint(out / 'latest.pt', model, optimizer, stream, carry, state, identity)
            atomic_json(out / 'status.json', {**state, 'nominal_offered_epochs': state['offered_examples'] / len(train_ids),
                        'accepted_epochs': state['accepted_examples'] / len(train_ids),
                        'complete': state['stopped_early'] or step == cfg['max_steps'],
                        'stop_requested': stop['requested']})
        if stop['requested'] or state['stopped_early']:
            break


def eval_checkpoint(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = saved['identity']['config']
    if sha(args.cache) != saved['identity']['cache_sha256']:
        raise ValueError('Evaluation cache differs from training')
    torch.set_num_threads(cfg['threads'])
    device = torch.device(args.device)
    model = build_model(cfg, device)
    model.load_state_dict(saved['model'], strict=True)
    with np.load(args.cache, allow_pickle=False) as data:
        puzzles, solutions = data['puzzles'].reshape(-1, 9, 9), data['solutions'].reshape(-1, 9, 9)
        ratings, splits = data['ratings'], data['splits']
    ids = np.load(args.ids, allow_pickle=False).astype(np.int64)
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique nonempty 1D global IDs required')
    if np.any(ids < 0) or np.any(ids >= len(puzzles)) or np.any(splits[ids] != args.split):
        raise ValueError('IDs outside requested split')
    metrics, raw, final = evaluate(model, cfg, puzzles, solutions, ratings, ids, device)
    np.savez_compressed(out / 'predictions.npz', global_ids=ids, raw=raw, final=final)
    atomic_json(out / 'metrics.json', {**metrics, 'status': 'complete', 'method': cfg['method'], 'seed': cfg['seed'],
                'teacher_labels_used_for_inference': False, 'model_valid_includes_clues': True,
                'selected_step': saved['state']['step'], 'checkpoint_sha256': sha(args.checkpoint),
                'ids_sha256': sha(args.ids), 'cache_sha256': saved['identity']['cache_sha256'],
                'config_sha256': digest(cfg), 'evaluation_sources': source_hashes()})
    print(json.dumps(metrics), flush=True)


def load_predictor(method, checkpoint_path, device):
    """Target-free A05 API; initialization is outside the timed callable.

    callable(np.uint8[B,9,9]) -> np.uint8[B,9,9]. Labels supplied internally to
    HRM's unused evaluation-loss branch are fixed dummy zeros, never references.
    Methods: satnet, abl_raw, abl_repair, hrm. No RNG resetting inside timed calls.
    """
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = saved['identity']['config']
    expected = 'abl' if method.startswith('abl_') else method
    if cfg['method'] != expected:
        raise ValueError('Requested method differs from checkpoint')
    device = torch.device(device)
    model = build_model(cfg, device)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    @torch.no_grad()
    def infer(puzzles):
        puzzles = np.asarray(puzzles, dtype=np.uint8).reshape(-1, 9, 9)
        p = torch.as_tensor(puzzles.astype(np.int64), device=device)
        pred, reflection = predict(model, cfg, p, torch.zeros_like(p))
        raw = pred.cpu().numpy().astype(np.uint8)
        if method != 'abl_repair':
            return raw
        return np.asarray([repair_abl(raw[i], reflection[i].cpu().numpy(), puzzles[i],
                          cfg['reflection_threshold'], cfg['repair_timeout'])[0]
                           for i in range(len(raw))], dtype=np.uint8)
    infer.config = cfg
    return infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--config', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int, default=0, help='Safe pilot pause; not a budget/config change')
    p.add_argument('--device', default='cuda')
    p = sub.add_parser('evaluate')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--ids', required=True)
    p.add_argument('--split', type=int, choices=(1, 2), required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    args = parser.parse_args()
    train(args) if args.command == 'train' else eval_checkpoint(args)


if __name__ == '__main__':
    main()
