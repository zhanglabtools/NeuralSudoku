"""Independent HRM execution entry; original scientific config is immutable.

Only the effective-update implementation and accuracy-evaluation batch change.
Production uses the original AdamATan2 optimizer. This is not an A05 timing
entry and does not stop workers, migrate checkpoints, or edit frozen sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

HERE = Path(__file__).resolve().parent
ENTRY_SCHEMA = 'hrm_train_execution_v2'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def scientific_part(identity):
    return {key: value for key, value in identity.items() if key != 'execution'}


def verify_stage_manifest(out, migration, ready, best_step):
    """Pure filesystem gate for first entry after an explicit conversion."""
    out = Path(out).resolve()
    if (ready.get('status') != 'ready'
            or ready.get('execution_migration_sha256') != sha(out / 'execution_migration.json')
            or ready.get('latest_sha256') != migration['new_checkpoint_sha256']
            or sha(out / 'latest.pt') != migration['new_checkpoint_sha256']):
        raise ValueError('Converted checkpoint is not explicitly staged and locked')
    files = ready.get('files_sha256', {})
    required = {'selection_ids.npz', 'metadata.json'}
    if best_step:
        required.update({'best.pt', f'validation_{best_step:07d}.json'})
    if (out / 'training.jsonl').exists():
        required.add('training.jsonl')
    if not required.issubset(files):
        raise ValueError('Staging manifest lacks committed selection/history files')
    for name, expected in files.items():
        path = (out / name).resolve()
        if path.parent != out or sha(path) != expected:
            raise ValueError('Staged artifact path or hash differs: ' + name)


def dependencies():
    import numpy as np
    import torch
    import a08_train_baseline as adapter
    import hrm_execution_v2 as execution
    return np, torch, adapter, execution


def verify_identity(identity, adapter, execution, expected_scientific=None):
    science = scientific_part(identity)
    cfg = science['config']
    if cfg.get('method') != 'hrm' or cfg['batch_size'] != 256 or cfg.get('micro_batch_size') != 4:
        raise ValueError('Expected original HRM effective batch256 / legacy micro4 configuration')
    execution.verify_sources(science)
    if science['source_sha256'] != adapter.source_hashes():
        raise ValueError('Current original source inventory differs from the checkpoint')
    if expected_scientific is not None and science != expected_scientific:
        raise ValueError('Scientific identity differs from data/config/runtime provenance')
    if 'execution' in identity:
        runtime = identity['execution']
        expected = execution.runtime_identity(science, runtime['runtime_train_micro'], runtime['runtime_eval_batch'])
        if identity != expected:
            raise ValueError('Versioned execution identity or module hash changed')
    return science


def configure_precision(torch, cfg):
    torch.set_num_threads(cfg['threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def assert_adam_atan2_state_devices(optimizer, torch_module=None):
    """Fail before the custom CUDA kernel can see a CPU/wrong-device pointer."""
    if torch_module is None:
        import torch as torch_module
    checked = 0

    def check(value, device, name):
        nonlocal checked
        if torch_module.is_tensor(value):
            if value.device != device:
                raise ValueError(f'AdamATan2 state device mismatch: {name}: {value.device} != {device}')
            checked += 1
        elif isinstance(value, dict):
            for key, item in value.items():
                check(item, device, name + '/' + str(key))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                check(item, device, name + '/' + str(index))

    group_parameters = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
    for parameter, state in optimizer.state.items():
        if id(parameter) not in group_parameters:
            raise ValueError('AdamATan2 state refers to a parameter outside its groups')
        if state:
            step = state.get('step')
            if not torch_module.is_tensor(step) or step.numel() != 1:
                raise ValueError('AdamATan2 step must be a one-element tensor')
            if not {'exp_avg', 'exp_avg_sq'}.issubset(state):
                raise ValueError('AdamATan2 state lacks moment tensors')
            check(state, parameter.device, 'parameter_state')
    return checked


def restore_adam_atan2_state_devices(optimizer, torch_module=None):
    """Repair only restored step placement; moments are validated, never moved.

    PyTorch's generic optimizer loader can retain a scalar step on CPU because
    this optimizer has no capturable/fused flag. Its custom CUDA kernel requires
    step on the parameter device. Device copy preserves the checkpoint value,
    shape and dtype; it does not advance the optimizer or consume RNG.
    """
    if torch_module is None:
        import torch as torch_module
    moved = 0
    for group in optimizer.param_groups:
        for parameter in group['params']:
            state = optimizer.state.get(parameter)
            if not state:
                continue
            step = state.get('step')
            if not torch_module.is_tensor(step) or step.numel() != 1:
                raise ValueError('AdamATan2 restored step must be a one-element tensor')
            if step.device != parameter.device:
                repaired = step.to(device=parameter.device)
                if (repaired.dtype != step.dtype or repaired.shape != step.shape
                        or not torch_module.equal(repaired.detach().cpu(), step.detach().cpu())):
                    raise ValueError('AdamATan2 step changed value, shape or dtype while moving device')
                state['step'] = repaired
                moved += 1
    assert_adam_atan2_state_devices(optimizer, torch_module)
    return moved


def entry_record(identity, adapter, execution):
    return {'schema': ENTRY_SCHEMA, 'entry_sha256': sha(__file__),
            'module_sha256': sha(execution.__file__), 'identity_sha256': adapter.digest(identity),
            'runtime_train_micro': identity['execution']['runtime_train_micro'],
            'runtime_eval_batch': identity['execution']['runtime_eval_batch'],
            'scientific_config_unchanged': True, 'production_optimizer': 'original AdamATan2',
            'numerical_note': execution.NUMERICAL_NOTE}


def train(args):
    np, torch, a, e = dependencies()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f'Refusing existing run: {out}')
    if args.resume and not (out / 'latest.pt').is_file():
        raise FileNotFoundError('Resume requires a committed latest.pt')
    # Validate execution options before loading the dataset or creating output.
    e.runtime_identity({'config': cfg}, args.runtime_train_micro, args.runtime_eval_batch)
    configure_precision(torch, cfg)
    a.seed_all(cfg['seed'])
    device = torch.device(args.device)
    puzzles, solutions, ratings, train_ids, val_ids = a.load_data(
        args.cache, cfg['val_limit'], cfg['selection_seed'], cfg.get('train_limit', 0), cfg.get('val_ids'))
    science = {'config': cfg, 'cache_sha256': a.sha(args.cache), 'source_sha256': a.source_hashes(),
               'torch_version': torch.__version__, 'numpy_version': np.__version__,
               'train_ids_sha256': hashlib.sha256(train_ids.tobytes()).hexdigest(),
               'val_ids_sha256': hashlib.sha256(val_ids.tobytes()).hexdigest()}
    identity = e.runtime_identity(science, args.runtime_train_micro, args.runtime_eval_batch)
    record = entry_record(identity, a, e)
    marker = out / 'execution_entry.json'
    if marker.exists() and json.loads(marker.read_text()) != record:
        raise ValueError('Entry, execution options, or scientific identity changed')
    model = a.build_model(cfg, device)
    plan = e.install_legacy_act_forward(model)
    optimizer = a.build_optimizer(model, cfg)  # original production AdamATan2
    stream = a.ShuffledStream(train_ids, cfg['seed'])
    state = {'step': 0, 'offered_examples': 0, 'accepted_examples': 0, 'completed_episodes': 0,
             'best_key': [-1, -1, -1], 'best_step': 0, 'no_improvement': 0,
             'wall_seconds': 0., 'stopped_early': False}
    carry = None
    if args.resume:
        saved = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        verify_identity(saved['identity'], a, e, science)
        if saved['identity'] != identity:
            raise ValueError('Resume runtime batches differ from the converted/current checkpoint')
        if not marker.exists():
            migration = json.loads((out / 'execution_migration.json').read_text())
            ready = json.loads((out / 'continuation_ready.json').read_text())
            if migration['new_identity'] != identity:
                raise ValueError('Migration identity differs')
            verify_stage_manifest(out, migration, ready, saved['state']['best_step'])
        with np.load(out / 'selection_ids.npz', allow_pickle=False) as selected:
            if not np.array_equal(selected['train'], train_ids) or not np.array_equal(selected['validation'], val_ids):
                raise ValueError('Staged training/validation IDs differ')
        if saved['state']['best_step']:
            best = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
            verify_identity(best['identity'], a, e, science)
            if (best['state']['step'] != saved['state']['best_step']
                    or best['state']['best_key'] != saved['state']['best_key']):
                raise ValueError('Selected best does not match committed resume selection')
        # Restore RNG last, after model/optimizer construction and provenance reads.
        carry, state = a.restore(out / 'latest.pt', model, optimizer, stream, identity, device)
        moved_steps = restore_adam_atan2_state_devices(optimizer, torch)
        print(json.dumps({'resume_optimizer_step_tensors_moved_to_parameter_device': moved_steps,
                          'optimizer_step_values_and_dtypes_preserved': True}), flush=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / 'selection_ids.npz', train=train_ids, validation=val_ids)
        a.atomic_json(out / 'metadata.json', {
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
        a.atomic_json(marker, record)
        a.checkpoint(out / 'latest.pt', model, optimizer, stream, carry, state, identity)
    a.atomic_json(marker, record)
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
        assert_adam_atan2_state_devices(optimizer, torch)
        ids = stream.take(cfg['batch_size'])
        p, s = a.make_batch(puzzles, solutions, ids, device)
        carry, update = e.effective_update(model, optimizer, cfg, p, s, carry, step,
                                         args.runtime_train_micro, plan, adapter=a)
        state['step'] = step
        state['offered_examples'] += len(ids)
        state['accepted_examples'] += update['accepted']
        state['completed_episodes'] += update['completed_episodes']
        state['wall_seconds'] = previous_wall + time.monotonic() - start_time
        if step == 1 or step % cfg['log_every'] == 0:
            row = {'step': step, 'loss': update['loss'], 'gradient_norm': update['gradient_norm'],
                   'lr': optimizer.param_groups[0]['lr'], 'offered': state['offered_examples'],
                   'accepted': state['accepted_examples'], 'wall_seconds': state['wall_seconds']}
            with (out / 'training.jsonl').open('a', encoding='utf-8') as stream_log:
                stream_log.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
        if step % cfg['eval_every'] == 0 or step == cfg['max_steps']:
            metrics, _, _ = e.runtime_evaluate(model, cfg, puzzles, solutions, ratings, val_ids, device,
                                              runtime_eval_batch=args.runtime_eval_batch, adapter=a)
            metrics['step'] = step
            a.atomic_json(out / f'validation_{step:07d}.json', metrics)
            if tuple(metrics['selection_key']) > tuple(state['best_key']):
                state['best_key'] = metrics['selection_key']
                state['best_step'] = step
                state['no_improvement'] = 0
                a.checkpoint(out / 'best.pt', model, optimizer, stream, carry, state, identity)
            else:
                state['no_improvement'] += 1
            if step >= cfg['min_steps'] and state['no_improvement'] >= cfg['patience']:
                state['stopped_early'] = True
            print(json.dumps({'validation': metrics}), flush=True)
        state['wall_seconds'] = previous_wall + time.monotonic() - start_time
        if step % cfg['save_every'] == 0 or step == stop_after or stop['requested'] or state['stopped_early']:
            a.checkpoint(out / 'latest.pt', model, optimizer, stream, carry, state, identity)
            a.atomic_json(out / 'status.json', {**state, 'nominal_offered_epochs': state['offered_examples'] / len(train_ids),
                          'accepted_epochs': state['accepted_examples'] / len(train_ids),
                          'complete': state['stopped_early'] or step == cfg['max_steps'],
                          'stop_requested': stop['requested']})
        if stop['requested'] or state['stopped_early']:
            break


def evaluate(args):
    np, torch, a, e = dependencies()
    checkpoint_sha = a.sha(args.checkpoint)
    if args.checkpoint_sha256 and checkpoint_sha != args.checkpoint_sha256:
        raise ValueError('Selected checkpoint hash changed')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if a.sha(args.checkpoint) != checkpoint_sha:
        raise ValueError('Checkpoint changed while loading')
    science = verify_identity(saved['identity'], a, e)
    if args.scientific_identity_sha256 and a.digest(science) != args.scientific_identity_sha256:
        raise ValueError('Unexpected scientific identity')
    cfg = science['config']
    if a.sha(args.cache) != science['cache_sha256']:
        raise ValueError('Evaluation cache differs from training')
    configure_precision(torch, cfg)
    device = torch.device(args.device)
    model = a.build_model(cfg, device)
    model.load_state_dict(saved['model'], strict=True)
    with np.load(args.cache, allow_pickle=False) as data:
        puzzles, solutions = data['puzzles'].reshape(-1, 9, 9), data['solutions'].reshape(-1, 9, 9)
        ratings, splits = data['ratings'], data['splits']
    ids = np.load(args.ids, allow_pickle=False).astype(np.int64)
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique nonempty 1D global IDs required')
    if np.any(ids < 0) or np.any(ids >= len(puzzles)) or np.any(splits[ids] != args.split):
        raise ValueError('IDs outside requested split')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    runtime = {'schema': ENTRY_SCHEMA, 'entry_sha256': sha(__file__), 'module_sha256': sha(e.__file__),
               'checkpoint_sha256': checkpoint_sha, 'checkpoint_identity_sha256': a.digest(saved['identity']),
               'scientific_identity_sha256': a.digest(science),
               'checkpoint_execution': saved['identity'].get('execution'),
               'checkpoint_origin': 'execution_v2' if 'execution' in saved['identity'] else 'original_execution',
               'configured_eval_batch': cfg['eval_batch_size'], 'runtime_eval_batch': args.runtime_eval_batch,
               'precision': cfg['model'].get('forward_dtype', 'float32'), 'tf32': False,
               'scientific_config_unchanged': True, 'for_accuracy_only_not_A05_timing': True,
               'numerical_note': e.NUMERICAL_NOTE}
    a.atomic_json(out / 'evaluation_execution.json', runtime)
    metrics, raw, final = e.runtime_evaluate(model, cfg, puzzles, solutions, ratings, ids, device,
                                           runtime_eval_batch=args.runtime_eval_batch, adapter=a)
    np.savez_compressed(out / 'predictions.npz', global_ids=ids, raw=raw, final=final)
    a.atomic_json(out / 'metrics.json', {**metrics, 'status': 'complete', 'method': cfg['method'], 'seed': cfg['seed'],
                  'teacher_labels_used_for_inference': False, 'model_valid_includes_clues': True,
                  'selected_step': saved['state']['step'], 'checkpoint_sha256': checkpoint_sha,
                  'ids_sha256': a.sha(args.ids), 'cache_sha256': science['cache_sha256'],
                  'config_sha256': a.digest(cfg), 'evaluation_sources': a.source_hashes()})
    print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--config', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int, default=0, help='Original safe pilot pause; not a budget change')
    p.add_argument('--runtime-train-micro', type=int, required=True)
    p.add_argument('--runtime-eval-batch', type=int, required=True)
    p.add_argument('--device', default='cuda')
    p = sub.add_parser('evaluate')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--checkpoint-sha256', default='')
    p.add_argument('--scientific-identity-sha256', default='')
    p.add_argument('--cache', required=True)
    p.add_argument('--ids', required=True)
    p.add_argument('--split', type=int, choices=(1, 2), required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--runtime-eval-batch', type=int, required=True)
    p.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.runtime_eval_batch <= 0 or getattr(args, 'stop_after', 0) < 0:
        parser.error('Positive runtime evaluation batch and nonnegative stop-after required')
    train(args) if args.command == 'train' else evaluate(args)


if __name__ == '__main__':
    main()
