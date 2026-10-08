"""Versioned HRM execution batching; the scientific configuration stays intact.

This module does not launch, stop, or alter a production worker.  Its update API
preserves the effective batch, loss weighting, optimizer schedule, slot order,
and legacy micro-batch random-call sequence.  Larger matrix batches and changed
gradient summation can change floating-point results and later ACT decisions.
It is explicitly NOT a bitwise-equivalent continuation.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import types

import numpy as np
import torch

OFFICIAL_ACT_SHA256 = 'b9c5195fc14c6f451569434905cd16849d6c56ff7723f5844604b48065ebfd39'
SCHEMA = 'hrm_execution_v2'
NUMERICAL_NOTE = ('Scientific effective batch, update budget, seeds, data and '
                  'optimizer schedule are unchanged. Runtime batching changes '
                  'BF16 kernels and gradient summation; bitwise equivalence and '
                  'identical future ACT acceptances are not claimed.')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def runtime_identity(scientific_identity, train_micro, eval_batch):
    """The original config is retained verbatim, including its legacy micro=4."""
    cfg = scientific_identity['config']
    if cfg.get('method') != 'hrm' or cfg['batch_size'] != 256:
        raise ValueError('Production migration requires HRM effective batch 256')
    if cfg.get('micro_batch_size') != 4:
        raise ValueError('This migration preserves the original micro-batch 4 RNG stream')
    if 'execution' in scientific_identity:
        raise ValueError('Already migrated; use exact v2 identity for normal resume')
    if train_micro not in (4, 8, 16, 32, 64, 128, 256) or eval_batch <= 0:
        raise ValueError('Invalid execution batch')
    return {**copy.deepcopy(scientific_identity), 'execution': {
        'schema': SCHEMA, 'runtime_train_micro': train_micro,
        'runtime_eval_batch': eval_batch, 'legacy_rng_micro': 4,
        'parent_identity_sha256': digest(scientific_identity),
        'wrapper_sha256': sha256(__file__), 'numerical_note': NUMERICAL_NOTE}}


def verify_sources(identity):
    sources = identity.get('source_sha256')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('An explicit nonempty original source manifest is required')
    for path, expected in sources.items():
        if sha256(path) != expected:
            raise ValueError(f'Original source hash changed: {path}')


def _cat_trees(values):
    first = values[0]
    if torch.is_tensor(first):
        if first.ndim < 1 or any(not torch.is_tensor(v) or v.shape[1:] != first.shape[1:]
                                or v.dtype != first.dtype or v.device != first.device for v in values):
            raise ValueError('Incompatible carry tensor')
        return torch.cat(values, dim=0)
    if dataclasses.is_dataclass(first):
        if any(type(v) is not type(first) for v in values):
            raise ValueError('Different carry dataclasses')
        return type(first)(**{f.name: _cat_trees([getattr(v, f.name) for v in values])
                              for f in dataclasses.fields(first)})
    if isinstance(first, dict):
        if any(v.keys() != first.keys() for v in values):
            raise ValueError('Different carry dictionary keys')
        return {k: _cat_trees([v[k] for v in values]) for k in first}
    raise TypeError(f'Unexpected non-tensor carry field: {type(first)}')


def _slice_tree(value, start, stop, total):
    if torch.is_tensor(value):
        if value.ndim < 1 or len(value) != total:
            raise ValueError('Every HRM carry tensor must have the same slot dimension')
        return value[start:stop].detach().clone()
    if dataclasses.is_dataclass(value):
        return type(value)(**{f.name: _slice_tree(getattr(value, f.name), start, stop, total)
                              for f in dataclasses.fields(value)})
    if isinstance(value, dict):
        return {k: _slice_tree(v, start, stop, total) for k, v in value.items()}
    raise TypeError(type(value))


def regroup_carry(carries, total, new_micro):
    """Concatenate old slot order and split it; never reset an existing episode."""
    if total <= 0 or new_micro <= 0:
        raise ValueError('Positive slot count and micro-batch required')
    n = math.ceil(total / new_micro)
    if carries is None:
        return [None] * n
    if not isinstance(carries, list) or not carries:
        raise ValueError('Expected nonempty carry list or uninitialized None')
    if all(c is None for c in carries):
        return [None] * n
    if any(c is None for c in carries):
        raise ValueError('Partially initialized carry is invalid at an update boundary')
    joined = _cat_trees(carries)
    return [_slice_tree(joined, i, min(i + new_micro, total), total)
            for i in range(0, total, new_micro)]


def legacy_min_halt_steps(total, device, probability, halt_max_steps, legacy_micro=4):
    """Issue precisely the old rand_like -> randint_like calls for each 4 slots.

    Official q_halt is float32 (a column view of [B,2]); steps is int32.
    Do not replace this with two draws of length `total`: CUDA generator offsets
    depend on call shape/count. Model forward/backward has no configured dropout.
    """
    if total <= 0 or legacy_micro <= 0 or halt_max_steps <= 1:
        raise ValueError('Invalid ACT random plan')
    values = []
    for start in range(0, total, legacy_micro):
        size = min(legacy_micro, total - start)
        q_template = torch.empty((size, 2), dtype=torch.float32, device=device)[:, 0]
        steps_template = torch.empty(size, dtype=torch.int32, device=device)
        explore = torch.rand_like(q_template) < probability
        minimum = torch.randint_like(steps_template, low=2, high=halt_max_steps + 1)
        values.append(explore * minimum)
    return torch.cat(values)


class LegacyDrawPlan:
    def __init__(self):
        self.values = None
        self.offset = 0

    def begin(self, values):
        if self.values is not None:
            raise RuntimeError('Previous random plan was not finished')
        self.values, self.offset = values, 0

    def take(self, steps):
        if self.values is None or self.offset + len(steps) > len(self.values):
            raise RuntimeError('Missing or exhausted ACT random plan')
        value = self.values[self.offset:self.offset + len(steps)]
        if value.device != steps.device or value.dtype != steps.dtype:
            raise ValueError('ACT plan dtype/device mismatch')
        self.offset += len(steps)
        return value

    def finish(self):
        if self.values is None or self.offset != len(self.values):
            raise RuntimeError('ACT random plan was not consumed exactly once')
        self.values, self.offset = None, 0


def _forward_with_legacy_plan(self, carry, batch):
    """Source-gated official ACT forward; only random minimum supply is replaced."""
    new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
    new_steps = torch.where(carry.halted, 0, carry.steps)
    current = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                              batch[k], v) for k, v in carry.current_data.items()}
    new_inner_carry, logits, (q_halt, q_continue) = self.inner(new_inner_carry, current)
    outputs = {'logits': logits, 'q_halt_logits': q_halt, 'q_continue_logits': q_continue}
    with torch.no_grad():
        new_steps = new_steps + 1
        last = new_steps >= self.config.halt_max_steps
        halted = last
        if self.training and self.config.halt_max_steps > 1:
            halted = halted | (q_halt > q_continue)
            minimum = self._review_legacy_draw_plan.take(new_steps)
            halted = halted & (new_steps >= minimum)
            next_halt, next_continue = self.inner(new_inner_carry, current)[-1]
            outputs['target_q_continue'] = torch.sigmoid(
                torch.where(last, next_halt, torch.maximum(next_halt, next_continue)))
    return type(carry)(new_inner_carry, new_steps, halted, current), outputs


def install_legacy_act_forward(loss_head, expected_source_sha=OFFICIAL_ACT_SHA256):
    """Patch only this model instance; no global torch or installed-file patches."""
    act = loss_head.model
    path = inspect.getsourcefile(type(act))
    if path is None or sha256(path) != expected_source_sha:
        raise ValueError('HRM ACT source differs from the audited implementation')
    if hasattr(act, '_review_legacy_draw_plan'):
        raise ValueError('Legacy ACT execution already installed')
    if act.config.puzzle_emb_ndim != 0:
        raise ValueError('Sparse puzzle embeddings were not audited for regrouping')
    plan = LegacyDrawPlan()
    act._review_legacy_draw_plan = plan
    act.forward = types.MethodType(_forward_with_legacy_plan, act)
    return plan


def effective_update(model, optimizer, cfg, puzzle, solution, carry, step,
                     runtime_train_micro, plan, adapter=None):
    """One optimizer update, with unchanged full-batch offered stream and scaling.

    Caller owns stream.take(256), LR/early-stop bookkeeping, and checkpointing.
    This helper sets the original schedule and clips/steps exactly once.
    """
    if adapter is None:
        import a08_train_baseline as adapter
    total = len(puzzle)
    if cfg['method'] != 'hrm' or total != cfg['batch_size'] or len(solution) != total:
        raise ValueError('Expected one complete HRM effective batch')
    if not model.training:
        raise ValueError('Training update requires model.train()')
    if runtime_train_micro <= 0 or total % runtime_train_micro:
        raise ValueError('Runtime micro-batch must divide the effective batch')
    carry = regroup_carry(carry, total, runtime_train_micro)
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = adapter.scheduled_lr(cfg, step)
    act_cfg = cfg['model']
    active_act = act_cfg['halt_max_steps'] > 1
    if active_act:
        plan.begin(legacy_min_halt_steps(total, puzzle.device, act_cfg['halt_exploration_prob'],
                                        act_cfg['halt_max_steps'], cfg['micro_batch_size']))
    accepted = 0
    value = 0.0
    for chunk, start in enumerate(range(0, total, runtime_train_micro)):
        pc, sc = puzzle[start:start + runtime_train_micro], solution[start:start + runtime_train_micro]
        loss, carry[chunk], fresh = adapter.train_loss(model, cfg, pc, sc, carry[chunk])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'Nonfinite HRM loss at step {step}')
        weight = len(pc) / total
        (loss * weight).backward()
        value += float(loss.detach()) * weight
        accepted += fresh
    if active_act:
        plan.finish()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    return carry, {'loss': value, 'gradient_norm': float(norm), 'accepted': accepted,
                   'offered': total, 'completed_episodes': sum(int(c.halted.sum()) for c in carry)}


def runtime_evaluate(model, cfg, *args, runtime_eval_batch, adapter=None, **kwargs):
    """Accuracy/validation batching only. Never apply this override to A05 timing."""
    if runtime_eval_batch <= 0 or cfg['method'] != 'hrm':
        raise ValueError('Invalid HRM evaluation execution batch')
    if adapter is None:
        import a08_train_baseline as adapter
    effective_cfg = copy.deepcopy(cfg)
    effective_cfg['eval_batch_size'] = runtime_eval_batch
    return adapter.evaluate(model, effective_cfg, *args, **kwargs)


def migrate_checkpoint(source, output_dir, expected_identity, expected_checkpoint_sha256,
                       runtime_train_micro, runtime_eval_batch):
    """Convert a STOPPED committed checkpoint into a new directory, never in place.

    No process signals are sent. The caller must independently establish a safe
    checkpoint boundary and handle the run's committed best/history artifacts.
    Source is copied into archive; all original scientific identity fields and
    model/optimizer/stream/RNG/state are retained. No frozen-source check bypass.
    """
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    if output_dir.exists() or output_dir == source.parent or source.parent in output_dir.parents:
        raise ValueError('Migration requires a new sibling/output directory outside the source run')
    if sha256(source) != expected_checkpoint_sha256:
        raise ValueError('Source checkpoint hash changed')
    verify_sources(expected_identity)
    identity = runtime_identity(expected_identity, runtime_train_micro, runtime_eval_batch)
    saved = torch.load(source, map_location='cpu', weights_only=False)
    if saved['identity'] != expected_identity:
        raise ValueError('Original checkpoint identity differs from the explicit expected identity')
    required = {'model', 'optimizer', 'stream', 'carry', 'rng', 'state', 'identity'}
    if not required.issubset(saved):
        raise ValueError('Checkpoint lacks full training resume state')
    if saved['state'].get('stopped_early') or saved['state']['step'] >= expected_identity['config']['max_steps']:
        raise ValueError('Already completed training must not be migrated as unfinished')
    saved['carry'] = regroup_carry(saved['carry'], 256, runtime_train_micro)
    saved['identity'] = identity
    # This is only a checkpoint conversion. New run identity explicitly signals
    # that committed best/history must be staged before a production continuation.
    output_dir.mkdir(parents=True, exist_ok=False)
    archive = output_dir / 'archive'
    archive.mkdir()
    shutil.copy2(source, archive / 'latest.original.pt')
    if sha256(source) != expected_checkpoint_sha256 or sha256(archive / 'latest.original.pt') != expected_checkpoint_sha256:
        raise ValueError('Source moved during migration; output is incomplete and must not run')
    temp = output_dir / 'latest.pt.tmp'
    torch.save(saved, temp)
    os.replace(temp, output_dir / 'latest.pt')
    record = {'status': 'converted_checkpoint_only', 'schema': SCHEMA,
              'parent_checkpoint': str(source), 'parent_checkpoint_sha256': expected_checkpoint_sha256,
              'parent_identity_sha256': digest(expected_identity), 'new_identity': identity,
              'new_checkpoint_sha256': sha256(output_dir / 'latest.pt'),
              'step': saved['state']['step'], 'numerical_note': NUMERICAL_NOTE,
              'requires_committed_best_and_history_staging': True}
    (output_dir / 'execution_migration.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record
