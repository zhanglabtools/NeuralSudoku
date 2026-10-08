"""Full-test evaluation with exact symbolic validation and active batching.

Valid Sudoku grids are accepted immediately at T64 or after any reflection
cycle.  Only unresolved puzzles remain in the recurrent batch.  This is exact
constraint checking, not a search procedure.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch

from eval_hybrid_hyper_rrn_restarts import hard_violation_count_batch, load_hybrid
from eval_iterative_hyper_reflection import SPLITS, load_reflector
from sudoku_exchange_experiment import set_seed, set_torch_threads
from sudoku_cache_utils import load_sudoku_dataset
from train_symbolic_primal_dual_reflection import (
    SymbolicPrimalDualCfg,
    SymbolicPrimalDualReflector,
)


def load_symbolic(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    backbone, _, _ = load_hybrid(checkpoint["base_checkpoint"], device)
    cfg = SymbolicPrimalDualCfg(**checkpoint["reflection_cfg"])
    model = SymbolicPrimalDualReflector(backbone, cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, cfg, checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", choices=["iterative", "symbolic"], required=True
    )
    parser.add_argument("--reflection_checkpoint", required=True)
    parser.add_argument(
        "--cache_path",
        default="data/cache_full_3m.npz",
    )
    parser.add_argument("--split", choices=sorted(SPLITS), default="test")
    parser.add_argument("--all_ratings", type=int, default=1)
    parser.add_argument("--strict_min_rating", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--progress_every", type=int, default=4096)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--detail_npz", default="")
    return parser.parse_args()


def _expand(model, tensor, slots):
    return model._expand_slots(tensor, slots)


def _keep_slots(tensor, keep, old_batch, slots):
    return tensor.reshape(old_batch, slots, *tensor.shape[1:])[keep].reshape(
        int(keep.sum()) * slots, *tensor.shape[1:]
    ).contiguous()


@torch.no_grad()
def active_batch(model, puzzle, solution, symbolic):
    batch = puzzle.shape[0]
    slots = int(model.cfg.slots)
    cycles = int(model.cfg.cycles)
    x0, unit_x0, parent_h, parent_u, anchor_logits = model.encode_parent(puzzle)
    anchor_pred = anchor_logits.argmax(dim=-1) + 1
    anchor_valid = hard_violation_count_batch(anchor_pred) == 0
    anchor_exact = (anchor_pred == solution).reshape(batch, -1).all(dim=-1)

    solved_step = torch.full(
        (batch,), -1, dtype=torch.int16, device=puzzle.device
    )
    solved_exact = torch.zeros(batch, dtype=torch.bool, device=puzzle.device)
    solved_step[anchor_valid] = 0
    solved_exact[anchor_valid] = anchor_exact[anchor_valid]
    equivalent_updates = torch.full(
        (batch,), int(model.cfg.parent_steps), dtype=torch.int32, device=puzzle.device
    )
    active_local = torch.nonzero(~anchor_valid, as_tuple=False).squeeze(-1)
    active_before = [int(active_local.numel())]
    new_valid = [int(anchor_valid.sum())]

    if active_local.numel() == 0:
        return solved_step, solved_exact, equivalent_updates, active_before, new_valid

    active_puzzle = puzzle.index_select(0, active_local)
    active_x0 = x0.index_select(0, active_local)
    active_unit_x0 = unit_x0.index_select(0, active_local)
    active_h = parent_h.index_select(0, active_local)
    active_u = parent_u.index_select(0, active_local)
    source_batch = int(active_local.numel())
    h = _expand(model, active_h, slots)
    unit_h = _expand(model, active_u, slots)
    slot_x0 = _expand(model, active_x0, slots)
    slot_unit_x0 = _expand(model, active_unit_x0, slots)
    slot_puzzle = _expand(model, active_puzzle, slots)
    dual = h.new_zeros(source_batch * slots, 27, 9) if symbolic else None

    for cycle_index in range(cycles):
        if source_batch == 0:
            active_before.append(0)
            new_valid.append(0)
            continue
        equivalent_updates[active_local] += slots * int(model.cfg.recovery_steps)
        if symbolic:
            h, unit_h, dual, _ = model._symbolic_reflect_once(
                h,
                unit_h,
                slot_x0,
                slot_unit_x0,
                slot_puzzle,
                dual,
                source_batch=source_batch,
                slots=slots,
                cycle_index=cycle_index,
            )
        else:
            h, unit_h, _ = model._reflect_once(
                h,
                unit_h,
                slot_x0,
                slot_unit_x0,
                slot_puzzle,
                source_batch=source_batch,
                slots=slots,
                cycle_index=cycle_index,
            )
        h, unit_h = model._rollout(
            h,
            unit_h,
            slot_x0,
            slot_unit_x0,
            model.cfg.recovery_steps,
        )
        logits = model.backbone.logits_from_state(h, slot_puzzle)
        pred = logits.argmax(dim=-1) + 1
        valid = (hard_violation_count_batch(pred) == 0).reshape(
            source_batch, slots
        )
        puzzle_solved = valid.any(dim=1)
        solved_count = int(puzzle_solved.sum())
        new_valid.append(solved_count)
        if solved_count:
            first_slot = valid.to(torch.int64).argmax(dim=1)
            pred_by_puzzle = pred.reshape(source_batch, slots, 9, 9)
            rows = torch.arange(source_batch, device=puzzle.device)
            chosen = pred_by_puzzle[rows, first_slot]
            active_solution = solution.index_select(0, active_local)
            exact = (chosen == active_solution).reshape(source_batch, -1).all(dim=-1)
            solved_indices = active_local[puzzle_solved]
            solved_step[solved_indices] = cycle_index + 1
            solved_exact[solved_indices] = exact[puzzle_solved]

        keep = ~puzzle_solved
        old_batch = source_batch
        active_local = active_local[keep]
        source_batch = int(keep.sum())
        active_before.append(source_batch)
        if source_batch == 0:
            continue
        h = _keep_slots(h, keep, old_batch, slots)
        unit_h = _keep_slots(unit_h, keep, old_batch, slots)
        slot_x0 = _keep_slots(slot_x0, keep, old_batch, slots)
        slot_unit_x0 = _keep_slots(slot_unit_x0, keep, old_batch, slots)
        slot_puzzle = _keep_slots(slot_puzzle, keep, old_batch, slots)
        if symbolic:
            dual = _keep_slots(dual, keep, old_batch, slots)

    return solved_step, solved_exact, equivalent_updates, active_before, new_valid


def summarize_bucket(name, mask, solved_step, solved_exact, updates, cycles, elapsed):
    n = int(mask.sum())
    if n == 0:
        return None
    steps = solved_step[mask]
    exact = solved_exact[mask]
    bucket_updates = updates[mask]
    row = {
        "bucket": name,
        "n": n,
        "symbolic_exact": float(exact.mean()),
        "valid_not_exact": int(((steps >= 0) & ~exact).sum()),
        "unsolved": int((steps < 0).sum()),
        "mean_equivalent_steps": float(bucket_updates.mean()),
        "median_equivalent_steps": float(np.median(bucket_updates)),
        "elapsed_full_run": elapsed,
        "full_run_puzzles_per_second": len(solved_step) / max(elapsed, 1e-9),
    }
    cumulative = steps == 0
    row["t64_exact"] = float(cumulative.mean())
    for cycle in range(1, cycles + 1):
        cumulative = cumulative | (steps == cycle)
        row[f"cycle{cycle}_cumulative_exact"] = float(cumulative.mean())
        row[f"cycle{cycle}_new"] = int((steps == cycle).sum())
    return row


@torch.no_grad()
def main():
    args = parse_args()
    set_torch_threads()
    set_seed(args.seed)
    device = torch.device(args.device)
    if args.model_type == "iterative":
        model, cfg, checkpoint = load_reflector(
            args.reflection_checkpoint, device
        )
        symbolic = False
    else:
        model, cfg, checkpoint = load_symbolic(
            args.reflection_checkpoint, device
        )
        symbolic = True

    dataset = load_sudoku_dataset(
        args.cache_path, split=SPLITS[args.split], limit=0
    )
    if args.all_ratings:
        indices = np.arange(len(dataset), dtype=np.int64)
    else:
        indices = np.flatnonzero(
            np.nan_to_num(dataset.ratings, nan=-np.inf)
            > args.strict_min_rating
        )
    if args.limit:
        indices = indices[: args.limit]

    all_steps = []
    all_exact = []
    all_updates = []
    all_ratings = []
    aggregate_active = np.zeros(cfg.cycles + 1, dtype=np.int64)
    aggregate_new = np.zeros(cfg.cycles + 1, dtype=np.int64)
    started = time.perf_counter()
    for offset in range(0, len(indices), args.batch_size):
        take = indices[offset : offset + args.batch_size]
        puzzle = torch.as_tensor(dataset.puzzles[take], dtype=torch.long, device=device)
        solution = torch.as_tensor(
            dataset.solutions[take], dtype=torch.long, device=device
        )
        solved_step, solved_exact, updates, active_before, new_valid = active_batch(
            model, puzzle, solution, symbolic
        )
        all_steps.append(solved_step.cpu().numpy())
        all_exact.append(solved_exact.cpu().numpy())
        all_updates.append(updates.cpu().numpy())
        all_ratings.append(np.asarray(dataset.ratings[take]))
        aggregate_active[: len(active_before)] += np.asarray(active_before)
        aggregate_new[: len(new_valid)] += np.asarray(new_valid)
        seen = offset + len(take)
        if args.progress_every and seen % args.progress_every < len(take):
            print(
                f"[progress] seen={seen}/{len(indices)} "
                f"solved={sum(int(x.sum()) for x in all_exact)} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    solved_step = np.concatenate(all_steps)
    solved_exact = np.concatenate(all_exact)
    updates = np.concatenate(all_updates)
    ratings = np.concatenate(all_ratings)
    finite_rating = np.nan_to_num(ratings, nan=-np.inf)
    masks = [
        ("all", np.ones(len(indices), dtype=bool)),
        ("rating_gt4", finite_rating > 4.0),
        ("rating_le4", finite_rating <= 4.0),
    ]
    rows = []
    for name, mask in masks:
        row = summarize_bucket(
            name, mask, solved_step, solved_exact, updates, cfg.cycles, elapsed
        )
        if row is not None:
            row.update(
                {
                    "model_type": args.model_type,
                    "variant": getattr(cfg, "variant", "symbolic_halt_c3"),
                    "slots": cfg.slots,
                    "cycles": cfg.cycles,
                    "recovery_steps": cfg.recovery_steps,
                    "checkpoint_step": checkpoint.get("step", -1),
                }
            )
            rows.append(row)
            print(f"[result] {row}", flush=True)
    print(
        f"[active] before_or_after={aggregate_active.tolist()} "
        f"new_valid={aggregate_new.tolist()}",
        flush=True,
    )

    if args.detail_npz:
        detail_target = Path(args.detail_npz)
        detail_target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            detail_target,
            test_position=np.asarray(indices, dtype=np.int64),
            rating=ratings.astype(np.float32, copy=False),
            solved_step=solved_step.astype(np.int16, copy=False),
            selected_exact=solved_exact.astype(bool, copy=False),
            equivalent_steps=updates.astype(np.int32, copy=False),
        )
        print(f"[save-detail] {detail_target}", flush=True)

    target = Path(args.output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save] {target}", flush=True)


if __name__ == "__main__":
    main()
