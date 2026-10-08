"""Evaluate noisy hidden-state restarts for a trained Hybrid Hyper-RRN.

Candidate zero is always the deterministic T-step Hybrid prediction.  Later
candidates perturb recurrent cell states, recurrent row/column/box unit states,
or both after every selected recurrent step.  Candidate selection only keeps a
strictly lower row/column/box violation count, so the deterministic candidate is
never discarded by an equally scoring stochastic candidate.

This is randomized neural trajectory sampling, not backtracking or exact repair.
"""

from __future__ import annotations

import argparse
import __main__
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from kaggle_sudoku_experiment import RATING_BUCKETS, rating_bucket_name
from kaggle_sudoku_hyper_rrn_experiment import HyperRRNCfg, SudokuHyperRRN, parameter_count
from sudoku_exchange_experiment import set_torch_threads


SPLITS = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class HybridRestartOptions:
    name: str
    restarts: int
    restart_chunk: int
    steps: int
    cell_noise_std: float
    unit_noise_std: float
    noise_warmup: int


class SudokuIndexDataset(Dataset):
    def __init__(self, npz, indices):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.puzzles = npz["puzzles"][self.indices].reshape(-1, 9, 9)
        self.solutions = npz["solutions"][self.indices].reshape(-1, 9, 9)
        self.ratings = (
            npz["ratings"][self.indices]
            if "ratings" in npz
            else np.full(len(self.indices), np.nan, dtype=np.float32)
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return {
            "dataset_index": int(self.indices[idx]),
            "puzzle": torch.from_numpy(self.puzzles[idx].astype(np.int64)),
            "solution": torch.from_numpy(self.solutions[idx].astype(np.int64)),
            "rating": torch.tensor(float(self.ratings[idx]), dtype=torch.float32),
        }


def parse_csv_set(value):
    return {item.strip() for item in value.split(",") if item.strip()}


def sample_indices(npz, args):
    if args.indices_file:
        values = np.loadtxt(args.indices_file, dtype=np.int64, ndmin=1)
        return np.asarray(values, dtype=np.int64).reshape(-1)
    split_indices = np.flatnonzero(npz["splits"] == SPLITS[args.split])
    ratings = npz.get("ratings", np.full(len(npz["splits"]), np.nan, dtype=np.float32))
    if args.bucket_names:
        wanted = parse_csv_set(args.bucket_names)
        keep = [rating_bucket_name(float(ratings[idx])) in wanted for idx in split_indices]
        split_indices = split_indices[np.asarray(keep, dtype=bool)]
    if args.min_rating is not None:
        values = np.nan_to_num(ratings[split_indices], nan=-math.inf)
        split_indices = split_indices[values >= args.min_rating]
    if args.strict_min_rating is not None:
        values = np.nan_to_num(ratings[split_indices], nan=-math.inf)
        split_indices = split_indices[values > args.strict_min_rating]
    if args.max_rating is not None:
        values = np.nan_to_num(ratings[split_indices], nan=math.inf)
        split_indices = split_indices[values <= args.max_rating]

    if args.samples_per_bucket > 0:
        rng = np.random.default_rng(args.bucket_sample_seed)
        selected = []
        groups = {}
        for idx in split_indices:
            groups.setdefault(rating_bucket_name(float(ratings[idx])), []).append(int(idx))
        ordered = [name for name, _, _ in RATING_BUCKETS] + ["rating_nan", "rating_other"]
        for name in ordered:
            values = groups.get(name, [])
            if not values:
                continue
            if len(values) > args.samples_per_bucket:
                values = rng.choice(values, size=args.samples_per_bucket, replace=False).tolist()
            selected.extend(values)
        return np.asarray(selected, dtype=np.int64)

    if args.limit > 0:
        split_indices = split_indices[: args.limit]
    return np.asarray(split_indices, dtype=np.int64)


def load_hybrid(checkpoint_path, device):
    __main__.HyperRRNCfg = HyperRRNCfg
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_cfg = checkpoint["cfg"]
    cfg = HyperRRNCfg(**saved_cfg) if isinstance(saved_cfg, dict) else saved_cfg
    if cfg.model_type != "hybrid":
        raise ValueError(f"Expected a hybrid checkpoint, got {cfg.model_type!r}")
    model = SudokuHyperRRN(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, cfg, checkpoint


def hard_violation_count_batch(pred):
    pred = pred.clamp(1, 9).long()
    onehot = F.one_hot(pred - 1, num_classes=9)
    row_bad = (onehot.sum(dim=2) != 1).any(dim=2)
    col_bad = (onehot.sum(dim=1) != 1).any(dim=2)
    boxes = onehot.reshape(pred.shape[0], 3, 3, 3, 3, 9).sum(dim=(2, 4)).reshape(
        pred.shape[0], 9, 9
    )
    box_bad = (boxes != 1).any(dim=2)
    return (
        row_bad.to(torch.int64).sum(dim=1)
        + col_bad.to(torch.int64).sum(dim=1)
        + box_bad.to(torch.int64).sum(dim=1)
    )


@torch.no_grad()
def stochastic_hybrid_forward(model, puzzle, options, restart_index):
    """Run one deterministic or stochastic recurrent Hybrid trajectory."""
    batch = puzzle.shape[0]
    x0 = model.input_features(puzzle)
    unit_x0 = model.initial_unit_features(x0)
    h = x0
    unit_h = unit_x0
    mutable_cell = (puzzle.reshape(batch, 81) == 0).unsqueeze(-1)

    for step in range(int(options.steps)):
        h, unit_h = model.step(h, unit_h, x0, unit_x0)
        noisy_step = step + 1 > int(options.noise_warmup)
        if restart_index > 0 and noisy_step:
            if options.cell_noise_std > 0.0:
                noise = torch.randn_like(h) * float(options.cell_noise_std)
                h = torch.where(mutable_cell, h + noise, h)
            if options.unit_noise_std > 0.0:
                unit_h = unit_h + torch.randn_like(unit_h) * float(options.unit_noise_std)

    logits = model.logits_from_state(h, puzzle)
    return logits, F.softmax(logits, dim=-1)


def base_options(args, cfg):
    return HybridRestartOptions(
        name="base",
        restarts=max(1, int(args.restarts)),
        restart_chunk=max(1, int(args.restart_chunk)),
        steps=int(args.eval_steps or cfg.eval_T),
        cell_noise_std=float(args.cell_noise_std),
        unit_noise_std=float(args.unit_noise_std),
        noise_warmup=max(0, int(args.noise_warmup)),
    )


def parse_portfolio_entry(entry, defaults, index):
    values = {
        "name": f"p{index}",
        "restarts": defaults.restarts,
        "restart_chunk": defaults.restart_chunk,
        "steps": defaults.steps,
        "cell_noise_std": defaults.cell_noise_std,
        "unit_noise_std": defaults.unit_noise_std,
        "noise_warmup": defaults.noise_warmup,
    }
    aliases = {
        "name": "name", "label": "name",
        "r": "restarts", "restarts": "restarts",
        "chunk": "restart_chunk", "restart_chunk": "restart_chunk",
        "steps": "steps", "step": "steps", "T": "steps", "t": "steps",
        "cell": "cell_noise_std", "c": "cell_noise_std",
        "cell_noise": "cell_noise_std", "cell_noise_std": "cell_noise_std",
        "unit": "unit_noise_std", "u": "unit_noise_std",
        "unit_noise": "unit_noise_std", "unit_noise_std": "unit_noise_std",
        "warmup": "noise_warmup", "w": "noise_warmup",
        "noise_warmup": "noise_warmup",
    }
    for token in entry.replace("|", ",").split(","):
        token = token.strip()
        if not token:
            continue
        sep = "=" if "=" in token else ":"
        if sep not in token:
            raise ValueError(f"Invalid portfolio token: {token}")
        raw_key, raw_value = token.split(sep, 1)
        key = aliases.get(raw_key.strip())
        if key is None:
            raise ValueError(f"Unknown portfolio key: {raw_key}")
        value = raw_value.strip()
        if key == "name":
            values[key] = value
        elif key in {"restarts", "restart_chunk", "steps", "noise_warmup"}:
            values[key] = int(value)
        else:
            values[key] = float(value)
    return HybridRestartOptions(**values)


def build_options(args, cfg):
    defaults = base_options(args, cfg)
    if not args.portfolio.strip():
        return [defaults]
    entries = [entry.strip() for entry in args.portfolio.split(";") if entry.strip()]
    return [parse_portfolio_entry(entry, defaults, i + 1) for i, entry in enumerate(entries)]


def describe_options(options):
    return "; ".join(
        f"{o.name}:T={o.steps},c={o.cell_noise_std},u={o.unit_noise_std},"
        f"w={o.noise_warmup},r={o.restarts},chunk={o.restart_chunk}"
        for o in options
    )


def parse_prefixes(value, total_budget):
    prefixes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    prefixes = [prefix for prefix in prefixes if 1 <= prefix <= total_budget]
    if total_budget not in prefixes:
        prefixes.append(total_budget)
    return prefixes


def empty_stats():
    return {
        "n": 0, "actual_exact": 0, "actual_valid": 0,
        "oracle_exact": 0, "cell": 0, "cells": 0, "viol": 0,
    }


def make_stats(prefixes):
    names = ["all"] + [name for name, _, _ in RATING_BUCKETS] + ["rating_nan", "rating_other"]
    return {prefix: {name: empty_stats() for name in names} for prefix in prefixes}


def snapshot_states(snapshots, index, best_pred, best_viol, oracle_exact, first_exact):
    snapshots["pred"][index].copy_(best_pred)
    snapshots["viol"][index].copy_(best_viol)
    snapshots["oracle_exact"][index].copy_(oracle_exact)
    snapshots["first_exact"][index].copy_(first_exact)


@torch.no_grad()
def evaluate_batch(model, puzzle, solution, options_list, prefixes, early_stop):
    device = puzzle.device
    batch_size = puzzle.shape[0]
    best_pred = torch.zeros((batch_size, 9, 9), dtype=torch.long, device=device)
    best_viol = torch.full((batch_size,), 10_000, dtype=torch.long, device=device)
    oracle_exact = torch.zeros(batch_size, dtype=torch.bool, device=device)
    first_exact = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    snapshots = {
        "pred": torch.empty((len(prefixes), batch_size, 9, 9), dtype=torch.long, device=device),
        "viol": torch.empty((len(prefixes), batch_size), dtype=torch.long, device=device),
        "oracle_exact": torch.empty((len(prefixes), batch_size), dtype=torch.bool, device=device),
        "first_exact": torch.empty((len(prefixes), batch_size), dtype=torch.long, device=device),
    }
    next_prefix = 0
    candidate_rank = 0
    active = torch.arange(batch_size, device=device)
    candidate_evals = 0
    baseline_elapsed = 0.0

    def process_candidate(pred, violations, candidate_active):
        nonlocal next_prefix, candidate_rank, candidate_evals
        candidate_rank += 1
        candidate_evals += pred.shape[0]
        target = solution.index_select(0, candidate_active)
        exact = (pred == target).reshape(pred.shape[0], -1).all(dim=1)
        old_viol = best_viol.index_select(0, candidate_active)
        better = violations < old_viol
        if bool(better.any().item()):
            update_indices = candidate_active[better]
            best_pred[update_indices] = pred[better]
            best_viol[update_indices] = violations[better]
        newly_exact = exact & (first_exact.index_select(0, candidate_active) < 0)
        if bool(newly_exact.any().item()):
            first_exact[candidate_active[newly_exact]] = candidate_rank
        oracle_exact[candidate_active] |= exact
        while next_prefix < len(prefixes) and candidate_rank >= prefixes[next_prefix]:
            snapshot_states(snapshots, next_prefix, best_pred, best_viol, oracle_exact, first_exact)
            next_prefix += 1

    for options in options_list:
        if active.numel() == 0:
            break
        candidate_active = active
        baseline_started = (
            time.perf_counter() if candidate_rank == 0 else None
        )
        _, probs = stochastic_hybrid_forward(
            model, puzzle.index_select(0, candidate_active), options, restart_index=0
        )
        pred = probs.argmax(dim=-1) + 1
        process_candidate(pred, hard_violation_count_batch(pred), candidate_active)
        if baseline_started is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            baseline_elapsed += time.perf_counter() - baseline_started
        completed = 1
        if early_stop:
            active = torch.nonzero(best_viol > 0, as_tuple=False).flatten()

        while completed < options.restarts and active.numel() > 0:
            chunk = min(int(options.restart_chunk), int(options.restarts) - completed)
            candidate_active = active
            active_puzzle = puzzle.index_select(0, candidate_active)
            active_size = active_puzzle.shape[0]
            puzzle_chunk = (
                active_puzzle[:, None].expand(active_size, chunk, 9, 9)
                .reshape(active_size * chunk, 9, 9).contiguous()
            )
            _, probs = stochastic_hybrid_forward(model, puzzle_chunk, options, restart_index=completed)
            pred = (probs.argmax(dim=-1) + 1).reshape(active_size, chunk, 9, 9)
            violations = hard_violation_count_batch(pred.reshape(-1, 9, 9)).reshape(active_size, chunk)
            for offset in range(chunk):
                process_candidate(pred[:, offset], violations[:, offset], candidate_active)
            completed += chunk
            if early_stop:
                active = torch.nonzero(best_viol > 0, as_tuple=False).flatten()

    while next_prefix < len(prefixes):
        snapshot_states(snapshots, next_prefix, best_pred, best_viol, oracle_exact, first_exact)
        next_prefix += 1
    snapshots["baseline_elapsed_seconds"] = baseline_elapsed
    return snapshots, candidate_evals


def update_item(item, exact, valid, oracle, cell_correct, violations):
    item["n"] += 1
    item["actual_exact"] += int(exact)
    item["actual_valid"] += int(valid)
    item["oracle_exact"] += int(oracle)
    item["cell"] += int(cell_correct)
    item["cells"] += 81
    item["viol"] += int(violations)


def aggregate(stats, prefixes, snapshots, solution, ratings, dataset_indices, detail_rows):
    solution = solution.cpu()
    pred_all = snapshots["pred"].cpu()
    viol_all = snapshots["viol"].cpu()
    oracle_all = snapshots["oracle_exact"].cpu()
    first_all = snapshots["first_exact"].cpu()
    for pi, prefix in enumerate(prefixes):
        pred = pred_all[pi]
        violations = viol_all[pi]
        exact = (pred == solution).reshape(pred.shape[0], -1).all(dim=1)
        valid = violations == 0
        cell_correct = (pred == solution).reshape(pred.shape[0], -1).sum(dim=1)
        for i in range(pred.shape[0]):
            bucket = rating_bucket_name(float(ratings[i]))
            for name in ("all", bucket):
                update_item(
                    stats[prefix][name], bool(exact[i]), bool(valid[i]), bool(oracle_all[pi, i]),
                    int(cell_correct[i]), int(violations[i]),
                )
            if detail_rows is not None:
                detail_rows.append({
                    "dataset_index": int(dataset_indices[i]), "rating": float(ratings[i]),
                    "bucket": bucket, "prefix": prefix, "actual_exact": int(exact[i]),
                    "actual_valid": int(valid[i]), "oracle_exact": int(oracle_all[pi, i]),
                    "best_violations": int(violations[i]), "cell_correct": int(cell_correct[i]),
                    "first_exact": int(first_all[pi, i]),
                })


def stats_rows(
    stats,
    prefixes,
    elapsed,
    baseline_elapsed,
    candidate_evals,
    seed,
    options_description,
):
    rows = []
    ordered = ["all"] + [name for name, _, _ in RATING_BUCKETS] + ["rating_nan", "rating_other"]
    for prefix in prefixes:
        for bucket in ordered:
            item = stats[prefix].get(bucket)
            if not item or item["n"] == 0:
                continue
            n = item["n"]
            rows.append({
                "prefix": prefix, "bucket": bucket, "n": n,
                "actual_exact": item["actual_exact"] / n,
                "actual_valid": item["actual_valid"] / n,
                "oracle_exact": item["oracle_exact"] / n,
                "selector_gap": (item["oracle_exact"] - item["actual_exact"]) / n,
                "cell_accuracy": item["cell"] / max(item["cells"], 1),
                "avg_violations": item["viol"] / n,
                "elapsed_seconds": elapsed, "candidate_evals": candidate_evals,
                "baseline_elapsed_seconds": baseline_elapsed,
                "seed": seed, "portfolio": options_description,
            })
    return rows


def write_csv(path, rows):
    if not path or not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save] {target}", flush=True)


@torch.no_grad()
def evaluate(model, cfg, loader, device, args):
    options = build_options(args, cfg)
    total_budget = sum(max(1, int(option.restarts)) for option in options)
    prefixes = parse_prefixes(args.prefixes, total_budget)
    stats = make_stats(prefixes)
    detail_rows = [] if args.detail_csv else None
    seen = 0
    candidate_evals = 0
    baseline_elapsed = 0.0
    started = time.perf_counter()
    description = describe_options(options)
    print(f"[hybrid-restart] portfolio {description}", flush=True)
    print(f"[hybrid-restart] total_budget={total_budget} prefixes={prefixes}", flush=True)

    for batch in loader:
        puzzle = batch["puzzle"].to(device)
        solution = batch["solution"].to(device)
        snapshots, batch_evals = evaluate_batch(
            model, puzzle, solution, options, prefixes, bool(args.early_stop)
        )
        candidate_evals += batch_evals
        baseline_elapsed += float(snapshots["baseline_elapsed_seconds"])
        aggregate(
            stats, prefixes, snapshots, solution, batch["rating"].numpy(),
            batch["dataset_index"].numpy(), detail_rows,
        )
        seen += puzzle.shape[0]
        if args.progress_every and seen % args.progress_every < puzzle.shape[0]:
            elapsed = time.perf_counter() - started
            print(
                f"[progress] seen={seen} elapsed={elapsed:.1f}s rate={seen/max(elapsed,1e-9):.2f}/s "
                f"candidate_evals={candidate_evals}", flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    rows = stats_rows(
        stats,
        prefixes,
        elapsed,
        baseline_elapsed,
        candidate_evals,
        args.seed,
        description,
    )
    print("[hybrid-restart] prefix bucket n actual oracle gap cell_acc avg_viol", flush=True)
    for row in rows:
        print(
            f"{row['prefix']:>6} {row['bucket']:<13} {row['n']:6d} "
            f"{row['actual_exact']:.6f} {row['oracle_exact']:.6f} {row['selector_gap']:.6f} "
            f"{row['cell_accuracy']:.6f} {row['avg_violations']:.4f}", flush=True,
        )
    print(
        f"[hybrid-restart] elapsed={elapsed:.1f}s sec_per_puzzle={elapsed/max(seen,1):.6f} "
        f"candidate_evals={candidate_evals}", flush=True,
    )
    write_csv(args.summary_csv, rows)
    write_csv(args.detail_csv, detail_rows)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default="data/cache_full_3m.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=sorted(SPLITS), default="val")
    parser.add_argument("--indices_file", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--samples_per_bucket", type=int, default=0)
    parser.add_argument("--bucket_sample_seed", type=int, default=20260722)
    parser.add_argument("--bucket_names", default="rating_4_plus")
    parser.add_argument("--min_rating", type=float, default=None)
    parser.add_argument("--strict_min_rating", type=float, default=None)
    parser.add_argument("--max_rating", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_steps", type=int, default=64)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--restart_chunk", type=int, default=4)
    parser.add_argument("--cell_noise_std", type=float, default=0.0)
    parser.add_argument("--unit_noise_std", type=float, default=0.0)
    parser.add_argument("--noise_warmup", type=int, default=0)
    parser.add_argument("--portfolio", default="")
    parser.add_argument("--prefixes", default="1,8,16,32,64,128,256")
    parser.add_argument("--early_stop", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--summary_csv", default="")
    parser.add_argument("--detail_csv", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    set_torch_threads()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    model, cfg, checkpoint = load_hybrid(args.checkpoint, device)
    npz = np.load(args.cache_path, allow_pickle=False)
    indices = sample_indices(npz, args)
    dataset = SudokuIndexDataset(npz, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print(
        f"[hybrid-restart] checkpoint={args.checkpoint} saved_step={checkpoint.get('step')} "
        f"val_exact={checkpoint.get('val_exact')} params={parameter_count(model)}", flush=True,
    )
    print(
        f"[hybrid-restart] split={args.split} n={len(dataset)} batch={args.batch_size} "
        f"bucket_names={args.bucket_names or 'all'} device={device}", flush=True,
    )
    print(
        f"[hybrid-restart] cfg D={cfg.D} msg_hidden={cfg.msg_hidden} "
        f"train_T={cfg.train_T} eval_T={cfg.eval_T}", flush=True,
    )
    evaluate(model, cfg, loader, device, args)


if __name__ == "__main__":
    main()
