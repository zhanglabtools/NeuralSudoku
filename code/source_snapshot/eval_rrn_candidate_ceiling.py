"""Estimate paper-style RRN network-only randomized candidate ceilings."""

from __future__ import annotations

import argparse
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
from kaggle_sudoku_rrn_paper_experiment import RRNPaperCfg, SudokuRRNPaper
from sudoku_exchange_experiment import set_torch_threads


SPLITS = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class RRNInferenceOptions:
    name: str
    restarts: int
    restart_chunk: int
    steps: int
    hidden_noise_std: float
    logit_noise_std: float


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
    ratings = npz["ratings"] if "ratings" in npz else np.full(len(npz["splits"]), np.nan, dtype=np.float32)

    if args.bucket_names:
        wanted = parse_csv_set(args.bucket_names)
        mask = [rating_bucket_name(float(ratings[idx])) in wanted for idx in split_indices]
        split_indices = split_indices[np.asarray(mask, dtype=bool)]
    if args.min_rating is not None:
        split_indices = split_indices[np.nan_to_num(ratings[split_indices], nan=-math.inf) >= args.min_rating]
    if args.max_rating is not None:
        split_indices = split_indices[np.nan_to_num(ratings[split_indices], nan=math.inf) <= args.max_rating]

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


def checkpoint_attr(checkpoint, cfg, name, default):
    if cfg is not None and hasattr(cfg, name):
        return getattr(cfg, name)
    return checkpoint.get("args", {}).get(name, default)


def load_rrn(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("cfg")
    model_cfg = RRNPaperCfg(
        D=checkpoint_attr(checkpoint, cfg, "D", 128),
        train_T=checkpoint_attr(checkpoint, cfg, "train_T", 32),
        eval_T=checkpoint_attr(checkpoint, cfg, "eval_T", 64),
        msg_hidden=checkpoint_attr(checkpoint, cfg, "msg_hidden", 256),
        dropout=checkpoint_attr(checkpoint, cfg, "dropout", 0.0),
        force_clues=checkpoint_attr(checkpoint, cfg, "force_clues", True),
    )
    model = SudokuRRNPaper(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, model_cfg


def hard_violation_count_batch(pred):
    pred = pred.clamp(1, 9).long()
    onehot = F.one_hot(pred - 1, num_classes=9)
    row_bad = (onehot.sum(dim=2) != 1).any(dim=2)
    col_bad = (onehot.sum(dim=1) != 1).any(dim=2)
    boxes = onehot.reshape(pred.shape[0], 3, 3, 3, 3, 9).sum(dim=(2, 4)).reshape(pred.shape[0], 9, 9)
    box_bad = (boxes != 1).any(dim=2)
    return (
        row_bad.to(torch.int64).sum(dim=1)
        + col_bad.to(torch.int64).sum(dim=1)
        + box_bad.to(torch.int64).sum(dim=1)
    )


@torch.no_grad()
def stochastic_rrn_forward(model, puzzle, options, restart_index):
    batch = puzzle.shape[0]
    x0 = model.input_features(puzzle)
    h = x0
    edge_e = model.edge_emb(model.edge_type).unsqueeze(0).expand(batch, -1, -1)
    mutable_state = (puzzle.reshape(batch, 81) == 0).unsqueeze(-1)

    for _ in range(int(options.steps)):
        h = model.step(h, x0, edge_e)
        if restart_index > 0 and options.hidden_noise_std > 0.0:
            noise = torch.randn_like(h) * float(options.hidden_noise_std)
            h = torch.where(mutable_state, h + noise, h)

    logits = model.classifier(model.out_norm(h)).reshape(batch, 9, 9, 9)
    if restart_index > 0 and options.logit_noise_std > 0.0:
        mutable_logits = (puzzle == 0).unsqueeze(-1)
        noise = torch.randn_like(logits) * float(options.logit_noise_std)
        logits = torch.where(mutable_logits, logits + noise, logits)
    logits = model.force_clue_logits(logits, puzzle)
    probs = F.softmax(logits, dim=-1)
    return logits, probs


def base_options(args, model_cfg):
    steps = int(args.eval_steps) if args.eval_steps else int(model_cfg.eval_T)
    return RRNInferenceOptions(
        name="base",
        restarts=max(1, int(args.restarts)),
        restart_chunk=max(1, int(args.restart_chunk)),
        steps=steps,
        hidden_noise_std=float(args.hidden_noise_std),
        logit_noise_std=float(args.logit_noise_std),
    )


def parse_portfolio_entry(entry, defaults, index):
    values = {
        "name": f"p{index}",
        "restarts": defaults.restarts,
        "restart_chunk": defaults.restart_chunk,
        "steps": defaults.steps,
        "hidden_noise_std": defaults.hidden_noise_std,
        "logit_noise_std": defaults.logit_noise_std,
    }
    aliases = {
        "name": "name",
        "label": "name",
        "r": "restarts",
        "restarts": "restarts",
        "chunk": "restart_chunk",
        "restart_chunk": "restart_chunk",
        "steps": "steps",
        "step": "steps",
        "eval_t": "steps",
        "eval_T": "steps",
        "T": "steps",
        "t": "steps",
        "h": "hidden_noise_std",
        "hidden": "hidden_noise_std",
        "hidden_noise": "hidden_noise_std",
        "hidden_noise_std": "hidden_noise_std",
        "l": "logit_noise_std",
        "logit": "logit_noise_std",
        "logits": "logit_noise_std",
        "logit_noise": "logit_noise_std",
        "logit_noise_std": "logit_noise_std",
    }

    for token in entry.replace("|", ",").split(","):
        token = token.strip()
        if not token:
            continue
        sep = "=" if "=" in token else ":"
        if sep not in token:
            raise ValueError(f"invalid portfolio token: {token}")
        raw_key, raw_value = token.split(sep, 1)
        key = aliases.get(raw_key.strip())
        if key is None:
            raise ValueError(f"unknown portfolio key: {raw_key}")
        value = raw_value.strip()
        if key == "name":
            values[key] = value
        elif key in ("restarts", "restart_chunk", "steps"):
            values[key] = int(value)
        else:
            values[key] = float(value)
    return RRNInferenceOptions(**values)


def build_options(args, model_cfg):
    defaults = base_options(args, model_cfg)
    if not args.portfolio.strip():
        return [defaults]
    entries = [entry.strip() for entry in args.portfolio.split(";") if entry.strip()]
    if not entries:
        return [defaults]
    return [parse_portfolio_entry(entry, defaults, i + 1) for i, entry in enumerate(entries)]


def describe_options(options_list):
    parts = []
    for opt in options_list:
        parts.append(
            f"{opt.name}:steps={opt.steps},h={opt.hidden_noise_std},l={opt.logit_noise_std},"
            f"r={opt.restarts},chunk={opt.restart_chunk}"
        )
    return "; ".join(parts)


def parse_prefixes(value, total_budget):
    prefixes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    prefixes = [prefix for prefix in prefixes if 1 <= prefix <= total_budget]
    if total_budget not in prefixes:
        prefixes.append(total_budget)
    return prefixes


def empty_stats():
    return {
        "n": 0,
        "actual_exact": 0,
        "actual_valid": 0,
        "oracle_exact": 0,
        "oracle_valid": 0,
        "cell": 0,
        "cells": 0,
        "viol": 0,
    }


def make_stats(prefixes):
    ordered = ["all"] + [name for name, _, _ in RATING_BUCKETS] + ["rating_nan", "rating_other"]
    return {prefix: {name: empty_stats() for name in ordered} for prefix in prefixes}


def update_one(item, actual_exact, actual_valid, oracle_exact, oracle_valid, cell_correct, violations):
    item["n"] += 1
    item["actual_exact"] += int(actual_exact)
    item["actual_valid"] += int(actual_valid)
    item["oracle_exact"] += int(oracle_exact)
    item["oracle_valid"] += int(oracle_valid)
    item["cell"] += int(cell_correct)
    item["cells"] += 81
    item["viol"] += int(violations)


def snapshot_states(snapshots, snapshot_index, best_pred, best_viol, oracle_exact, oracle_valid):
    snapshots["pred"][snapshot_index].copy_(best_pred)
    snapshots["viol"][snapshot_index].copy_(best_viol)
    snapshots["oracle_exact"][snapshot_index].copy_(oracle_exact)
    snapshots["oracle_valid"][snapshot_index].copy_(oracle_valid)


@torch.no_grad()
def evaluate_batch(model, puzzle, solution, options_list, prefixes, args):
    device = puzzle.device
    batch_size = puzzle.shape[0]
    best_pred = torch.zeros((batch_size, 9, 9), dtype=torch.long, device=device)
    best_viol = torch.full((batch_size,), 10_000, dtype=torch.long, device=device)
    oracle_exact = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    oracle_valid = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    snapshots = {
        "pred": torch.empty((len(prefixes), batch_size, 9, 9), dtype=torch.long, device=device),
        "viol": torch.empty((len(prefixes), batch_size), dtype=torch.long, device=device),
        "oracle_exact": torch.empty((len(prefixes), batch_size), dtype=torch.bool, device=device),
        "oracle_valid": torch.empty((len(prefixes), batch_size), dtype=torch.bool, device=device),
    }

    next_prefix = 0
    candidate_rank = 0
    active = torch.arange(batch_size, device=device)

    def process_candidate(pred, violations, candidate_active):
        nonlocal next_prefix, candidate_rank, active
        candidate_rank += 1
        exact = (pred == solution.index_select(0, candidate_active)).reshape(pred.shape[0], -1).all(dim=1)
        valid = violations == 0
        old_viol = best_viol.index_select(0, candidate_active)
        better = violations < old_viol
        if bool(better.any().item()):
            update_indices = candidate_active[better]
            best_pred[update_indices] = pred[better]
            best_viol[update_indices] = violations[better]
        oracle_exact[candidate_active] |= exact
        oracle_valid[candidate_active] |= valid

        while next_prefix < len(prefixes) and candidate_rank >= prefixes[next_prefix]:
            snapshot_states(snapshots, next_prefix, best_pred, best_viol, oracle_exact, oracle_valid)
            next_prefix += 1
        if args.stop_on_oracle_exact:
            active = torch.nonzero(~oracle_exact, as_tuple=False).flatten()

    for options in options_list:
        if active.numel() == 0:
            break
        completed = 0
        active_puzzle = puzzle.index_select(0, active)
        _, probs = stochastic_rrn_forward(model, active_puzzle, options, 0)
        pred = probs.argmax(dim=-1) + 1
        violations = hard_violation_count_batch(pred)
        process_candidate(pred, violations, active)
        completed = 1

        while completed < options.restarts and active.numel() > 0:
            chunk = min(options.restart_chunk, options.restarts - completed)
            active_puzzle = puzzle.index_select(0, active)
            active_size = active_puzzle.shape[0]
            puzzle_chunk = (
                active_puzzle[:, None, :, :]
                .expand(active_size, chunk, 9, 9)
                .reshape(active_size * chunk, 9, 9)
                .contiguous()
            )
            _, probs = stochastic_rrn_forward(model, puzzle_chunk, options, completed)
            pred = (probs.argmax(dim=-1) + 1).reshape(active_size, chunk, 9, 9)
            violations = hard_violation_count_batch(pred.reshape(active_size * chunk, 9, 9)).reshape(active_size, chunk)
            candidate_active = active
            for offset in range(chunk):
                process_candidate(pred[:, offset], violations[:, offset], candidate_active)
                if active.numel() == 0:
                    break
            completed += chunk

    while next_prefix < len(prefixes):
        snapshot_states(snapshots, next_prefix, best_pred, best_viol, oracle_exact, oracle_valid)
        next_prefix += 1
    return snapshots


def aggregate(stats, prefixes, snapshots, solution, ratings):
    solution_cpu = solution.cpu()
    pred_cpu = snapshots["pred"].cpu()
    viol_cpu = snapshots["viol"].cpu()
    oracle_exact_cpu = snapshots["oracle_exact"].cpu()
    oracle_valid_cpu = snapshots["oracle_valid"].cpu()

    for prefix_index, prefix in enumerate(prefixes):
        pred = pred_cpu[prefix_index]
        violations = viol_cpu[prefix_index]
        actual_exact = (pred == solution_cpu).reshape(pred.shape[0], -1).all(dim=1)
        actual_valid = violations == 0
        cell_correct = (pred == solution_cpu).reshape(pred.shape[0], -1).sum(dim=1)
        for idx in range(pred.shape[0]):
            bucket = rating_bucket_name(float(ratings[idx]))
            for name in ("all", bucket):
                update_one(
                    stats[prefix][name],
                    bool(actual_exact[idx].item()),
                    bool(actual_valid[idx].item()),
                    bool(oracle_exact_cpu[prefix_index, idx].item()),
                    bool(oracle_valid_cpu[prefix_index, idx].item()),
                    int(cell_correct[idx].item()),
                    int(violations[idx].item()),
                )


def print_stats(stats, prefixes, elapsed, args):
    print("[rrn-ceiling] prefix bucket n actual_exact actual_valid oracle_exact oracle_valid gap cell_acc avg_viol", flush=True)
    ordered = ["all"] + [name for name, _, _ in RATING_BUCKETS] + ["rating_nan", "rating_other"]
    rows = []
    for prefix in prefixes:
        for bucket in ordered:
            item = stats[prefix].get(bucket)
            if not item or item["n"] == 0:
                continue
            n = item["n"]
            actual_exact = item["actual_exact"] / n
            oracle_exact = item["oracle_exact"] / n
            rows.append({
                "prefix": prefix, "bucket": bucket, "n": n,
                "actual_exact_count": item["actual_exact"],
                "actual_valid_count": item["actual_valid"],
                "oracle_exact_count": item["oracle_exact"],
                "oracle_valid_count": item["oracle_valid"],
                "cell_correct": item["cell"], "cells": item["cells"],
                "violation_sum": item["viol"],
                "actual_exact": actual_exact,
                "actual_valid": item["actual_valid"] / n,
                "oracle_exact": oracle_exact,
                "oracle_valid": item["oracle_valid"] / n,
                "selector_gap": oracle_exact - actual_exact,
                "cell_accuracy": item["cell"] / max(item["cells"], 1),
                "avg_violations": item["viol"] / n,
                "elapsed_seconds": elapsed, "seed": args.seed,
            })
            print(
                f"{prefix:>6d} {bucket:<13} {n:6d} "
                f"{actual_exact:12.4f} {item['actual_valid'] / n:12.4f} "
                f"{oracle_exact:12.4f} {item['oracle_valid'] / n:12.4f} "
                f"{oracle_exact - actual_exact:7.4f} "
                f"{item['cell'] / max(item['cells'], 1):8.4f} {item['viol'] / n:8.3f}",
                flush=True,
            )
    n_all = stats[prefixes[-1]]["all"]["n"]
    if n_all:
        print(f"[rrn-ceiling] elapsed={elapsed:.1f}s sec_per_puzzle={elapsed / n_all:.4f}", flush=True)
    if args.summary_csv and rows:
        target = Path(args.summary_csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[save] {target}", flush=True)


@torch.no_grad()
def evaluate(model, model_cfg, loader, device, args):
    model.eval()
    options_list = build_options(args, model_cfg)
    total_budget = sum(max(1, int(option.restarts)) for option in options_list)
    prefixes = parse_prefixes(args.prefixes, total_budget)
    stats = make_stats(prefixes)
    seen = 0
    next_progress = int(args.progress_every) if args.progress_every else 0
    start_all = time.perf_counter()

    print(f"[rrn-ceiling] portfolio {describe_options(options_list)}", flush=True)
    print(f"[rrn-ceiling] total_budget={total_budget} prefixes={','.join(map(str, prefixes))}", flush=True)

    for batch in loader:
        puzzle = batch["puzzle"].to(device)
        solution = batch["solution"].to(device)
        ratings = batch["rating"].numpy()
        snapshots = evaluate_batch(model, puzzle, solution, options_list, prefixes, args)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        aggregate(stats, prefixes, snapshots, solution, ratings)
        seen += puzzle.shape[0]
        if next_progress and seen >= next_progress:
            elapsed = time.perf_counter() - start_all
            print(f"[progress] seen={seen} elapsed={elapsed:.1f}s rate={seen / max(elapsed, 1e-9):.2f}/s", flush=True)
            while next_progress and next_progress <= seen:
                next_progress += int(args.progress_every)

    elapsed = time.perf_counter() - start_all
    print_stats(stats, prefixes, elapsed, args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default="data/cache_full_3m.npz")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/model_rrn_paper_d128_t32_eval64_50k_20260610.pt",
    )
    parser.add_argument("--split", choices=sorted(SPLITS), default="test")
    parser.add_argument("--indices_file", default="")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--samples_per_bucket", type=int, default=0)
    parser.add_argument("--bucket_sample_seed", type=int, default=20260622)
    parser.add_argument("--bucket_names", default="")
    parser.add_argument("--min_rating", type=float, default=None)
    parser.add_argument("--max_rating", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--restart_chunk", type=int, default=1)
    parser.add_argument("--hidden_noise_std", type=float, default=0.0)
    parser.add_argument("--logit_noise_std", type=float, default=0.0)
    parser.add_argument("--portfolio", default="")
    parser.add_argument("--prefixes", default="1,8,16,32,64,128,256")
    parser.add_argument("--stop_on_oracle_exact", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--summary_csv", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    set_torch_threads()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    npz = np.load(args.cache_path, allow_pickle=False)
    indices = sample_indices(npz, args)
    dataset = SudokuIndexDataset(npz, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device(args.device)
    model, model_cfg = load_rrn(args.checkpoint, device)

    print(f"[rrn-ceiling] checkpoint={args.checkpoint}", flush=True)
    print(f"[rrn-ceiling] cache={args.cache_path}", flush=True)
    print(
        f"[rrn-ceiling] split={args.split} n={len(dataset)} batch={args.batch_size} "
        f"bucket_names={args.bucket_names or 'all'} device={device}",
        flush=True,
    )
    print(
        f"[rrn-ceiling] cfg D={model_cfg.D} train_T={model_cfg.train_T} eval_T={model_cfg.eval_T} "
        f"msg_hidden={model_cfg.msg_hidden} force_clues={model_cfg.force_clues}",
        flush=True,
    )
    evaluate(model, model_cfg, loader, device, args)


if __name__ == "__main__":
    main()
