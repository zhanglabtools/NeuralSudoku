"""Evaluate a trained pure-neural iterative hypergraph reflector."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch

from eval_hybrid_hyper_rrn_restarts import load_hybrid
from kaggle_sudoku_experiment import KaggleSudokuDataset
from sudoku_exchange_experiment import set_seed, set_torch_threads
from train_iterative_hyper_reflection import (
    IterativeHyperReflector,
    IterativeReflectionCfg,
    evaluate_indices,
)


SPLITS = {"train": 0, "val": 1, "test": 2}


def load_reflector(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    backbone, _, _ = load_hybrid(checkpoint["base_checkpoint"], device)
    cfg = IterativeReflectionCfg(**checkpoint["reflection_cfg"])
    model = IterativeHyperReflector(backbone, cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, cfg, checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reflection_checkpoint", required=True)
    parser.add_argument(
        "--cache_path",
        default="data/cache_full_3m.npz",
    )
    parser.add_argument("--split", choices=sorted(SPLITS), default="val")
    parser.add_argument("--strict_min_rating", type=float, default=4.0)
    parser.add_argument("--all_ratings", type=int, default=0)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_csv", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    set_torch_threads()
    set_seed(args.seed)
    device = torch.device(args.device)
    model, cfg, checkpoint = load_reflector(
        args.reflection_checkpoint, device
    )
    npz = np.load(args.cache_path, allow_pickle=False)
    dataset = KaggleSudokuDataset(npz, split=SPLITS[args.split], limit=0)
    if args.all_ratings:
        indices = np.arange(len(dataset), dtype=np.int64)
    else:
        indices = np.flatnonzero(
            np.nan_to_num(dataset.ratings, nan=-np.inf)
            > args.strict_min_rating
        )
    if args.limit:
        indices = indices[: args.limit]

    started = time.time()
    metrics = evaluate_indices(
        model, dataset, indices, device, args.batch_size
    )
    elapsed = time.time() - started
    row = {
        "mode": cfg.mode,
        "slots": cfg.slots,
        "parent_steps": cfg.parent_steps,
        "cycles": cfg.cycles,
        "recovery_steps": cfg.recovery_steps,
        "inference_equivalent_steps": (
            cfg.parent_steps + cfg.slots * cfg.cycles * cfg.recovery_steps
        ),
        "split": args.split,
        "strict_min_rating": (
            "all" if args.all_ratings else args.strict_min_rating
        ),
        **metrics,
        "elapsed": elapsed,
        "puzzles_per_second": len(indices) / max(elapsed, 1e-9),
        "checkpoint_step": checkpoint.get("step", -1),
    }
    print(f"[result] {row}", flush=True)
    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        print(f"[save] {path}", flush=True)


if __name__ == "__main__":
    main()
