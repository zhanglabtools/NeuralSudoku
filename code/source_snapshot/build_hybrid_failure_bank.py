"""Build an on-policy Hybrid T-step failure bank from the official train split."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from eval_hybrid_hyper_rrn_restarts import load_hybrid
from kaggle_sudoku_experiment import KaggleSudokuDataset
from sudoku_exchange_experiment import set_seed, set_torch_threads


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/model_hybrid_hyper_rrn_d128_t32_eval64_50k_20260717_best.pt",
    )
    parser.add_argument(
        "--cache_path",
        default="data/cache_full_3m.npz",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--strict_min_rating", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=4096)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    set_torch_threads()
    set_seed(args.seed)
    device = torch.device(args.device)
    model, _, checkpoint = load_hybrid(args.checkpoint, device)
    npz = np.load(args.cache_path, allow_pickle=False)
    dataset = KaggleSudokuDataset(npz, split=0, limit=0)
    indices = np.flatnonzero(
        np.nan_to_num(dataset.ratings, nan=-np.inf) > args.strict_min_rating
    )
    if args.limit:
        indices = indices[: args.limit]

    failures = []
    seen = 0
    started = time.time()
    for offset in range(0, len(indices), args.batch_size):
        take = indices[offset : offset + args.batch_size]
        puzzle = torch.as_tensor(
            dataset.puzzles[take], dtype=torch.long, device=device
        )
        solution = torch.as_tensor(
            dataset.solutions[take], dtype=torch.long, device=device
        )
        logits, _ = model(puzzle, steps=args.steps)
        pred = logits.argmax(dim=-1) + 1
        exact = (pred == solution).reshape(len(take), -1).all(dim=-1)
        if (~exact).any():
            failures.extend(take[(~exact).cpu().numpy()].tolist())
        seen += len(take)
        if (
            args.progress_every
            and seen % args.progress_every < len(take)
        ):
            print(
                f"[progress] seen={seen}/{len(indices)} failures={len(failures)} "
                f"elapsed={time.time()-started:.1f}s",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        indices=np.asarray(failures, dtype=np.int64),
        scanned_indices=indices,
        split=np.asarray(0, dtype=np.int64),
        strict_min_rating=np.asarray(args.strict_min_rating),
        steps=np.asarray(args.steps, dtype=np.int64),
        checkpoint_step=np.asarray(checkpoint.get("step", -1), dtype=np.int64),
    )
    print(
        f"[save] {output} scanned={len(indices)} failures={len(failures)} "
        f"rate={len(failures)/max(len(indices),1):.6f} "
        f"elapsed={time.time()-started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
