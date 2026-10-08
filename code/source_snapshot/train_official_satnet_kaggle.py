"""Train official SATNet on the shared Kaggle Sudoku 3M split.

This adapter uses the official `locuslab/SATNet` layer and keeps the
Sudoku representation used by the original experiment:

  - 729 Boolean/probability variables for 81 cells x 9 digits.
  - clue cells are one-hot inputs and their full 9-variable cell is masked
    as known, matching SATNet's `exps/sudoku.py`.
  - labels are the full one-hot solution and the loss is BCE over 729 vars.

The data split is the existing project cache split:
  split 0 train, split 1 val, split 2 test.
"""

import argparse
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REMOTE_HELPER_DIR = Path("code/source_snapshot")
if REMOTE_HELPER_DIR.exists() and str(REMOTE_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(REMOTE_HELPER_DIR))

from kaggle_sudoku_experiment import KaggleSudokuDataset, RATING_BUCKETS, rating_bucket_name
from sudoku_exchange_experiment import is_valid_sudoku, set_seed, set_torch_threads


def clue_ok(pred, puzzle):
    mask = puzzle > 0
    return bool(np.array_equal(pred[mask], puzzle[mask]))


def hard_violations(grid):
    target = set(range(1, 10))
    viol = 0
    for i in range(9):
        viol += int(set(grid[i, :].tolist()) != target)
        viol += int(set(grid[:, i].tolist()) != target)
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            viol += int(set(grid[br : br + 3, bc : bc + 3].reshape(-1).tolist()) != target)
    return viol


def empty_stats():
    return {"n": 0, "exact": 0, "valid": 0, "clue_ok": 0, "cell": 0, "cells": 0, "viol": 0.0}


def one_hot_board(board):
    """Return flattened one-hot B x 729 for a B x 9 x 9 board with 0 allowed."""
    idx = (board - 1).clamp(0, 8)
    one_hot = F.one_hot(idx, num_classes=9).float()
    one_hot = one_hot * (board > 0).unsqueeze(-1).float()
    return one_hot.reshape(board.shape[0], 729).contiguous()


def satnet_inputs(puzzle, solution):
    z = one_hot_board(puzzle)
    label = one_hot_board(solution)
    clue = (puzzle > 0).unsqueeze(-1).expand(-1, -1, -1, 9)
    is_input = clue.reshape(puzzle.shape[0], 729).to(torch.int32).contiguous()
    return z, is_input, label


def predictions_from_probs(probs):
    return probs.reshape(probs.shape[0], 9, 9, 9).argmax(dim=-1) + 1


@dataclass
class SATNetCfg:
    board_sz: int = 3
    aux: int = 300
    m: int = 600
    max_iter: int = 40
    eps: float = 1e-4
    prox_lam: float = 1e-2


class OfficialSATNetSudoku(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import satnet

        n_vars = cfg.board_sz ** 6
        self.cfg = cfg
        self.sat = satnet.SATNet(
            n_vars,
            cfg.m,
            cfg.aux,
            max_iter=cfg.max_iter,
            eps=cfg.eps,
            prox_lam=cfg.prox_lam,
        )

    def forward(self, puzzle, solution=None):
        if solution is None:
            solution = puzzle
        z, is_input, _ = satnet_inputs(puzzle, solution)
        return self.sat(z, is_input)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0, progress_every=0):
    model.eval()
    stats = {"all": empty_stats()}
    for name, _, _ in RATING_BUCKETS:
        stats[name] = empty_stats()
    stats["rating_nan"] = empty_stats()

    seen = 0
    start = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
        puzzle = batch["puzzle"].to(device)
        solution = batch["solution"].to(device)
        ratings = batch["rating"].numpy()
        probs = model(puzzle, solution)
        pred = predictions_from_probs(probs)

        pred_np = pred.cpu().numpy()
        puzzle_np = puzzle.cpu().numpy()
        solution_np = solution.cpu().numpy()

        for i in range(pred_np.shape[0]):
            bucket = rating_bucket_name(float(ratings[i]))
            for name in ("all", bucket):
                item = stats.setdefault(name, empty_stats())
                item["n"] += 1
                item["exact"] += int(np.array_equal(pred_np[i], solution_np[i]))
                item["valid"] += int(is_valid_sudoku(pred_np[i]))
                item["clue_ok"] += int(clue_ok(pred_np[i], puzzle_np[i]))
                item["cell"] += int((pred_np[i] == solution_np[i]).sum())
                item["cells"] += 81
                item["viol"] += float(hard_violations(pred_np[i]))
            seen += 1

        if progress_every and seen % progress_every < pred_np.shape[0]:
            elapsed = time.time() - start
            print(f"[eval-progress] seen={seen} elapsed={elapsed:.1f}s rate={seen / max(elapsed, 1e-6):.2f}/s", flush=True)

        if max_batches and batch_idx >= max_batches:
            break

    return rows_from_stats(stats)


def rows_from_stats(stats):
    rows = []
    ordered = ["all"] + [name for name, _, _ in RATING_BUCKETS] + ["rating_nan"]
    for name in ordered:
        item = stats.get(name)
        if not item or item["n"] == 0:
            continue
        n = item["n"]
        rows.append(
            (
                name,
                n,
                item["exact"] / n,
                item["valid"] / n,
                item["clue_ok"] / n,
                item["cell"] / item["cells"],
                item["viol"] / n,
            )
        )
    return rows


def print_rows(rows, prefix):
    print(prefix, flush=True)
    print("bucket          n      exact   valid   clue_ok cell_acc avg_viol", flush=True)
    for bucket, n, exact, valid, clue, cell, viol in rows:
        print(f"{bucket:<13} {n:6d}  {exact:7.4f} {valid:7.4f} {clue:7.4f} {cell:8.4f} {viol:8.3f}", flush=True)


def save_checkpoint(path, model, cfg, args, step, best_val_exact):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg,
            "cfg_dict": asdict(cfg),
            "model_type": "OfficialSATNetSudoku",
            "args": vars(args),
            "step": step,
            "best_val_exact": best_val_exact,
        },
        path,
    )


def train(args):
    set_torch_threads()
    set_seed(args.seed)

    satnet_root = Path(args.satnet_root)
    if str(satnet_root) not in sys.path:
        sys.path.insert(0, str(satnet_root))

    npz = np.load(args.cache_path, allow_pickle=False)
    train_ds = KaggleSudokuDataset(npz, split=0, limit=args.train_limit)
    val_ds = KaggleSudokuDataset(npz, split=1, limit=args.val_limit)
    test_ds = KaggleSudokuDataset(npz, split=2, limit=args.test_limit)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} cache={args.cache_path}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)

    device = torch.device(args.device)
    cfg = SATNetCfg(aux=args.aux, m=args.m, max_iter=args.max_iter, eps=args.eps, prox_lam=args.prox_lam)
    model = OfficialSATNetSudoku(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[resume] {args.resume}", flush=True)

    print(
        f"[model] OfficialSATNet n=729 aux={cfg.aux} m={cfg.m} max_iter={cfg.max_iter} "
        f"lr={args.lr} device={device}",
        flush=True,
    )

    if not args.skip_initial_eval:
        print_rows(evaluate(model, val_loader, device, progress_every=args.eval_progress_every), "[val] step=0")

    train_iter = iter(train_loader)
    best_val_exact = -math.inf
    start = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        puzzle = batch["puzzle"].to(device)
        solution = batch["solution"].to(device)
        z, is_input, label = satnet_inputs(puzzle, solution)
        pred = model.sat(z, is_input)
        loss = F.binary_cross_entropy(pred, label)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - start
            print(f"step {step:6d}/{args.steps} loss={loss.item():.6f} elapsed={elapsed:.1f}s", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            val_rows = evaluate(model, val_loader, device, progress_every=args.eval_progress_every)
            print_rows(val_rows, f"[val] step={step}")
            val_exact = val_rows[0][2]
            if val_exact > best_val_exact:
                best_val_exact = val_exact
                save_checkpoint(args.save_path, model, cfg, args, step, best_val_exact)
                print(f"[save] {args.save_path} best_val_exact={best_val_exact:.4f}", flush=True)

    if args.save_final_path:
        save_checkpoint(args.save_final_path, model, cfg, args, args.steps, best_val_exact)
        print(f"[save-final] {args.save_final_path}", flush=True)

    print_rows(evaluate(model, test_loader, device, progress_every=args.eval_progress_every), "[test] final")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default="data/cache_full_3m.npz")
    parser.add_argument("--satnet_root", default="code/source_snapshot/external/SATNet-master")
    parser.add_argument("--save_path", default="checkpoints/model_official_satnet_full.pt")
    parser.add_argument("--save_final_path", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--eval_batch_size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--aux", type=int, default=300)
    parser.add_argument("--m", type=int, default=600)
    parser.add_argument("--max_iter", type=int, default=40)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--prox_lam", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--eval_progress_every", type=int, default=0)
    parser.add_argument("--skip_initial_eval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
