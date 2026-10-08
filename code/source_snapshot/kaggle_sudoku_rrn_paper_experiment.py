import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kaggle_sudoku_experiment import KaggleSudokuDataset, RATING_BUCKETS, rating_bucket_name
from sudoku_exchange_experiment import is_valid_sudoku, set_seed, set_torch_threads


def build_rrn_edges():
    src = []
    dst = []
    edge_type = []
    for row in range(9):
        for col in range(9):
            idx = row * 9 + col
            for other_col in range(9):
                if other_col == col:
                    continue
                src.append(row * 9 + other_col)
                dst.append(idx)
                edge_type.append(0)
            for other_row in range(9):
                if other_row == row:
                    continue
                src.append(other_row * 9 + col)
                dst.append(idx)
                edge_type.append(1)
            box_row = (row // 3) * 3
            box_col = (col // 3) * 3
            for rr in range(box_row, box_row + 3):
                for cc in range(box_col, box_col + 3):
                    if rr == row and cc == col:
                        continue
                    src.append(rr * 9 + cc)
                    dst.append(idx)
                    edge_type.append(2)
    return (
        torch.tensor(src, dtype=torch.long),
        torch.tensor(dst, dtype=torch.long),
        torch.tensor(edge_type, dtype=torch.long),
    )


@dataclass
class RRNPaperCfg:
    D: int = 128
    train_T: int = 32
    eval_T: int = 64
    msg_hidden: int = 256
    dropout: float = 0.0
    force_clues: bool = True


class SudokuRRNPaper(nn.Module):
    """RRN-style Sudoku model with every-step input injection and stepwise logits."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.value_emb = nn.Embedding(10, cfg.D)
        self.row_emb = nn.Embedding(9, cfg.D)
        self.col_emb = nn.Embedding(9, cfg.D)
        self.box_emb = nn.Embedding(9, cfg.D)
        self.clue_emb = nn.Embedding(2, cfg.D)
        self.edge_emb = nn.Embedding(3, cfg.D)
        self.in_norm = nn.LayerNorm(cfg.D)
        self.msg_mlp = nn.Sequential(
            nn.Linear(3 * cfg.D, cfg.msg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.msg_hidden, cfg.D),
        )
        self.msg_norm = nn.LayerNorm(cfg.D)
        self.gru = nn.GRUCell(2 * cfg.D, cfg.D)
        self.out_norm = nn.LayerNorm(cfg.D)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.D, cfg.D),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.D, 9),
        )

        row_ids = torch.arange(9).view(9, 1).expand(9, 9).reshape(81)
        col_ids = torch.arange(9).view(1, 9).expand(9, 9).reshape(81)
        box_ids = ((row_ids // 3) * 3 + (col_ids // 3)).reshape(81)
        edge_src, edge_dst, edge_type = build_rrn_edges()
        self.register_buffer("row_ids", row_ids, persistent=False)
        self.register_buffer("col_ids", col_ids, persistent=False)
        self.register_buffer("box_ids", box_ids, persistent=False)
        self.register_buffer("edge_src", edge_src, persistent=False)
        self.register_buffer("edge_dst", edge_dst, persistent=False)
        self.register_buffer("edge_type", edge_type, persistent=False)

    def input_features(self, puzzle_b99):
        batch = puzzle_b99.shape[0]
        flat = puzzle_b99.reshape(batch, 81).clamp(0, 9)
        clue = (flat > 0).long()
        x = self.value_emb(flat)
        x = x + self.clue_emb(clue)
        x = x + self.row_emb(self.row_ids).unsqueeze(0)
        x = x + self.col_emb(self.col_ids).unsqueeze(0)
        x = x + self.box_emb(self.box_ids).unsqueeze(0)
        return self.in_norm(x)

    def force_clue_logits(self, logits, puzzle_b99):
        if not self.cfg.force_clues:
            return logits
        batch = puzzle_b99.shape[0]
        logits_shape = logits.shape
        logits = logits.reshape(batch, 81, 9)
        flat = puzzle_b99.reshape(batch, 81)
        clue_mask = flat > 0
        clue_idx = (flat - 1).clamp(0, 8)
        clue_onehot = F.one_hot(clue_idx, num_classes=9).float()
        forced = clue_onehot * 30.0 + (1.0 - clue_onehot) * -30.0
        logits = torch.where(clue_mask.unsqueeze(-1), forced, logits)
        return logits.reshape(logits_shape)

    def step(self, h, x0, edge_e):
        batch = h.shape[0]
        h_src = h.index_select(1, self.edge_src)
        h_dst = h.index_select(1, self.edge_dst)
        msg = self.msg_mlp(torch.cat([h_src, h_dst, edge_e], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(1, self.edge_dst, msg)
        agg = self.msg_norm(agg)
        update_input = torch.cat([agg, x0], dim=-1)
        h = self.gru(update_input.reshape(batch * 81, -1), h.reshape(batch * 81, -1))
        return h.reshape(batch, 81, -1)

    def logits_from_state(self, h, puzzle_b99):
        batch = h.shape[0]
        logits = self.classifier(self.out_norm(h)).reshape(batch, 9, 9, 9)
        return self.force_clue_logits(logits, puzzle_b99)

    def forward(self, puzzle_b99, steps=None, return_all=False):
        steps = self.cfg.train_T if steps is None else int(steps)
        batch = puzzle_b99.shape[0]
        x0 = self.input_features(puzzle_b99)
        h = x0
        edge_e = self.edge_emb(self.edge_type).unsqueeze(0).expand(batch, -1, -1)
        logits_steps = []

        for _ in range(steps):
            h = self.step(h, x0, edge_e)
            logits = self.logits_from_state(h, puzzle_b99)
            if return_all:
                logits_steps.append(logits)

        if return_all:
            logits = logits_steps[-1]
            probs = F.softmax(logits, dim=-1)
            return logits, probs, logits_steps

        logits = self.logits_from_state(h, puzzle_b99)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


def empty_stats():
    return {"n": 0, "exact": 0, "valid": 0, "clue_ok": 0, "cell": 0, "cells": 0}


@torch.no_grad()
def evaluate_rrn(model, loader, device, steps):
    model.eval()
    stats = {"all": empty_stats()}
    for name, _, _ in RATING_BUCKETS:
        stats[name] = empty_stats()
    stats["rating_nan"] = empty_stats()

    for batch in loader:
        puzzle = batch["puzzle"].to(device)
        solution = batch["solution"].to(device)
        ratings = batch["rating"].numpy()
        _, probs = model(puzzle, steps=steps)
        pred = probs.argmax(dim=-1) + 1

        pred_np = pred.cpu().numpy()
        solution_np = solution.cpu().numpy()
        puzzle_np = puzzle.cpu().numpy()
        for i in range(pred_np.shape[0]):
            names = ["all", rating_bucket_name(float(ratings[i]))]
            for name in names:
                item = stats.setdefault(name, empty_stats())
                item["n"] += 1
                item["exact"] += int(np.array_equal(pred_np[i], solution_np[i]))
                item["valid"] += int(is_valid_sudoku(pred_np[i]))
                clue_mask = puzzle_np[i] > 0
                item["clue_ok"] += int(np.array_equal(pred_np[i][clue_mask], puzzle_np[i][clue_mask]))
                item["cell"] += int((pred_np[i] == solution_np[i]).sum())
                item["cells"] += 81

    rows = []
    ordered = ["all"] + [name for name, _, _ in RATING_BUCKETS] + ["rating_nan"]
    for name in ordered:
        item = stats.get(name)
        if not item or item["n"] == 0:
            continue
        rows.append(
            (
                name,
                item["n"],
                item["exact"] / item["n"],
                item["valid"] / item["n"],
                item["clue_ok"] / item["n"],
                item["cell"] / item["cells"],
            )
        )
    return rows


def print_eval(rows, prefix):
    print(prefix)
    print("bucket          n      exact   valid   clue_ok cell_acc")
    for bucket, n, exact, valid, clue_ok, cell_acc in rows:
        print(f"{bucket:<13} {n:6d}  {exact:7.4f} {valid:7.4f} {clue_ok:7.4f} {cell_acc:8.4f}")


def supervised_loss(logits, puzzle, solution, empty_weight):
    target = (solution - 1).clamp(0, 8)
    ce = F.cross_entropy(logits.reshape(-1, 9), target.reshape(-1), reduction="none").view(-1, 9, 9)
    weights = 1.0 + (empty_weight - 1.0) * (puzzle == 0).float()
    return (ce * weights).mean()


def stepwise_loss(logits_steps, puzzle, solution, empty_weight, loss_steps, loss_tail):
    if loss_steps == "all":
        selected = logits_steps
    elif loss_steps == "tail":
        selected = logits_steps[-loss_tail:]
    elif loss_steps == "final":
        selected = [logits_steps[-1]]
    else:
        raise ValueError(f"Unknown loss_steps={loss_steps!r}")
    return torch.stack([supervised_loss(item, puzzle, solution, empty_weight) for item in selected]).mean()


def train(args):
    set_torch_threads()
    set_seed(args.seed)

    npz = np.load(args.cache_path, allow_pickle=False)
    train_ds = KaggleSudokuDataset(npz, split=0, limit=args.train_limit)
    val_ds = KaggleSudokuDataset(npz, split=1, limit=args.val_limit)
    test_ds = KaggleSudokuDataset(npz, split=2, limit=args.test_limit)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} cache={args.cache_path}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)

    device = torch.device(args.device)
    cfg = RRNPaperCfg(
        D=args.D,
        train_T=args.train_T,
        eval_T=args.eval_T,
        msg_hidden=args.msg_hidden,
        dropout=args.dropout,
        force_clues=not args.no_force_clues,
    )
    model = SudokuRRNPaper(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"[model] RRN-paper D={cfg.D} train_T={cfg.train_T} eval_T={cfg.eval_T} "
        f"msg_hidden={cfg.msg_hidden} loss_steps={args.loss_steps} force_clues={cfg.force_clues} device={device}"
    )
    if not args.skip_initial_eval:
        print_eval(evaluate_rrn(model, val_loader, device, steps=args.eval_T), f"[val] step=0 eval_T={args.eval_T}")

    best_exact = -1.0
    train_iter = iter(train_loader)
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
        _, _, logits_steps = model(puzzle, steps=args.train_T, return_all=True)
        loss = stepwise_loss(logits_steps, puzzle, solution, args.empty_weight, args.loss_steps, args.loss_tail)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - start
            print(f"step {step:5d}/{args.steps} loss={loss.item():.4f} elapsed={elapsed:.1f}s")

        if step % args.eval_every == 0 or step == args.steps:
            rows_train_t = evaluate_rrn(model, val_loader, device, steps=args.train_T)
            print_eval(rows_train_t, f"[val] step={step} eval_T={args.train_T}")
            rows_eval_t = evaluate_rrn(model, val_loader, device, steps=args.eval_T)
            print_eval(rows_eval_t, f"[val] step={step} eval_T={args.eval_T}")
            val_exact = rows_eval_t[0][2]
            if val_exact > best_exact:
                best_exact = val_exact
                Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "model_type": "SudokuRRNPaper",
                        "args": vars(args),
                    },
                    args.save_path,
                )
                print(f"[save] {args.save_path} best_val_exact_eval_T{args.eval_T}={best_exact:.4f}")

    print_eval(evaluate_rrn(model, test_loader, device, steps=args.train_T), f"[test] final eval_T={args.train_T}")
    print_eval(evaluate_rrn(model, test_loader, device, steps=args.eval_T), f"[test] final eval_T={args.eval_T}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default="data/cache_full_3m.npz")
    parser.add_argument("--save_path", default="checkpoints/model_rrn_paper_d128_t32_eval64_50k.pt")
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--train_T", type=int, default=32)
    parser.add_argument("--eval_T", type=int, default=64)
    parser.add_argument("--msg_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no_force_clues", action="store_true")
    parser.add_argument("--loss_steps", choices=["final", "tail", "all"], default="all")
    parser.add_argument("--loss_tail", type=int, default=8)
    parser.add_argument("--empty_weight", type=float, default=2.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--skip_initial_eval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
