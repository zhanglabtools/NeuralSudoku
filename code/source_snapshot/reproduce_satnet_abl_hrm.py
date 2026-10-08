"""Reproduce Sudoku neural / neuro-symbolic baselines on the Kaggle 3M split.

This script intentionally reuses the existing Kaggle cache split:
  split 0: train, split 1: val, split 2: test.

Methods:
  hrm         - local HRM-style two-timescale recurrent Transformer.
  abl_refl    - local ABL-Refl-style GNN with reflection head and one repair call.
  satnet_lite - differentiable SAT/Sudoku constraint relaxation fallback.

The official SATNet/HRM repositories were not assumed to be available at runtime.
Logs should label satnet_lite as a fallback rather than the original CUDA SATNet.
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
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


def build_peer_edges():
    src, dst, typ = [], [], []
    for row in range(9):
        for col in range(9):
            idx = row * 9 + col
            for other_col in range(9):
                if other_col != col:
                    src.append(row * 9 + other_col)
                    dst.append(idx)
                    typ.append(0)
            for other_row in range(9):
                if other_row != row:
                    src.append(other_row * 9 + col)
                    dst.append(idx)
                    typ.append(1)
            br, bc = (row // 3) * 3, (col // 3) * 3
            for rr in range(br, br + 3):
                for cc in range(bc, bc + 3):
                    if rr != row or cc != col:
                        src.append(rr * 9 + cc)
                        dst.append(idx)
                        typ.append(2)
    return (
        torch.tensor(src, dtype=torch.long),
        torch.tensor(dst, dtype=torch.long),
        torch.tensor(typ, dtype=torch.long),
    )


def row_col_box_ids():
    row_ids = torch.arange(9).view(9, 1).expand(9, 9).reshape(81)
    col_ids = torch.arange(9).view(1, 9).expand(9, 9).reshape(81)
    box_ids = ((row_ids // 3) * 3 + (col_ids // 3)).reshape(81)
    return row_ids, col_ids, box_ids


def force_clue_logits(logits, puzzle):
    batch = puzzle.shape[0]
    shape = logits.shape
    logits = logits.reshape(batch, 81, 9)
    flat = puzzle.reshape(batch, 81)
    clue_mask = flat > 0
    clue_idx = (flat - 1).clamp(0, 8)
    clue_onehot = F.one_hot(clue_idx, num_classes=9).float()
    forced = clue_onehot * 30.0 + (1.0 - clue_onehot) * -30.0
    logits = torch.where(clue_mask.unsqueeze(-1), forced, logits)
    return logits.reshape(shape)


class SudokuInputEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.value_emb = nn.Embedding(10, d_model)
        self.row_emb = nn.Embedding(9, d_model)
        self.col_emb = nn.Embedding(9, d_model)
        self.box_emb = nn.Embedding(9, d_model)
        self.clue_emb = nn.Embedding(2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        row_ids, col_ids, box_ids = row_col_box_ids()
        self.register_buffer("row_ids", row_ids, persistent=False)
        self.register_buffer("col_ids", col_ids, persistent=False)
        self.register_buffer("box_ids", box_ids, persistent=False)

    def forward(self, puzzle):
        batch = puzzle.shape[0]
        flat = puzzle.reshape(batch, 81).clamp(0, 9)
        clue = (flat > 0).long()
        x = self.value_emb(flat)
        x = x + self.row_emb(self.row_ids).unsqueeze(0)
        x = x + self.col_emb(self.col_ids).unsqueeze(0)
        x = x + self.box_emb(self.box_ids).unsqueeze(0)
        x = x + self.clue_emb(clue)
        return self.drop(self.norm(x))


class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model, heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class HRMCfg:
    d_model: int = 128
    heads: int = 8
    h_cycles: int = 4
    l_steps: int = 4
    dropout: float = 0.0
    force_clues: bool = True


class SudokuHRM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = SudokuInputEmbedding(cfg.d_model, cfg.dropout)
        self.h0 = nn.Parameter(torch.zeros(1, 81, cfg.d_model))
        self.l0 = nn.Parameter(torch.zeros(1, 81, cfg.d_model))
        self.low = SimpleTransformerBlock(cfg.d_model, cfg.heads, dropout=cfg.dropout)
        self.high = SimpleTransformerBlock(cfg.d_model, cfg.heads, dropout=cfg.dropout)
        self.low_merge = nn.Linear(3 * cfg.d_model, cfg.d_model)
        self.high_merge = nn.Linear(3 * cfg.d_model, cfg.d_model)
        self.out = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 9))

    def forward(self, puzzle):
        batch = puzzle.shape[0]
        x = self.embed(puzzle)
        h = self.h0.expand(batch, -1, -1) + x
        l = self.l0.expand(batch, -1, -1) + x
        for _ in range(self.cfg.h_cycles):
            for _ in range(self.cfg.l_steps):
                l = self.low(self.low_merge(torch.cat([l, h, x], dim=-1)))
            h = self.high(self.high_merge(torch.cat([h, l, x], dim=-1)))
        logits = self.out(h).reshape(batch, 9, 9, 9)
        if self.cfg.force_clues:
            logits = force_clue_logits(logits, puzzle)
        return logits, F.softmax(logits, dim=-1)


@dataclass
class GNNCfg:
    d_model: int = 128
    steps: int = 8
    msg_hidden: int = 256
    dropout: float = 0.0
    force_clues: bool = True


class ABLReflGNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = SudokuInputEmbedding(cfg.d_model, cfg.dropout)
        self.edge_emb = nn.Embedding(3, cfg.d_model)
        self.msg = nn.Sequential(
            nn.Linear(3 * cfg.d_model, cfg.msg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.msg_hidden, cfg.d_model),
        )
        self.msg_norm = nn.LayerNorm(cfg.d_model)
        self.gru = nn.GRUCell(2 * cfg.d_model, cfg.d_model)
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.digit_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 9),
        )
        self.refl_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )
        edge_src, edge_dst, edge_type = build_peer_edges()
        self.register_buffer("edge_src", edge_src, persistent=False)
        self.register_buffer("edge_dst", edge_dst, persistent=False)
        self.register_buffer("edge_type", edge_type, persistent=False)

    def step(self, h, x0, edge_e):
        batch = h.shape[0]
        h_src = h.index_select(1, self.edge_src)
        h_dst = h.index_select(1, self.edge_dst)
        msg = self.msg(torch.cat([h_src, h_dst, edge_e], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(1, self.edge_dst, msg)
        agg = self.msg_norm(agg)
        h = self.gru(torch.cat([agg, x0], dim=-1).reshape(batch * 81, -1), h.reshape(batch * 81, -1))
        return h.reshape(batch, 81, -1)

    def forward(self, puzzle):
        batch = puzzle.shape[0]
        x0 = self.embed(puzzle)
        h = x0
        edge_e = self.edge_emb(self.edge_type).unsqueeze(0).expand(batch, -1, -1)
        for _ in range(self.cfg.steps):
            h = self.step(h, x0, edge_e)
        h = self.out_norm(h)
        logits = self.digit_head(h).reshape(batch, 9, 9, 9)
        if self.cfg.force_clues:
            logits = force_clue_logits(logits, puzzle)
        refl_logits = self.refl_head(h).reshape(batch, 9, 9)
        return logits, F.softmax(logits, dim=-1), refl_logits


@dataclass
class SATLiteCfg:
    d_model: int = 128
    layers: int = 4
    proj_steps: int = 8
    dropout: float = 0.0
    force_clues: bool = True


class SATNetLite(nn.Module):
    """Differentiable Sudoku constraint relaxation fallback for unavailable SATNet CUDA."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = SudokuInputEmbedding(cfg.d_model, cfg.dropout)
        blocks = []
        for _ in range(cfg.layers):
            blocks.extend(
                [
                    nn.Linear(cfg.d_model, cfg.d_model),
                    nn.GELU(),
                    nn.LayerNorm(cfg.d_model),
                    nn.Dropout(cfg.dropout),
                ]
            )
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(cfg.d_model, 9)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.25))

    @staticmethod
    def box_sums(p):
        b = p.shape[0]
        x = p.reshape(b, 3, 3, 3, 3, 9).sum(dim=(2, 4), keepdim=True)
        return x.expand(b, 3, 3, 3, 3, 9).reshape(b, 9, 9, 9)

    def relax(self, logits, puzzle):
        logits = force_clue_logits(logits, puzzle) if self.cfg.force_clues else logits
        alpha = F.softplus(self.alpha)
        beta = F.softplus(self.beta)
        for _ in range(self.cfg.proj_steps):
            p = F.softmax(logits, dim=-1)
            row = p.sum(dim=2, keepdim=True).expand_as(p) - p
            col = p.sum(dim=1, keepdim=True).expand_as(p) - p
            box = self.box_sums(p) - p
            conflict = row + col + box
            centered = conflict - conflict.mean(dim=-1, keepdim=True)
            logits = logits - alpha * centered
            logits = logits + beta * torch.log(p.clamp_min(1e-6))
            if self.cfg.force_clues:
                logits = force_clue_logits(logits, puzzle)
        return logits

    def forward(self, puzzle):
        batch = puzzle.shape[0]
        h = self.body(self.embed(puzzle))
        logits = self.head(h).reshape(batch, 9, 9, 9)
        logits = self.relax(logits, puzzle)
        return logits, F.softmax(logits, dim=-1)


def empty_stats():
    return {"n": 0, "exact": 0, "valid": 0, "clue_ok": 0, "cell": 0, "cells": 0, "viol": 0.0, "time": 0.0}


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


def clue_ok(pred, puzzle):
    mask = puzzle > 0
    return bool(np.array_equal(pred[mask], puzzle[mask]))


def add_metric(stats, bucket, pred, puzzle, solution, elapsed):
    for name in ("all", bucket):
        item = stats.setdefault(name, empty_stats())
        item["n"] += 1
        item["exact"] += int(np.array_equal(pred, solution))
        item["valid"] += int(is_valid_sudoku(pred))
        item["clue_ok"] += int(clue_ok(pred, puzzle))
        item["cell"] += int((pred == solution).sum())
        item["cells"] += 81
        item["viol"] += float(hard_violations(pred))
        item["time"] += float(elapsed)


def rows_from_stats(stats):
    rows = []
    ordered = ["all"] + [name for name, _, _ in RATING_BUCKETS] + ["rating_nan", "rating_other"]
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
                item["time"] / n,
            )
        )
    return rows


def print_rows(rows, prefix):
    print(prefix, flush=True)
    print("bucket          n      exact   valid   clue_ok cell_acc avg_viol sec/puz", flush=True)
    for bucket, n, exact, valid, clue, cell, viol, sec in rows:
        print(
            f"{bucket:<13} {n:6d}  {exact:7.4f} {valid:7.4f} {clue:7.4f} "
            f"{cell:8.4f} {viol:8.3f} {sec:7.4f}",
            flush=True,
        )


def candidates_from_grid(grid):
    used_row = [set(grid[r, grid[r] > 0].tolist()) for r in range(9)]
    used_col = [set(grid[grid[:, c] > 0, c].tolist()) for c in range(9)]
    used_box = []
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            vals = grid[br : br + 3, bc : bc + 3].reshape(-1)
            used_box.append(set(vals[vals > 0].tolist()))
    cand = {}
    digits = set(range(1, 10))
    for r in range(9):
        for c in range(9):
            if grid[r, c] == 0:
                b = (r // 3) * 3 + (c // 3)
                vals = digits - used_row[r] - used_col[c] - used_box[b]
                if not vals:
                    return None
                cand[(r, c)] = vals
    return cand


def solve_mrv(grid, max_nodes=20000):
    grid = grid.copy()
    nodes = 0

    def rec():
        nonlocal nodes
        cand = candidates_from_grid(grid)
        if cand is None:
            return False
        if not cand:
            return True
        (r, c), vals = min(cand.items(), key=lambda item: len(item[1]))
        for v in sorted(vals):
            nodes += 1
            if nodes > max_nodes:
                return False
            grid[r, c] = v
            if rec():
                return True
            grid[r, c] = 0
        return False

    ok = rec()
    return (grid if ok else None), nodes


def repair_with_reflection(pred, refl_prob, puzzle, threshold, max_nodes):
    givens = np.zeros((9, 9), dtype=np.uint8)
    clue_mask = puzzle > 0
    keep_mask = (refl_prob < threshold) & (~clue_mask)
    givens[clue_mask] = puzzle[clue_mask]
    for r in range(9):
        for c in range(9):
            if keep_mask[r, c]:
                v = int(pred[r, c])
                if v < 1 or v > 9:
                    continue
                row = givens[r, :]
                col = givens[:, c]
                box = givens[(r // 3) * 3 : (r // 3) * 3 + 3, (c // 3) * 3 : (c // 3) * 3 + 3]
                if v in row or v in col or v in box:
                    continue
                givens[r, c] = v
    sol, nodes = solve_mrv(givens, max_nodes=max_nodes)
    return sol, nodes, int(keep_mask.sum())


def supervised_loss(logits, puzzle, solution, empty_weight):
    target = (solution - 1).clamp(0, 8)
    ce = F.cross_entropy(logits.reshape(-1, 9), target.reshape(-1), reduction="none").view(-1, 9, 9)
    weights = 1.0 + (empty_weight - 1.0) * (puzzle == 0).float()
    return (ce * weights).mean()


def constraint_loss_from_probs(probs):
    row = (probs.sum(dim=2) - 1.0).pow(2).mean()
    col = (probs.sum(dim=1) - 1.0).pow(2).mean()
    box = probs.reshape(probs.shape[0], 3, 3, 3, 3, 9).sum(dim=(2, 4)).sub(1.0).pow(2).mean()
    return row + col + box


def make_model(args):
    if args.method == "hrm":
        cfg = HRMCfg(
            d_model=args.d_model,
            heads=args.heads,
            h_cycles=args.h_cycles,
            l_steps=args.l_steps,
            dropout=args.dropout,
            force_clues=not args.no_force_clues,
        )
        return SudokuHRM(cfg), cfg
    if args.method == "abl_refl":
        cfg = GNNCfg(
            d_model=args.d_model,
            steps=args.gnn_steps,
            msg_hidden=args.msg_hidden,
            dropout=args.dropout,
            force_clues=not args.no_force_clues,
        )
        return ABLReflGNN(cfg), cfg
    if args.method == "satnet_lite":
        cfg = SATLiteCfg(
            d_model=args.d_model,
            layers=args.sat_layers,
            proj_steps=args.sat_proj_steps,
            dropout=args.dropout,
            force_clues=not args.no_force_clues,
        )
        return SATNetLite(cfg), cfg
    raise ValueError(args.method)


def forward_model(model, method, puzzle):
    if method == "abl_refl":
        logits, probs, refl_logits = model(puzzle)
        return logits, probs, refl_logits
    logits, probs = model(puzzle)
    return logits, probs, None


@torch.no_grad()
def evaluate(model, method, loader, device, args, prefix):
    model.eval()
    raw_stats = {"all": empty_stats()}
    repair_stats = {"all": empty_stats()}
    repair_extra = {"nodes": 0, "kept": 0, "repaired": 0, "attempted": 0}

    for batch in loader:
        puzzle = batch["puzzle"].to(device)
        solution = batch["solution"].to(device)
        ratings = batch["rating"].numpy()
        start = time.time()
        logits, probs, refl_logits = forward_model(model, method, puzzle)
        pred = probs.argmax(dim=-1).cpu().numpy() + 1
        elapsed_batch = (time.time() - start) / max(1, pred.shape[0])

        puzzle_np = puzzle.cpu().numpy()
        solution_np = solution.cpu().numpy()
        refl_prob = torch.sigmoid(refl_logits).cpu().numpy() if refl_logits is not None else None
        for i in range(pred.shape[0]):
            bucket = rating_bucket_name(float(ratings[i]))
            add_metric(raw_stats, bucket, pred[i], puzzle_np[i], solution_np[i], elapsed_batch)

            if method == "abl_refl" and args.eval_repair:
                r0 = time.time()
                sol, nodes, kept = repair_with_reflection(
                    pred[i],
                    refl_prob[i],
                    puzzle_np[i],
                    args.refl_threshold,
                    args.repair_max_nodes,
                )
                repair_extra["attempted"] += 1
                repair_extra["nodes"] += nodes
                repair_extra["kept"] += kept
                if sol is None:
                    sol = pred[i]
                else:
                    repair_extra["repaired"] += int(not np.array_equal(sol, pred[i]))
                add_metric(repair_stats, bucket, sol, puzzle_np[i], solution_np[i], time.time() - r0 + elapsed_batch)

    raw_rows = rows_from_stats(raw_stats)
    print_rows(raw_rows, prefix + " raw")
    if method == "abl_refl" and args.eval_repair:
        repair_rows = rows_from_stats(repair_stats)
        print_rows(repair_rows, prefix + f" repair(th={args.refl_threshold})")
        n = max(1, repair_extra["attempted"])
        print(
            f"[repair-extra] attempted={repair_extra['attempted']} repaired={repair_extra['repaired']} "
            f"avg_nodes={repair_extra['nodes'] / n:.2f} avg_kept={repair_extra['kept'] / n:.2f}",
            flush=True,
        )
    return raw_rows


def train(args):
    set_torch_threads()
    set_seed(args.seed)
    device = torch.device(args.device)

    npz = np.load(args.cache_path, allow_pickle=False)
    train_ds = KaggleSudokuDataset(npz, split=0, limit=args.train_limit)
    val_ds = KaggleSudokuDataset(npz, split=1, limit=args.val_limit)
    test_ds = KaggleSudokuDataset(npz, split=2, limit=args.test_limit)
    print(
        f"[data] cache={args.cache_path} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"split_counts={np.bincount(npz['splits'], minlength=3).tolist()}",
        flush=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)

    model, cfg = make_model(args)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[model] method={args.method} cfg={cfg} device={device}", flush=True)
    if args.note:
        print(f"[note] {args.note}", flush=True)

    if args.eval_only:
        checkpoint_path = args.checkpoint or args.save_path
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[eval-only] loaded={checkpoint_path}", flush=True)
        evaluate(model, args.method, val_loader, device, args, "[val] eval_only")
        evaluate(model, args.method, test_loader, device, args, "[test] eval_only")
        return

    if not args.skip_initial_eval:
        evaluate(model, args.method, val_loader, device, args, "[val] step=0")

    best = -1.0
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
        logits, probs, refl_logits = forward_model(model, args.method, puzzle)
        loss = supervised_loss(logits, puzzle, solution, args.empty_weight)
        if args.constraint_weight:
            loss = loss + args.constraint_weight * constraint_loss_from_probs(probs)
        if args.method == "abl_refl":
            with torch.no_grad():
                pred = probs.argmax(dim=-1) + 1
                target_refl = ((pred != solution) & (puzzle == 0)).float()
            refl_loss = F.binary_cross_entropy_with_logits(refl_logits, target_refl)
            loss = loss + args.refl_weight * refl_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step == 1 or step % args.log_every == 0:
            print(f"step {step:6d}/{args.steps} loss={loss.item():.5f} elapsed={time.time() - start:.1f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            rows = evaluate(model, args.method, val_loader, device, args, f"[val] step={step}")
            val_exact = rows[0][2]
            if val_exact > best:
                best = val_exact
                Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "method": args.method,
                        "args": vars(args),
                    },
                    args.save_path,
                )
                print(f"[save] {args.save_path} best_val_exact={best:.4f}", flush=True)

    evaluate(model, args.method, test_loader, device, args, "[test] final")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["hrm", "abl_refl", "satnet_lite"], required=True)
    parser.add_argument("--cache_path", default="data/cache_full_3m.npz")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--empty_weight", type=float, default=2.0)
    parser.add_argument("--constraint_weight", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--skip_initial_eval", action="store_true")
    parser.add_argument("--note", default="")
    parser.add_argument("--no_force_clues", action="store_true")

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--h_cycles", type=int, default=4)
    parser.add_argument("--l_steps", type=int, default=4)

    parser.add_argument("--gnn_steps", type=int, default=8)
    parser.add_argument("--msg_hidden", type=int, default=256)
    parser.add_argument("--refl_weight", type=float, default=1.0)
    parser.add_argument("--eval_repair", action="store_true")
    parser.add_argument("--refl_threshold", type=float, default=0.5)
    parser.add_argument("--repair_max_nodes", type=int, default=20000)

    parser.add_argument("--sat_layers", type=int, default=4)
    parser.add_argument("--sat_proj_steps", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
