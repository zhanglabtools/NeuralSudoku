"""Paper-faithful ABL-Refl Sudoku experiment on the Kaggle Sudoku 3M split.

The paper-level inference protocol is:
  1. one GNN forward pass produces an intuitive Sudoku solution;
  2. the same forward pass produces a reflection vector over 81 cells;
  3. one symbolic abduction call repairs reflected cells.

Reflection training follows the paper's equations: a REINFORCE surrogate uses
the improvement in discrete KB consistency, while a squared-hinge size loss
keeps at least C of the intuitive outputs. Reflection training never reads the
solution label. PySAT's Minisat22 performs the single symbolic repair call.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REMOTE_HELPER_DIR = Path("code/source_snapshot")
if REMOTE_HELPER_DIR.exists() and str(REMOTE_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(REMOTE_HELPER_DIR))

PYDEPS_PYSAT = Path("code/optional_dependencies/python_sat")
if PYDEPS_PYSAT.exists() and str(PYDEPS_PYSAT) not in sys.path:
    sys.path.insert(0, str(PYDEPS_PYSAT))

from kaggle_sudoku_experiment import KaggleSudokuDataset, RATING_BUCKETS, rating_bucket_name
from reproduce_satnet_abl_hrm import ABLReflGNN, GNNCfg, supervised_loss
from sudoku_exchange_experiment import is_valid_sudoku, set_seed, set_torch_threads


def varnum(row, col, digit):
    return row * 81 + col * 9 + digit + 1


def build_sudoku_cnf():
    clauses = []
    for r in range(9):
        for c in range(9):
            clauses.append([varnum(r, c, d) for d in range(9)])
            for d1 in range(9):
                for d2 in range(d1 + 1, 9):
                    clauses.append([-varnum(r, c, d1), -varnum(r, c, d2)])

    for r in range(9):
        for d in range(9):
            for c1 in range(9):
                for c2 in range(c1 + 1, 9):
                    clauses.append([-varnum(r, c1, d), -varnum(r, c2, d)])

    for c in range(9):
        for d in range(9):
            for r1 in range(9):
                for r2 in range(r1 + 1, 9):
                    clauses.append([-varnum(r1, c, d), -varnum(r2, c, d)])

    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            cells = [(br + dr, bc + dc) for dr in range(3) for dc in range(3)]
            for d in range(9):
                for i in range(9):
                    for j in range(i + 1, 9):
                        r1, c1 = cells[i]
                        r2, c2 = cells[j]
                        clauses.append([-varnum(r1, c1, d), -varnum(r2, c2, d)])
    return clauses


BASE_CNF = build_sudoku_cnf()


def grid_from_model(model):
    grid = np.zeros((9, 9), dtype=np.uint8)
    positives = {lit for lit in model if 1 <= lit <= 729}
    for r in range(9):
        for c in range(9):
            for d in range(9):
                if varnum(r, c, d) in positives:
                    grid[r, c] = d + 1
                    break
    return grid


def minisat_abduction_repair(pred, refl_prob, puzzle, threshold):
    from pysat.solvers import Minisat22

    clauses = list(BASE_CNF)
    clue_mask = puzzle > 0
    reflected = (refl_prob >= threshold) & (~clue_mask)
    kept = (~reflected) & (~clue_mask)

    for r in range(9):
        for c in range(9):
            if clue_mask[r, c]:
                clauses.append([varnum(r, c, int(puzzle[r, c]) - 1)])
            elif kept[r, c]:
                clauses.append([varnum(r, c, int(pred[r, c]) - 1)])

    start = time.time()
    with Minisat22(bootstrap_with=clauses) as solver:
        ok = solver.solve()
        elapsed = time.time() - start
        if not ok:
            return None, "unsat", int(reflected.sum()), int(kept.sum()), elapsed
        return grid_from_model(solver.get_model()), "solved", int(reflected.sum()), int(kept.sum()), elapsed


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
    return {
        "n": 0,
        "exact": 0,
        "valid": 0,
        "clue_ok": 0,
        "cell": 0,
        "cells": 0,
        "viol": 0.0,
        "solved": 0,
        "unsat": 0,
        "reflected": 0,
        "kept": 0,
        "repair_time": 0.0,
    }


def update_stats(stats, bucket, pred, puzzle, solution, repair_status=None, reflected=0, kept=0, repair_time=0.0):
    for name in ("all", bucket):
        item = stats.setdefault(name, empty_stats())
        item["n"] += 1
        item["exact"] += int(np.array_equal(pred, solution))
        item["valid"] += int(is_valid_sudoku(pred))
        item["clue_ok"] += int(clue_ok(pred, puzzle))
        item["cell"] += int((pred == solution).sum())
        item["cells"] += 81
        item["viol"] += float(hard_violations(pred))
        item["solved"] += int(repair_status == "solved")
        item["unsat"] += int(repair_status == "unsat")
        item["reflected"] += int(reflected)
        item["kept"] += int(kept)
        item["repair_time"] += float(repair_time)


def rows_from_stats(stats):
    rows = []
    ordered = ["all"] + [name for name, _, _ in RATING_BUCKETS] + ["rating_nan"]
    for name in ordered:
        item = stats.get(name)
        if not item or item["n"] == 0:
            continue
        n = item["n"]
        rows.append(
            {
                "bucket": name,
                "n": n,
                "exact": item["exact"] / n,
                "valid": item["valid"] / n,
                "clue_ok": item["clue_ok"] / n,
                "cell_acc": item["cell"] / item["cells"],
                "avg_viol": item["viol"] / n,
                "solved": item["solved"] / n,
                "unsat": item["unsat"] / n,
                "avg_reflected": item["reflected"] / n,
                "avg_kept": item["kept"] / n,
                "repair_time": item["repair_time"] / n,
            }
        )
    return rows


def print_rows(rows, prefix, repair=False):
    print(prefix, flush=True)
    if repair:
        print(
            "bucket          n      exact   valid   clue_ok cell_acc avg_viol solved  unsat avg_refl avg_kept repair_s",
            flush=True,
        )
        for r in rows:
            print(
                f"{r['bucket']:<13} {r['n']:6d}  {r['exact']:7.4f} {r['valid']:7.4f} "
                f"{r['clue_ok']:7.4f} {r['cell_acc']:8.4f} {r['avg_viol']:8.3f} "
                f"{r['solved']:6.4f} {r['unsat']:6.4f} {r['avg_reflected']:8.2f} "
                f"{r['avg_kept']:8.2f} {r['repair_time']:8.4f}",
                flush=True,
            )
    else:
        print("bucket          n      exact   valid   clue_ok cell_acc avg_viol", flush=True)
        for r in rows:
            print(
                f"{r['bucket']:<13} {r['n']:6d}  {r['exact']:7.4f} {r['valid']:7.4f} "
                f"{r['clue_ok']:7.4f} {r['cell_acc']:8.4f} {r['avg_viol']:8.3f}",
                flush=True,
            )


@torch.no_grad()
def evaluate(model, loader, device, args, prefix):
    model.eval()
    raw_stats = {"all": empty_stats()}
    repair_stats = {"all": empty_stats()}
    for name, _, _ in RATING_BUCKETS:
        raw_stats[name] = empty_stats()
        repair_stats[name] = empty_stats()
    raw_stats["rating_nan"] = empty_stats()
    repair_stats["rating_nan"] = empty_stats()

    seen = 0
    start = time.time()
    for batch in loader:
        puzzle = batch["puzzle"].to(device)
        solution = batch["solution"].to(device)
        ratings = batch["rating"].numpy()
        logits, probs, refl_logits = model(puzzle)
        pred = probs.argmax(dim=-1).cpu().numpy() + 1
        refl_prob = torch.sigmoid(refl_logits).cpu().numpy()
        puzzle_np = puzzle.cpu().numpy()
        solution_np = solution.cpu().numpy()

        for i in range(pred.shape[0]):
            bucket = rating_bucket_name(float(ratings[i]))
            update_stats(raw_stats, bucket, pred[i], puzzle_np[i], solution_np[i])
            if args.eval_repair:
                repaired, status, reflected, kept, repair_time = minisat_abduction_repair(
                    pred[i], refl_prob[i], puzzle_np[i], args.refl_threshold
                )
                final = repaired if repaired is not None else pred[i]
                update_stats(
                    repair_stats,
                    bucket,
                    final,
                    puzzle_np[i],
                    solution_np[i],
                    repair_status=status,
                    reflected=reflected,
                    kept=kept,
                    repair_time=repair_time,
                )
            seen += 1
            if args.progress_every and seen % args.progress_every == 0:
                elapsed = time.time() - start
                print(f"[progress] seen={seen} elapsed={elapsed:.1f}s rate={seen / max(elapsed, 1e-6):.2f}/s", flush=True)

    raw_rows = rows_from_stats(raw_stats)
    print_rows(raw_rows, prefix + " raw", repair=False)
    if args.eval_repair:
        repair_rows = rows_from_stats(repair_stats)
        print_rows(repair_rows, prefix + f" repair(th={args.refl_threshold})", repair=True)
    return raw_rows


def sudoku_consistency_score(grids):
    """Paper KB score: 1 per duplicate-free group and +10 if all 27 pass."""
    batch = grids.shape[0]
    rows = grids
    cols = grids.transpose(1, 2)
    boxes = (
        grids.reshape(batch, 3, 3, 3, 3)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, 9, 9)
    )
    groups = torch.cat((rows, cols, boxes), dim=1)
    left = groups.unsqueeze(-1)
    right = groups.unsqueeze(-2)
    nonzero = (left > 0) & (right > 0)
    eye = torch.eye(9, dtype=torch.bool, device=grids.device).view(1, 1, 9, 9)
    has_duplicate = ((left == right) & nonzero & ~eye).any(dim=(-1, -2))
    valid_groups = ~has_duplicate
    group_score = valid_groups.float().sum(dim=1)
    return group_score + 10.0 * valid_groups.all(dim=1).float()


def reflection_size_loss(refl_logits, puzzle, reflection_size_c):
    empty = puzzle == 0
    effective_logits = torch.where(empty, refl_logits, torch.full_like(refl_logits, -20.0))
    retained_fraction = 1.0 - torch.sigmoid(effective_logits).mean(dim=(1, 2))
    loss = torch.relu(reflection_size_c - retained_fraction).pow(2).mean()
    return loss, effective_logits, retained_fraction


def paper_reflection_losses(refl_logits, probs, puzzle, reflection_size_c, baseline="none"):
    """REINFORCE Lcon and the exact retained-fraction Lsize from the paper."""
    empty = puzzle == 0

    digit_dist = torch.distributions.Categorical(probs=probs)
    sampled_digits = digit_dist.sample() + 1
    intuitive = torch.where(empty, sampled_digits, puzzle)

    # Given cells are always retained. Their fixed negative logits carry no gradient.
    size_loss, effective_refl_logits, retained_fraction = reflection_size_loss(
        refl_logits, puzzle, reflection_size_c
    )
    refl_dist = torch.distributions.Bernoulli(logits=effective_refl_logits)
    reflection = refl_dist.sample()
    reflected_output = intuitive.masked_fill(reflection.bool(), 0)

    with torch.no_grad():
        score_before = sudoku_consistency_score(intuitive)
        score_after = sudoku_consistency_score(reflected_output)
        raw_delta = score_after - score_before
        # Con is a consistency measurement, so normalize the paper's 37-point
        # Sudoku score before combining it with alpha=beta=1 losses.
        delta = raw_delta / 37.0
        if baseline == "batch":
            advantage = delta - delta.mean()
        else:
            advantage = delta

    digit_log_prob = digit_dist.log_prob(sampled_digits - 1)
    digit_log_prob = (digit_log_prob * empty.float()).sum(dim=(1, 2))
    refl_log_prob = refl_dist.log_prob(reflection).sum(dim=(1, 2))
    joint_log_prob = (digit_log_prob + refl_log_prob) / 81.0
    consistency_loss = -(advantage * joint_log_prob).mean()

    stats = {
        "score_before": score_before.mean(),
        "score_after": score_after.mean(),
        "delta": raw_delta.mean(),
        "normalized_delta": delta.mean(),
        "reflected": reflection.sum(dim=(1, 2)).float().mean(),
        "retained_fraction": retained_fraction.mean().detach(),
    }
    return consistency_loss, size_loss, stats


def build_model(args):
    cfg = GNNCfg(
        d_model=args.d_model,
        steps=args.gnn_steps,
        msg_hidden=args.msg_hidden,
        dropout=args.dropout,
        force_clues=args.force_clues,
    )
    return ABLReflGNN(cfg), cfg


def train(args):
    set_torch_threads()
    set_seed(args.seed)
    device = torch.device(args.device)

    npz = np.load(args.cache_path, allow_pickle=False)
    train_ds = KaggleSudokuDataset(npz, split=0, limit=args.train_limit)
    val_ds = KaggleSudokuDataset(npz, split=1, limit=args.val_limit)
    test_ds = KaggleSudokuDataset(npz, split=2, limit=args.test_limit)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} cache={args.cache_path}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)

    model, cfg = build_model(args)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[load] {args.checkpoint}", flush=True)

    print(
        f"[model] ABL-Refl GNN d={args.d_model} steps={args.gnn_steps} msg_hidden={args.msg_hidden} "
        f"C={args.reflection_size_c} th={args.refl_threshold} device={device}",
        flush=True,
    )

    if args.eval_only:
        eval_loader = val_loader if args.eval_split == "val" else test_loader
        evaluate(model, eval_loader, device, args, f"[{args.eval_split}] checkpoint")
        return

    best_val_exact = -1.0
    milestone_steps = {int(x) for x in args.milestone_steps.split(",") if x.strip()}
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
        logits, probs, refl_logits = model(puzzle)
        labeled_loss = supervised_loss(logits, puzzle, solution, args.empty_weight)
        consistency_loss, size_loss, refl_stats = paper_reflection_losses(
            refl_logits,
            probs,
            puzzle,
            args.reflection_size_c,
            baseline=args.reinforce_baseline,
        )
        loss = labeled_loss + args.alpha * consistency_loss + args.beta * size_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - start
            print(
                f"step {step:6d}/{args.steps} loss={loss.item():.5f} labeled={labeled_loss.item():.5f} "
                f"lcon={consistency_loss.item():.5f} lsize={size_loss.item():.5f} "
                f"kb={float(refl_stats['score_before']):.2f}->{float(refl_stats['score_after']):.2f} "
                f"delta={float(refl_stats['delta']):.2f} reflected={float(refl_stats['reflected']):.2f} "
                f"retained={float(refl_stats['retained_fraction']):.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps or step in milestone_steps:
            old_eval_repair = args.eval_repair
            args.eval_repair = False
            rows = evaluate(model, val_loader, device, args, f"[val] step={step}")
            args.eval_repair = old_eval_repair
            val_exact = rows[0]["exact"]
            if val_exact > best_val_exact:
                best_val_exact = val_exact
                Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "model_type": "ABLReflGNN",
                        "args": vars(args),
                        "step": step,
                        "best_val_exact": best_val_exact,
                    },
                    args.save_path,
                )
                print(f"[save] {args.save_path} best_val_exact={best_val_exact:.4f}", flush=True)
            if step in milestone_steps:
                milestone_path = Path(args.save_path).with_name(
                    f"{Path(args.save_path).stem}_step{step}{Path(args.save_path).suffix}"
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "model_type": "ABLReflGNN_PaperV2",
                        "args": vars(args),
                        "step": step,
                        "best_val_exact": best_val_exact,
                    },
                    milestone_path,
                )
                print(f"[save-milestone] {milestone_path}", flush=True)

    if args.save_final_path:
        torch.save(
            {
                "model": model.state_dict(),
                "cfg": cfg,
                "model_type": "ABLReflGNN",
                "args": vars(args),
                "step": args.steps,
                "best_val_exact": best_val_exact,
            },
            args.save_final_path,
        )
        print(f"[save-final] {args.save_final_path}", flush=True)

    evaluate(model, test_loader, device, args, "[test] final")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default="data/cache_full_3m.npz")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--save_final_path", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_split", choices=["val", "test"], default="test")
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--gnn_steps", type=int, default=8)
    parser.add_argument("--msg_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--force_clues", action="store_true")
    parser.add_argument("--empty_weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--reinforce_baseline", choices=["none", "batch"], default="none")
    parser.add_argument("--reflection_size_c", type=float, default=0.8)
    parser.add_argument("--refl_threshold", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_repair", action="store_true")
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--milestone_steps", default="")
    parser.add_argument("--progress_every", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
