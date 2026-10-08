"""Train official HRM architecture on the shared Kaggle Sudoku 3M split.

This script reuses sapientinc/HRM's HierarchicalReasoningModel_ACTV1 and
ACTLossHead, but replaces the dataset stack with the existing Kaggle cache.
It keeps the Full Sudoku-Hard architectural overrides from the HRM README:

  - loss: softmax_cross_entropy
  - L_cycles: 8
  - halt_max_steps: 8
  - pos_encodings: learned

Inference is a single HRM evaluation pass that internally iterates until
halting/max-steps. There is no external restart or symbolic repair.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REMOTE_HELPER_DIR = Path("code/source_snapshot")
if REMOTE_HELPER_DIR.exists() and str(REMOTE_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(REMOTE_HELPER_DIR))

HRM_PYDEPS = Path("code/optional_dependencies/hrm")
if HRM_PYDEPS.exists() and str(HRM_PYDEPS) not in sys.path:
    sys.path.insert(0, str(HRM_PYDEPS))

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
    return {"n": 0, "exact": 0, "valid": 0, "clue_ok": 0, "cell": 0, "cells": 0, "viol": 0.0, "steps": 0.0}


def make_hrm_batch(puzzle, solution):
    batch_size = puzzle.shape[0]
    return {
        "inputs": (puzzle.reshape(batch_size, 81).to(torch.int32) + 1).contiguous(),
        "labels": (solution.reshape(batch_size, 81).to(torch.int32) + 1).contiguous(),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.int32, device=puzzle.device),
    }


def pred_tokens_to_grid(tokens):
    # HRM Sudoku data uses token 1 for blank "0" and tokens 2..10 for digits 1..9.
    return (tokens - 1).clamp(0, 9)


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
                item["steps"] / n,
            )
        )
    return rows


def print_rows(rows, prefix):
    print(prefix, flush=True)
    print("bucket          n      exact   valid   clue_ok cell_acc avg_viol avg_steps", flush=True)
    for bucket, n, exact, valid, clue, cell, viol, steps in rows:
        print(
            f"{bucket:<13} {n:6d}  {exact:7.4f} {valid:7.4f} {clue:7.4f} "
            f"{cell:8.4f} {viol:8.3f} {steps:9.3f}",
            flush=True,
        )


@torch.no_grad()
def evaluate(model, loader, device, args, prefix):
    model.eval()
    stats = {"all": empty_stats()}
    for name, _, _ in RATING_BUCKETS:
        stats[name] = empty_stats()
    stats["rating_nan"] = empty_stats()

    seen = 0
    start = time.time()
    for batch in loader:
        puzzle = batch["puzzle"].to(device)
        solution = batch["solution"].to(device)
        ratings = batch["rating"].numpy()
        hrm_batch = make_hrm_batch(puzzle, solution)
        with torch.device(device):
            carry = model.initial_carry(hrm_batch)
        preds = None
        while True:
            carry, _, _, preds, all_finish = model(carry=carry, batch=hrm_batch, return_keys=["logits"])
            if all_finish:
                break
        pred_tokens = torch.argmax(preds["logits"], dim=-1)
        pred = pred_tokens_to_grid(pred_tokens).cpu().numpy().reshape(-1, 9, 9)
        puzzle_np = puzzle.cpu().numpy()
        solution_np = solution.cpu().numpy()
        steps_np = carry.steps.cpu().numpy()

        for i in range(pred.shape[0]):
            bucket = rating_bucket_name(float(ratings[i]))
            for name in ("all", bucket):
                item = stats.setdefault(name, empty_stats())
                item["n"] += 1
                item["exact"] += int(np.array_equal(pred[i], solution_np[i]))
                item["valid"] += int(is_valid_sudoku(pred[i]))
                item["clue_ok"] += int(clue_ok(pred[i], puzzle_np[i]))
                item["cell"] += int((pred[i] == solution_np[i]).sum())
                item["cells"] += 81
                item["viol"] += float(hard_violations(pred[i]))
                item["steps"] += float(steps_np[i])
            seen += 1
            if args.progress_every and seen % args.progress_every == 0:
                elapsed = time.time() - start
                print(f"[progress] seen={seen} elapsed={elapsed:.1f}s rate={seen / max(elapsed, 1e-6):.2f}/s", flush=True)

    rows = rows_from_stats(stats)
    print_rows(rows, prefix)
    return rows


def build_model(args):
    hrm_root = Path(args.hrm_root)
    if str(hrm_root) not in sys.path:
        sys.path.insert(0, str(hrm_root))

    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    from models.losses import ACTLossHead

    cfg = {
        "batch_size": args.batch_size,
        "seq_len": 81,
        "puzzle_emb_ndim": args.puzzle_emb_ndim,
        "num_puzzle_identifiers": 1,
        "vocab_size": 11,
        "H_cycles": args.H_cycles,
        "L_cycles": args.L_cycles,
        "H_layers": args.H_layers,
        "L_layers": args.L_layers,
        "hidden_size": args.hidden_size,
        "expansion": args.expansion,
        "num_heads": args.num_heads,
        "pos_encodings": args.pos_encodings,
        "halt_max_steps": args.halt_max_steps,
        "halt_exploration_prob": args.halt_exploration_prob,
        "forward_dtype": args.forward_dtype,
    }
    return ACTLossHead(HierarchicalReasoningModel_ACTV1(cfg), loss_type=args.loss_type), cfg


def save_checkpoint(path, model, cfg, args, step, best_val_exact, optimizer=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg,
            "model_type": "HRM_ACTV1_Official",
            "args": vars(args),
            "step": step,
            "best_val_exact": best_val_exact,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
        },
        path,
    )


def scheduled_lr(step, total_steps, base_lr, warmup_steps, min_ratio):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def train(args):
    set_torch_threads()
    set_seed(args.seed)
    device = torch.device(args.device)

    npz = np.load(args.cache_path, allow_pickle=False)
    train_ds = KaggleSudokuDataset(npz, split=0, limit=args.train_limit)
    val_ds = KaggleSudokuDataset(npz, split=1, limit=args.val_limit)
    test_ds = KaggleSudokuDataset(npz, split=2, limit=args.test_limit)
    if args.epochs > 0:
        args.steps = math.ceil(args.epochs * len(train_ds) / args.batch_size)
    if args.eval_every_epochs > 0:
        args.eval_every = max(1, round(args.eval_every_epochs * len(train_ds) / args.batch_size))
    target_epochs = args.steps * args.batch_size / len(train_ds)
    reference_steps = math.ceil(target_epochs * len(train_ds) / args.reference_global_batch_size)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} cache={args.cache_path}", flush=True)
    print(
        f"[budget] steps={args.steps} batch={args.batch_size} "
        f"sample_exposures={args.steps * args.batch_size} equivalent_epochs={target_epochs:.3f} "
        f"reference_global_batch={args.reference_global_batch_size} reference_optimizer_steps={reference_steps} "
        f"eval_every={args.eval_every}",
        flush=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)

    model, cfg = build_model(args)
    model = model.to(device)
    from adam_atan2 import AdamATan2

    optimizer = AdamATan2(
        model.parameters(), lr=0.0, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay
    )

    start_step = 0
    best_val_exact = -1.0
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = int(ckpt.get("step", 0))
        best_val_exact = float(ckpt.get("best_val_exact", -1.0))
        if ckpt.get("optimizer") is not None and not args.reset_optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
            print(f"[load-optimizer] restored at step={start_step}", flush=True)
        elif start_step and not args.eval_only:
            print("[load-optimizer] unavailable or reset; optimizer restarts", flush=True)
        print(f"[load] {args.checkpoint} step={start_step}", flush=True)

    print(
        f"[model] HRM official hidden={args.hidden_size} H={args.H_cycles} L={args.L_cycles} "
        f"halt={args.halt_max_steps} layers=({args.H_layers},{args.L_layers}) "
        f"pos={args.pos_encodings} loss={args.loss_type} dtype={args.forward_dtype} device={device}",
        flush=True,
    )

    if args.eval_only:
        evaluate(model, test_loader, device, args, "[test] checkpoint")
        return

    train_iter = iter(train_loader)
    carry = None
    start = time.time()
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        puzzle = batch["puzzle"].to(device)
        solution = batch["solution"].to(device)
        hrm_batch = make_hrm_batch(puzzle, solution)
        if carry is None:
            with torch.device(device):
                carry = model.initial_carry(hrm_batch)

        carry, loss, metrics, _, all_finish = model(carry=carry, batch=hrm_batch, return_keys=[])
        loss_scaled = loss / args.batch_size
        lr_this_step = scheduled_lr(
            step, args.steps, args.lr, args.lr_warmup_steps, args.lr_min_ratio
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step

        optimizer.zero_grad(set_to_none=True)
        loss_scaled.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if bool(all_finish):
            carry = None

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - start
            equivalent_epoch = step * args.batch_size / len(train_ds)
            metric_text = ""
            if metrics:
                count = float(metrics["count"].detach().cpu().item())
                denom = max(count, 1.0)
                metric_text = (
                    f" acc={float(metrics['accuracy'].detach().cpu().item()) / denom:.4f}"
                    f" exact={float(metrics['exact_accuracy'].detach().cpu().item()) / denom:.4f}"
                    f" steps={float(metrics['steps'].detach().cpu().item()) / denom:.2f}"
                )
            print(
                f"step {step:6d}/{args.steps} loss={float(loss_scaled.detach().cpu()):.5f} "
                f"epoch={equivalent_epoch:.3f} lr={lr_this_step:.7g}{metric_text} "
                f"halted_all={bool(all_finish)} elapsed={elapsed:.1f}s",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            rows = evaluate(model, val_loader, device, args, f"[val] step={step}")
            val_exact = rows[0][2]
            if val_exact > best_val_exact:
                best_val_exact = val_exact
                save_checkpoint(args.save_path, model, cfg, args, step, best_val_exact, optimizer)
                print(f"[save] {args.save_path} best_val_exact={best_val_exact:.4f}", flush=True)

    if args.save_final_path:
        save_checkpoint(args.save_final_path, model, cfg, args, args.steps, best_val_exact, optimizer)
        print(f"[save-final] {args.save_final_path}", flush=True)

    evaluate(model, test_loader, device, args, "[test] final")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default="data/cache_full_3m.npz")
    parser.add_argument("--hrm_root", default="code/source_snapshot/external/HRM")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--save_final_path", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--epochs", type=float, default=0.0, help="If positive, overrides --steps from dataset size and batch size.")
    parser.add_argument("--reference_global_batch_size", type=int, default=2_304)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=2_000)
    parser.add_argument("--lr_min_ratio", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.add_argument("--loss_type", choices=["softmax_cross_entropy", "stablemax_cross_entropy"], default="softmax_cross_entropy")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--expansion", type=float, default=4.0)
    parser.add_argument("--H_cycles", type=int, default=2)
    parser.add_argument("--L_cycles", type=int, default=8)
    parser.add_argument("--H_layers", type=int, default=4)
    parser.add_argument("--L_layers", type=int, default=4)
    parser.add_argument("--halt_max_steps", type=int, default=8)
    parser.add_argument("--halt_exploration_prob", type=float, default=0.1)
    parser.add_argument("--pos_encodings", choices=["learned", "rope"], default="learned")
    parser.add_argument("--puzzle_emb_ndim", type=int, default=0)
    parser.add_argument("--forward_dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--eval_every_epochs", type=float, default=0.0)
    parser.add_argument("--progress_every", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
