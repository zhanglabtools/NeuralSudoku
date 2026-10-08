import argparse
import hashlib
import math
import os
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sudoku_exchange_experiment import (
    ModelCfg,
    SudokuAnnealETF,
    constraint_loss,
    entropy_loss,
    integrity_loss,
    is_valid_sudoku,
    set_seed,
    set_torch_threads,
)


RATING_BUCKETS = [
    ("rating_0", -math.inf, 0.0),
    ("rating_0_1", 0.0, 1.0),
    ("rating_1_2", 1.0, 2.0),
    ("rating_2_4", 2.0, 4.0),
    ("rating_4_plus", 4.0, math.inf),
]


def find_column(columns, candidates):
    normalized = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def detect_columns(columns):
    puzzle_col = find_column(columns, ["puzzle", "quiz", "quizzes", "question", "givens", "board"])
    solution_col = find_column(columns, ["solution", "solutions", "answer", "answers", "solved"])
    clues_col = find_column(columns, ["clues", "num_clues", "n_clues", "clue_count"])
    rating_col = find_column(columns, ["rating", "difficulty", "difficulty_rating", "score"])

    if puzzle_col is None or solution_col is None:
        raise ValueError(
            "Could not detect puzzle/solution columns. "
            f"columns={list(columns)}; expected names like puzzle/solution or quizzes/solutions."
        )
    return puzzle_col, solution_col, clues_col, rating_col


def iter_csv_chunks(dataset_path, chunksize):
    dataset_path = Path(dataset_path)
    if dataset_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(dataset_path) as zf:
            csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError(f"No CSV found in {dataset_path}")
            csv_names.sort(key=lambda name: zf.getinfo(name).file_size, reverse=True)
            csv_name = csv_names[0]
            print(f"[csv] zip={dataset_path} member={csv_name} size={zf.getinfo(csv_name).file_size}")
            with zf.open(csv_name) as handle:
                yield from pd.read_csv(handle, chunksize=chunksize)
    else:
        print(f"[csv] path={dataset_path}")
        yield from pd.read_csv(dataset_path, chunksize=chunksize)


def normalize_grid_strings(series):
    values = series.astype(str).str.strip().str.replace(".", "0", regex=False)
    mask = values.str.len().eq(81) & values.str.match(r"^[0-9]+$")
    return values, mask.to_numpy()


def strings_to_uint8(strings):
    joined = "".join(strings)
    data = np.frombuffer(joined.encode("ascii"), dtype=np.uint8).reshape(-1, 81)
    return (data - ord("0")).astype(np.uint8, copy=False)


def split_from_puzzle(puzzle, seed):
    digest = hashlib.blake2b((str(seed) + puzzle).encode("ascii"), digest_size=2).digest()
    bucket = int.from_bytes(digest, "little") % 10
    if bucket < 8:
        return 0
    if bucket == 8:
        return 1
    return 2


def build_cache(args):
    out = Path(args.cache_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    puzzle_chunks = []
    solution_chunks = []
    clue_chunks = []
    rating_chunks = []
    split_chunks = []

    accepted = 0
    seen = 0
    start = time.time()
    detected = None

    for chunk in iter_csv_chunks(args.dataset_path, args.chunk_size):
        if detected is None:
            detected = detect_columns(chunk.columns)
            print(
                "[columns] puzzle=%s solution=%s clues=%s rating=%s"
                % tuple(str(x) for x in detected)
            )
        puzzle_col, solution_col, clues_col, rating_col = detected

        puzzle_str, puzzle_ok = normalize_grid_strings(chunk[puzzle_col])
        solution_str, solution_ok = normalize_grid_strings(chunk[solution_col])
        mask = puzzle_ok & solution_ok
        if args.drop_zero_rating and rating_col is not None:
            ratings_tmp = pd.to_numeric(chunk[rating_col], errors="coerce").to_numpy(dtype=np.float32)
            mask &= np.nan_to_num(ratings_tmp, nan=-1.0) > 0.0

        puzzle_list = puzzle_str[mask].tolist()
        solution_list = solution_str[mask].tolist()
        if not puzzle_list:
            seen += len(chunk)
            continue

        puzzles = strings_to_uint8(puzzle_list)
        solutions = strings_to_uint8(solution_list)
        if clues_col is None:
            clues = (puzzles > 0).sum(axis=1).astype(np.uint8)
        else:
            clues = pd.to_numeric(chunk.loc[mask, clues_col], errors="coerce").fillna(0).to_numpy(dtype=np.uint8)

        if rating_col is None:
            ratings = np.full(len(puzzle_list), np.nan, dtype=np.float32)
        else:
            ratings = pd.to_numeric(chunk.loc[mask, rating_col], errors="coerce").to_numpy(dtype=np.float32)

        splits = np.array([split_from_puzzle(puzzle, args.seed) for puzzle in puzzle_list], dtype=np.uint8)

        if args.max_rows and accepted + len(puzzles) > args.max_rows:
            keep = args.max_rows - accepted
            puzzles = puzzles[:keep]
            solutions = solutions[:keep]
            clues = clues[:keep]
            ratings = ratings[:keep]
            splits = splits[:keep]
            puzzle_list = puzzle_list[:keep]

        puzzle_chunks.append(puzzles)
        solution_chunks.append(solutions)
        clue_chunks.append(clues)
        rating_chunks.append(ratings)
        split_chunks.append(splits)
        accepted += len(puzzles)
        seen += len(chunk)

        if accepted % args.log_cache_every < len(puzzles):
            elapsed = time.time() - start
            split_counts = np.bincount(np.concatenate(split_chunks), minlength=3)
            print(
                f"[cache] accepted={accepted} seen={seen} "
                f"train/val/test={split_counts.tolist()} elapsed={elapsed:.1f}s"
            )
        if args.max_rows and accepted >= args.max_rows:
            break

    if accepted == 0:
        raise RuntimeError("No usable rows found in dataset.")

    puzzles = np.concatenate(puzzle_chunks, axis=0)
    solutions = np.concatenate(solution_chunks, axis=0)
    clues = np.concatenate(clue_chunks, axis=0)
    ratings = np.concatenate(rating_chunks, axis=0)
    splits = np.concatenate(split_chunks, axis=0)

    np.savez_compressed(
        out,
        puzzles=puzzles,
        solutions=solutions,
        clues=clues,
        ratings=ratings,
        splits=splits,
    )
    print(f"[cache] saved={out} rows={accepted} split={np.bincount(splits, minlength=3).tolist()}")


class KaggleSudokuDataset(Dataset):
    def __init__(self, npz, split, limit=0):
        mask = npz["splits"] == split
        indices = np.flatnonzero(mask)
        if limit and len(indices) > limit:
            indices = indices[:limit]
        self.puzzles = npz["puzzles"][indices].reshape(-1, 9, 9)
        self.solutions = npz["solutions"][indices].reshape(-1, 9, 9)
        self.clues = npz["clues"][indices]
        self.ratings = npz["ratings"][indices]

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        return {
            "puzzle": torch.from_numpy(self.puzzles[idx].astype(np.int64)),
            "solution": torch.from_numpy(self.solutions[idx].astype(np.int64)),
            "clues": int(self.clues[idx]),
            "rating": float(self.ratings[idx]),
        }


def rating_bucket_name(rating):
    if math.isnan(float(rating)):
        return "rating_nan"
    for name, lo, hi in RATING_BUCKETS:
        if lo < rating <= hi:
            return name
    return "rating_other"


def empty_stats():
    return {"n": 0, "exact": 0, "valid": 0, "clue_ok": 0, "cell": 0, "cells": 0}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    stats = {"all": empty_stats()}
    for name, _, _ in RATING_BUCKETS:
        stats[name] = empty_stats()
    stats["rating_nan"] = empty_stats()

    for batch in loader:
        puzzle = batch["puzzle"].to(device)
        solution = batch["solution"].to(device)
        ratings = batch["rating"].numpy()
        _, probs = model(puzzle)
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


def maybe_kaggle_download(args):
    if not args.kaggle_download:
        return
    out_dir = Path(args.dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "radcliffe/3-million-sudoku-puzzles-with-ratings",
        "-p",
        str(out_dir),
    ]
    print("[kaggle]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train(args):
    set_torch_threads()
    set_seed(args.seed)
    maybe_kaggle_download(args)

    if args.rebuild_cache or not Path(args.cache_path).exists():
        if not args.dataset_path:
            raise ValueError("--dataset_path is required when cache does not exist.")
        build_cache(args)
    if args.prepare_only:
        return

    npz = np.load(args.cache_path, allow_pickle=False)
    train_ds = KaggleSudokuDataset(npz, split=0, limit=args.train_limit)
    val_ds = KaggleSudokuDataset(npz, split=1, limit=args.val_limit)
    test_ds = KaggleSudokuDataset(npz, split=2, limit=args.test_limit)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} cache={args.cache_path}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)

    device = torch.device(args.device)
    cfg = ModelCfg(
        D=args.D,
        heads=args.heads,
        depth=args.depth,
        T=args.T,
        tau_max=args.tau_max,
        tau_min=args.tau_min,
        dropout=args.dropout,
        etf_seed=args.seed,
    )
    resume_checkpoint = None
    if args.resume_checkpoint:
        resume_checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        cfg = resume_checkpoint["cfg"]
        if args.resume_override_T:
            cfg.T = args.resume_override_T
        if args.resume_override_tau_max > 0.0:
            cfg.tau_max = args.resume_override_tau_max
        if args.resume_override_tau_min > 0.0:
            cfg.tau_min = args.resume_override_tau_min
        print(f"[resume] checkpoint={args.resume_checkpoint}")
        print("[resume] using checkpoint cfg; optimizer is reinitialized")

    model = SudokuAnnealETF(cfg).to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"[model] D={cfg.D} heads={cfg.heads} depth={cfg.depth} T={cfg.T} device={device}")
    initial_rows = evaluate(model, val_loader, device)
    print_eval(initial_rows, "[val] step=0")

    best_exact = initial_rows[0][2] if args.resume_checkpoint else -1.0
    if args.resume_checkpoint:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "cfg": cfg,
                "args": vars(args),
                "resume_checkpoint": args.resume_checkpoint,
            },
            args.save_path,
        )
        print(f"[save] {args.save_path} best_val_exact={best_exact:.4f} resume_start")

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
        logits, probs = model(puzzle)
        target = (solution - 1).clamp(0, 8)
        ce = F.cross_entropy(logits.reshape(-1, 9), target.reshape(-1), reduction="none").view(-1, 9, 9)
        empty_mask = (puzzle == 0).float()
        weights = 1.0 + (args.empty_weight - 1.0) * empty_mask
        loss_ce = (ce * weights).mean()
        loss_cstr = constraint_loss(probs)
        loss_ent = entropy_loss(probs)
        loss_int = integrity_loss(probs)
        loss = loss_ce + args.w_cstr * loss_cstr + args.w_ent * loss_ent + args.w_int * loss_int

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - start
            print(
                f"step {step:5d}/{args.steps} loss={loss.item():.4f} "
                f"ce={loss_ce.item():.4f} cstr={loss_cstr.item():.4f} "
                f"ent={loss_ent.item():.4f} int={loss_int.item():.4f} elapsed={elapsed:.1f}s"
            )

        if step % args.eval_every == 0 or step == args.steps:
            rows = evaluate(model, val_loader, device)
            print_eval(rows, f"[val] step={step}")
            val_exact = rows[0][2]
            if val_exact > best_exact:
                best_exact = val_exact
                Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "args": vars(args),
                        "resume_checkpoint": args.resume_checkpoint,
                    },
                    args.save_path,
                )
                print(f"[save] {args.save_path} best_val_exact={best_exact:.4f}")

    print_eval(evaluate(model, test_loader, device), "[test] final")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--dataset_dir", default="data")
    parser.add_argument("--cache_path", default="data/cache_full_3m.npz")
    parser.add_argument("--save_path", default="checkpoints/model_full_3m_depth6_t12_50k.pt")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--resume_override_T", type=int, default=0)
    parser.add_argument("--resume_override_tau_max", type=float, default=0.0)
    parser.add_argument("--resume_override_tau_min", type=float, default=0.0)
    parser.add_argument("--kaggle_download", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=100_000)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--log_cache_every", type=int, default=200_000)
    parser.add_argument("--drop_zero_rating", action="store_true")
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--T", type=int, default=12)
    parser.add_argument("--tau_max", type=float, default=1.5)
    parser.add_argument("--tau_min", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--w_cstr", type=float, default=0.3)
    parser.add_argument("--w_ent", type=float, default=0.02)
    parser.add_argument("--w_int", type=float, default=0.2)
    parser.add_argument("--empty_weight", type=float, default=2.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=5_000)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
