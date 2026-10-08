import argparse
import math
import os
import random
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


BUCKETS = ["easy", "medium", "hard", "diabolical"]
URLS = {
    "easy": "https://raw.githubusercontent.com/grantm/sudoku-exchange-puzzle-bank/master/easy.txt",
    "medium": "https://raw.githubusercontent.com/grantm/sudoku-exchange-puzzle-bank/master/medium.txt",
    "hard": "https://raw.githubusercontent.com/grantm/sudoku-exchange-puzzle-bank/master/hard.txt",
    "diabolical": "https://raw.githubusercontent.com/grantm/sudoku-exchange-puzzle-bank/master/diabolical.txt",
}
EXPECTED_SIZES = {
    "easy": 10_000_000,
    "medium": 35_264_300,
    "hard": 32_159_200,
    "diabolical": 11_968_100,
}


def set_torch_threads():
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_sudoku_exchange(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for bucket, url in URLS.items():
        out = data_dir / f"{bucket}.txt"
        expected = EXPECTED_SIZES[bucket]
        if out.exists() and out.stat().st_size == expected:
            print(f"[download] exists: {out} ({out.stat().st_size} bytes)")
            continue
        if out.exists():
            print(f"[download] removing incomplete file: {out} ({out.stat().st_size}/{expected} bytes)")
            out.unlink()

        print(f"[download] {bucket}: {url}")
        tmp = out.with_suffix(".txt.tmp")
        if tmp.exists():
            tmp.unlink()

        wget = shutil.which("wget")
        if wget:
            cmd = [
                wget,
                "--tries=10",
                "--timeout=60",
                "--waitretry=5",
                "-O",
                str(tmp),
                url,
            ]
            subprocess.run(cmd, check=True)
        else:
            last_error = None
            for attempt in range(1, 6):
                try:
                    urllib.request.urlretrieve(url, tmp)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"[download] retry {attempt}/5 failed: {exc}")
                    if tmp.exists():
                        tmp.unlink()
                    time.sleep(5)
            if last_error is not None:
                raise last_error

        actual = tmp.stat().st_size
        if actual != expected:
            raise RuntimeError(f"Downloaded {tmp} has {actual} bytes, expected {expected}")
        os.replace(tmp, out)
        print(f"[download] saved: {out} ({out.stat().st_size} bytes)")


ALL_MASK = (1 << 9) - 1


def bit(d):
    return 1 << (d - 1)


def block_index(r, c):
    return (r // 3) * 3 + (c // 3)


def solve_puzzle_string(puzzle):
    grid = [0 if ch in "0." else int(ch) for ch in puzzle]
    row_used = [0] * 9
    col_used = [0] * 9
    box_used = [0] * 9

    for idx, val in enumerate(grid):
        if val == 0:
            continue
        r, c = divmod(idx, 9)
        m = bit(val)
        b = block_index(r, c)
        if (row_used[r] & m) or (col_used[c] & m) or (box_used[b] & m):
            return None
        row_used[r] |= m
        col_used[c] |= m
        box_used[b] |= m

    def candidates(r, c):
        used = row_used[r] | col_used[c] | box_used[block_index(r, c)]
        return ALL_MASK & (~used)

    def choose_cell():
        best_idx = -1
        best_mask = 0
        best_count = 10
        for idx, val in enumerate(grid):
            if val != 0:
                continue
            r, c = divmod(idx, 9)
            m = candidates(r, c)
            count = m.bit_count()
            if count == 0:
                return idx, 0
            if count < best_count:
                best_idx = idx
                best_mask = m
                best_count = count
                if count == 1:
                    break
        return best_idx, best_mask

    def dfs():
        idx, mask = choose_cell()
        if idx == -1:
            return True
        if mask == 0:
            return False

        r, c = divmod(idx, 9)
        b = block_index(r, c)
        m = mask
        while m:
            low = m & -m
            d = low.bit_length()
            grid[idx] = d
            row_used[r] |= low
            col_used[c] |= low
            box_used[b] |= low

            if dfs():
                return True

            grid[idx] = 0
            row_used[r] ^= low
            col_used[c] ^= low
            box_used[b] ^= low
            m ^= low
        return False

    return "".join(str(v) for v in grid) if dfs() else None


def parse_record(line):
    parts = line.strip().split()
    if len(parts) < 3:
        return None
    puzzle = parts[1].replace(".", "0")
    rating = float(parts[2])
    if len(puzzle) != 81:
        return None
    return puzzle, rating


def build_cache(args):
    data_dir = Path(args.data_dir)
    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    need_per_bucket = args.per_bucket_train + args.per_bucket_val

    puzzles = []
    solutions = []
    ratings = []
    bucket_ids = []
    splits = []

    for bucket_id, bucket in enumerate(BUCKETS):
        path = data_dir / f"{bucket}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run with --download or check data_dir.")

        solved = 0
        seen = 0
        start = time.time()
        print(f"[cache] bucket={bucket} need={need_per_bucket}")
        with path.open("r", encoding="ascii") as handle:
            for line in handle:
                rec = parse_record(line)
                if rec is None:
                    continue
                puzzle, rating = rec
                solution = solve_puzzle_string(puzzle)
                seen += 1
                if solution is None:
                    continue

                split = 0 if solved < args.per_bucket_train else 1
                puzzles.append([0 if ch in "0." else int(ch) for ch in puzzle])
                solutions.append([int(ch) for ch in solution])
                ratings.append(rating)
                bucket_ids.append(bucket_id)
                splits.append(split)
                solved += 1

                if solved % 500 == 0:
                    print(f"[cache] {bucket}: solved={solved}/{need_per_bucket} seen={seen}")
                if solved >= need_per_bucket:
                    break

        if solved < need_per_bucket:
            raise RuntimeError(f"Only solved {solved} records for {bucket}, need {need_per_bucket}")
        print(f"[cache] {bucket}: done in {time.time() - start:.1f}s")

    np.savez_compressed(
        cache_path,
        puzzles=np.array(puzzles, dtype=np.uint8),
        solutions=np.array(solutions, dtype=np.uint8),
        ratings=np.array(ratings, dtype=np.float32),
        bucket_ids=np.array(bucket_ids, dtype=np.int64),
        splits=np.array(splits, dtype=np.int64),
        buckets=np.array(BUCKETS),
    )
    print(f"[cache] saved: {cache_path} samples={len(puzzles)}")


class SudokuNpzDataset(Dataset):
    def __init__(self, npz, split):
        mask = npz["splits"] == split
        self.puzzles = npz["puzzles"][mask].reshape(-1, 9, 9)
        self.solutions = npz["solutions"][mask].reshape(-1, 9, 9)
        self.bucket_ids = npz["bucket_ids"][mask]
        self.ratings = npz["ratings"][mask]

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        return {
            "puzzle": torch.from_numpy(self.puzzles[idx].astype(np.int64)),
            "solution": torch.from_numpy(self.solutions[idx].astype(np.int64)),
            "bucket_id": int(self.bucket_ids[idx]),
            "rating": float(self.ratings[idx]),
        }


def onehot10(puzzle_b99):
    x = F.one_hot(puzzle_b99.clamp(0, 9), num_classes=10).float()
    return x.permute(0, 3, 1, 2).contiguous()


def is_valid_sudoku(grid):
    target = set(range(1, 10))
    for r in range(9):
        if set(grid[r, :].tolist()) != target:
            return False
    for c in range(9):
        if set(grid[:, c].tolist()) != target:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            if set(grid[br : br + 3, bc : bc + 3].reshape(-1).tolist()) != target:
                return False
    return True


def simplex_etf(K=9, D=128, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    eye = torch.eye(K)
    centered = eye - torch.ones(K, K) / K
    _, eigvecs = torch.linalg.eigh(centered)
    basis = eigvecs[:, 1:]
    simplex = math.sqrt(K / (K - 1)) * basis.T
    weights = torch.zeros(D, K)
    weights[: K - 1, :] = simplex
    rotation, _ = torch.linalg.qr(torch.randn(D, D, generator=generator))
    return F.normalize(rotation @ weights, dim=0)


class ReasoningBlock(nn.Module):
    def __init__(self, dim=128, heads=8, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.row_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.col_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.box_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def _attn_rows(self, x):
        batch, _, _, dim = x.shape
        row = x.reshape(batch * 9, 9, dim)
        out, _ = self.row_attn(row, row, row, need_weights=False)
        return out.reshape(batch, 9, 9, dim)

    def _attn_cols(self, x):
        batch, _, _, dim = x.shape
        col = x.permute(0, 2, 1, 3).reshape(batch * 9, 9, dim)
        out, _ = self.col_attn(col, col, col, need_weights=False)
        return out.reshape(batch, 9, 9, dim).permute(0, 2, 1, 3)

    def _attn_boxes(self, x):
        batch, _, _, dim = x.shape
        box = x.view(batch, 3, 3, 3, 3, dim)
        box = box.permute(0, 1, 3, 2, 4, 5).reshape(batch * 9, 9, dim)
        out, _ = self.box_attn(box, box, box, need_weights=False)
        out = out.reshape(batch, 3, 3, 3, 3, dim).permute(0, 1, 3, 2, 4, 5)
        return out.reshape(batch, 9, 9, dim)

    def forward(self, x):
        h = self.ln1(x)
        msg = self._attn_rows(h) + self._attn_cols(h) + self._attn_boxes(h)
        x = x + self.drop(self.proj(msg))
        h2 = self.ln2(x)
        return x + self.drop(self.ffn(h2))


@dataclass
class ModelCfg:
    D: int = 128
    heads: int = 8
    depth: int = 6
    T: int = 12
    tau_max: float = 1.5
    tau_min: float = 0.2
    dropout: float = 0.0
    etf_seed: int = 0


class SudokuAnnealETF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Conv2d(19, cfg.D, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.D, cfg.D, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pos = nn.Parameter(torch.zeros(9, 9, cfg.D))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([ReasoningBlock(cfg.D, cfg.heads, cfg.dropout) for _ in range(cfg.depth)])
        self.to_feat = nn.Linear(cfg.D, cfg.D)
        self.register_buffer("W_etf", simplex_etf(K=9, D=cfg.D, seed=cfg.etf_seed))
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def tau_schedule(self):
        t = torch.linspace(0, 1, self.cfg.T)
        return self.cfg.tau_max * (self.cfg.tau_min / self.cfg.tau_max) ** t

    def forward(self, puzzle_b99):
        batch = puzzle_b99.shape[0]
        device = puzzle_b99.device
        tau_list = self.tau_schedule().to(device)
        clue_mask = puzzle_b99 > 0
        clue_idx = (puzzle_b99 - 1).clamp(0, 8)
        clue_onehot = F.one_hot(clue_idx, num_classes=9).float()
        forced_logits = clue_onehot * 30.0 + (1.0 - clue_onehot) * -30.0
        probs = torch.full((batch, 9, 9, 9), 1.0 / 9.0, device=device)
        probs = torch.where(clue_mask.unsqueeze(-1), clue_onehot, probs)
        logits = None

        for step in range(self.cfg.T):
            x_puzzle = onehot10(puzzle_b99)
            x_probs = probs.permute(0, 3, 1, 2).contiguous()
            x = torch.cat([x_puzzle, x_probs], dim=1)
            h = self.encoder(x).permute(0, 2, 3, 1).contiguous()
            h = h + self.pos.unsqueeze(0)
            for block in self.blocks:
                h = block(h)
            feat = F.normalize(self.to_feat(h), dim=-1)
            weights = F.normalize(self.W_etf, dim=0)
            scale = self.logit_scale.clamp(1.0, 100.0)
            logits = scale * torch.einsum("brcd,dk->brck", feat, weights)
            logits = torch.where(clue_mask.unsqueeze(-1), forced_logits, logits)
            probs = F.softmax(logits / tau_list[step], dim=-1)
        return logits, probs


def constraint_loss(probs):
    row_sum = probs.sum(dim=2)
    col_sum = probs.sum(dim=1)
    batch = probs.shape[0]
    box_sum = probs.view(batch, 3, 3, 3, 3, 9).sum(dim=(2, 4)).view(batch, 9, 9)
    return (row_sum - 1).pow(2).mean() + (col_sum - 1).pow(2).mean() + (box_sum - 1).pow(2).mean()


def entropy_loss(probs, eps=1e-8):
    p = probs.clamp(min=eps)
    return -(p * p.log()).sum(dim=-1).mean()


def integrity_loss(probs):
    return (1.0 - probs.pow(2).sum(dim=-1)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {name: {"n": 0, "exact": 0, "valid": 0, "clue_ok": 0, "cell": 0, "cells": 0} for name in BUCKETS}
    totals["all"] = {"n": 0, "exact": 0, "valid": 0, "clue_ok": 0, "cell": 0, "cells": 0}

    for batch in loader:
        puzzle = batch["puzzle"].to(device)
        solution = batch["solution"].to(device)
        bucket_ids = batch["bucket_id"].numpy()
        _, probs = model(puzzle)
        pred = probs.argmax(dim=-1) + 1

        pred_np = pred.cpu().numpy()
        solution_np = solution.cpu().numpy()
        puzzle_np = puzzle.cpu().numpy()
        for i in range(pred_np.shape[0]):
            bucket = BUCKETS[int(bucket_ids[i])]
            for key in (bucket, "all"):
                totals[key]["n"] += 1
                totals[key]["exact"] += int(np.array_equal(pred_np[i], solution_np[i]))
                totals[key]["valid"] += int(is_valid_sudoku(pred_np[i]))
                clue_mask = puzzle_np[i] > 0
                totals[key]["clue_ok"] += int(np.array_equal(pred_np[i][clue_mask], puzzle_np[i][clue_mask]))
                totals[key]["cell"] += int((pred_np[i] == solution_np[i]).sum())
                totals[key]["cells"] += 81

    rows = []
    for key in ["all"] + BUCKETS:
        item = totals[key]
        if item["n"] == 0:
            continue
        rows.append(
            (
                key,
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
    print("bucket       n      exact   valid   clue_ok cell_acc")
    for bucket, n, exact, valid, clue_ok, cell_acc in rows:
        print(f"{bucket:<10} {n:5d}  {exact:7.4f} {valid:7.4f} {clue_ok:7.4f} {cell_acc:8.4f}")


def run_training(args):
    set_torch_threads()
    set_seed(args.seed)
    if args.download:
        download_sudoku_exchange(args.data_dir)
    if args.rebuild_cache or not Path(args.cache_path).exists():
        build_cache(args)
    if args.prepare_only:
        return

    npz = np.load(args.cache_path, allow_pickle=True)
    train_ds = SudokuNpzDataset(npz, split=0)
    val_ds = SudokuNpzDataset(npz, split=1)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} cache={args.cache_path}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)

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
    model = SudokuAnnealETF(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"[model] D={cfg.D} heads={cfg.heads} depth={cfg.depth} T={cfg.T} device={device}")
    print(f"[train] steps={args.steps} batch={args.batch_size} lr={args.lr}")
    print_eval(evaluate(model, val_loader, device), "[eval] step=0")

    train_iter = iter(train_loader)
    best_exact = -1.0
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

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - start
            print(
                f"step {step:5d}/{args.steps} loss={loss.item():.4f} "
                f"ce={loss_ce.item():.4f} cstr={loss_cstr.item():.4f} "
                f"ent={loss_ent.item():.4f} int={loss_int.item():.4f} elapsed={elapsed:.1f}s"
            )

        if step % args.eval_every == 0 or step == args.steps:
            rows = evaluate(model, val_loader, device)
            print_eval(rows, f"[eval] step={step}")
            all_exact = rows[0][2]
            if all_exact > best_exact:
                best_exact = all_exact
                Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "cfg": cfg, "args": vars(args)}, args.save_path)
                print(f"[save] {args.save_path} best_exact={best_exact:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="sudoku_exchange_data")
    parser.add_argument("--cache_path", default="sudoku_exchange_data/cache_small.npz")
    parser.add_argument("--save_path", default="sudoku_exchange_data/sudoku_exchange_model.pt")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--per_bucket_train", type=int, default=1000)
    parser.add_argument("--per_bucket_val", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--tau_max", type=float, default=1.5)
    parser.add_argument("--tau_min", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--w_cstr", type=float, default=0.3)
    parser.add_argument("--w_ent", type=float, default=0.02)
    parser.add_argument("--w_int", type=float, default=0.2)
    parser.add_argument("--empty_weight", type=float, default=2.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=250)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
