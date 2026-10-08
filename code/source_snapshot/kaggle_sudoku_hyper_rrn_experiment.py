"""Train hypergraph variants of the paper-style recurrent relational Sudoku model.

The experiment intentionally keeps the paper-style RRN data split, recurrent
schedule, every-step input injection, stepwise cross entropy, and evaluation
metrics.  The controlled variable is the message-passing structure:

* hyper_only: 81 cell nodes communicate through 27 recurrent row/column/box
  hyperedge (unit) states.
* hybrid: the same hypergraph channel plus the original typed pairwise RRN
  channel, fused as a gated residual over the pairwise message.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kaggle_sudoku_experiment import KaggleSudokuDataset
from kaggle_sudoku_rrn_paper_experiment import (
    build_rrn_edges,
    evaluate_rrn,
    print_eval,
    stepwise_loss,
)
from sudoku_exchange_experiment import set_seed, set_torch_threads


UNIT_COUNT = 27
MEMBERSHIP_COUNT = 27 * 9


def build_hypergraph_memberships():
    """Return cell-to-unit memberships for 9 rows, 9 columns, and 9 boxes."""
    cell_ids = []
    unit_ids = []
    unit_types = []
    for row in range(9):
        for col in range(9):
            cell = row * 9 + col
            box = (row // 3) * 3 + col // 3
            cell_ids.extend([cell, cell, cell])
            unit_ids.extend([row, 9 + col, 18 + box])
            unit_types.extend([0, 1, 2])
    return (
        torch.tensor(cell_ids, dtype=torch.long),
        torch.tensor(unit_ids, dtype=torch.long),
        torch.tensor(unit_types, dtype=torch.long),
    )


@dataclass
class HyperRRNCfg:
    model_type: str = "hyper_only"
    D: int = 128
    train_T: int = 32
    eval_T: int = 64
    msg_hidden: int = 256
    dropout: float = 0.0
    force_clues: bool = True
    hybrid_gate_bias: float = -2.0


class SudokuHyperRRN(nn.Module):
    """Paper-style recurrent Sudoku model with typed recurrent hyperedges."""

    def __init__(self, cfg):
        super().__init__()
        if cfg.model_type not in {"hyper_only", "hybrid"}:
            raise ValueError(f"Unknown model_type={cfg.model_type!r}")
        self.cfg = cfg
        # Evaluation-only causal intervention at the final message fusion.
        # It is deliberately not part of state_dict/checkpoint parameters.
        self.ablation_mode = "none"

        self.value_emb = nn.Embedding(10, cfg.D)
        self.row_emb = nn.Embedding(9, cfg.D)
        self.col_emb = nn.Embedding(9, cfg.D)
        self.box_emb = nn.Embedding(9, cfg.D)
        self.clue_emb = nn.Embedding(2, cfg.D)
        self.unit_type_emb = nn.Embedding(3, cfg.D)
        self.in_norm = nn.LayerNorm(cfg.D)
        self.unit_in_norm = nn.LayerNorm(cfg.D)

        self.v2u_mlp = nn.Sequential(
            nn.Linear(3 * cfg.D, cfg.msg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.msg_hidden, cfg.D),
        )
        self.v2u_norm = nn.LayerNorm(cfg.D)
        self.unit_gru = nn.GRUCell(2 * cfg.D, cfg.D)

        self.u2v_mlp = nn.Sequential(
            nn.Linear(3 * cfg.D, cfg.msg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.msg_hidden, cfg.D),
        )
        self.u2v_norm = nn.LayerNorm(cfg.D)

        if cfg.model_type == "hybrid":
            self.pair_edge_emb = nn.Embedding(3, cfg.D)
            self.pair_msg_mlp = nn.Sequential(
                nn.Linear(3 * cfg.D, cfg.msg_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.msg_hidden, cfg.D),
            )
            self.pair_msg_norm = nn.LayerNorm(cfg.D)
            self.fusion_gate = nn.Linear(3 * cfg.D, cfg.D)
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.constant_(self.fusion_gate.bias, cfg.hybrid_gate_bias)

        # Match the paper-style RRN update: aggregate message plus x0 is
        # injected into the cell GRU at every reasoning step.
        self.cell_gru = nn.GRUCell(2 * cfg.D, cfg.D)
        self.out_norm = nn.LayerNorm(cfg.D)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.D, cfg.D),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.D, 9),
        )

        row_ids = torch.arange(9).view(9, 1).expand(9, 9).reshape(81)
        col_ids = torch.arange(9).view(1, 9).expand(9, 9).reshape(81)
        box_ids = ((row_ids // 3) * 3 + col_ids // 3).reshape(81)
        unit_types = torch.arange(UNIT_COUNT, dtype=torch.long) // 9
        member_cell, member_unit, member_type = build_hypergraph_memberships()
        pair_src, pair_dst, pair_type = build_rrn_edges()

        self.register_buffer("row_ids", row_ids, persistent=False)
        self.register_buffer("col_ids", col_ids, persistent=False)
        self.register_buffer("box_ids", box_ids, persistent=False)
        self.register_buffer("unit_types", unit_types, persistent=False)
        self.register_buffer("member_cell", member_cell, persistent=False)
        self.register_buffer("member_unit", member_unit, persistent=False)
        self.register_buffer("member_type", member_type, persistent=False)
        self.register_buffer("pair_src", pair_src, persistent=False)
        self.register_buffer("pair_dst", pair_dst, persistent=False)
        self.register_buffer("pair_type", pair_type, persistent=False)

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

    def initial_unit_features(self, x0):
        batch = x0.shape[0]
        member_x0 = x0.index_select(1, self.member_cell)
        unit_x0 = x0.new_zeros(batch, UNIT_COUNT, self.cfg.D)
        unit_x0.index_add_(1, self.member_unit, member_x0)
        unit_x0 = unit_x0 / 9.0
        unit_type = self.unit_type_emb(self.unit_types).unsqueeze(0)
        return self.unit_in_norm(unit_x0 + unit_type)

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

    def update_units(self, h, unit_h, unit_x0):
        batch = h.shape[0]
        member_h = h.index_select(1, self.member_cell)
        member_unit_h = unit_h.index_select(1, self.member_unit)
        member_type = self.unit_type_emb(self.member_type).unsqueeze(0).expand(batch, -1, -1)
        msg = self.v2u_mlp(torch.cat([member_h, member_unit_h, member_type], dim=-1))
        agg = h.new_zeros(batch, UNIT_COUNT, self.cfg.D)
        agg.index_add_(1, self.member_unit, msg)
        agg = self.v2u_norm(agg / 9.0)
        update_input = torch.cat([agg, unit_x0], dim=-1)
        unit_h = self.unit_gru(
            update_input.reshape(batch * UNIT_COUNT, -1),
            unit_h.reshape(batch * UNIT_COUNT, -1),
        )
        return unit_h.reshape(batch, UNIT_COUNT, self.cfg.D)

    def hyper_messages(self, h, unit_h):
        batch = h.shape[0]
        member_h = h.index_select(1, self.member_cell)
        member_unit_h = unit_h.index_select(1, self.member_unit)
        member_type = self.unit_type_emb(self.member_type).unsqueeze(0).expand(batch, -1, -1)
        msg = self.u2v_mlp(torch.cat([member_unit_h, member_h, member_type], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(1, self.member_cell, msg)
        return self.u2v_norm(agg)

    def pair_messages(self, h):
        batch = h.shape[0]
        h_src = h.index_select(1, self.pair_src)
        h_dst = h.index_select(1, self.pair_dst)
        edge_e = self.pair_edge_emb(self.pair_type).unsqueeze(0).expand(batch, -1, -1)
        msg = self.pair_msg_mlp(torch.cat([h_src, h_dst, edge_e], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(1, self.pair_dst, msg)
        return self.pair_msg_norm(agg)

    def step(self, h, unit_h, x0, unit_x0):
        batch = h.shape[0]
        unit_h = self.update_units(h, unit_h, unit_x0)
        hyper_agg = self.hyper_messages(h, unit_h)

        if self.cfg.model_type == "hybrid":
            pair_agg = self.pair_messages(h)
            gate_input = torch.cat([h, pair_agg, hyper_agg], dim=-1)
            gate = torch.sigmoid(self.fusion_gate(gate_input))
            hyper_contribution = gate * hyper_agg
            if self.ablation_mode == "none":
                cell_agg = pair_agg + hyper_contribution
            elif self.ablation_mode == "no_hyper":
                cell_agg = pair_agg
            elif self.ablation_mode == "no_pair":
                # Hold the learned gate fixed and remove only the additive
                # pairwise contribution at the causal fusion site.
                cell_agg = hyper_contribution
            else:
                raise ValueError(f"Unknown ablation_mode={self.ablation_mode!r}")
        else:
            if self.ablation_mode != "none":
                raise ValueError("Branch ablation is only defined for the hybrid model")
            cell_agg = hyper_agg

        update_input = torch.cat([cell_agg, x0], dim=-1)
        h = self.cell_gru(
            update_input.reshape(batch * 81, -1),
            h.reshape(batch * 81, -1),
        )
        return h.reshape(batch, 81, self.cfg.D), unit_h

    def logits_from_state(self, h, puzzle_b99):
        batch = h.shape[0]
        logits = self.classifier(self.out_norm(h)).reshape(batch, 9, 9, 9)
        return self.force_clue_logits(logits, puzzle_b99)

    def forward(self, puzzle_b99, steps=None, return_all=False):
        steps = self.cfg.train_T if steps is None else int(steps)
        x0 = self.input_features(puzzle_b99)
        unit_x0 = self.initial_unit_features(x0)
        h = x0
        unit_h = unit_x0
        logits_steps = []

        for _ in range(steps):
            h, unit_h = self.step(h, unit_h, x0, unit_x0)
            if return_all:
                logits_steps.append(self.logits_from_state(h, puzzle_b99))

        logits = logits_steps[-1] if return_all else self.logits_from_state(h, puzzle_b99)
        probs = F.softmax(logits, dim=-1)
        if return_all:
            return logits, probs, logits_steps
        return logits, probs


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def checkpoint_payload(model, cfg, args, step, val_exact):
    return {
        "model": model.state_dict(),
        # Save a plain dictionary rather than a dataclass pickled from
        # __main__, so future checkpoints can be loaded from any entry point.
        "cfg": vars(cfg),
        "model_type": "SudokuHyperRRN",
        "variant": cfg.model_type,
        "step": int(step),
        "val_exact": float(val_exact),
        "args": vars(args),
    }


def train(args):
    set_torch_threads()
    set_seed(args.seed)

    npz = np.load(args.cache_path, allow_pickle=False)
    train_ds = KaggleSudokuDataset(npz, split=0, limit=args.train_limit)
    val_ds = KaggleSudokuDataset(npz, split=1, limit=args.val_limit)
    test_ds = KaggleSudokuDataset(npz, split=2, limit=args.test_limit)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} cache={args.cache_path}")

    train_generator = torch.Generator()
    train_generator.manual_seed(args.data_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    device = torch.device(args.device)
    cfg = HyperRRNCfg(
        model_type=args.model_type,
        D=args.D,
        train_T=args.train_T,
        eval_T=args.eval_T,
        msg_hidden=args.msg_hidden,
        dropout=args.dropout,
        force_clues=not args.no_force_clues,
        hybrid_gate_bias=args.hybrid_gate_bias,
    )
    model = SudokuHyperRRN(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"[model] Hyper-RRN variant={cfg.model_type} D={cfg.D} train_T={cfg.train_T} "
        f"eval_T={cfg.eval_T} msg_hidden={cfg.msg_hidden} loss_steps={args.loss_steps} "
        f"force_clues={cfg.force_clues} params={parameter_count(model)} device={device}"
    )
    print(
        f"[graph] units={UNIT_COUNT} memberships={MEMBERSHIP_COUNT} "
        f"pair_directed_edges={model.pair_src.numel() if cfg.model_type == 'hybrid' else 0} "
        f"data_seed={args.data_seed}"
    )
    if not args.skip_initial_eval:
        rows = evaluate_rrn(model, val_loader, device, steps=args.eval_T)
        print_eval(rows, f"[val] step=0 eval_T={args.eval_T}")

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
        loss = stepwise_loss(
            logits_steps,
            puzzle,
            solution,
            args.empty_weight,
            args.loss_steps,
            args.loss_tail,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - start
            print(
                f"step {step:5d}/{args.steps} loss={loss.item():.4f} "
                f"grad_norm={float(grad_norm):.4f} elapsed={elapsed:.1f}s"
            )

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
                    checkpoint_payload(model, cfg, args, step=step, val_exact=val_exact),
                    args.save_path,
                )
                print(
                    f"[save] {args.save_path} step={step} "
                    f"best_val_exact_eval_T{args.eval_T}={best_exact:.4f}"
                )

    if args.final_save_path:
        Path(args.final_save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint_payload(model, cfg, args, step=args.steps, val_exact=rows_eval_t[0][2]),
            args.final_save_path,
        )
        print(f"[save-final] {args.final_save_path}")

    print_eval(
        evaluate_rrn(model, test_loader, device, steps=args.train_T),
        f"[test] final eval_T={args.train_T}",
    )
    print_eval(
        evaluate_rrn(model, test_loader, device, steps=args.eval_T),
        f"[test] final eval_T={args.eval_T}",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["hyper_only", "hybrid"], required=True)
    parser.add_argument("--cache_path", default="data/cache_full_3m.npz")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--final_save_path", default="")
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_seed", type=int, default=0)
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
    parser.add_argument("--hybrid_gate_bias", type=float, default=-2.0)
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
