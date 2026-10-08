"""Train verifier-guided primal-dual hypergraph reflection for Sudoku.

The model extends a trained eight-slot iterative reflector.  Exact Sudoku
validation is used only for evaluation/halting; training-time symbolic inputs
are structured row/column/box residuals.  There is no DFS, DLX, MRV, or
enumeration of candidate assignments.

Two experiment variants share this implementation:

* primal_dual: five-cycle reflection with soft/hard constraint tokens and
  persistent dual variables;
* diverse_primal_dual: the same model plus symbolic attention coverage,
  DPP trajectory diversity, top-four quality, and slot-dropout losses.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_hybrid_hyper_rrn_restarts import hard_violation_count_batch, load_hybrid
from sudoku_exchange_experiment import set_seed, set_torch_threads
from sudoku_cache_utils import load_sudoku_dataset
from train_iterative_hyper_reflection import (
    IterativeHyperReflector,
    IterativeReflectionCfg,
    per_sample_ce,
    per_slot_ce,
    reflection_loss,
)


@dataclass
class SymbolicPrimalDualCfg(IterativeReflectionCfg):
    dual_decay: float = 0.80
    symbolic_cell_scale: float = 0.20
    symbolic_unit_scale: float = 0.15
    attention_floor: float = 0.25
    attention_cap: float = 4.0
    variant: str = "primal_dual"


class SymbolicPrimalDualReflector(IterativeHyperReflector):
    """Five-cycle reflector augmented with persistent constraint state."""

    def __init__(self, backbone, cfg: SymbolicPrimalDualCfg):
        super().__init__(backbone, cfg)
        if cfg.variant not in {"primal_dual", "diverse_primal_dual"}:
            raise ValueError(f"Unknown symbolic variant={cfg.variant!r}")
        dim = cfg.D

        # Per-unit input: soft residual, detached hard count residual, dual.
        self.symbolic_encoder = nn.Sequential(
            nn.LayerNorm(27),
            nn.Linear(27, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.attention_norm = nn.LayerNorm(4 * dim)
        self.constraint_attention = nn.Sequential(
            nn.Linear(4 * dim, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, 1),
        )

        unit_feature_dim = 7 * dim
        self.symbolic_unit_norm = nn.LayerNorm(unit_feature_dim)
        self.symbolic_unit_body = nn.Sequential(
            nn.Linear(unit_feature_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.symbolic_unit_delta = nn.Linear(cfg.hidden, dim)
        self.symbolic_unit_gate = nn.Linear(cfg.hidden, 1)

        cell_feature_dim = 7 * dim
        self.symbolic_cell_norm = nn.LayerNorm(cell_feature_dim)
        self.symbolic_cell_body = nn.Sequential(
            nn.Linear(cell_feature_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.symbolic_cell_delta = nn.Linear(cfg.hidden, dim)
        self.symbolic_cell_gate = nn.Linear(cfg.hidden, 1)

        # Zero initialization makes a loaded three-cycle model the exact
        # starting point; symbolic behavior is learned rather than injected.
        nn.init.zeros_(self.symbolic_unit_delta.weight)
        nn.init.zeros_(self.symbolic_unit_delta.bias)
        nn.init.constant_(self.symbolic_unit_gate.bias, -1.0)
        nn.init.zeros_(self.symbolic_cell_delta.weight)
        nn.init.zeros_(self.symbolic_cell_delta.bias)
        nn.init.constant_(self.symbolic_cell_gate.bias, -1.0)

    def _hard_residuals(self, logits):
        pred = logits.reshape(logits.shape[0], 81, 9).argmax(dim=-1)
        onehot = F.one_hot(pred, num_classes=9).to(logits.dtype)
        counts = torch.einsum("uc,bcd->bud", self.incidence, onehot)
        # The discrete path is a diagnostic/token input, not a gradient path.
        return ((counts - 1.0) / 8.0).detach()

    def _symbolic_reflect_once(
        self,
        h,
        unit_h,
        x0,
        unit_x0,
        puzzle,
        dual,
        *,
        source_batch,
        slots,
        cycle_index,
    ):
        total = source_batch * slots
        pre_h = h
        pre_u = unit_h
        logits = self.backbone.logits_from_state(pre_h, puzzle)
        _, soft_residual = self._soft_residuals(logits)
        hard_residual = self._hard_residuals(logits)
        dual = (
            float(self.cfg.dual_decay) * dual + soft_residual
        ).clamp(-4.0, 4.0)

        symbolic = self.symbolic_encoder(
            torch.cat([soft_residual, hard_residual, dual], dim=-1)
        )
        slot_ids = torch.arange(slots, device=h.device)[None].expand(
            source_batch, -1
        ).reshape(-1)
        slot_emb = self.slot_embedding(slot_ids)
        cycle_ids = torch.full(
            (total,), int(cycle_index), dtype=torch.long, device=h.device
        )
        cycle_emb = self.cycle_embedding(cycle_ids)
        global_h = pre_h.mean(dim=1)
        global_u = pre_u.mean(dim=1)

        attention_features = self.attention_norm(
            torch.cat(
                [
                    pre_u,
                    symbolic,
                    slot_emb[:, None].expand(-1, 27, -1),
                    cycle_emb[:, None].expand(-1, 27, -1),
                ],
                dim=-1,
            )
        )
        attention_logits = self.constraint_attention(
            attention_features
        ).squeeze(-1)
        attention = torch.softmax(attention_logits, dim=-1)
        attention_scale = (
            float(self.cfg.attention_floor)
            + 27.0 * attention
        ).clamp_max(float(self.cfg.attention_cap))
        focused_symbolic = symbolic * attention_scale[:, :, None]

        # Preserve the already successful learned reflection path.
        h, unit_h, base_diagnostics = self._reflect_once(
            h,
            unit_h,
            x0,
            unit_x0,
            puzzle,
            source_batch=source_batch,
            slots=slots,
            cycle_index=cycle_index,
        )

        pooled_h = torch.einsum("uc,bcd->bud", self.incidence / 9.0, h)
        unit_features = self.symbolic_unit_norm(
            torch.cat(
                [
                    unit_h,
                    unit_x0,
                    pooled_h,
                    focused_symbolic,
                    global_h[:, None].expand(-1, 27, -1),
                    slot_emb[:, None].expand(-1, 27, -1),
                    cycle_emb[:, None].expand(-1, 27, -1),
                ],
                dim=-1,
            )
        )
        unit_body = self.symbolic_unit_body(unit_features)
        symbolic_unit_delta = torch.tanh(
            self.symbolic_unit_delta(unit_body)
        )
        symbolic_unit_gate = torch.sigmoid(
            self.symbolic_unit_gate(unit_body)
        )
        unit_h = unit_h + float(self.cfg.symbolic_unit_scale) * (
            attention_scale[:, :, None]
            * symbolic_unit_gate
            * symbolic_unit_delta
        )

        rows = self.backbone.row_ids
        cols = self.backbone.col_ids
        boxes = self.backbone.box_ids
        unit_context = (
            unit_h.index_select(1, rows)
            + unit_h.index_select(1, 9 + cols)
            + unit_h.index_select(1, 18 + boxes)
        ) / 3.0
        cell_symbolic = torch.einsum(
            "uc,bud->bcd", self.incidence / 3.0, focused_symbolic
        )
        cell_features = self.symbolic_cell_norm(
            torch.cat(
                [
                    h,
                    x0,
                    unit_context,
                    cell_symbolic,
                    global_h[:, None].expand(-1, 81, -1),
                    slot_emb[:, None].expand(-1, 81, -1),
                    cycle_emb[:, None].expand(-1, 81, -1),
                ],
                dim=-1,
            )
        )
        cell_body = self.symbolic_cell_body(cell_features)
        symbolic_cell_delta = torch.tanh(
            self.symbolic_cell_delta(cell_body)
        )
        symbolic_cell_gate = torch.sigmoid(
            self.symbolic_cell_gate(cell_body)
        )
        mutable = (puzzle.reshape(total, 81) == 0).to(h.dtype)
        h = h + float(self.cfg.symbolic_cell_scale) * (
            mutable[:, :, None]
            * symbolic_cell_gate
            * symbolic_cell_delta
        )

        diagnostics = dict(base_diagnostics)
        diagnostics.update(
            {
                "soft_residual": soft_residual,
                "hard_residual": hard_residual,
                "dual": dual,
                "constraint_attention": attention,
                "symbolic_cell_delta": symbolic_cell_delta,
                "symbolic_unit_delta": symbolic_unit_delta,
            }
        )
        return h, unit_h, dual, diagnostics

    def forward(self, puzzle, *, include_continuation=False):
        source_batch = puzzle.shape[0]
        slots = int(self.cfg.slots)
        x0, unit_x0, parent_h, parent_u, anchor_logits = self.encode_parent(puzzle)

        continuation_logits = []
        if include_continuation:
            with torch.no_grad():
                control_h, control_u = parent_h, parent_u
                for _ in range(int(self.cfg.cycles)):
                    control_h, control_u = self._rollout(
                        control_h,
                        control_u,
                        x0,
                        unit_x0,
                        self.cfg.recovery_steps,
                    )
                    continuation_logits.append(
                        self.backbone.logits_from_state(control_h, puzzle)
                    )

        h = self._expand_slots(parent_h, slots)
        unit_h = self._expand_slots(parent_u, slots)
        slot_x0 = self._expand_slots(x0, slots)
        slot_unit_x0 = self._expand_slots(unit_x0, slots)
        slot_puzzle = self._expand_slots(puzzle, slots)
        dual = h.new_zeros(source_batch * slots, 27, 9)
        cycle_logits = []
        cycle_diagnostics = []

        for cycle_index in range(int(self.cfg.cycles)):
            h, unit_h, dual = h.detach(), unit_h.detach(), dual.detach()
            h, unit_h, dual, diagnostics = self._symbolic_reflect_once(
                h,
                unit_h,
                slot_x0,
                slot_unit_x0,
                slot_puzzle,
                dual,
                source_batch=source_batch,
                slots=slots,
                cycle_index=cycle_index,
            )
            h, unit_h = self._rollout(
                h,
                unit_h,
                slot_x0,
                slot_unit_x0,
                self.cfg.recovery_steps,
            )
            logits = self.backbone.logits_from_state(h, slot_puzzle)
            cycle_logits.append(
                logits.reshape(source_batch, slots, 9, 9, 9)
            )
            cycle_diagnostics.append(diagnostics)

        slot_logits = cycle_logits[-1]
        anchor_ids = torch.zeros(
            source_batch, dtype=torch.long, device=puzzle.device
        )
        anchor_scores = self._critic_features(
            parent_h, parent_u, anchor_logits, puzzle, anchor_ids
        )
        slot_ids = (
            torch.arange(1, slots + 1, device=puzzle.device)[None]
            .expand(source_batch, -1)
            .reshape(-1)
        )
        slot_scores = self._critic_features(
            h,
            unit_h,
            slot_logits.reshape(source_batch * slots, 9, 9, 9),
            slot_puzzle,
            slot_ids,
        ).reshape(source_batch, slots)

        return {
            "anchor_logits": anchor_logits,
            "slot_logits": slot_logits,
            "cycle_logits": torch.stack(cycle_logits, dim=1),
            "continuation_logits": (
                torch.stack(continuation_logits, dim=1)
                if continuation_logits
                else None
            ),
            "scores": torch.cat([anchor_scores[:, None], slot_scores], dim=1),
            "cycle_diagnostics": cycle_diagnostics,
        }


def initialize_from_iterative(path, device, cfg):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    backbone, backbone_cfg, _ = load_hybrid(checkpoint["base_checkpoint"], device)
    cfg.D = int(backbone_cfg.D)
    model = SymbolicPrimalDualReflector(backbone, cfg).to(device)
    target = model.state_dict()
    copied = 0
    skipped = []
    for name, value in checkpoint["model_state"].items():
        if name == "cycle_embedding.weight":
            rows = min(value.shape[0], target[name].shape[0])
            target[name][:rows].copy_(value[:rows])
            for row in range(rows, target[name].shape[0]):
                target[name][row].copy_(value[rows - 1])
                target[name][row].add_(0.005 * torch.randn_like(target[name][row]))
            copied += 1
        elif name in target and target[name].shape == value.shape:
            target[name].copy_(value)
            copied += 1
        else:
            skipped.append(name)
    model.load_state_dict(target, strict=True)
    return model, checkpoint, copied, skipped


def exact_and_symbolic_metrics(output, puzzle, solution):
    batch = puzzle.shape[0]
    anchor_pred = output["anchor_logits"].argmax(dim=-1) + 1
    cycle_pred = output["cycle_logits"].argmax(dim=-1) + 1
    anchor_exact = (anchor_pred == solution).reshape(batch, -1).all(dim=-1)
    cycle_exact = (cycle_pred == solution[:, None, None]).reshape(
        batch, cycle_pred.shape[1], cycle_pred.shape[2], -1
    ).all(dim=-1)
    anchor_valid = hard_violation_count_batch(anchor_pred) == 0
    flat_cycle = cycle_pred.reshape(-1, 9, 9)
    cycle_valid = (hard_violation_count_batch(flat_cycle) == 0).reshape(
        batch, cycle_pred.shape[1], cycle_pred.shape[2]
    )
    symbolic_success = anchor_valid | cycle_valid.any(dim=(1, 2))
    exact_success = anchor_exact | cycle_exact.any(dim=(1, 2))
    return {
        "anchor_exact": anchor_exact,
        "cycle_exact": cycle_exact,
        "symbolic_success": symbolic_success,
        "exact_success": exact_success,
    }


def symbolic_diversity_terms(model, output, solution, args):
    batch, slots = output["slot_logits"].shape[:2]
    slot_ce = per_slot_ce(output["slot_logits"], solution)
    top_count = min(int(args.quality_topk), slots)
    top_quality = slot_ce.sort(dim=1).values[:, :top_count].mean()

    final_diag = output["cycle_diagnostics"][-1]
    attention = final_diag["constraint_attention"].reshape(batch, slots, 27)
    correction = final_diag["symbolic_unit_delta"].reshape(
        batch, slots, 27 * model.cfg.D
    )

    attention_signature = F.normalize(attention, p=2, dim=-1)
    correction_signature = F.normalize(correction, p=2, dim=-1)
    kernel = (
        0.60
        * torch.matmul(
            attention_signature, attention_signature.transpose(1, 2)
        )
        + 0.40
        * torch.matmul(
            correction_signature, correction_signature.transpose(1, 2)
        )
    )
    eye = torch.eye(slots, device=kernel.device, dtype=kernel.dtype)[None]
    sign, logabsdet = torch.linalg.slogdet(kernel + 0.05 * eye)
    dpp = torch.where(sign > 0, -logabsdet, logabsdet.new_full((), 20.0)).mean()

    hard = final_diag["hard_residual"].reshape(batch, slots, 27, 9)
    active = (hard.abs().sum(dim=-1) > 1e-6).any(dim=1).to(attention.dtype)
    covered = attention.max(dim=1).values
    coverage = -(
        (covered * active).sum(dim=1)
        / active.sum(dim=1).clamp_min(1.0)
    ).mean()

    candidate_ce = torch.cat(
        [per_sample_ce(output["anchor_logits"], solution)[:, None], slot_ce],
        dim=1,
    )
    masked_scores = output["scores"].clone()
    if slots > 1:
        drop_n = torch.randint(
            1, min(3, slots - 1) + 1, (batch,), device=solution.device
        )
        order = slot_ce.argsort(dim=1)
        for row in range(batch):
            masked_scores[row, 1 + order[row, : int(drop_n[row])]] = -1e9
    drop_prob = torch.softmax(
        masked_scores / float(args.critic_temperature), dim=1
    )
    slot_dropout = (drop_prob * candidate_ce).sum(dim=1).mean()
    return top_quality, dpp, coverage, slot_dropout


def total_loss(model, output, solution, args):
    base_loss, pieces = reflection_loss(model, output, solution, args)
    if model.cfg.variant == "diverse_primal_dual":
        top_quality, dpp, coverage, slot_dropout = symbolic_diversity_terms(
            model, output, solution, args
        )
        base_loss = (
            base_loss
            + args.top_quality_coef * top_quality
            + args.symbolic_dpp_coef * dpp
            + args.coverage_coef * coverage
            + args.slot_dropout_coef * slot_dropout
        )
    else:
        zero = base_loss.new_zeros(())
        top_quality = dpp = coverage = slot_dropout = zero
    pieces.update(
        {
            "top4": top_quality.detach(),
            "dpp": dpp.detach(),
            "coverage": coverage.detach(),
            "slotdrop": slot_dropout.detach(),
        }
    )
    return base_loss, pieces


@torch.no_grad()
def evaluate_validation(model, dataset, indices, device, batch_size):
    model.eval()
    totals = {
        "n": 0,
        "anchor": 0,
        "symbolic": 0,
        "exact_oracle": 0,
        "valid_not_exact": 0,
        "cycle": [0 for _ in range(model.cfg.cycles)],
    }
    for offset in range(0, len(indices), batch_size):
        take = indices[offset : offset + batch_size]
        puzzle = torch.as_tensor(dataset.puzzles[take], dtype=torch.long, device=device)
        solution = torch.as_tensor(
            dataset.solutions[take], dtype=torch.long, device=device
        )
        output = model(puzzle, include_continuation=False)
        metrics = exact_and_symbolic_metrics(output, puzzle, solution)
        totals["n"] += len(take)
        totals["anchor"] += int(metrics["anchor_exact"].sum())
        totals["symbolic"] += int(metrics["symbolic_success"].sum())
        totals["exact_oracle"] += int(metrics["exact_success"].sum())
        totals["valid_not_exact"] += int(
            (metrics["symbolic_success"] & ~metrics["exact_success"]).sum()
        )
        cumulative = metrics["anchor_exact"]
        for cycle in range(model.cfg.cycles):
            cumulative = cumulative | metrics["cycle_exact"][:, cycle].any(dim=1)
            totals["cycle"][cycle] += int(cumulative.sum())
    n = max(totals["n"], 1)
    return {
        "n": totals["n"],
        "anchor_exact": totals["anchor"] / n,
        "symbolic_exact": totals["symbolic"] / n,
        "oracle_exact": totals["exact_oracle"] / n,
        "valid_not_exact": totals["valid_not_exact"],
        "cycle_oracle": [value / n for value in totals["cycle"]],
    }


def load_indices(path):
    payload = np.load(path, allow_pickle=False)
    if isinstance(payload, np.ndarray):
        return np.asarray(payload, dtype=np.int64)
    return np.asarray(payload["indices"], dtype=np.int64)


def save_checkpoint(path, model, cfg, args, step, metrics):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "symbolic_primal_dual_reflection",
            "variant": cfg.variant,
            "base_checkpoint": model.backbone_checkpoint_path,
            "init_reflection_checkpoint": args.init_reflection_checkpoint,
            "reflection_cfg": asdict(cfg),
            "model_state": model.state_dict(),
            "args": vars(args),
            "step": int(step),
            "metrics": metrics,
        },
        target,
    )
    print(
        f"[save] {target} step={step} symbolic={metrics['symbolic_exact']:.6f} "
        f"cycles={metrics['cycle_oracle']}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_reflection_checkpoint", required=True)
    parser.add_argument("--residual_indices", required=True)
    parser.add_argument("--failure_indices", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument(
        "--cache_path",
        default="data/cache_full_3m.npz",
    )
    parser.add_argument(
        "--variant",
        choices=["primal_dual", "diverse_primal_dual"],
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--train_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--recovery_steps", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--residual_fraction", type=float, default=0.70)
    parser.add_argument("--failure_fraction", type=float, default=0.25)
    parser.add_argument("--best_ce_coef", type=float, default=1.0)
    parser.add_argument("--mean_ce_coef", type=float, default=0.10)
    parser.add_argument("--selected_ce_coef", type=float, default=0.20)
    parser.add_argument("--critic_coef", type=float, default=0.15)
    parser.add_argument("--cycle_ce_coef", type=float, default=0.35)
    parser.add_argument("--progress_coef", type=float, default=0.0)
    parser.add_argument("--progress_margin", type=float, default=0.002)
    parser.add_argument("--diversity_coef", type=float, default=0.0)
    parser.add_argument("--constraint_coef", type=float, default=0.03)
    parser.add_argument("--correction_reg_coef", type=float, default=0.001)
    parser.add_argument("--critic_temperature", type=float, default=0.25)
    parser.add_argument("--critic_target_temperature", type=float, default=0.05)
    parser.add_argument("--quality_topk", type=int, default=4)
    parser.add_argument("--top_quality_coef", type=float, default=0.15)
    parser.add_argument("--symbolic_dpp_coef", type=float, default=0.01)
    parser.add_argument("--coverage_coef", type=float, default=0.05)
    parser.add_argument("--slot_dropout_coef", type=float, default=0.10)
    parser.add_argument("--dual_decay", type=float, default=0.80)
    parser.add_argument("--symbolic_cell_scale", type=float, default=0.20)
    parser.add_argument("--symbolic_unit_scale", type=float, default=0.15)
    parser.add_argument("--val_hard_n", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=12)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    set_torch_threads()
    set_seed(args.seed)
    device = torch.device(args.device)
    cfg = SymbolicPrimalDualCfg(
        mode="global8",
        hidden=args.hidden,
        slots=8,
        cycles=args.cycles,
        recovery_steps=args.recovery_steps,
        parent_steps=64,
        correction_scale=0.35,
        unit_correction_scale=0.25,
        dropout=0.0,
        max_slots=8,
        dual_decay=args.dual_decay,
        symbolic_cell_scale=args.symbolic_cell_scale,
        symbolic_unit_scale=args.symbolic_unit_scale,
        variant=args.variant,
    )
    model, init_checkpoint, copied, skipped = initialize_from_iterative(
        args.init_reflection_checkpoint, device, cfg
    )
    model.backbone_checkpoint_path = init_checkpoint["base_checkpoint"]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )

    train_ds = load_sudoku_dataset(args.cache_path, split=0, limit=0)
    val_ds = load_sudoku_dataset(args.cache_path, split=1, limit=0)
    residual_indices = load_indices(args.residual_indices)
    failure_indices = load_indices(args.failure_indices)
    val_hard = np.flatnonzero(
        np.nan_to_num(val_ds.ratings, nan=-np.inf) > 4.0
    )[: args.val_hard_n]
    rng = np.random.default_rng(args.seed)

    print(
        f"[config] variant={args.variant} cycles={cfg.cycles} slots={cfg.slots} "
        f"residual_bank={len(residual_indices)} failure_bank={len(failure_indices)} "
        f"copied={copied} skipped={len(skipped)} trainable="
        f"{sum(parameter.numel() for parameter in trainable)} device={device}",
        flush=True,
    )
    best_score = -math.inf
    started = time.time()
    for step in range(1, args.train_steps + 1):
        residual_n = int(round(args.batch_size * args.residual_fraction))
        failure_n = int(round(args.batch_size * args.failure_fraction))
        if residual_n + failure_n > args.batch_size:
            failure_n = args.batch_size - residual_n
        random_n = args.batch_size - residual_n - failure_n
        parts = [
            rng.choice(residual_indices, size=residual_n, replace=True),
            rng.choice(failure_indices, size=failure_n, replace=True),
            rng.integers(0, len(train_ds), size=random_n),
        ]
        take = np.concatenate([part for part in parts if len(part)])
        rng.shuffle(take)
        puzzle = torch.as_tensor(
            train_ds.puzzles[take], dtype=torch.long, device=device
        )
        solution = torch.as_tensor(
            train_ds.solutions[take], dtype=torch.long, device=device
        )

        model.train()
        output = model(puzzle, include_continuation=False)
        loss, pieces = total_loss(model, output, solution, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            print(
                f"[train] step={step}/{args.train_steps} loss={float(loss.detach()):.4f} "
                + " ".join(
                    f"{name}={float(value):.4f}" for name, value in pieces.items()
                )
                + f" grad={float(grad_norm):.3f} elapsed={time.time()-started:.1f}s",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.train_steps:
            metrics = evaluate_validation(
                model, val_ds, val_hard, device, args.eval_batch_size
            )
            score = metrics["symbolic_exact"]
            print(f"[val] step={step} metrics={metrics} score={score:.6f}", flush=True)
            if score > best_score:
                best_score = score
                save_checkpoint(args.save_path, model, cfg, args, step, metrics)

    print(
        f"[done] variant={args.variant} best={best_score:.6f} "
        f"elapsed={time.time()-started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
