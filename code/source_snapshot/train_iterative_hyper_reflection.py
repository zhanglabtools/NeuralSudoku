"""Train pure-neural iterative hypergraph reflection for Sudoku.

The frozen Hybrid Hyper-RRN runs to a shared T64 parent.  A learned reflector
then rewrites recurrent cell/unit state and observes the result after a short
recovery rollout.  No candidate masks, DFS, MRV, DLX, or symbolic assignments
are used during inference.

Three modes share one implementation:

* local: one straight-through selected row/column/box is rewritten;
* global1: one global recurrent reflection slot;
* global8: eight deterministic latent slots with global rewrites.

The unchanged T64 prediction remains a candidate.  A neural value head chooses
between it and the reflected slots.  A T64+recovery continuation is computed
only as a diagnostic/training control and is not an inference candidate.
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
from kaggle_sudoku_experiment import KaggleSudokuDataset
from sudoku_exchange_experiment import set_seed, set_torch_threads


@dataclass
class IterativeReflectionCfg:
    mode: str = "global1"
    D: int = 128
    hidden: int = 256
    slots: int = 1
    cycles: int = 3
    recovery_steps: int = 8
    parent_steps: int = 64
    correction_scale: float = 0.35
    unit_correction_scale: float = 0.25
    local_tau: float = 0.75
    dropout: float = 0.0
    max_slots: int = 8


def unit_incidence(device=None):
    matrix = torch.zeros(27, 81, dtype=torch.float32, device=device)
    for row in range(9):
        matrix[row, row * 9 : (row + 1) * 9] = 1.0
    for col in range(9):
        matrix[9 + col, col::9] = 1.0
    for box in range(9):
        top, left = (box // 3) * 3, (box % 3) * 3
        for dr in range(3):
            for dc in range(3):
                matrix[18 + box, (top + dr) * 9 + left + dc] = 1.0
    return matrix


def per_sample_ce(logits, solution):
    batch = logits.shape[0]
    return F.cross_entropy(
        logits.reshape(batch * 81, 9),
        (solution.reshape(batch, 81) - 1).long().reshape(-1),
        reduction="none",
    ).reshape(batch, 81).mean(dim=-1)


def per_slot_ce(logits, solution):
    batch, slots = logits.shape[:2]
    targets = solution[:, None].expand(-1, slots, -1, -1)
    return F.cross_entropy(
        logits.reshape(batch * slots * 81, 9),
        (targets.reshape(batch * slots * 81) - 1).long(),
        reduction="none",
    ).reshape(batch, slots, 81).mean(dim=-1)


class IterativeHyperReflector(nn.Module):
    def __init__(self, backbone, cfg: IterativeReflectionCfg):
        super().__init__()
        if cfg.mode not in {"local", "global1", "global8"}:
            raise ValueError(f"Unknown reflection mode={cfg.mode!r}")
        expected_slots = 8 if cfg.mode == "global8" else 1
        if cfg.slots != expected_slots:
            raise ValueError(f"{cfg.mode} requires slots={expected_slots}, got {cfg.slots}")
        if int(backbone.cfg.D) != int(cfg.D):
            raise ValueError("Backbone and reflector dimensions differ")
        self.backbone = backbone
        self.cfg = cfg
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        dim = cfg.D
        self.slot_embedding = nn.Embedding(cfg.max_slots, dim)
        self.cycle_embedding = nn.Embedding(cfg.cycles, dim)
        self.candidate_embedding = nn.Embedding(cfg.max_slots + 1, dim)

        self.residual_encoder = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        feature_dim = 9 * dim
        self.cell_norm = nn.LayerNorm(feature_dim)
        self.cell_body = nn.Sequential(
            nn.Linear(feature_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.cell_delta = nn.Linear(cfg.hidden, dim)
        self.cell_gate = nn.Linear(cfg.hidden, 1)

        self.unit_norm = nn.LayerNorm(feature_dim)
        self.unit_body = nn.Sequential(
            nn.Linear(feature_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.unit_delta = nn.Linear(cfg.hidden, dim)
        self.unit_gate = nn.Linear(cfg.hidden, 1)

        local_dim = 7 * dim
        self.local_norm = nn.LayerNorm(local_dim)
        self.local_score = nn.Sequential(
            nn.Linear(local_dim, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, 1),
        )

        critic_dim = 3 * dim + 6
        self.critic_norm = nn.LayerNorm(critic_dim)
        self.critic = nn.Sequential(
            nn.Linear(critic_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )

        nn.init.zeros_(self.cell_delta.weight)
        nn.init.zeros_(self.cell_delta.bias)
        nn.init.constant_(self.cell_gate.bias, -1.0)
        nn.init.zeros_(self.unit_delta.weight)
        nn.init.zeros_(self.unit_delta.bias)
        nn.init.constant_(self.unit_gate.bias, -1.0)
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

        incidence = unit_incidence(torch.device("cpu"))
        self.register_buffer("incidence", incidence, persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _rollout(self, h, unit_h, x0, unit_x0, steps):
        for _ in range(int(steps)):
            h, unit_h = self.backbone.step(h, unit_h, x0, unit_x0)
        return h, unit_h

    @torch.no_grad()
    def encode_parent(self, puzzle):
        x0 = self.backbone.input_features(puzzle)
        unit_x0 = self.backbone.initial_unit_features(x0)
        h, unit_h = x0, unit_x0
        for _ in range(int(self.cfg.parent_steps)):
            h, unit_h = self.backbone.step(h, unit_h, x0, unit_x0)
        logits = self.backbone.logits_from_state(h, puzzle)
        return x0, unit_x0, h, unit_h, logits

    def _soft_residuals(self, logits):
        probs = torch.softmax(logits.reshape(logits.shape[0], 81, 9), dim=-1)
        unit_digit_sum = torch.einsum("uc,bcd->bud", self.incidence, probs)
        residual = unit_digit_sum - 1.0
        return probs, residual

    def _expand_slots(self, tensor, slots):
        shape = tensor.shape
        return (
            tensor[:, None]
            .expand(shape[0], slots, *shape[1:])
            .reshape(shape[0] * slots, *shape[1:])
            .contiguous()
        )

    def _choose_local_unit(self, unit_features):
        scores = self.local_score(self.local_norm(unit_features)).squeeze(-1)
        if self.training:
            choice = F.gumbel_softmax(
                scores, tau=float(self.cfg.local_tau), hard=True, dim=-1
            )
        else:
            index = scores.argmax(dim=-1)
            choice = F.one_hot(index, num_classes=27).to(scores.dtype)
        return scores, choice

    def _reflect_once(
        self,
        h,
        unit_h,
        x0,
        unit_x0,
        puzzle,
        *,
        source_batch,
        slots,
        cycle_index,
    ):
        total = source_batch * slots
        logits = self.backbone.logits_from_state(h, puzzle)
        probs, residual = self._soft_residuals(logits)
        residual_h = self.residual_encoder(residual)
        pooled_h = torch.einsum("uc,bcd->bud", self.incidence / 9.0, h)
        cell_residual_h = torch.einsum(
            "uc,bud->bcd", self.incidence / 3.0, residual_h
        )

        rows = self.backbone.row_ids
        cols = self.backbone.col_ids
        boxes = self.backbone.box_ids
        unit_context = (
            unit_h.index_select(1, rows)
            + unit_h.index_select(1, 9 + cols)
            + unit_h.index_select(1, 18 + boxes)
        ) / 3.0

        h_slots = h.reshape(source_batch, slots, 81, self.cfg.D)
        u_slots = unit_h.reshape(source_batch, slots, 27, self.cfg.D)
        mean_h = h_slots.mean(dim=1, keepdim=True).expand_as(h_slots).reshape_as(h)
        mean_u = u_slots.mean(dim=1, keepdim=True).expand_as(u_slots).reshape_as(unit_h)

        slot_ids = torch.arange(slots, device=h.device)[None].expand(
            source_batch, -1
        ).reshape(-1)
        slot_emb = self.slot_embedding(slot_ids)
        cycle_ids = torch.full(
            (total,), int(cycle_index), dtype=torch.long, device=h.device
        )
        cycle_emb = self.cycle_embedding(cycle_ids)
        global_h = h.mean(dim=1)
        global_u = unit_h.mean(dim=1)

        cell_features = self.cell_norm(
            torch.cat(
                [
                    h,
                    x0,
                    unit_context,
                    cell_residual_h,
                    global_h[:, None].expand(-1, 81, -1),
                    global_u[:, None].expand(-1, 81, -1),
                    slot_emb[:, None].expand(-1, 81, -1),
                    cycle_emb[:, None].expand(-1, 81, -1),
                    h - mean_h,
                ],
                dim=-1,
            )
        )
        cell_body = self.cell_body(cell_features)
        cell_delta = torch.tanh(self.cell_delta(cell_body))
        cell_gate = torch.sigmoid(self.cell_gate(cell_body))

        unit_features = self.unit_norm(
            torch.cat(
                [
                    unit_h,
                    unit_x0,
                    pooled_h,
                    residual_h,
                    global_h[:, None].expand(-1, 27, -1),
                    global_u[:, None].expand(-1, 27, -1),
                    slot_emb[:, None].expand(-1, 27, -1),
                    cycle_emb[:, None].expand(-1, 27, -1),
                    unit_h - mean_u,
                ],
                dim=-1,
            )
        )
        unit_body = self.unit_body(unit_features)
        unit_delta = torch.tanh(self.unit_delta(unit_body))
        unit_gate = torch.sigmoid(self.unit_gate(unit_body))

        local_scores = None
        if self.cfg.mode == "local":
            local_features = torch.cat(
                [
                    unit_h,
                    pooled_h,
                    residual_h,
                    global_h[:, None].expand(-1, 27, -1),
                    global_u[:, None].expand(-1, 27, -1),
                    slot_emb[:, None].expand(-1, 27, -1),
                    cycle_emb[:, None].expand(-1, 27, -1),
                ],
                dim=-1,
            )
            local_scores, unit_mask = self._choose_local_unit(local_features)
            cell_mask = torch.matmul(unit_mask, self.incidence).clamp_max(1.0)
        else:
            unit_mask = unit_h.new_ones(total, 27)
            cell_mask = h.new_ones(total, 81)

        mutable = (puzzle.reshape(total, 81) == 0).to(h.dtype)
        cell_mask = cell_mask * mutable
        h = h + float(self.cfg.correction_scale) * (
            cell_mask[:, :, None] * cell_gate * cell_delta
        )
        unit_h = unit_h + float(self.cfg.unit_correction_scale) * (
            unit_mask[:, :, None] * unit_gate * unit_delta
        )
        diagnostics = {
            "cell_gate": cell_gate,
            "unit_gate": unit_gate,
            "cell_delta": cell_delta,
            "unit_delta": unit_delta,
            "local_scores": local_scores,
            "probs": probs,
            "residual": residual,
        }
        return h, unit_h, diagnostics

    def _critic_features(self, h, unit_h, logits, puzzle, candidate_ids):
        probs, residual = self._soft_residuals(logits)
        confidence = probs.max(dim=-1).values
        entropy = -(
            probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()
        ).sum(dim=-1) / math.log(9.0)
        mutable = puzzle.reshape(puzzle.shape[0], 81) == 0
        denom = mutable.sum(dim=-1).clamp_min(1).float()
        mean_confidence = (confidence * mutable.float()).sum(dim=-1) / denom
        mean_entropy = (entropy * mutable.float()).sum(dim=-1) / denom
        mean_abs_residual = residual.abs().mean(dim=(1, 2))
        max_abs_residual = residual.abs().amax(dim=(1, 2))
        mean_square_residual = residual.square().mean(dim=(1, 2))
        mutable_fraction = denom / 81.0
        candidate_emb = self.candidate_embedding(candidate_ids)
        features = torch.cat(
            [
                h.mean(dim=1),
                unit_h.mean(dim=1),
                candidate_emb,
                mean_confidence[:, None],
                mean_entropy[:, None],
                mean_abs_residual[:, None],
                max_abs_residual[:, None],
                mean_square_residual[:, None],
                mutable_fraction[:, None],
            ],
            dim=-1,
        )
        return self.critic(self.critic_norm(features)).squeeze(-1)

    def forward(self, puzzle, *, include_continuation=True):
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
        cycle_logits = []
        cycle_diagnostics = []
        for cycle_index in range(int(self.cfg.cycles)):
            # Truncated gradients keep each reflection event trainable while
            # preserving the event-to-event state feedback at inference.
            h, unit_h = h.detach(), unit_h.detach()
            h, unit_h, diagnostics = self._reflect_once(
                h,
                unit_h,
                slot_x0,
                slot_unit_x0,
                slot_puzzle,
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
        anchor_ids = torch.zeros(source_batch, dtype=torch.long, device=puzzle.device)
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
        scores = torch.cat([anchor_scores[:, None], slot_scores], dim=1)

        return {
            "anchor_logits": anchor_logits,
            "slot_logits": slot_logits,
            "cycle_logits": torch.stack(cycle_logits, dim=1),
            "continuation_logits": (
                torch.stack(continuation_logits, dim=1)
                if continuation_logits
                else None
            ),
            "scores": scores,
            "cycle_diagnostics": cycle_diagnostics,
        }


def soft_constraint_loss(logits, incidence):
    probs = torch.softmax(logits.reshape(-1, 81, 9), dim=-1)
    residual = torch.einsum("uc,bcd->bud", incidence, probs) - 1.0
    return residual.square().mean()


def reflection_loss(model, output, solution, args):
    anchor_ce = per_sample_ce(output["anchor_logits"], solution)
    slot_ce = per_slot_ce(output["slot_logits"], solution)
    candidate_ce = torch.cat([anchor_ce[:, None], slot_ce], dim=1)

    best_loss = slot_ce.min(dim=1).values.mean()
    mean_loss = slot_ce.mean()
    score_prob = torch.softmax(
        output["scores"] / float(args.critic_temperature), dim=1
    )
    selected_loss = (score_prob * candidate_ce).sum(dim=1).mean()
    target_prob = torch.softmax(
        -candidate_ce.detach() / float(args.critic_target_temperature), dim=1
    )
    critic_loss = -(
        target_prob
        * F.log_softmax(
            output["scores"] / float(args.critic_temperature), dim=1
        )
    ).sum(dim=1).mean()

    cycle_best = []
    progress = []
    for cycle_index in range(model.cfg.cycles):
        reflected_ce = per_slot_ce(
            output["cycle_logits"][:, cycle_index], solution
        ).min(dim=1).values
        cycle_best.append(reflected_ce.mean())
        if output["continuation_logits"] is not None:
            control_ce = per_sample_ce(
                output["continuation_logits"][:, cycle_index], solution
            )
            progress.append(
                F.relu(
                    reflected_ce
                    - control_ce.detach()
                    + float(args.progress_margin)
                ).mean()
            )
    cycle_loss = torch.stack(cycle_best).mean()
    progress_loss = (
        torch.stack(progress).mean()
        if progress
        else anchor_ce.new_zeros(())
    )

    slots = output["slot_logits"].shape[1]
    if slots > 1:
        probs = torch.softmax(
            output["slot_logits"].reshape(
                output["slot_logits"].shape[0], slots, 81, 9
            ),
            dim=-1,
        )
        mean_probs = probs.mean(dim=1)
        entropy_mean = -(
            mean_probs.clamp_min(1e-12) * mean_probs.clamp_min(1e-12).log()
        ).sum(dim=-1)
        mean_entropy = -(
            probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()
        ).sum(dim=-1).mean(dim=1)
        js_diversity = (entropy_mean - mean_entropy).mean()
        diversity_loss = -js_diversity
    else:
        diversity_loss = anchor_ce.new_zeros(())

    constraint = soft_constraint_loss(
        output["slot_logits"], model.incidence
    )
    delta_terms = []
    for diagnostics in output["cycle_diagnostics"]:
        delta_terms.append(
            diagnostics["cell_delta"].square().mean()
            + diagnostics["unit_delta"].square().mean()
        )
    correction_reg = torch.stack(delta_terms).mean()

    loss = (
        args.best_ce_coef * best_loss
        + args.mean_ce_coef * mean_loss
        + args.selected_ce_coef * selected_loss
        + args.critic_coef * critic_loss
        + args.cycle_ce_coef * cycle_loss
        + args.progress_coef * progress_loss
        + args.diversity_coef * diversity_loss
        + args.constraint_coef * constraint
        + args.correction_reg_coef * correction_reg
    )
    pieces = {
        "best": best_loss.detach(),
        "mean": mean_loss.detach(),
        "selected": selected_loss.detach(),
        "critic": critic_loss.detach(),
        "cycle": cycle_loss.detach(),
        "progress": progress_loss.detach(),
        "diversity": (-diversity_loss).detach(),
        "constraint": constraint.detach(),
        "reg": correction_reg.detach(),
    }
    return loss, pieces


def selected_logits(output):
    candidates = torch.cat(
        [output["anchor_logits"][:, None], output["slot_logits"]], dim=1
    )
    choice = output["scores"].argmax(dim=1)
    rows = torch.arange(candidates.shape[0], device=candidates.device)
    return candidates[rows, choice], choice, candidates


@torch.no_grad()
def evaluate_indices(model, dataset, indices, device, batch_size):
    model.eval()
    totals = {
        "n": 0,
        "anchor_exact": 0,
        "continuation_exact": 0,
        "slot0_exact": 0,
        "selected_exact": 0,
        "selected_valid": 0,
        "oracle_exact": 0,
        "gain_vs_anchor": 0,
        "loss_vs_anchor": 0,
        "gain_vs_continuation": 0,
        "loss_vs_continuation": 0,
        "unique_grids": 0.0,
        "selected_slot": 0,
    }
    for offset in range(0, len(indices), batch_size):
        take = indices[offset : offset + batch_size]
        puzzle = torch.as_tensor(
            dataset.puzzles[take], dtype=torch.long, device=device
        )
        solution = torch.as_tensor(
            dataset.solutions[take], dtype=torch.long, device=device
        )
        output = model(puzzle, include_continuation=True)
        chosen_logits, choice, candidates = selected_logits(output)
        candidate_pred = candidates.argmax(dim=-1) + 1
        exact = (candidate_pred == solution[:, None]).reshape(
            len(take), candidates.shape[1], -1
        ).all(dim=-1)
        anchor_exact = exact[:, 0]
        slot0_exact = exact[:, 1]
        selected_pred = chosen_logits.argmax(dim=-1) + 1
        selected_exact = (selected_pred == solution).reshape(len(take), -1).all(dim=-1)
        oracle_exact = exact.any(dim=1)
        continuation_pred = output["continuation_logits"][:, -1].argmax(dim=-1) + 1
        continuation_exact = (continuation_pred == solution).reshape(
            len(take), -1
        ).all(dim=-1)
        valid = hard_violation_count_batch(selected_pred) == 0

        totals["n"] += len(take)
        totals["anchor_exact"] += int(anchor_exact.sum())
        totals["continuation_exact"] += int(continuation_exact.sum())
        totals["slot0_exact"] += int(slot0_exact.sum())
        totals["selected_exact"] += int(selected_exact.sum())
        totals["selected_valid"] += int(valid.sum())
        totals["oracle_exact"] += int(oracle_exact.sum())
        totals["gain_vs_anchor"] += int((selected_exact & ~anchor_exact).sum())
        totals["loss_vs_anchor"] += int((~selected_exact & anchor_exact).sum())
        totals["gain_vs_continuation"] += int(
            (selected_exact & ~continuation_exact).sum()
        )
        totals["loss_vs_continuation"] += int(
            (~selected_exact & continuation_exact).sum()
        )
        totals["selected_slot"] += int((choice > 0).sum())
        for sample in candidate_pred:
            totals["unique_grids"] += float(
                torch.unique(sample.reshape(sample.shape[0], 81), dim=0).shape[0]
            )
    n = max(totals["n"], 1)
    return {
        "n": totals["n"],
        "anchor_exact": totals["anchor_exact"] / n,
        "continuation_exact": totals["continuation_exact"] / n,
        "slot0_exact": totals["slot0_exact"] / n,
        "selected_exact": totals["selected_exact"] / n,
        "selected_valid": totals["selected_valid"] / n,
        "oracle_exact": totals["oracle_exact"] / n,
        "gain_vs_anchor": totals["gain_vs_anchor"],
        "loss_vs_anchor": totals["loss_vs_anchor"],
        "gain_vs_continuation": totals["gain_vs_continuation"],
        "loss_vs_continuation": totals["loss_vs_continuation"],
        "unique_grids": totals["unique_grids"] / n,
        "reflection_rate": totals["selected_slot"] / n,
    }


def save_checkpoint(path, model, cfg, args, step, metrics):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "iterative_hyper_reflection",
            "base_checkpoint": args.checkpoint,
            "reflection_cfg": asdict(cfg),
            "model_state": model.state_dict(),
            "args": vars(args),
            "step": int(step),
            "metrics": metrics,
        },
        target,
    )
    print(
        f"[save] {target} step={step} selected={metrics['selected_exact']:.6f} "
        f"oracle={metrics['oracle_exact']:.6f}",
        flush=True,
    )


def load_failure_indices(path):
    payload = np.load(path, allow_pickle=False)
    if isinstance(payload, np.ndarray):
        return np.asarray(payload, dtype=np.int64)
    return np.asarray(payload["indices"], dtype=np.int64)


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
    parser.add_argument("--failure_indices", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--mode", choices=["local", "global1", "global8"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--train_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--parent_steps", type=int, default=64)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--recovery_steps", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--correction_scale", type=float, default=0.35)
    parser.add_argument("--unit_correction_scale", type=float, default=0.25)
    parser.add_argument("--local_tau", type=float, default=0.75)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--failure_fraction", type=float, default=0.75)
    parser.add_argument("--hard_fraction", type=float, default=0.20)
    parser.add_argument("--best_ce_coef", type=float, default=1.0)
    parser.add_argument("--mean_ce_coef", type=float, default=0.10)
    parser.add_argument("--selected_ce_coef", type=float, default=0.25)
    parser.add_argument("--critic_coef", type=float, default=0.20)
    parser.add_argument("--cycle_ce_coef", type=float, default=0.35)
    parser.add_argument("--progress_coef", type=float, default=0.50)
    parser.add_argument("--progress_margin", type=float, default=0.002)
    parser.add_argument("--diversity_coef", type=float, default=0.02)
    parser.add_argument("--constraint_coef", type=float, default=0.02)
    parser.add_argument("--correction_reg_coef", type=float, default=0.001)
    parser.add_argument("--critic_temperature", type=float, default=0.25)
    parser.add_argument("--critic_target_temperature", type=float, default=0.05)
    parser.add_argument("--val_hard_n", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    set_torch_threads()
    set_seed(args.seed)
    device = torch.device(args.device)
    backbone, backbone_cfg, source = load_hybrid(args.checkpoint, device)
    slots = 8 if args.mode == "global8" else 1
    cfg = IterativeReflectionCfg(
        mode=args.mode,
        D=int(backbone_cfg.D),
        hidden=args.hidden,
        slots=slots,
        cycles=args.cycles,
        recovery_steps=args.recovery_steps,
        parent_steps=args.parent_steps,
        correction_scale=args.correction_scale,
        unit_correction_scale=args.unit_correction_scale,
        local_tau=args.local_tau,
        dropout=args.dropout,
        max_slots=max(8, slots),
    )
    model = IterativeHyperReflector(backbone, cfg).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )

    npz = np.load(args.cache_path, allow_pickle=False)
    train_ds = KaggleSudokuDataset(npz, split=0, limit=0)
    val_ds = KaggleSudokuDataset(npz, split=1, limit=0)
    failure_indices = load_failure_indices(args.failure_indices)
    train_hard = np.flatnonzero(
        np.nan_to_num(train_ds.ratings, nan=-np.inf) > 4.0
    )
    val_hard = np.flatnonzero(
        np.nan_to_num(val_ds.ratings, nan=-np.inf) > 4.0
    )[: args.val_hard_n]
    rng = np.random.default_rng(args.seed)

    print(
        f"[config] mode={args.mode} slots={slots} parent={args.parent_steps} "
        f"cycles={args.cycles} recovery={args.recovery_steps} "
        f"inference_equivalent_steps={args.parent_steps + slots * args.cycles * args.recovery_steps} "
        f"source_step={source.get('step')} failure_bank={len(failure_indices)} "
        f"train_hard={len(train_hard)} val_hard={len(val_hard)} "
        f"trainable={sum(p.numel() for p in trainable)} batch={args.batch_size} "
        f"device={device}",
        flush=True,
    )

    best_score = -math.inf
    started = time.time()
    for step in range(1, args.train_steps + 1):
        failure_n = int(round(args.batch_size * args.failure_fraction))
        hard_n = int(round(args.batch_size * args.hard_fraction))
        if failure_n + hard_n > args.batch_size:
            hard_n = args.batch_size - failure_n
        random_n = args.batch_size - failure_n - hard_n
        parts = [
            rng.choice(failure_indices, size=failure_n, replace=True),
            rng.choice(train_hard, size=hard_n, replace=True),
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
        output = model(puzzle, include_continuation=True)
        loss, pieces = reflection_loss(model, output, solution, args)
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
            metrics = evaluate_indices(
                model, val_ds, val_hard, device, args.eval_batch_size
            )
            score = (
                metrics["selected_exact"]
                + 0.10 * metrics["oracle_exact"]
                - 0.001 * metrics["loss_vs_anchor"] / max(metrics["n"], 1)
            )
            print(
                f"[val] step={step} metrics={metrics} score={score:.6f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                save_checkpoint(args.save_path, model, cfg, args, step, metrics)

    print(
        f"[done] mode={args.mode} best_score={best_score:.6f} "
        f"elapsed={time.time()-started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
