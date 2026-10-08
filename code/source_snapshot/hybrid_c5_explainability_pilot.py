"""Mechanistic explainability pilot for Hybrid Hyper-RRN + Primal-Dual C5.

The script provides four deliberately separate evidence layers:

1. stratified validation target discovery;
2. recurrent Hybrid state/constraint trajectories;
3. candidate-specific causal gates over pair, cell->unit, and unit->cell messages;
4. inference-time interventions on C5 residual, dual, attention, correction,
   and recovery pathways.

Solution labels are used for evaluation and oracle diagnostic target selection
only. They are never passed to the model, verifier, or candidate selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from eval_hybrid_hyper_rrn_restarts import hard_violation_count_batch
from eval_symbolic_active_reflection import load_symbolic
from sudoku_cache_utils import load_sudoku_dataset
from sudoku_exchange_experiment import set_seed, set_torch_threads


SPLITS = {"train": 0, "val": 1, "test": 2}
GROUPS = ("anchor_success", "c5_early", "c5_late", "c5_failure")
MESSAGE_KINDS = ("pair", "v2u", "u2v")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


def as_tensor_batch(dataset, positions, device):
    positions = np.asarray(positions, dtype=np.int64)
    puzzle = torch.as_tensor(
        np.asarray(dataset.puzzles[positions]), dtype=torch.long, device=device
    ).reshape(-1, 9, 9)
    solution = torch.as_tensor(
        np.asarray(dataset.solutions[positions]), dtype=torch.long, device=device
    ).reshape(-1, 9, 9)
    ratings = np.asarray(dataset.ratings[positions], dtype=np.float32)
    clues = np.asarray((puzzle.detach().cpu().numpy() > 0).sum(axis=(1, 2)))
    return puzzle, solution, ratings, clues


def fixed_margin(logits, cell_index, positive_digit, negative_digit):
    flat = logits.reshape(logits.shape[0], 81, 9)
    return (
        flat[:, int(cell_index), int(positive_digit)]
        - flat[:, int(cell_index), int(negative_digit)]
    )


def strongest_competitor(logits, cell_index, positive_digit):
    values = logits.reshape(logits.shape[0], 81, 9)[0, int(cell_index)].detach()
    masked = values.clone()
    masked[int(positive_digit)] = -torch.inf
    return int(masked.argmax())


def unit_residuals(model, logits):
    probs = torch.softmax(logits.reshape(logits.shape[0], 81, 9), dim=-1)
    residual = torch.einsum("uc,bcd->bud", model.incidence, probs) - 1.0
    return probs, residual


def state_metric_tensors(model, logits, puzzle, solution):
    batch = logits.shape[0]
    probs, residual = unit_residuals(model, logits)
    pred = logits.argmax(dim=-1) + 1
    target = (solution.reshape(batch, 81) - 1).clamp(0, 8)
    flat_logits = logits.reshape(batch, 81, 9)
    correct = flat_logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    alternatives = flat_logits.masked_fill(
        F.one_hot(target, num_classes=9).bool(), -torch.inf
    ).amax(dim=-1)
    margin = correct - alternatives
    entropy = -(
        probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()
    ).sum(dim=-1) / math.log(9.0)
    mutable = puzzle.reshape(batch, 81) == 0
    denom = mutable.sum(dim=-1).clamp_min(1).to(logits.dtype)
    exact = (pred == solution).reshape(batch, -1).all(dim=-1)
    violations = hard_violation_count_batch(pred)
    return {
        "pred": pred,
        "exact": exact,
        "valid": violations == 0,
        "hard_violations": violations,
        "residual_mse": residual.square().mean(dim=(1, 2)),
        "residual_mae": residual.abs().mean(dim=(1, 2)),
        "residual_max": residual.abs().amax(dim=(1, 2)),
        "entropy": (entropy * mutable).sum(dim=-1) / denom,
        "correct_margin": (margin * mutable).sum(dim=-1) / denom,
        "min_correct_margin": margin.masked_fill(~mutable, torch.inf).amin(dim=-1),
        "cell_accuracy": (
            ((pred == solution).reshape(batch, 81) & mutable).sum(dim=-1)
            / denom
        ),
    }


def scalar_metrics(metrics, index):
    result = {}
    for key, value in metrics.items():
        if key == "pred":
            continue
        item = value[index]
        if item.dtype == torch.bool:
            result[key] = int(item)
        elif item.dtype.is_floating_point:
            result[key] = float(item)
        else:
            result[key] = int(item)
    return result


def hybrid_step_intervened(
    model,
    h,
    unit_h,
    x0,
    unit_x0,
    *,
    pair_mask=None,
    v2u_mask=None,
    u2v_mask=None,
):
    """Exact Hybrid step with optional multiplicative message gates."""

    batch = h.shape[0]
    member_h = h.index_select(1, model.member_cell)
    member_unit_h = unit_h.index_select(1, model.member_unit)
    member_type = model.unit_type_emb(model.member_type).unsqueeze(0).expand(
        batch, -1, -1
    )
    v2u_msg = model.v2u_mlp(
        torch.cat([member_h, member_unit_h, member_type], dim=-1)
    )
    if v2u_mask is not None:
        v2u_msg = v2u_msg * v2u_mask.reshape(1, -1, 1)
    v2u_agg = h.new_zeros(batch, 27, model.cfg.D)
    v2u_agg.index_add_(1, model.member_unit, v2u_msg)
    v2u_agg = model.v2u_norm(v2u_agg / 9.0)
    unit_update_input = torch.cat([v2u_agg, unit_x0], dim=-1)
    new_unit_h = model.unit_gru(
        unit_update_input.reshape(batch * 27, -1),
        unit_h.reshape(batch * 27, -1),
    ).reshape(batch, 27, model.cfg.D)

    member_h = h.index_select(1, model.member_cell)
    member_unit_h = new_unit_h.index_select(1, model.member_unit)
    u2v_msg = model.u2v_mlp(
        torch.cat([member_unit_h, member_h, member_type], dim=-1)
    )
    if u2v_mask is not None:
        u2v_msg = u2v_msg * u2v_mask.reshape(1, -1, 1)
    hyper_agg = torch.zeros_like(h)
    hyper_agg.index_add_(1, model.member_cell, u2v_msg)
    hyper_agg = model.u2v_norm(hyper_agg)

    h_src = h.index_select(1, model.pair_src)
    h_dst = h.index_select(1, model.pair_dst)
    edge_type = model.pair_edge_emb(model.pair_type).unsqueeze(0).expand(
        batch, -1, -1
    )
    pair_msg = model.pair_msg_mlp(torch.cat([h_src, h_dst, edge_type], dim=-1))
    if pair_mask is not None:
        pair_msg = pair_msg * pair_mask.reshape(1, -1, 1)
    pair_agg = torch.zeros_like(h)
    pair_agg.index_add_(1, model.pair_dst, pair_msg)
    pair_agg = model.pair_msg_norm(pair_agg)

    gate = torch.sigmoid(
        model.fusion_gate(torch.cat([h, pair_agg, hyper_agg], dim=-1))
    )
    hyper_contribution = gate * hyper_agg
    update_input = torch.cat([pair_agg + hyper_contribution, x0], dim=-1)
    new_h = model.cell_gru(
        update_input.reshape(batch * 81, -1),
        h.reshape(batch * 81, -1),
    ).reshape(batch, 81, model.cfg.D)
    diagnostics = {
        "v2u_msg": v2u_msg,
        "u2v_msg": u2v_msg,
        "pair_msg": pair_msg,
        "v2u_agg": v2u_agg,
        "hyper_agg": hyper_agg,
        "pair_agg": pair_agg,
        "fusion_gate": gate,
    }
    return new_h, new_unit_h, diagnostics


def base_reflect_intervened(
    model,
    h,
    unit_h,
    x0,
    unit_x0,
    puzzle,
    residual,
    *,
    source_batch,
    slots,
    cycle_index,
    disable_cell=False,
    disable_unit=False,
):
    total = source_batch * slots
    probs = torch.softmax(
        model.backbone.logits_from_state(h, puzzle).reshape(total, 81, 9),
        dim=-1,
    )
    residual_h = model.residual_encoder(residual)
    pooled_h = torch.einsum("uc,bcd->bud", model.incidence / 9.0, h)
    cell_residual_h = torch.einsum(
        "uc,bud->bcd", model.incidence / 3.0, residual_h
    )
    rows = model.backbone.row_ids
    cols = model.backbone.col_ids
    boxes = model.backbone.box_ids
    unit_context = (
        unit_h.index_select(1, rows)
        + unit_h.index_select(1, 9 + cols)
        + unit_h.index_select(1, 18 + boxes)
    ) / 3.0
    h_slots = h.reshape(source_batch, slots, 81, model.cfg.D)
    u_slots = unit_h.reshape(source_batch, slots, 27, model.cfg.D)
    mean_h = h_slots.mean(dim=1, keepdim=True).expand_as(h_slots).reshape_as(h)
    mean_u = (
        u_slots.mean(dim=1, keepdim=True).expand_as(u_slots).reshape_as(unit_h)
    )
    slot_ids = (
        torch.arange(slots, device=h.device)[None]
        .expand(source_batch, -1)
        .reshape(-1)
    )
    slot_emb = model.slot_embedding(slot_ids)
    cycle_ids = torch.full(
        (total,), int(cycle_index), dtype=torch.long, device=h.device
    )
    cycle_emb = model.cycle_embedding(cycle_ids)
    global_h = h.mean(dim=1)
    global_u = unit_h.mean(dim=1)
    cell_features = model.cell_norm(
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
    cell_body = model.cell_body(cell_features)
    cell_delta = torch.tanh(model.cell_delta(cell_body))
    cell_gate = torch.sigmoid(model.cell_gate(cell_body))
    unit_features = model.unit_norm(
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
    unit_body = model.unit_body(unit_features)
    unit_delta = torch.tanh(model.unit_delta(unit_body))
    unit_gate = torch.sigmoid(model.unit_gate(unit_body))
    mutable = (puzzle.reshape(total, 81) == 0).to(h.dtype)
    if not disable_cell:
        h = h + float(model.cfg.correction_scale) * (
            mutable[:, :, None] * cell_gate * cell_delta
        )
    if not disable_unit:
        unit_h = unit_h + float(model.cfg.unit_correction_scale) * (
            unit_gate * unit_delta
        )
    return h, unit_h, {
        "cell_gate": cell_gate,
        "unit_gate": unit_gate,
        "cell_delta": cell_delta,
        "unit_delta": unit_delta,
        "probs": probs,
        "residual": residual,
    }


def modify_residual(residual, ablation):
    if ablation == "zero_residual":
        return torch.zeros_like(residual)
    if ablation == "shuffle_unit":
        return residual.roll(shifts=1, dims=1)
    if ablation == "shuffle_digit":
        return residual.roll(shifts=1, dims=2)
    if ablation == "sign_flip":
        return -residual
    return residual


def symbolic_reflect_intervened(
    model,
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
    ablation="full",
    stale_residual=None,
):
    """C5 reflection event with explicit inference-time interventions."""

    total = source_batch * slots
    pre_h = h
    pre_u = unit_h
    logits = model.backbone.logits_from_state(pre_h, puzzle)
    _, current_residual = model._soft_residuals(logits)
    current_hard = model._hard_residuals(logits)
    residual = (
        stale_residual
        if ablation == "stale_residual" and stale_residual is not None
        else current_residual
    )
    residual = modify_residual(residual, ablation)
    hard_residual = (
        torch.zeros_like(current_hard) if ablation == "no_hard" else current_hard
    )
    if ablation == "reset_dual":
        dual = torch.zeros_like(dual)
    dual = (
        float(model.cfg.dual_decay) * dual + residual
    ).clamp(-4.0, 4.0)
    symbolic_dual = torch.zeros_like(dual) if ablation == "no_dual" else dual
    symbolic = model.symbolic_encoder(
        torch.cat([residual, hard_residual, symbolic_dual], dim=-1)
    )
    slot_ids = (
        torch.arange(slots, device=h.device)[None]
        .expand(source_batch, -1)
        .reshape(-1)
    )
    slot_emb = model.slot_embedding(slot_ids)
    cycle_ids = torch.full(
        (total,), int(cycle_index), dtype=torch.long, device=h.device
    )
    cycle_emb = model.cycle_embedding(cycle_ids)
    global_h = pre_h.mean(dim=1)
    global_u = pre_u.mean(dim=1)
    attention_features = model.attention_norm(
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
    attention_logits = model.constraint_attention(attention_features).squeeze(-1)
    if ablation == "flat_attention":
        attention = torch.full_like(attention_logits, 1.0 / 27.0)
    elif ablation == "shuffle_attention":
        attention = torch.softmax(attention_logits, dim=-1).roll(
            shifts=1, dims=1
        )
    else:
        attention = torch.softmax(attention_logits, dim=-1)
    attention_scale = (
        float(model.cfg.attention_floor) + 27.0 * attention
    ).clamp_max(float(model.cfg.attention_cap))
    focused_symbolic = symbolic * attention_scale[:, :, None]

    if ablation in {"recovery_only", "no_base"}:
        base_diagnostics = {
            "cell_gate": torch.zeros(
                total, 81, 1, dtype=h.dtype, device=h.device
            ),
            "unit_gate": torch.zeros(
                total, 27, 1, dtype=h.dtype, device=h.device
            ),
            "cell_delta": torch.zeros_like(h),
            "unit_delta": torch.zeros_like(unit_h),
            "probs": torch.softmax(logits.reshape(total, 81, 9), dim=-1),
            "residual": residual,
        }
    else:
        h, unit_h, base_diagnostics = base_reflect_intervened(
            model,
            h,
            unit_h,
            x0,
            unit_x0,
            puzzle,
            residual,
            source_batch=source_batch,
            slots=slots,
            cycle_index=cycle_index,
            disable_cell=ablation in {"no_cell", "no_base_cell"},
            disable_unit=ablation in {"no_unit", "no_base_unit"},
        )

    symbolic_unit_delta = torch.zeros_like(unit_h)
    symbolic_unit_gate = torch.zeros(
        total, 27, 1, dtype=h.dtype, device=h.device
    )
    symbolic_cell_delta = torch.zeros_like(h)
    symbolic_cell_gate = torch.zeros(
        total, 81, 1, dtype=h.dtype, device=h.device
    )
    if ablation not in {"recovery_only", "no_symbolic"}:
        pooled_h = torch.einsum("uc,bcd->bud", model.incidence / 9.0, h)
        unit_features = model.symbolic_unit_norm(
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
        unit_body = model.symbolic_unit_body(unit_features)
        symbolic_unit_delta = torch.tanh(model.symbolic_unit_delta(unit_body))
        symbolic_unit_gate = torch.sigmoid(model.symbolic_unit_gate(unit_body))
        if ablation not in {"no_unit", "no_symbolic_unit"}:
            unit_h = unit_h + float(model.cfg.symbolic_unit_scale) * (
                attention_scale[:, :, None]
                * symbolic_unit_gate
                * symbolic_unit_delta
            )
        rows = model.backbone.row_ids
        cols = model.backbone.col_ids
        boxes = model.backbone.box_ids
        unit_context = (
            unit_h.index_select(1, rows)
            + unit_h.index_select(1, 9 + cols)
            + unit_h.index_select(1, 18 + boxes)
        ) / 3.0
        cell_symbolic = torch.einsum(
            "uc,bud->bcd", model.incidence / 3.0, focused_symbolic
        )
        cell_features = model.symbolic_cell_norm(
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
        cell_body = model.symbolic_cell_body(cell_features)
        symbolic_cell_delta = torch.tanh(model.symbolic_cell_delta(cell_body))
        symbolic_cell_gate = torch.sigmoid(model.symbolic_cell_gate(cell_body))
        mutable = (puzzle.reshape(total, 81) == 0).to(h.dtype)
        if ablation not in {"no_cell", "no_symbolic_cell"}:
            h = h + float(model.cfg.symbolic_cell_scale) * (
                mutable[:, :, None]
                * symbolic_cell_gate
                * symbolic_cell_delta
            )

    diagnostics = dict(base_diagnostics)
    diagnostics.update(
        {
            "soft_residual": residual,
            "raw_soft_residual": current_residual,
            "hard_residual": hard_residual,
            "dual": dual,
            "symbolic_dual": symbolic_dual,
            "constraint_attention": attention,
            "symbolic_cell_gate": symbolic_cell_gate,
            "symbolic_unit_gate": symbolic_unit_gate,
            "symbolic_cell_delta": symbolic_cell_delta,
            "symbolic_unit_delta": symbolic_unit_delta,
        }
    )
    return h, unit_h, dual, diagnostics


@torch.no_grad()
def scan_target_groups(model, dataset, args, device):
    ratings = np.nan_to_num(np.asarray(dataset.ratings), nan=-np.inf)
    eligible = np.flatnonzero(ratings > float(args.strict_min_rating))
    rng = np.random.default_rng(args.seed + 11)
    rng.shuffle(eligible)
    if args.scan_limit > 0:
        eligible = eligible[: args.scan_limit]
    found = {name: [] for name in GROUPS}
    scanned = 0
    for start in range(0, len(eligible), args.scan_batch_size):
        positions = eligible[start : start + args.scan_batch_size]
        puzzle, solution, batch_ratings, clues = as_tensor_batch(
            dataset, positions, device
        )
        x0, unit_x0, parent_h, parent_u, anchor_logits = model.encode_parent(
            puzzle
        )
        anchor_pred = anchor_logits.argmax(dim=-1) + 1
        anchor_valid = hard_violation_count_batch(anchor_pred) == 0
        first_cycle = torch.full(
            (len(positions),), -1, dtype=torch.int64, device=device
        )
        failures = torch.nonzero(~anchor_valid, as_tuple=False).squeeze(-1)
        if failures.numel():
            fail_output = model(
                puzzle.index_select(0, failures), include_continuation=False
            )
            cycle_pred = fail_output["cycle_logits"].argmax(dim=-1) + 1
            flat = cycle_pred.reshape(
                len(failures) * model.cfg.cycles * model.cfg.slots, 9, 9
            )
            valid = (
                hard_violation_count_batch(flat) == 0
            ).reshape(len(failures), model.cfg.cycles, model.cfg.slots)
            for local_index, original_index in enumerate(failures.tolist()):
                hit = torch.nonzero(
                    valid[local_index].any(dim=1), as_tuple=False
                ).squeeze(-1)
                if hit.numel():
                    first_cycle[original_index] = int(hit[0]) + 1
        for i, position in enumerate(positions):
            if anchor_valid[i]:
                group = "anchor_success"
            elif 1 <= first_cycle[i] <= 2:
                group = "c5_early"
            elif 3 <= first_cycle[i] <= int(model.cfg.cycles):
                group = "c5_late"
            else:
                group = "c5_failure"
            found[group].append(
                {
                    "split_position": int(position),
                    "group": group,
                    "rating": float(batch_ratings[i]),
                    "clues": int(clues[i]),
                    "anchor_valid": int(anchor_valid[i]),
                    "first_valid_cycle": int(first_cycle[i]),
                }
            )
        scanned += len(positions)
        if all(len(found[name]) >= args.quota_per_group for name in GROUPS):
            break
    selected = []
    py_rng = random.Random(args.seed + 17)
    for name in GROUPS:
        rows = found[name]
        py_rng.shuffle(rows)
        selected.extend(rows[: args.quota_per_group])
    print(
        "[scan] "
        + " ".join(f"{name}={len(found[name])}" for name in GROUPS)
        + f" scanned={scanned} selected={len(selected)}",
        flush=True,
    )
    return selected, {name: len(found[name]) for name in GROUPS}, scanned


@torch.no_grad()
def collect_hybrid_trajectories(model, dataset, targets, device, save_states):
    rows = []
    state_payload = {
        "split_position": np.asarray(
            [item["split_position"] for item in targets], dtype=np.int64
        )
    }
    puzzles, solutions, _, _ = as_tensor_batch(
        dataset, state_payload["split_position"], device
    )
    x0 = model.backbone.input_features(puzzles)
    unit_x0 = model.backbone.initial_unit_features(x0)
    h, unit_h = x0, unit_x0
    previous_h = h
    previous_u = unit_h
    previous_pred = None
    final_predictions = []
    all_h = []
    all_u = []
    all_logits = []
    for step in range(1, int(model.cfg.parent_steps) + 1):
        h, unit_h = model.backbone.step(h, unit_h, x0, unit_x0)
        logits = model.backbone.logits_from_state(h, puzzles)
        metrics = state_metric_tensors(model, logits, puzzles, solutions)
        pred = metrics["pred"]
        final_predictions.append(pred.detach().cpu())
        h_delta = (h - previous_h).square().mean(dim=(1, 2)).sqrt()
        u_delta = (unit_h - previous_u).square().mean(dim=(1, 2)).sqrt()
        flips = (
            torch.zeros(len(targets), dtype=torch.int64, device=device)
            if previous_pred is None
            else (pred != previous_pred).reshape(len(targets), -1).sum(dim=-1)
        )
        for i, target in enumerate(targets):
            row = {
                "split_position": target["split_position"],
                "group": target["group"],
                "rating": target["rating"],
                "clues": target["clues"],
                "step": step,
                "cell_state_delta_rms": float(h_delta[i]),
                "unit_state_delta_rms": float(u_delta[i]),
                "prediction_flips": int(flips[i]),
            }
            row.update(scalar_metrics(metrics, i))
            rows.append(row)
        if save_states:
            all_h.append(h.detach().cpu().to(torch.float16).numpy())
            all_u.append(unit_h.detach().cpu().to(torch.float16).numpy())
            all_logits.append(
                logits.detach().cpu().to(torch.float16).numpy()
            )
        previous_h, previous_u, previous_pred = h, unit_h, pred
    final_pred = final_predictions[-1]
    for step, pred in enumerate(final_predictions, start=1):
        agreement = (pred == final_pred).reshape(len(targets), -1).float().mean(
            dim=-1
        )
        offset = (step - 1) * len(targets)
        for i in range(len(targets)):
            rows[offset + i]["final_prediction_agreement"] = float(agreement[i])
    if save_states:
        state_payload.update(
            {
                "cell_h": np.stack(all_h, axis=1),
                "unit_h": np.stack(all_u, axis=1),
                "logits": np.stack(all_logits, axis=1),
                "puzzle": puzzles.detach().cpu().numpy().astype(np.uint8),
                "solution": solutions.detach().cpu().numpy().astype(np.uint8),
            }
        )
    return rows, state_payload


@torch.no_grad()
def run_c5_ablation(model, puzzle, solution, ablation):
    batch = puzzle.shape[0]
    slots = int(model.cfg.slots)
    x0, unit_x0, parent_h, parent_u, anchor_logits = model.encode_parent(puzzle)
    h = model._expand_slots(parent_h, slots)
    unit_h = model._expand_slots(parent_u, slots)
    slot_x0 = model._expand_slots(x0, slots)
    slot_unit_x0 = model._expand_slots(unit_x0, slots)
    slot_puzzle = model._expand_slots(puzzle, slots)
    slot_solution = model._expand_slots(solution, slots)
    dual = h.new_zeros(batch * slots, 27, 9)
    stale_residual = None
    rows = []
    first_valid = torch.full(
        (batch,), -1, dtype=torch.int64, device=puzzle.device
    )
    for cycle in range(int(model.cfg.cycles)):
        pre_logits = model.backbone.logits_from_state(h, slot_puzzle)
        pre_metrics = state_metric_tensors(
            model, pre_logits, slot_puzzle, slot_solution
        )
        h_before, u_before = h, unit_h
        h, unit_h, dual, diagnostics = symbolic_reflect_intervened(
            model,
            h,
            unit_h,
            slot_x0,
            slot_unit_x0,
            slot_puzzle,
            dual,
            source_batch=batch,
            slots=slots,
            cycle_index=cycle,
            ablation=ablation,
            stale_residual=stale_residual,
        )
        if stale_residual is None:
            stale_residual = diagnostics["raw_soft_residual"].detach()
        rewrite_logits = model.backbone.logits_from_state(h, slot_puzzle)
        rewrite_metrics = state_metric_tensors(
            model, rewrite_logits, slot_puzzle, slot_solution
        )
        rewrite_h, rewrite_u = h, unit_h
        recovery_steps = (
            0 if ablation == "no_recovery" else int(model.cfg.recovery_steps)
        )
        for _ in range(recovery_steps):
            h, unit_h = model.backbone.step(
                h, unit_h, slot_x0, slot_unit_x0
            )
        post_logits = model.backbone.logits_from_state(h, slot_puzzle)
        post_metrics = state_metric_tensors(
            model, post_logits, slot_puzzle, slot_solution
        )
        valid = post_metrics["valid"].reshape(batch, slots)
        newly_valid = (first_valid < 0) & valid.any(dim=1)
        first_valid[newly_valid] = cycle + 1
        attention = diagnostics["constraint_attention"].reshape(
            batch, slots, 27
        )
        raw_residual_strength = (
            diagnostics["raw_soft_residual"]
            .abs()
            .mean(dim=-1)
            .reshape(batch, slots, 27)
        )
        used_residual_strength = (
            diagnostics["soft_residual"]
            .abs()
            .mean(dim=-1)
            .reshape(batch, slots, 27)
        )
        centered_attention = attention - attention.mean(dim=-1, keepdim=True)
        centered_residual = (
            raw_residual_strength
            - raw_residual_strength.mean(dim=-1, keepdim=True)
        )
        attention_residual_corr = (
            (centered_attention * centered_residual).sum(dim=-1)
            / (
                centered_attention.square().sum(dim=-1).sqrt()
                * centered_residual.square().sum(dim=-1).sqrt()
            ).clamp_min(1e-12)
        )
        base_cell_norm = diagnostics["cell_delta"].reshape(
            batch, slots, 81, -1
        ).square().mean(dim=(2, 3)).sqrt()
        base_unit_norm = diagnostics["unit_delta"].reshape(
            batch, slots, 27, -1
        ).square().mean(dim=(2, 3)).sqrt()
        symbolic_cell_norm = diagnostics["symbolic_cell_delta"].reshape(
            batch, slots, 81, -1
        ).square().mean(dim=(2, 3)).sqrt()
        symbolic_unit_norm = diagnostics["symbolic_unit_delta"].reshape(
            batch, slots, 27, -1
        ).square().mean(dim=(2, 3)).sqrt()
        h_rewrite = (h_before - rewrite_h).reshape(
            batch, slots, 81, -1
        ).square().mean(dim=(2, 3)).sqrt()
        u_rewrite = (u_before - rewrite_u).reshape(
            batch, slots, 27, -1
        ).square().mean(dim=(2, 3)).sqrt()
        cell_rewrite = (rewrite_h - h_before).reshape(
            batch, slots, 81, -1
        ).square().mean(dim=-1).sqrt()
        pre_pred = pre_metrics["pred"].reshape(batch, slots, 81)
        truth = slot_solution.reshape(batch, slots, 81)
        mutable = slot_puzzle.reshape(batch, slots, 81) == 0
        wrong = mutable & (pre_pred != truth)
        correct = mutable & (pre_pred == truth)
        wrong_correction = (
            (cell_rewrite * wrong).sum(dim=-1)
            / wrong.sum(dim=-1).clamp_min(1)
        )
        correct_correction = (
            (cell_rewrite * correct).sum(dim=-1)
            / correct.sum(dim=-1).clamp_min(1)
        )
        for b in range(batch):
            for slot in range(slots):
                flat_index = b * slots + slot
                row = {
                    "batch_index": b,
                    "ablation": ablation,
                    "cycle": cycle + 1,
                    "slot": slot,
                    "attention_top_unit": int(attention[b, slot].argmax()),
                    "attention_max": float(attention[b, slot].max()),
                    "raw_residual_top_unit": int(
                        raw_residual_strength[b, slot].argmax()
                    ),
                    "used_residual_top_unit": int(
                        used_residual_strength[b, slot].argmax()
                    ),
                    "attention_hits_raw_residual_top": int(
                        attention[b, slot].argmax()
                        == raw_residual_strength[b, slot].argmax()
                    ),
                    "attention_residual_corr": float(
                        attention_residual_corr[b, slot]
                    ),
                    "base_cell_delta_rms": float(base_cell_norm[b, slot]),
                    "base_unit_delta_rms": float(base_unit_norm[b, slot]),
                    "symbolic_cell_delta_rms": float(
                        symbolic_cell_norm[b, slot]
                    ),
                    "symbolic_unit_delta_rms": float(
                        symbolic_unit_norm[b, slot]
                    ),
                    "state_rewrite_cell_rms": float(h_rewrite[b, slot]),
                    "state_rewrite_unit_rms": float(u_rewrite[b, slot]),
                    "wrong_cell_correction_rms": float(
                        wrong_correction[b, slot]
                    ),
                    "correct_cell_correction_rms": float(
                        correct_correction[b, slot]
                    ),
                    "pre_residual_mse": float(
                        pre_metrics["residual_mse"][flat_index]
                    ),
                    "rewrite_residual_mse": float(
                        rewrite_metrics["residual_mse"][flat_index]
                    ),
                    "post_residual_mse": float(
                        post_metrics["residual_mse"][flat_index]
                    ),
                    "pre_correct_margin": float(
                        pre_metrics["correct_margin"][flat_index]
                    ),
                    "rewrite_correct_margin": float(
                        rewrite_metrics["correct_margin"][flat_index]
                    ),
                    "post_correct_margin": float(
                        post_metrics["correct_margin"][flat_index]
                    ),
                    "post_hard_violations": int(
                        post_metrics["hard_violations"][flat_index]
                    ),
                    "post_exact": int(post_metrics["exact"][flat_index]),
                    "post_valid": int(post_metrics["valid"][flat_index]),
                }
                rows.append(row)
    anchor_metrics = state_metric_tensors(
        model, anchor_logits, puzzle, solution
    )
    summary = []
    post_valid = post_metrics["valid"].reshape(batch, slots)
    post_exact = post_metrics["exact"].reshape(batch, slots)
    for b in range(batch):
        summary.append(
            {
                "batch_index": b,
                "ablation": ablation,
                "anchor_exact": int(anchor_metrics["exact"][b]),
                "anchor_valid": int(anchor_metrics["valid"][b]),
                "first_valid_cycle": int(first_valid[b]),
                "final_any_valid": int(post_valid[b].any()),
                "final_any_exact": int(post_exact[b].any()),
                "final_best_residual_mse": float(
                    post_metrics["residual_mse"]
                    .reshape(batch, slots)[b]
                    .min()
                ),
                "final_best_hard_violations": int(
                    post_metrics["hard_violations"]
                    .reshape(batch, slots)[b]
                    .min()
                ),
            }
        )
    return rows, summary


def choose_message_target(anchor_logits, puzzle, solution):
    flat_pred = anchor_logits.argmax(dim=-1).reshape(81) + 1
    flat_solution = solution.reshape(81)
    mutable = puzzle.reshape(81) == 0
    wrong = torch.nonzero(
        mutable & (flat_pred != flat_solution), as_tuple=False
    ).squeeze(-1)
    if wrong.numel():
        cell = int(wrong[0])
        kind = "wrong"
    else:
        target = (flat_solution - 1).clamp(0, 8)
        flat = anchor_logits.reshape(81, 9)
        correct = flat.gather(-1, target[:, None]).squeeze(-1)
        alternatives = flat.masked_fill(
            F.one_hot(target, 9).bool(), -torch.inf
        ).amax(dim=-1)
        margin = (correct - alternatives).masked_fill(~mutable, torch.inf)
        cell = int(margin.argmin())
        kind = "low_margin_correct"
    positive = int(flat_solution[cell]) - 1
    negative = strongest_competitor(anchor_logits, cell, positive)
    return cell, positive, negative, kind


def message_records(model, gradients):
    records = []
    for index, value in enumerate(gradients["pair"].detach().cpu().tolist()):
        records.append(
            {
                "kind": "pair",
                "index": index,
                "source_cell": int(model.backbone.pair_src[index]),
                "target_cell": int(model.backbone.pair_dst[index]),
                "unit": -1,
                "edge_type": int(model.backbone.pair_type[index]),
                "attribution": float(value),
            }
        )
    for kind in ("v2u", "u2v"):
        values = gradients[kind].detach().cpu().tolist()
        for index, value in enumerate(values):
            records.append(
                {
                    "kind": kind,
                    "index": index,
                    "source_cell": int(model.backbone.member_cell[index]),
                    "target_cell": int(model.backbone.member_cell[index]),
                    "unit": int(model.backbone.member_unit[index]),
                    "edge_type": int(model.backbone.member_type[index]),
                    "attribution": float(value),
                }
            )
    return records


def masks_from_records(model, records, device):
    masks = {
        "pair": torch.ones(
            model.backbone.pair_src.numel(), device=device
        ),
        "v2u": torch.ones(
            model.backbone.member_cell.numel(), device=device
        ),
        "u2v": torch.ones(
            model.backbone.member_cell.numel(), device=device
        ),
    }
    for record in records:
        masks[record["kind"]][record["index"]] = 0.0
    return masks


def evaluate_final_step_masks(
    model,
    h,
    unit_h,
    x0,
    unit_x0,
    puzzle,
    masks,
):
    next_h, next_u, _ = hybrid_step_intervened(
        model.backbone,
        h,
        unit_h,
        x0,
        unit_x0,
        pair_mask=masks["pair"],
        v2u_mask=masks["v2u"],
        u2v_mask=masks["u2v"],
    )
    return model.backbone.logits_from_state(next_h, puzzle), next_h, next_u


def causal_message_target(
    model, puzzle, solution, target_meta, args, inference_step
):
    backbone = model.backbone
    x0 = backbone.input_features(puzzle)
    unit_x0 = backbone.initial_unit_features(x0)
    h, unit_h = x0, unit_x0
    collect_step = int(inference_step) - 1
    if collect_step < 0 or inference_step > int(model.cfg.parent_steps):
        raise ValueError(
            f"message inference_step={inference_step} is outside "
            f"1..{int(model.cfg.parent_steps)}"
        )
    for _ in range(collect_step):
        h, unit_h = backbone.step(h, unit_h, x0, unit_x0)
    anchor_h, anchor_u = h.detach(), unit_h.detach()
    full_masks = {
        "pair": torch.ones(
            backbone.pair_src.numel(), device=puzzle.device, requires_grad=True
        ),
        "v2u": torch.ones(
            backbone.member_cell.numel(),
            device=puzzle.device,
            requires_grad=True,
        ),
        "u2v": torch.ones(
            backbone.member_cell.numel(),
            device=puzzle.device,
            requires_grad=True,
        ),
    }
    logits, _, _ = evaluate_final_step_masks(
        model, h, unit_h, x0, unit_x0, puzzle, full_masks
    )
    cell, positive, negative, target_kind = choose_message_target(
        logits.detach(), puzzle[0], solution[0]
    )
    margin = fixed_margin(logits, cell, positive, negative).sum()
    grad_values = torch.autograd.grad(
        margin,
        [full_masks["pair"], full_masks["v2u"], full_masks["u2v"]],
    )
    gradients = dict(zip(MESSAGE_KINDS, grad_values))
    records = message_records(model, gradients)
    candidates = [
        row
        for row in records
        if (
            row["target_cell"] == cell
            or (
                row["kind"] == "v2u"
                and cell
                in torch.nonzero(
                    model.incidence[row["unit"]], as_tuple=False
                )
                .squeeze(-1)
                .tolist()
            )
        )
    ]
    if len(candidates) < args.message_top_k:
        candidates = records
    support = sorted(
        candidates, key=lambda row: row["attribution"], reverse=True
    )[: args.message_top_k]
    opposition = sorted(
        candidates, key=lambda row: row["attribution"]
    )[: args.message_top_k]
    rng = random.Random(
        args.seed + int(target_meta["split_position"]) * 1009
    )
    random_edges = rng.sample(candidates, min(args.message_top_k, len(candidates)))
    base_margin = float(margin.detach())
    intervention_rows = []
    for name, chosen in (
        ("remove_support", support),
        ("remove_opposition", opposition),
        ("remove_random", random_edges),
    ):
        masks = masks_from_records(model, chosen, puzzle.device)
        changed_logits, _, _ = evaluate_final_step_masks(
            model,
            anchor_h,
            anchor_u,
            x0,
            unit_x0,
            puzzle,
            masks,
        )
        changed_margin = float(
            fixed_margin(
                changed_logits, cell, positive, negative
            ).detach()
        )
        changed_pred = changed_logits.argmax(dim=-1) + 1
        intervention_rows.append(
            {
                **target_meta,
                "inference_step": int(inference_step),
                "target_kind": target_kind,
                "target_cell": cell,
                "target_row": cell // 9,
                "target_col": cell % 9,
                "positive_digit": positive + 1,
                "negative_digit": negative + 1,
                "method": name,
                "k": len(chosen),
                "base_margin": base_margin,
                "changed_margin": changed_margin,
                "delta_margin": changed_margin - base_margin,
                "target_prediction_after": int(
                    changed_pred.reshape(81)[cell]
                ),
                "board_exact_after": int(
                    (changed_pred == solution).reshape(-1).all()
                ),
                "hard_violations_after": int(
                    hard_violation_count_batch(changed_pred)[0]
                ),
            }
        )
    top_records = sorted(
        records, key=lambda row: abs(row["attribution"]), reverse=True
    )[: args.save_top_edges]
    attribution_rows = [
        {
            **target_meta,
            "inference_step": int(inference_step),
            "target_kind": target_kind,
            "target_cell": cell,
            "positive_digit": positive + 1,
            "negative_digit": negative + 1,
            "rank_abs": rank + 1,
            **record,
        }
        for rank, record in enumerate(top_records)
    ]
    return attribution_rows, intervention_rows


def summarize_c5(rows):
    grouped = {}
    for row in rows:
        key = row["ablation"]
        grouped.setdefault(key, []).append(row)
    result = {}
    for key, values in grouped.items():
        result[key] = {
            "n": len(values),
            "solve_rate": float(
                np.mean([row["first_valid_cycle"] >= 0 for row in values])
            ),
            "final_any_exact": float(
                np.mean([row["final_any_exact"] for row in values])
            ),
            "mean_first_valid_cycle_solved": float(
                np.mean(
                    [
                        row["first_valid_cycle"]
                        for row in values
                        if row["first_valid_cycle"] >= 0
                    ]
                )
            )
            if any(row["first_valid_cycle"] >= 0 for row in values)
            else float("nan"),
            "mean_best_residual_mse": float(
                np.mean(
                    [row["final_best_residual_mse"] for row in values]
                )
            ),
        }
    return result


def stratified_message_targets(targets, limit):
    """Round-robin target groups so an early group cannot exhaust the quota."""
    buckets = {
        group: [row for row in targets if row["group"] == group]
        for group in GROUPS
    }
    selected = []
    offset = 0
    while len(selected) < limit:
        added = False
        for group in GROUPS:
            if offset < len(buckets[group]):
                selected.append(buckets[group][offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reflection_checkpoint", required=True)
    parser.add_argument(
        "--cache_path",
        default="data/cache_full_3m.npz",
    )
    parser.add_argument("--split", choices=sorted(SPLITS), default="val")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--strict_min_rating", type=float, default=4.0)
    parser.add_argument("--scan_limit", type=int, default=4096)
    parser.add_argument("--scan_batch_size", type=int, default=16)
    parser.add_argument("--quota_per_group", type=int, default=16)
    parser.add_argument("--save_states", type=int, default=1)
    parser.add_argument("--message_targets", type=int, default=16)
    parser.add_argument("--message_top_k", type=int, default=5)
    parser.add_argument("--save_top_edges", type=int, default=40)
    parser.add_argument("--collect_step", type=int, default=63)
    parser.add_argument(
        "--message_steps",
        default="",
        help=(
            "Comma-separated 1-based Hybrid inference steps. When empty, "
            "uses collect_step + 1 for backward compatibility."
        ),
    )
    parser.add_argument(
        "--ablations",
        default=(
            "full,zero_residual,shuffle_unit,shuffle_digit,sign_flip,"
            "stale_residual,no_hard,no_dual,reset_dual,flat_attention,"
            "shuffle_attention,no_base,no_symbolic,no_cell,no_unit,"
            "no_recovery,recovery_only"
        ),
    )
    return parser.parse_args()


def main(args):
    set_torch_threads()
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg, checkpoint = load_symbolic(
        args.reflection_checkpoint, device
    )
    dataset = load_sudoku_dataset(
        args.cache_path, split=SPLITS[args.split], limit=0
    )
    started = time.time()
    targets, found_counts, scanned = scan_target_groups(
        model, dataset, args, device
    )
    if not targets:
        raise RuntimeError("Target scan returned no puzzles")
    write_csv(out_dir / "targets.csv", targets)

    trajectory_rows, state_payload = collect_hybrid_trajectories(
        model, dataset, targets, device, bool(args.save_states)
    )
    write_csv(out_dir / "hybrid_trajectory.csv", trajectory_rows)
    if args.save_states:
        np.savez_compressed(out_dir / "hybrid_states.npz", **state_payload)

    positions = [row["split_position"] for row in targets]
    puzzle, solution, _, _ = as_tensor_batch(dataset, positions, device)
    ablations = [
        value.strip() for value in args.ablations.split(",") if value.strip()
    ]
    c5_cycle_rows = []
    c5_summary_rows = []
    for ablation in ablations:
        print(f"[c5] ablation={ablation}", flush=True)
        cycle_rows, summary_rows = run_c5_ablation(
            model, puzzle, solution, ablation
        )
        for row in cycle_rows:
            target = targets[row.pop("batch_index")]
            row.update(
                {
                    "split_position": target["split_position"],
                    "group": target["group"],
                    "rating": target["rating"],
                    "clues": target["clues"],
                }
            )
        for row in summary_rows:
            target = targets[row.pop("batch_index")]
            row.update(
                {
                    "split_position": target["split_position"],
                    "group": target["group"],
                    "rating": target["rating"],
                    "clues": target["clues"],
                }
            )
        c5_cycle_rows.extend(cycle_rows)
        c5_summary_rows.extend(summary_rows)
    write_csv(out_dir / "c5_cycle_trace.csv", c5_cycle_rows)
    write_csv(out_dir / "c5_ablation_summary.csv", c5_summary_rows)

    attribution_rows = []
    intervention_rows = []
    message_target_rows = stratified_message_targets(
        targets, args.message_targets
    )
    message_steps = (
        [
            int(value.strip())
            for value in args.message_steps.split(",")
            if value.strip()
        ]
        if args.message_steps
        else [int(args.collect_step) + 1]
    )
    completed_message_runs = 0
    total_message_runs = len(message_target_rows) * len(message_steps)
    for target in message_target_rows:
        one_puzzle, one_solution, _, _ = as_tensor_batch(
            dataset, [target["split_position"]], device
        )
        for inference_step in message_steps:
            current_attr, current_int = causal_message_target(
                model,
                one_puzzle,
                one_solution,
                target,
                args,
                inference_step,
            )
            attribution_rows.extend(current_attr)
            intervention_rows.extend(current_int)
            completed_message_runs += 1
            print(
                f"[message] {completed_message_runs}/{total_message_runs} "
                f"group={target['group']} "
                f"position={target['split_position']} step={inference_step}",
                flush=True,
            )
    write_csv(out_dir / "message_attribution.csv", attribution_rows)
    write_csv(out_dir / "message_interventions.csv", intervention_rows)

    summary = {
        "reflection_checkpoint": args.reflection_checkpoint,
        "base_checkpoint": checkpoint.get("base_checkpoint"),
        "split": args.split,
        "strict_min_rating": args.strict_min_rating,
        "scan_limit": args.scan_limit,
        "scanned": scanned,
        "found_counts": found_counts,
        "selected_counts": {
            group: sum(row["group"] == group for row in targets)
            for group in GROUPS
        },
        "targets": len(targets),
        "hybrid_steps": int(cfg.parent_steps),
        "c5_slots": int(cfg.slots),
        "c5_cycles": int(cfg.cycles),
        "c5_recovery_steps": int(cfg.recovery_steps),
        "message_steps": message_steps,
        "ablations": ablations,
        "c5_ablation": summarize_c5(c5_summary_rows),
        "elapsed_seconds": time.time() - started,
    }
    if intervention_rows:
        for method in ("remove_support", "remove_opposition", "remove_random"):
            values = [
                row["delta_margin"]
                for row in intervention_rows
                if row["method"] == method
            ]
            summary.setdefault("message_interventions", {})[method] = {
                "n": len(values),
                "mean_delta_margin": float(np.mean(values)),
                "median_delta_margin": float(np.median(values)),
            }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(parse_args())
