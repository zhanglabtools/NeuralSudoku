#!/usr/bin/env python3
"""Shared Hybrid+C5 utilities for exact-two-solution explainability."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from eval_hybrid_hyper_rrn_restarts import hard_violation_count_batch


CLASS_INVALID = 0
CLASS_A = 1
CLASS_B = 2
CLASS_OTHER_VALID = 3
CLASS_NAMES = {
    CLASS_INVALID: "invalid_or_unsolved",
    CLASS_A: "solution_a",
    CLASS_B: "solution_b",
    CLASS_OTHER_VALID: "other_valid",
}


@dataclass(frozen=True)
class DualSolutionDataset:
    puzzles: np.ndarray
    solutions_a: np.ndarray
    solutions_b: np.ndarray
    diff_masks: np.ndarray
    controls_a: np.ndarray | None
    controls_b: np.ndarray | None
    source_index: np.ndarray
    metadata: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.puzzles)


def load_dual_dataset(path: str | Path, *, limit: int = 0) -> DualSolutionDataset:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    with np.load(path, allow_pickle=False) as data:
        required = {"puzzles", "solutions_a", "solutions_b"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Missing dual-solution dataset keys: {sorted(missing)}")
        has_control_a = "controls_a" in data.files
        has_control_b = "controls_b" in data.files
        if has_control_a != has_control_b:
            raise KeyError("controls_a and controls_b must either both be present or both absent")

        puzzle_array = np.asarray(data["puzzles"])
        if puzzle_array.ndim != 3 or puzzle_array.shape[1:] != (9, 9):
            raise ValueError(
                f"puzzles must have exact shape N x 9 x 9, got {puzzle_array.shape}"
            )
        total = len(puzzle_array)
        take = total if not limit else min(limit, total)

        def grid_array(name: str, dtype) -> np.ndarray:
            raw = np.asarray(data[name])
            if raw.shape != (total, 9, 9):
                raise ValueError(
                    f"{name} must have exact shape {(total, 9, 9)}, got {raw.shape}"
                )
            if not np.issubdtype(raw.dtype, np.integer) and raw.dtype != np.bool_:
                raise TypeError(f"{name} must contain integer-valued grids")
            if name in {"solutions_a", "solutions_b"}:
                lower, upper = 1, 9
            elif name == "diff_masks":
                lower, upper = 0, 1
            else:
                lower, upper = 0, 9
            if np.any((raw < lower) | (raw > upper)):
                raise ValueError(
                    f"{name} entries must lie in {lower}..{upper} before casting"
                )
            return np.asarray(raw[:take], dtype=dtype)

        puzzles = grid_array("puzzles", np.uint8)
        solutions_a = grid_array("solutions_a", np.uint8)
        solutions_b = grid_array("solutions_b", np.uint8)
        diff_masks = (
            grid_array("diff_masks", bool)
            if "diff_masks" in data.files
            else solutions_a != solutions_b
        )
        controls_a = grid_array("controls_a", np.uint8) if has_control_a else None
        controls_b = grid_array("controls_b", np.uint8) if has_control_b else None
        if "source_index" in data.files:
            raw_source_index = np.asarray(data["source_index"])
            if raw_source_index.shape != (total,):
                raise ValueError(
                    f"source_index must have exact shape {(total,)}, "
                    f"got {raw_source_index.shape}"
                )
            source_index = np.asarray(raw_source_index[:take], dtype=np.int64)
        else:
            source_index = np.arange(take, dtype=np.int64)
        excluded = {
            "puzzles",
            "solutions_a",
            "solutions_b",
            "diff_masks",
            "controls_a",
            "controls_b",
            "source_index",
        }
        metadata = {
            key: np.asarray(data[key][:take])
            for key in data.files
            if key not in excluded
        }
    dataset = DualSolutionDataset(
        puzzles=puzzles,
        solutions_a=solutions_a,
        solutions_b=solutions_b,
        diff_masks=diff_masks,
        controls_a=controls_a,
        controls_b=controls_b,
        source_index=source_index,
        metadata=metadata,
    )
    validate_dual_dataset(dataset)
    return dataset


def validate_dual_dataset(dataset: DualSolutionDataset) -> None:
    n = len(dataset)
    expected_grid_shape = (n, 9, 9)
    for name, value in (
        ("puzzles", dataset.puzzles),
        ("solutions_a", dataset.solutions_a),
        ("solutions_b", dataset.solutions_b),
        ("diff_masks", dataset.diff_masks),
    ):
        if np.asarray(value).shape != expected_grid_shape:
            raise ValueError(
                f"{name} must have exact shape {expected_grid_shape}, "
                f"got {np.asarray(value).shape}"
            )
    if np.asarray(dataset.source_index).shape != (n,):
        raise ValueError(
            f"source_index must have exact shape {(n,)}, "
            f"got {np.asarray(dataset.source_index).shape}"
        )
    if (dataset.controls_a is None) != (dataset.controls_b is None):
        raise ValueError("controls_a and controls_b must both be present or both be absent")
    for name, value in (
        ("controls_a", dataset.controls_a),
        ("controls_b", dataset.controls_b),
    ):
        if value is not None and np.asarray(value).shape != expected_grid_shape:
            raise ValueError(
                f"{name} must have exact shape {expected_grid_shape}, "
                f"got {np.asarray(value).shape}"
            )

    puzzles = np.asarray(dataset.puzzles)
    solutions_a = np.asarray(dataset.solutions_a)
    solutions_b = np.asarray(dataset.solutions_b)
    if np.any((puzzles < 0) | (puzzles > 9)):
        raise AssertionError("Puzzle entries must lie in 0..9")
    for label, solutions in (("A", solutions_a), ("B", solutions_b)):
        if np.any((solutions < 1) | (solutions > 9)):
            raise AssertionError(f"Solution {label} entries must lie in 1..9")
        target = np.arange(1, 10, dtype=solutions.dtype)
        row_valid = (
            np.sort(solutions, axis=2) == target[None, None, :]
        ).all(axis=(1, 2))
        column_valid = (
            np.sort(solutions, axis=1) == target[None, :, None]
        ).all(axis=(1, 2))
        boxes = (
            solutions.reshape(n, 3, 3, 3, 3)
            .transpose(0, 1, 3, 2, 4)
            .reshape(n, 9, 9)
        )
        box_valid = (
            np.sort(boxes, axis=2) == target[None, None, :]
        ).all(axis=(1, 2))
        invalid = np.flatnonzero(~(row_valid & column_valid & box_valid))
        if invalid.size:
            raise AssertionError(
                f"Solution {label} was not a legal Sudoku at rows "
                f"{invalid[:8].tolist()}"
            )

    distinct = (solutions_a != solutions_b).reshape(n, -1).any(axis=1)
    if not distinct.all():
        raise AssertionError(
            "A and B must be distinct for every puzzle; equal rows "
            f"{np.flatnonzero(~distinct)[:8].tolist()}"
        )
    actual_diff = dataset.solutions_a != dataset.solutions_b
    if not np.array_equal(actual_diff, dataset.diff_masks):
        raise AssertionError("diff_masks did not equal solutions_a != solutions_b")
    clues = dataset.puzzles > 0
    if not np.all(dataset.puzzles[clues] == dataset.solutions_a[clues]):
        raise AssertionError("A solution violated a clue")
    if not np.all(dataset.puzzles[clues] == dataset.solutions_b[clues]):
        raise AssertionError("B solution violated a clue")
    if np.any(dataset.diff_masks & clues):
        raise AssertionError("A clue appeared in the A/B difference set")

    def validate_control(
        control: np.ndarray, own_solution: np.ndarray, other_solution: np.ndarray, label: str
    ) -> None:
        values = np.asarray(control)
        if np.any((values < 0) | (values > 9)):
            raise AssertionError(f"Control {label} entries must lie in 0..9")
        if not np.array_equal(values[clues], puzzles[clues]):
            raise AssertionError(f"Control {label} did not preserve every base clue")
        control_clues = values > 0
        if not np.all(values[control_clues] == own_solution[control_clues]):
            raise AssertionError(
                f"Control {label} contained a clue inconsistent with solution {label}"
            )
        excludes_other = (
            control_clues & (values != other_solution)
        ).reshape(n, -1).any(axis=1)
        if not excludes_other.all():
            raise AssertionError(
                f"Control {label} did not exclude the other certified solution at rows "
                f"{np.flatnonzero(~excludes_other)[:8].tolist()}"
            )

    if dataset.controls_a is not None and dataset.controls_b is not None:
        validate_control(dataset.controls_a, solutions_a, solutions_b, "A")
        validate_control(dataset.controls_b, solutions_b, solutions_a, "B")


def swap_ab_view(dataset: DualSolutionDataset) -> DualSolutionDataset:
    return DualSolutionDataset(
        puzzles=dataset.puzzles,
        solutions_a=dataset.solutions_b,
        solutions_b=dataset.solutions_a,
        diff_masks=dataset.diff_masks,
        controls_a=dataset.controls_b,
        controls_b=dataset.controls_a,
        source_index=dataset.source_index,
        metadata=dataset.metadata,
    )


def flatten_c5_candidates(output: dict[str, torch.Tensor]) -> torch.Tensor:
    anchor = output["anchor_logits"][:, None]
    cycles = output["cycle_logits"]
    batch = cycles.shape[0]
    return torch.cat([anchor, cycles.reshape(batch, -1, 9, 9, 9)], dim=1)


@torch.no_grad()
def c5_trace_from_parent(
    model,
    puzzle: torch.Tensor,
    x0: torch.Tensor,
    unit_x0: torch.Tensor,
    parent_h: torch.Tensor,
    parent_u: torch.Tensor,
    anchor_logits: torch.Tensor,
) -> dict[str, object]:
    """Run the unchanged C5 recurrence and attach exact applied-update diagnostics.

    The production symbolic reflector does not return its two symbolic gate
    activations.  Temporary forward hooks observe the pre-sigmoid linear outputs;
    hooks never replace outputs and are removed in ``finally`` on both success and
    failure.  The returned ``*_effect`` tensors are therefore the exact additive
    updates applied before recovery rollout, while the original delta/gate fields
    remain available for backwards-compatible raw-proposal analyses.
    """
    source_batch = puzzle.shape[0]
    slots = int(model.cfg.slots)
    h = model._expand_slots(parent_h, slots)
    unit_h = model._expand_slots(parent_u, slots)
    slot_x0 = model._expand_slots(x0, slots)
    slot_unit_x0 = model._expand_slots(unit_x0, slots)
    slot_puzzle = model._expand_slots(puzzle, slots)
    dual = h.new_zeros(source_batch * slots, 27, 9)
    cycle_logits: list[torch.Tensor] = []
    cycle_diagnostics: list[dict[str, torch.Tensor]] = []
    symbolic_cell_gates: list[torch.Tensor] = []
    symbolic_unit_gates: list[torch.Tensor] = []

    def capture_gate(storage: list[torch.Tensor]):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise TypeError("A symbolic gate hook received a non-tensor output")
            storage.append(torch.sigmoid(output))

        return hook

    handles = []
    try:
        handles.append(
            model.symbolic_cell_gate.register_forward_hook(
                capture_gate(symbolic_cell_gates)
            )
        )
        handles.append(
            model.symbolic_unit_gate.register_forward_hook(
                capture_gate(symbolic_unit_gates)
            )
        )
        for cycle_index in range(int(model.cfg.cycles)):
            cell_gate_count = len(symbolic_cell_gates)
            unit_gate_count = len(symbolic_unit_gates)
            h, unit_h, dual = h.detach(), unit_h.detach(), dual.detach()
            h, unit_h, dual, native_diagnostics = model._symbolic_reflect_once(
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
            if len(symbolic_cell_gates) != cell_gate_count + 1:
                raise RuntimeError(
                    "Expected exactly one symbolic-cell gate call per C5 cycle"
                )
            if len(symbolic_unit_gates) != unit_gate_count + 1:
                raise RuntimeError(
                    "Expected exactly one symbolic-unit gate call per C5 cycle"
                )
            diagnostics = dict(native_diagnostics)
            symbolic_cell_gate = symbolic_cell_gates[-1]
            symbolic_unit_gate = symbolic_unit_gates[-1]
            if symbolic_cell_gate.shape[:-1] != diagnostics[
                "symbolic_cell_delta"
            ].shape[:-1]:
                raise RuntimeError("Captured symbolic-cell gate had an invalid shape")
            if symbolic_unit_gate.shape[:-1] != diagnostics[
                "symbolic_unit_delta"
            ].shape[:-1]:
                raise RuntimeError("Captured symbolic-unit gate had an invalid shape")

            total = source_batch * slots
            mutable = (slot_puzzle.reshape(total, 81) == 0).to(h.dtype)
            local_scores = diagnostics.get("local_scores")
            if local_scores is None:
                base_unit_mask = unit_h.new_ones(total, 27)
                base_cell_mask = h.new_ones(total, 81)
            else:
                if model.training:
                    raise RuntimeError(
                        "Exact local-mode effect instrumentation requires model.eval(); "
                        "training-mode Gumbel choices are not returned by the model"
                    )
                local_index = local_scores.argmax(dim=-1)
                base_unit_mask = F.one_hot(local_index, num_classes=27).to(h.dtype)
                base_cell_mask = torch.matmul(
                    base_unit_mask, model.incidence.to(h.dtype)
                ).clamp_max(1.0)
            base_cell_mask = base_cell_mask * mutable
            attention = diagnostics["constraint_attention"]
            attention_scale = (
                float(model.cfg.attention_floor) + 27.0 * attention
            ).clamp_max(float(model.cfg.attention_cap))

            base_cell_effect_raw = diagnostics["cell_gate"] * diagnostics["cell_delta"]
            base_unit_effect_raw = diagnostics["unit_gate"] * diagnostics["unit_delta"]
            symbolic_cell_effect_raw = (
                symbolic_cell_gate * diagnostics["symbolic_cell_delta"]
            )
            symbolic_unit_effect_raw = (
                symbolic_unit_gate * diagnostics["symbolic_unit_delta"]
            )
            diagnostics.update(
                {
                    "symbolic_cell_gate": symbolic_cell_gate,
                    "symbolic_unit_gate": symbolic_unit_gate,
                    "attention_scale": attention_scale,
                    "base_cell_effect_raw": base_cell_effect_raw,
                    "base_unit_effect_raw": base_unit_effect_raw,
                    "symbolic_cell_delta_raw": diagnostics[
                        "symbolic_cell_delta"
                    ],
                    "symbolic_unit_delta_raw": diagnostics[
                        "symbolic_unit_delta"
                    ],
                    "symbolic_cell_effect_raw": symbolic_cell_effect_raw,
                    "symbolic_unit_effect_raw": symbolic_unit_effect_raw,
                    "base_cell_effect": float(model.cfg.correction_scale)
                    * base_cell_mask[:, :, None]
                    * base_cell_effect_raw,
                    "base_unit_effect": float(model.cfg.unit_correction_scale)
                    * base_unit_mask[:, :, None]
                    * base_unit_effect_raw,
                    "symbolic_cell_effect": float(model.cfg.symbolic_cell_scale)
                    * mutable[:, :, None]
                    * symbolic_cell_effect_raw,
                    "symbolic_unit_effect": float(model.cfg.symbolic_unit_scale)
                    * attention_scale[:, :, None]
                    * symbolic_unit_effect_raw,
                }
            )
            h, unit_h = model._rollout(
                h,
                unit_h,
                slot_x0,
                slot_unit_x0,
                model.cfg.recovery_steps,
            )
            logits = model.backbone.logits_from_state(h, slot_puzzle)
            cycle_logits.append(logits.reshape(source_batch, slots, 9, 9, 9))
            cycle_diagnostics.append(diagnostics)
    finally:
        for handle in handles:
            handle.remove()
    cycles = torch.stack(cycle_logits, dim=1)
    return {
        "anchor_logits": anchor_logits,
        "slot_logits": cycles[:, -1],
        "cycle_logits": cycles,
        "cycle_diagnostics": cycle_diagnostics,
        "final_cell_state": h,
        "final_unit_state": unit_h,
    }


def candidate_coordinates(candidate_count: int, slots: int) -> tuple[np.ndarray, np.ndarray]:
    stage = np.zeros(candidate_count, dtype=np.int16)
    slot = np.full(candidate_count, -1, dtype=np.int16)
    for index in range(1, candidate_count):
        stage[index] = (index - 1) // slots + 1
        slot[index] = (index - 1) % slots
    return stage, slot


def classify_candidate_tensor(
    predictions: torch.Tensor,
    puzzles: torch.Tensor,
    solutions_a: torch.Tensor,
    solutions_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify BxCx9x9 predictions into invalid/A/B/other-valid."""
    if predictions.ndim == 3:
        predictions = predictions[:, None]
    batch, candidates = predictions.shape[:2]
    flat = predictions.reshape(batch * candidates, 9, 9)
    sudoku_valid = (hard_violation_count_batch(flat) == 0).reshape(batch, candidates)
    clue_ok = (
        (puzzles[:, None] == 0) | (predictions == puzzles[:, None])
    ).reshape(batch, candidates, -1).all(dim=-1)
    valid = sudoku_valid & clue_ok
    match_a = (predictions == solutions_a[:, None]).reshape(
        batch, candidates, -1
    ).all(dim=-1)
    match_b = (predictions == solutions_b[:, None]).reshape(
        batch, candidates, -1
    ).all(dim=-1)
    classes = torch.zeros_like(valid, dtype=torch.int8)
    classes[valid] = CLASS_OTHER_VALID
    classes[valid & match_a] = CLASS_A
    classes[valid & match_b] = CLASS_B
    return classes, valid, clue_ok


def branch_observables(
    logits: torch.Tensor,
    solutions_a: torch.Tensor,
    solutions_b: torch.Tensor,
    diff_masks: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute A/B observables for Bx...x9x9x9 logits.

    A candidate dimension is optional.  Unlike the historical implementation,
    targets are explicitly expanded over it, so all 41 C5 candidates receive a
    separate preference margin.
    """
    squeeze_candidate = logits.ndim == 4
    if squeeze_candidate:
        logits = logits[:, None]
    if logits.ndim != 5:
        raise ValueError(f"Expected BxCx9x9x9 logits, got {tuple(logits.shape)}")
    target_a = solutions_a[:, None].expand(-1, logits.shape[1], -1, -1)
    target_b = solutions_b[:, None].expand(-1, logits.shape[1], -1, -1)
    mask = diff_masks[:, None].to(logits.dtype)
    probability = F.softmax(logits, dim=-1)
    log_probability = F.log_softmax(logits, dim=-1)
    a_probability = probability.gather(-1, (target_a - 1).unsqueeze(-1)).squeeze(-1)
    b_probability = probability.gather(-1, (target_b - 1).unsqueeze(-1)).squeeze(-1)
    a_log = log_probability.gather(-1, (target_a - 1).unsqueeze(-1)).squeeze(-1)
    b_log = log_probability.gather(-1, (target_b - 1).unsqueeze(-1)).squeeze(-1)
    count = mask.sum(dim=(-1, -2)).clamp_min(1.0)
    log_score_a = (a_log * mask).sum(dim=(-1, -2))
    log_score_b = (b_log * mask).sum(dim=(-1, -2))
    mean_delta = (log_score_a - log_score_b) / count
    branch_probability_a = torch.sigmoid(mean_delta)
    branch_entropy = -(
        branch_probability_a * torch.log2(branch_probability_a.clamp_min(1e-12))
        + (1.0 - branch_probability_a)
        * torch.log2((1.0 - branch_probability_a).clamp_min(1e-12))
    )
    top_two = logits.topk(2, dim=-1).values
    cell_gap = top_two[..., 0] - top_two[..., 1]
    diff_gap = torch.where(mask.bool(), cell_gap, torch.full_like(cell_gap, torch.inf))
    result = {
        "log_score_a": log_score_a,
        "log_score_b": log_score_b,
        "preference_delta": log_score_a - log_score_b,
        "mean_preference_delta": mean_delta,
        "branch_probability_a": branch_probability_a,
        "branch_entropy": branch_entropy,
        "pair_mass": ((a_probability + b_probability) * mask).sum(dim=(-1, -2)) / count,
        "mean_a_probability": (a_probability * mask).sum(dim=(-1, -2)) / count,
        "mean_b_probability": (b_probability * mask).sum(dim=(-1, -2)) / count,
        "minimum_cell_logit_gap": cell_gap.amin(dim=(-1, -2)),
        "minimum_diff_logit_gap": diff_gap.amin(dim=(-1, -2)),
    }
    if squeeze_candidate:
        result = {key: value[:, 0] for key, value in result.items()}
    return result


def deploy_first_valid(
    classes: np.ndarray,
    *,
    slots: int,
    slot_order: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct anchor-first, cycle-major, first-valid deployment."""
    values = np.asarray(classes, dtype=np.int8)
    if values.ndim != 2:
        raise ValueError("classes must be N x candidate_count")
    candidate_count = values.shape[1]
    if (candidate_count - 1) % slots:
        raise ValueError("candidate_count was not 1 + cycles * slots")
    cycles = (candidate_count - 1) // slots
    order = list(range(slots)) if slot_order is None else [int(x) for x in slot_order]
    if sorted(order) != list(range(slots)):
        raise ValueError("slot_order must be a permutation of range(slots)")
    scan = [0]
    for cycle in range(cycles):
        scan.extend(1 + cycle * slots + slot for slot in order)
    deployed_class = np.zeros(len(values), dtype=np.int8)
    deployed_index = np.full(len(values), -1, dtype=np.int16)
    unresolved = np.ones(len(values), dtype=bool)
    for index in scan:
        accept = unresolved & (values[:, index] > CLASS_INVALID)
        deployed_class[accept] = values[accept, index]
        deployed_index[accept] = index
        unresolved[accept] = False
    return deployed_class, deployed_index


def candidate_prediction_hashes(predictions: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.uint8)
    if values.ndim != 4:
        raise ValueError("predictions must be N x C x 9 x 9")
    result = np.empty(values.shape[:2], dtype="S32")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            result[row, col] = hashlib.blake2b(
                values[row, col].tobytes(), digest_size=16
            ).hexdigest().encode("ascii")
    return result


def touching_unit_masks(diff_masks: torch.Tensor) -> torch.Tensor:
    masks = torch.zeros(
        len(diff_masks), 27, dtype=torch.bool, device=diff_masks.device
    )
    for batch_index in range(len(diff_masks)):
        cells = torch.nonzero(
            diff_masks[batch_index].reshape(-1), as_tuple=False
        ).reshape(-1)
        for flat in cells.tolist():
            row, col = divmod(int(flat), 9)
            box = (row // 3) * 3 + col // 3
            masks[batch_index, row] = True
            masks[batch_index, 9 + col] = True
            masks[batch_index, 18 + box] = True
    return masks


def deterministic_matched_masks(
    eligible: torch.Tensor,
    counts: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """Choose deterministic matched controls without using global RNG state."""
    result = torch.zeros_like(eligible, dtype=torch.bool)
    for row in range(len(eligible)):
        candidates = torch.nonzero(eligible[row], as_tuple=False).reshape(-1)
        count = int(counts[row])
        if len(candidates) < count:
            raise ValueError("Not enough eligible entries for matched controls")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(keys[row].detach().cpu()))
        order = torch.randperm(len(candidates), generator=generator)[:count]
        chosen = candidates.detach().cpu()[order].to(eligible.device)
        result[row, chosen] = True
    return result


def deterministic_type_matched_unit_masks(
    touch_units: torch.Tensor, keys: torch.Tensor
) -> torch.Tensor:
    """Match untouched controls separately within rows, columns, and boxes."""
    if touch_units.ndim != 2 or touch_units.shape[1] != 27:
        raise ValueError("touch_units must have shape B x 27")
    if keys.shape != (len(touch_units),):
        raise ValueError("keys must have shape B")
    result = torch.zeros_like(touch_units, dtype=torch.bool)
    for unit_type, start in enumerate((0, 9, 18)):
        stop = start + 9
        touched_type = touch_units[:, start:stop]
        selected = deterministic_matched_masks(
            ~touched_type,
            touched_type.sum(dim=1),
            keys + 1_000_003 * (unit_type + 1),
        )
        result[:, start:stop] = selected
    return result


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask_float = mask.to(values.dtype)
    numerator = (values * mask_float).sum(dim=dims)
    denominator = mask_float.expand_as(values).sum(dim=dims).clamp_min(1.0)
    return numerator / denominator


def aggregate_cycle_diagnostics(
    diagnostics: dict[str, torch.Tensor],
    *,
    batch_size: int,
    slots: int,
    puzzles: torch.Tensor,
    solutions_a: torch.Tensor,
    diff_masks: torch.Tensor,
    source_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Reduce one C5 cycle's high-dimensional diagnostics to BxS scalars."""
    diff_cells = diff_masks.reshape(batch_size, 81)
    mutable_non_diff = (puzzles.reshape(batch_size, 81) == 0) & (~diff_cells)
    diff_counts = diff_cells.sum(dim=1)
    control_cells = deterministic_matched_masks(
        mutable_non_diff, diff_counts, source_indices + 17
    )
    touch_units = touching_unit_masks(diff_masks)
    control_units = deterministic_type_matched_unit_masks(
        touch_units, source_indices + 29
    )
    trade_digits = torch.zeros(
        batch_size, 9, dtype=torch.bool, device=puzzles.device
    )
    for row in range(batch_size):
        digits = solutions_a[row][diff_masks[row]].long() - 1
        trade_digits[row, digits] = True

    def expand(mask: torch.Tensor) -> torch.Tensor:
        return mask[:, None].expand(-1, slots, -1).reshape(batch_size * slots, -1)

    diff_expanded = expand(diff_cells)
    control_cell_expanded = expand(control_cells)
    touch_expanded = expand(touch_units)
    control_unit_expanded = expand(control_units)
    digit_expanded = expand(trade_digits)

    required_effects = {
        "base_cell_effect",
        "base_unit_effect",
        "symbolic_cell_effect",
        "symbolic_unit_effect",
    }
    missing_effects = required_effects - set(diagnostics)
    if missing_effects:
        raise KeyError(
            "aggregate_cycle_diagnostics requires instrumented actual effects; "
            f"missing {sorted(missing_effects)}. Use c5_trace_from_parent."
        )
    base_cell_raw = diagnostics.get(
        "base_cell_effect_raw", diagnostics["cell_gate"] * diagnostics["cell_delta"]
    )
    base_unit_raw = diagnostics.get(
        "base_unit_effect_raw", diagnostics["unit_gate"] * diagnostics["unit_delta"]
    )
    symbolic_cell_delta_raw = diagnostics.get(
        "symbolic_cell_delta_raw", diagnostics["symbolic_cell_delta"]
    )
    symbolic_unit_delta_raw = diagnostics.get(
        "symbolic_unit_delta_raw", diagnostics["symbolic_unit_delta"]
    )
    symbolic_cell_effect_raw = diagnostics.get(
        "symbolic_cell_effect_raw", symbolic_cell_delta_raw
    )
    symbolic_unit_effect_raw = diagnostics.get(
        "symbolic_unit_effect_raw", symbolic_unit_delta_raw
    )

    def rms_norm(value: torch.Tensor) -> torch.Tensor:
        return value.square().mean(dim=-1).sqrt()

    base_cell_norm = rms_norm(diagnostics["base_cell_effect"])
    base_unit_norm = rms_norm(diagnostics["base_unit_effect"])
    symbolic_cell_effect_norm = rms_norm(diagnostics["symbolic_cell_effect"])
    symbolic_unit_effect_norm = rms_norm(diagnostics["symbolic_unit_effect"])
    base_cell_raw_norm = rms_norm(base_cell_raw)
    base_unit_raw_norm = rms_norm(base_unit_raw)
    symbolic_cell_delta_raw_norm = rms_norm(symbolic_cell_delta_raw)
    symbolic_unit_delta_raw_norm = rms_norm(symbolic_unit_delta_raw)
    symbolic_cell_effect_raw_norm = rms_norm(symbolic_cell_effect_raw)
    symbolic_unit_effect_raw_norm = rms_norm(symbolic_unit_effect_raw)
    attention = diagnostics["constraint_attention"]
    soft_abs = diagnostics["soft_residual"].abs()
    hard_abs = diagnostics["hard_residual"].abs()
    dual_abs = diagnostics["dual"].abs()

    touch_trade = touch_expanded[:, :, None] & digit_expanded[:, None, :]
    touch_nontrade = touch_expanded[:, :, None] & (~digit_expanded[:, None, :])
    control_trade = control_unit_expanded[:, :, None] & digit_expanded[:, None, :]
    control_nontrade = control_unit_expanded[:, :, None] & (
        ~digit_expanded[:, None, :]
    )

    def reshape(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(batch_size, slots)

    result = {
        "base_cell_effect_diff": reshape(
            _masked_mean(base_cell_norm, diff_expanded, (1,))
        ),
        "base_cell_effect_control": reshape(
            _masked_mean(base_cell_norm, control_cell_expanded, (1,))
        ),
        "base_cell_effect_actual_diff": reshape(
            _masked_mean(base_cell_norm, diff_expanded, (1,))
        ),
        "base_cell_effect_actual_control": reshape(
            _masked_mean(base_cell_norm, control_cell_expanded, (1,))
        ),
        "base_cell_effect_raw_diff": reshape(
            _masked_mean(base_cell_raw_norm, diff_expanded, (1,))
        ),
        "base_cell_effect_raw_control": reshape(
            _masked_mean(base_cell_raw_norm, control_cell_expanded, (1,))
        ),
        "symbolic_cell_effect_diff": reshape(
            _masked_mean(symbolic_cell_effect_norm, diff_expanded, (1,))
        ),
        "symbolic_cell_effect_control": reshape(
            _masked_mean(symbolic_cell_effect_norm, control_cell_expanded, (1,))
        ),
        "symbolic_cell_effect_actual_diff": reshape(
            _masked_mean(symbolic_cell_effect_norm, diff_expanded, (1,))
        ),
        "symbolic_cell_effect_actual_control": reshape(
            _masked_mean(symbolic_cell_effect_norm, control_cell_expanded, (1,))
        ),
        "symbolic_cell_effect_raw_diff": reshape(
            _masked_mean(symbolic_cell_effect_raw_norm, diff_expanded, (1,))
        ),
        "symbolic_cell_effect_raw_control": reshape(
            _masked_mean(symbolic_cell_effect_raw_norm, control_cell_expanded, (1,))
        ),
        "symbolic_cell_delta_diff": reshape(
            _masked_mean(symbolic_cell_delta_raw_norm, diff_expanded, (1,))
        ),
        "symbolic_cell_delta_control": reshape(
            _masked_mean(symbolic_cell_delta_raw_norm, control_cell_expanded, (1,))
        ),
        "symbolic_cell_delta_raw_diff": reshape(
            _masked_mean(symbolic_cell_delta_raw_norm, diff_expanded, (1,))
        ),
        "symbolic_cell_delta_raw_control": reshape(
            _masked_mean(symbolic_cell_delta_raw_norm, control_cell_expanded, (1,))
        ),
        "base_unit_effect_touch": reshape(
            _masked_mean(base_unit_norm, touch_expanded, (1,))
        ),
        "base_unit_effect_control": reshape(
            _masked_mean(base_unit_norm, control_unit_expanded, (1,))
        ),
        "base_unit_effect_actual_touch": reshape(
            _masked_mean(base_unit_norm, touch_expanded, (1,))
        ),
        "base_unit_effect_actual_control": reshape(
            _masked_mean(base_unit_norm, control_unit_expanded, (1,))
        ),
        "base_unit_effect_raw_touch": reshape(
            _masked_mean(base_unit_raw_norm, touch_expanded, (1,))
        ),
        "base_unit_effect_raw_control": reshape(
            _masked_mean(base_unit_raw_norm, control_unit_expanded, (1,))
        ),
        "symbolic_unit_effect_touch": reshape(
            _masked_mean(symbolic_unit_effect_norm, touch_expanded, (1,))
        ),
        "symbolic_unit_effect_control": reshape(
            _masked_mean(symbolic_unit_effect_norm, control_unit_expanded, (1,))
        ),
        "symbolic_unit_effect_actual_touch": reshape(
            _masked_mean(symbolic_unit_effect_norm, touch_expanded, (1,))
        ),
        "symbolic_unit_effect_actual_control": reshape(
            _masked_mean(symbolic_unit_effect_norm, control_unit_expanded, (1,))
        ),
        "symbolic_unit_effect_raw_touch": reshape(
            _masked_mean(symbolic_unit_effect_raw_norm, touch_expanded, (1,))
        ),
        "symbolic_unit_effect_raw_control": reshape(
            _masked_mean(symbolic_unit_effect_raw_norm, control_unit_expanded, (1,))
        ),
        "symbolic_unit_delta_touch": reshape(
            _masked_mean(symbolic_unit_delta_raw_norm, touch_expanded, (1,))
        ),
        "symbolic_unit_delta_control": reshape(
            _masked_mean(symbolic_unit_delta_raw_norm, control_unit_expanded, (1,))
        ),
        "symbolic_unit_delta_raw_touch": reshape(
            _masked_mean(symbolic_unit_delta_raw_norm, touch_expanded, (1,))
        ),
        "symbolic_unit_delta_raw_control": reshape(
            _masked_mean(symbolic_unit_delta_raw_norm, control_unit_expanded, (1,))
        ),
        "attention_touch_mass": reshape(
            (attention * touch_expanded.to(attention.dtype)).sum(dim=1)
        ),
        "attention_touch_excess": reshape(
            (attention * touch_expanded.to(attention.dtype)).sum(dim=1)
            - touch_expanded.to(attention.dtype).mean(dim=1)
        ),
    }
    for prefix, values in (
        ("soft_residual", soft_abs),
        ("hard_residual", hard_abs),
        ("dual", dual_abs),
    ):
        touch_trade_value = reshape(_masked_mean(values, touch_trade, (1, 2)))
        touch_nontrade_value = reshape(
            _masked_mean(values, touch_nontrade, (1, 2))
        )
        control_trade_value = reshape(
            _masked_mean(values, control_trade, (1, 2))
        )
        control_nontrade_value = reshape(
            _masked_mean(values, control_nontrade, (1, 2))
        )
        result.update(
            {
                f"{prefix}_touch_trade": touch_trade_value,
                f"{prefix}_touch_nontrade": touch_nontrade_value,
                f"{prefix}_matched_control_trade": control_trade_value,
                f"{prefix}_matched_control_nontrade": control_nontrade_value,
                # Historical names remain aliases for downstream compatibility.
                f"{prefix}_trade_touch": touch_trade_value,
                f"{prefix}_control": control_nontrade_value,
            }
        )
    return result
