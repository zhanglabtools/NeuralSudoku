#!/usr/bin/env python3
"""Audit external-isomorphism equivariance of the current Hybrid+C5 solver.

This experiment applies invertible *external* standard-Sudoku isomorphisms
``g: P -> gP``.  It does not claim that ``g`` fixes a puzzle, and therefore it
does not test membership in an instance stabilizer ``Aut(P)``.

For every family/repeat the script runs all 41 Hybrid+C5 candidates on the
transformed puzzles, maps every predicted grid back with ``g^{-1}``, and
compares it with the candidate at the same anchor/cycle/slot index on the
original puzzle.  Both index-wise and unordered-set comparisons are retained.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from eval_symbolic_active_reflection import load_symbolic
from hybrid_dual_explainability_core import (
    CLASS_A,
    CLASS_B,
    CLASS_INVALID,
    CLASS_OTHER_VALID,
    branch_observables,
    candidate_coordinates,
    candidate_prediction_hashes,
    classify_candidate_tensor,
    deploy_first_valid,
    flatten_c5_candidates,
    load_dual_dataset,
)
from sudoku_exchange_experiment import set_seed, set_torch_threads


FAMILIES = (
    "identity",
    "digit",
    "row",
    "column",
    "transpose",
    "composite",
)
SEED_MODULUS = (1 << 63) - 25


@dataclass(frozen=True)
class SudokuIsomorphism:
    """One invertible standard 9x9 Sudoku transformation.

    Forward convention matches ``eval_hybrid_c5_two_solution.transform_grid``:
    optional transpose, output-to-input row/column indexing, then digit relabel.
    """

    family: str
    seed: int
    row_order: np.ndarray
    col_order: np.ndarray
    digit_map: np.ndarray
    transpose: bool

    def __post_init__(self) -> None:
        row = np.asarray(self.row_order, dtype=np.int64)
        col = np.asarray(self.col_order, dtype=np.int64)
        digit = np.asarray(self.digit_map, dtype=np.int64)
        if row.shape != (9,) or sorted(row.tolist()) != list(range(9)):
            raise ValueError("row_order must be a permutation of range(9)")
        if col.shape != (9,) or sorted(col.tolist()) != list(range(9)):
            raise ValueError("col_order must be a permutation of range(9)")
        if digit.shape != (10,) or digit[0] != 0:
            raise ValueError("digit_map must have shape (10,) and fix zero")
        if sorted(digit[1:].tolist()) != list(range(1, 10)):
            raise ValueError("digit_map[1:] must permute digits 1..9")

    @property
    def inverse_row_order(self) -> np.ndarray:
        return np.argsort(self.row_order)

    @property
    def inverse_col_order(self) -> np.ndarray:
        return np.argsort(self.col_order)

    @property
    def inverse_digit_map(self) -> np.ndarray:
        inverse = np.empty(10, dtype=np.uint8)
        inverse[np.asarray(self.digit_map, dtype=np.int64)] = np.arange(
            10, dtype=np.uint8
        )
        return inverse

    def apply(self, grids: np.ndarray) -> np.ndarray:
        values = np.asarray(grids)
        if values.shape[-2:] != (9, 9):
            raise ValueError(f"Expected trailing 9x9 grid axes, got {values.shape}")
        result = np.swapaxes(values, -2, -1) if self.transpose else values
        result = result[..., np.asarray(self.row_order), :]
        result = result[..., :, np.asarray(self.col_order)]
        return np.asarray(self.digit_map, dtype=np.uint8)[result]

    def inverse(self, grids: np.ndarray) -> np.ndarray:
        values = np.asarray(grids)
        if values.shape[-2:] != (9, 9):
            raise ValueError(f"Expected trailing 9x9 grid axes, got {values.shape}")
        result = self.inverse_digit_map[values]
        result = result[..., self.inverse_row_order, :]
        result = result[..., :, self.inverse_col_order]
        if self.transpose:
            result = np.swapaxes(result, -2, -1)
        return result


def _nonidentity_digit_map(rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(9) + 1
    if np.array_equal(order, np.arange(1, 10)):
        order[[0, 1]] = order[[1, 0]]
    result = np.zeros(10, dtype=np.uint8)
    result[1:] = order
    return result


def _legal_axis_order(rng: np.random.Generator) -> np.ndarray:
    groups = rng.permutation(3)
    order = np.concatenate(
        [3 * group + rng.permutation(3) for group in groups]
    ).astype(np.int8)
    if np.array_equal(order, np.arange(9)):
        order[[0, 1]] = order[[1, 0]]
    return order


def transform_seed(
    seed: int, family_index: int, repeat: int, sample_index: int, source_index: int
) -> int:
    value = (
        int(seed)
        + 100_000_007 * (family_index + 1)
        + 10_000_019 * (repeat + 1)
        + 1_000_003 * (int(source_index) + 1)
        + 97_003 * (sample_index + 1)
    )
    return int(value % SEED_MODULUS)


def make_isomorphism(family: str, seed: int) -> SudokuIsomorphism:
    if family not in FAMILIES:
        raise ValueError(f"Unknown transformation family {family!r}")
    rng = np.random.default_rng(int(seed))
    identity_axis = np.arange(9, dtype=np.int8)
    identity_digit = np.arange(10, dtype=np.uint8)
    row_order = identity_axis.copy()
    col_order = identity_axis.copy()
    digit_map = identity_digit.copy()
    transpose = False
    if family == "identity":
        pass
    elif family == "digit":
        digit_map = _nonidentity_digit_map(rng)
    elif family == "row":
        row_order = _legal_axis_order(rng)
    elif family == "column":
        col_order = _legal_axis_order(rng)
    elif family == "transpose":
        transpose = True
    elif family == "composite":
        row_order = _legal_axis_order(rng)
        col_order = _legal_axis_order(rng)
        digit_map = _nonidentity_digit_map(rng)
        transpose = True
    return SudokuIsomorphism(
        family=family,
        seed=int(seed),
        row_order=row_order,
        col_order=col_order,
        digit_map=digit_map,
        transpose=transpose,
    )


def build_transformed_batch(
    *,
    family: str,
    family_index: int,
    repeat: int,
    seed: int,
    source_indices: np.ndarray,
    puzzles: np.ndarray,
    solutions_a: np.ndarray,
    solutions_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[SudokuIsomorphism]]:
    transformed_puzzles = np.empty_like(puzzles)
    transformed_a = np.empty_like(solutions_a)
    transformed_b = np.empty_like(solutions_b)
    transforms: list[SudokuIsomorphism] = []
    for index, source_index in enumerate(source_indices):
        current_seed = transform_seed(
            seed, family_index, repeat, index, int(source_index)
        )
        transform = make_isomorphism(family, current_seed)
        transformed_puzzles[index] = transform.apply(puzzles[index])
        transformed_a[index] = transform.apply(solutions_a[index])
        transformed_b[index] = transform.apply(solutions_b[index])
        if not np.array_equal(transform.inverse(transformed_puzzles[index]), puzzles[index]):
            raise AssertionError("Puzzle isomorphism round-trip failed")
        if not np.array_equal(transform.inverse(transformed_a[index]), solutions_a[index]):
            raise AssertionError("Solution-A isomorphism round-trip failed")
        if not np.array_equal(transform.inverse(transformed_b[index]), solutions_b[index]):
            raise AssertionError("Solution-B isomorphism round-trip failed")
        clue_mask = transformed_puzzles[index] > 0
        if not np.array_equal(
            transformed_puzzles[index][clue_mask], transformed_a[index][clue_mask]
        ):
            raise AssertionError("Transformed A violated a transformed clue")
        if not np.array_equal(
            transformed_puzzles[index][clue_mask], transformed_b[index][clue_mask]
        ):
            raise AssertionError("Transformed B violated a transformed clue")
        transforms.append(transform)
    return transformed_puzzles, transformed_a, transformed_b, transforms


@torch.no_grad()
def run_c5_candidates(
    model,
    *,
    puzzles: np.ndarray,
    solutions_a: np.ndarray,
    solutions_b: np.ndarray,
    batch_size: int,
    device: torch.device,
    label: str,
) -> dict[str, np.ndarray]:
    predictions: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    clue_masks: list[np.ndarray] = []
    preference: list[np.ndarray] = []
    n = len(puzzles)
    started = time.perf_counter()
    model.eval()
    for offset in range(0, n, batch_size):
        stop = min(offset + batch_size, n)
        puzzle = torch.as_tensor(
            puzzles[offset:stop], dtype=torch.long, device=device
        )
        solution_a = torch.as_tensor(
            solutions_a[offset:stop], dtype=torch.long, device=device
        )
        solution_b = torch.as_tensor(
            solutions_b[offset:stop], dtype=torch.long, device=device
        )
        diff = solution_a != solution_b
        output = model(puzzle, include_continuation=False)
        logits = flatten_c5_candidates(output)
        prediction = logits.argmax(dim=-1) + 1
        current_class, valid, clue_ok = classify_candidate_tensor(
            prediction, puzzle, solution_a, solution_b
        )
        observables = branch_observables(
            logits, solution_a, solution_b, diff
        )
        predictions.append(prediction.to(torch.uint8).cpu().numpy())
        classes.append(current_class.cpu().numpy().astype(np.int8))
        valid_masks.append(valid.cpu().numpy())
        clue_masks.append(clue_ok.cpu().numpy())
        preference.append(
            observables["preference_delta"].float().cpu().numpy()
        )
        if stop == n or stop % max(256, batch_size) < batch_size:
            print(
                f"[c5:{label}] {stop}/{n} elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    result = {
        "predictions": np.concatenate(predictions, axis=0),
        "classes": np.concatenate(classes, axis=0),
        "valid": np.concatenate(valid_masks, axis=0),
        "clue_ok": np.concatenate(clue_masks, axis=0),
        "preference_delta": np.concatenate(preference, axis=0).astype(np.float32),
        "elapsed_seconds": np.asarray(time.perf_counter() - started),
    }
    expected = 1 + int(model.cfg.cycles) * int(model.cfg.slots)
    if result["predictions"].shape[1] != expected:
        raise AssertionError("Unexpected C5 candidate count")
    if expected != 41:
        raise AssertionError(
            f"This audit requires the current 1+5x8=41 C5 candidates, got {expected}"
        )
    return result


@torch.no_grad()
def classify_numpy_candidates(
    predictions: np.ndarray,
    puzzles: np.ndarray,
    solutions_a: np.ndarray,
    solutions_b: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    classes: list[np.ndarray] = []
    validity: list[np.ndarray] = []
    for offset in range(0, len(predictions), batch_size):
        stop = min(offset + batch_size, len(predictions))
        pred = torch.as_tensor(
            predictions[offset:stop], dtype=torch.long, device=device
        )
        puzzle = torch.as_tensor(
            puzzles[offset:stop], dtype=torch.long, device=device
        )
        solution_a = torch.as_tensor(
            solutions_a[offset:stop], dtype=torch.long, device=device
        )
        solution_b = torch.as_tensor(
            solutions_b[offset:stop], dtype=torch.long, device=device
        )
        current_class, valid, _ = classify_candidate_tensor(
            pred, puzzle, solution_a, solution_b
        )
        classes.append(current_class.cpu().numpy().astype(np.int8))
        validity.append(valid.cpu().numpy())
    return np.concatenate(classes), np.concatenate(validity)


def inverse_candidate_batch(
    predictions: np.ndarray, transforms: list[SudokuIsomorphism]
) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.empty_like(predictions)
    roundtrip = np.ones(len(predictions), dtype=bool)
    for index, transform in enumerate(transforms):
        inverse[index] = transform.inverse(predictions[index])
        roundtrip[index] = np.array_equal(
            transform.apply(inverse[index]), predictions[index]
        )
    if not roundtrip.all():
        raise AssertionError("Candidate-grid isomorphism round-trip failed")
    return inverse, roundtrip


def deployed_grids(
    predictions: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    result = np.zeros((len(predictions), 9, 9), dtype=np.uint8)
    available = indices >= 0
    rows = np.flatnonzero(available)
    result[rows] = predictions[rows, indices[rows]]
    return result


def _grid_set(grids: np.ndarray, mask: np.ndarray | None = None) -> set[bytes]:
    values = grids if mask is None else grids[np.asarray(mask, dtype=bool)]
    return {np.asarray(grid, dtype=np.uint8).tobytes() for grid in values}


def _jaccard(left: set[bytes], right: set[bytes]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def set_metrics(
    base_predictions: np.ndarray,
    base_valid: np.ndarray,
    base_classes: np.ndarray,
    inverse_predictions: np.ndarray,
    inverse_valid: np.ndarray,
    inverse_classes: np.ndarray,
) -> dict[str, np.ndarray]:
    n = len(base_predictions)
    candidate_jaccard = np.empty(n, dtype=np.float32)
    valid_jaccard = np.empty(n, dtype=np.float32)
    valid_set_equal = np.empty(n, dtype=bool)
    certified_closed = np.empty(n, dtype=bool)
    semantic_branch_set_equal = np.empty(n, dtype=bool)
    for index in range(n):
        base_set = _grid_set(base_predictions[index])
        inverse_set = _grid_set(inverse_predictions[index])
        base_legal = _grid_set(base_predictions[index], base_valid[index])
        inverse_legal = _grid_set(inverse_predictions[index], inverse_valid[index])
        candidate_jaccard[index] = _jaccard(base_set, inverse_set)
        valid_jaccard[index] = _jaccard(base_legal, inverse_legal)
        valid_set_equal[index] = base_legal == inverse_legal
        certified_closed[index] = not np.any(
            base_classes[index] == CLASS_OTHER_VALID
        ) and not np.any(inverse_classes[index] == CLASS_OTHER_VALID)
        base_branches = set(
            base_classes[index][
                np.isin(base_classes[index], (CLASS_A, CLASS_B))
            ].tolist()
        )
        inverse_branches = set(
            inverse_classes[index][
                np.isin(inverse_classes[index], (CLASS_A, CLASS_B))
            ].tolist()
        )
        semantic_branch_set_equal[index] = base_branches == inverse_branches
    return {
        "candidate_set_jaccard": candidate_jaccard,
        "valid_set_jaccard": valid_jaccard,
        "valid_set_equal": valid_set_equal,
        "certified_solution_closed": certified_closed,
        "semantic_branch_set_equal": semantic_branch_set_equal,
    }


def safe_rate(values: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    array = np.asarray(values)
    if mask is not None:
        array = array[np.asarray(mask, dtype=bool)]
    return float(array.mean()) if array.size else None


def safe_float(value: float | np.floating) -> float:
    return float(np.asarray(value))


def summarize_slice(
    *,
    family: str,
    repeat: int | str,
    candidate_exact: np.ndarray,
    candidate_class_equal: np.ndarray,
    candidate_valid_equal: np.ndarray,
    base_candidate_valid: np.ndarray,
    transformed_candidate_valid: np.ndarray,
    preference_residual: np.ndarray,
    set_values: dict[str, np.ndarray],
    deployed_grid_exact: np.ndarray,
    deployed_class_equal: np.ndarray,
    deployed_index_equal: np.ndarray,
    base_deployed_class: np.ndarray,
    transformed_deployed_class: np.ndarray,
    roundtrip: np.ndarray,
    inverse_class_equal: np.ndarray,
) -> dict[str, Any]:
    candidate_union_valid = base_candidate_valid | transformed_candidate_valid
    candidate_base_valid = np.asarray(base_candidate_valid, dtype=bool)
    candidate_base_invalid = ~candidate_base_valid
    both_valid = (base_deployed_class > CLASS_INVALID) & (
        transformed_deployed_class > CLASS_INVALID
    )
    both_branch = np.isin(base_deployed_class, (CLASS_A, CLASS_B)) & np.isin(
        transformed_deployed_class, (CLASS_A, CLASS_B)
    )
    residual = np.asarray(preference_residual, dtype=np.float64)
    return {
        "family": family,
        "repeat": repeat,
        "n_puzzles": int(len(candidate_exact)),
        "candidate_count": int(candidate_exact.shape[-1]),
        "candidate_index_exact_rate": safe_rate(candidate_exact),
        "all_41_candidates_exact_rate": safe_rate(candidate_exact.all(axis=-1)),
        "candidate_semantic_class_rate": safe_rate(candidate_class_equal),
        "candidate_validity_rate": safe_rate(candidate_valid_equal),
        "candidate_exact_rate_union_valid": safe_rate(
            candidate_exact, candidate_union_valid
        ),
        "candidate_semantic_class_rate_union_valid": safe_rate(
            candidate_class_equal, candidate_union_valid
        ),
        "candidate_valid_to_invalid_rate": safe_rate(
            ~transformed_candidate_valid, candidate_base_valid
        ),
        "candidate_invalid_to_valid_rate": safe_rate(
            transformed_candidate_valid, candidate_base_invalid
        ),
        "mean_candidate_set_jaccard": safe_float(
            set_values["candidate_set_jaccard"].mean()
        ),
        "mean_valid_set_jaccard": safe_float(set_values["valid_set_jaccard"].mean()),
        "valid_set_equal_rate": safe_rate(set_values["valid_set_equal"]),
        "certified_solution_closure_rate": safe_rate(
            set_values["certified_solution_closed"]
        ),
        "semantic_branch_set_equal_rate": safe_rate(
            set_values["semantic_branch_set_equal"]
        ),
        "deployed_grid_exact_rate_all": safe_rate(deployed_grid_exact),
        "deployed_grid_exact_rate_both_valid": safe_rate(
            deployed_grid_exact, both_valid
        ),
        "deployed_semantic_class_rate": safe_rate(deployed_class_equal),
        "deployed_candidate_index_rate": safe_rate(deployed_index_equal),
        "ab_semantic_branch_preservation": safe_rate(
            base_deployed_class == transformed_deployed_class, both_branch
        ),
        "both_deployed_valid_n": int(both_valid.sum()),
        "both_deployed_ab_n": int(both_branch.sum()),
        "preference_residual_mean": safe_float(residual.mean()),
        "preference_residual_mean_abs": safe_float(np.abs(residual).mean()),
        "preference_residual_rms": safe_float(np.sqrt(np.mean(residual**2))),
        "preference_residual_max_abs": safe_float(np.abs(residual).max()),
        "strict_roundtrip_rate": safe_rate(roundtrip),
        "inverse_classification_rate": safe_rate(inverse_class_equal),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main(args: argparse.Namespace) -> None:
    set_torch_threads()
    set_seed(args.seed)
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = load_dual_dataset(args.dataset, limit=args.limit)
    if not len(dataset):
        raise ValueError("The selected dual-solution dataset is empty")
    model, cfg, checkpoint = load_symbolic(args.checkpoint, device)
    model.eval()
    slots = int(cfg.slots)
    cycles = int(cfg.cycles)
    candidate_count = 1 + slots * cycles
    if (slots, cycles, candidate_count) != (8, 5, 41):
        raise AssertionError(
            f"Expected current C5 slots=8 cycles=5 candidates=41, got "
            f"slots={slots} cycles={cycles} candidates={candidate_count}"
        )

    base = run_c5_candidates(
        model,
        puzzles=dataset.puzzles,
        solutions_a=dataset.solutions_a,
        solutions_b=dataset.solutions_b,
        batch_size=args.batch_size,
        device=device,
        label="original",
    )
    base_deployed_class, base_deployed_index = deploy_first_valid(
        base["classes"], slots=slots
    )
    base_deployed_grid = deployed_grids(
        base["predictions"], base_deployed_index
    )
    base_hashes = candidate_prediction_hashes(base["predictions"])

    family_count = len(FAMILIES)
    shape_candidate = (family_count, args.repeats, len(dataset), candidate_count)
    shape_puzzle = (family_count, args.repeats, len(dataset))
    candidate_exact = np.empty(shape_candidate, dtype=bool)
    candidate_class_equal = np.empty(shape_candidate, dtype=bool)
    candidate_valid_equal = np.empty(shape_candidate, dtype=bool)
    transformed_candidate_class = np.empty(shape_candidate, dtype=np.int8)
    transformed_candidate_valid = np.empty(shape_candidate, dtype=bool)
    preference_residual = np.empty(shape_candidate, dtype=np.float32)
    inverse_hashes = np.empty(shape_candidate, dtype="S32")
    candidate_set_jaccard = np.empty(shape_puzzle, dtype=np.float32)
    valid_set_jaccard = np.empty(shape_puzzle, dtype=np.float32)
    valid_set_equal = np.empty(shape_puzzle, dtype=bool)
    certified_solution_closed = np.empty(shape_puzzle, dtype=bool)
    semantic_branch_set_equal = np.empty(shape_puzzle, dtype=bool)
    deployed_grid_exact = np.empty(shape_puzzle, dtype=bool)
    deployed_class_equal = np.empty(shape_puzzle, dtype=bool)
    deployed_index_equal = np.empty(shape_puzzle, dtype=bool)
    transformed_deployed_class = np.empty(shape_puzzle, dtype=np.int8)
    transformed_deployed_index = np.empty(shape_puzzle, dtype=np.int16)
    inverse_deployed_predictions = np.empty(
        shape_puzzle + (9, 9), dtype=np.uint8
    )
    strict_roundtrip = np.empty(shape_puzzle, dtype=bool)
    inverse_classification_equal = np.empty(shape_puzzle, dtype=bool)
    transformation_seeds = np.empty(shape_puzzle, dtype=np.int64)
    row_orders = np.empty(shape_puzzle + (9,), dtype=np.int8)
    col_orders = np.empty(shape_puzzle + (9,), dtype=np.int8)
    digit_maps = np.empty(shape_puzzle + (10,), dtype=np.uint8)
    transpose_flags = np.empty(shape_puzzle, dtype=bool)

    repeat_rows: list[dict[str, Any]] = []
    family_collections: dict[str, list[dict[str, np.ndarray]]] = {
        family: [] for family in FAMILIES
    }
    for family_index, family in enumerate(FAMILIES):
        for repeat in range(args.repeats):
            transformed_puzzle, transformed_a, transformed_b, transforms = (
                build_transformed_batch(
                    family=family,
                    family_index=family_index,
                    repeat=repeat,
                    seed=args.seed,
                    source_indices=dataset.source_index,
                    puzzles=dataset.puzzles,
                    solutions_a=dataset.solutions_a,
                    solutions_b=dataset.solutions_b,
                )
            )
            transformed = run_c5_candidates(
                model,
                puzzles=transformed_puzzle,
                solutions_a=transformed_a,
                solutions_b=transformed_b,
                batch_size=args.batch_size,
                device=device,
                label=f"{family}:r{repeat}",
            )
            inverse_predictions, roundtrip = inverse_candidate_batch(
                transformed["predictions"], transforms
            )
            inverse_classes, inverse_valid = classify_numpy_candidates(
                inverse_predictions,
                dataset.puzzles,
                dataset.solutions_a,
                dataset.solutions_b,
                batch_size=args.batch_size,
                device=device,
            )
            if not np.array_equal(inverse_classes, transformed["classes"]):
                raise AssertionError(
                    "Semantic class changed under an asserted inverse isomorphism"
                )
            if not np.array_equal(inverse_valid, transformed["valid"]):
                raise AssertionError(
                    "Candidate validity changed under an asserted inverse isomorphism"
                )

            current_exact = (
                inverse_predictions == base["predictions"]
            ).reshape(len(dataset), candidate_count, -1).all(axis=-1)
            current_class_equal = inverse_classes == base["classes"]
            current_valid_equal = inverse_valid == base["valid"]
            current_preference_residual = (
                transformed["preference_delta"] - base["preference_delta"]
            ).astype(np.float32)
            current_sets = set_metrics(
                base["predictions"],
                base["valid"],
                base["classes"],
                inverse_predictions,
                inverse_valid,
                inverse_classes,
            )
            deployed_class, deployed_index = deploy_first_valid(
                transformed["classes"], slots=slots
            )
            inverse_deployed = deployed_grids(
                inverse_predictions, deployed_index
            )
            current_deployed_exact = (
                inverse_deployed == base_deployed_grid
            ).reshape(len(dataset), -1).all(axis=1)
            current_deployed_exact &= (base_deployed_index >= 0) & (
                deployed_index >= 0
            )

            destination = (family_index, repeat)
            candidate_exact[destination] = current_exact
            candidate_class_equal[destination] = current_class_equal
            candidate_valid_equal[destination] = current_valid_equal
            transformed_candidate_class[destination] = inverse_classes
            transformed_candidate_valid[destination] = inverse_valid
            preference_residual[destination] = current_preference_residual
            inverse_hashes[destination] = candidate_prediction_hashes(
                inverse_predictions
            )
            candidate_set_jaccard[destination] = current_sets[
                "candidate_set_jaccard"
            ]
            valid_set_jaccard[destination] = current_sets["valid_set_jaccard"]
            valid_set_equal[destination] = current_sets["valid_set_equal"]
            certified_solution_closed[destination] = current_sets[
                "certified_solution_closed"
            ]
            semantic_branch_set_equal[destination] = current_sets[
                "semantic_branch_set_equal"
            ]
            deployed_grid_exact[destination] = current_deployed_exact
            deployed_class_equal[destination] = (
                deployed_class == base_deployed_class
            )
            deployed_index_equal[destination] = (
                deployed_index == base_deployed_index
            )
            transformed_deployed_class[destination] = deployed_class
            transformed_deployed_index[destination] = deployed_index
            inverse_deployed_predictions[destination] = inverse_deployed
            strict_roundtrip[destination] = roundtrip
            inverse_classification_equal[destination] = (
                (inverse_classes == transformed["classes"]).all(axis=1)
                & (inverse_valid == transformed["valid"]).all(axis=1)
            )
            for index, transform in enumerate(transforms):
                transformation_seeds[destination + (index,)] = transform.seed
                row_orders[destination + (index,)] = transform.row_order
                col_orders[destination + (index,)] = transform.col_order
                digit_maps[destination + (index,)] = transform.digit_map
                transpose_flags[destination + (index,)] = transform.transpose

            row = summarize_slice(
                family=family,
                repeat=repeat,
                candidate_exact=current_exact,
                candidate_class_equal=current_class_equal,
                candidate_valid_equal=current_valid_equal,
                base_candidate_valid=base["valid"],
                transformed_candidate_valid=inverse_valid,
                preference_residual=current_preference_residual,
                set_values=current_sets,
                deployed_grid_exact=current_deployed_exact,
                deployed_class_equal=deployed_class == base_deployed_class,
                deployed_index_equal=deployed_index == base_deployed_index,
                base_deployed_class=base_deployed_class,
                transformed_deployed_class=deployed_class,
                roundtrip=roundtrip,
                inverse_class_equal=inverse_classification_equal[destination],
            )
            repeat_rows.append(row)
            family_collections[family].append(
                {
                    "candidate_exact": current_exact,
                    "candidate_class_equal": current_class_equal,
                    "candidate_valid_equal": current_valid_equal,
                    "base_candidate_valid": base["valid"],
                    "transformed_candidate_valid": inverse_valid,
                    "preference_residual": current_preference_residual,
                    "candidate_set_jaccard": current_sets["candidate_set_jaccard"],
                    "valid_set_jaccard": current_sets["valid_set_jaccard"],
                    "valid_set_equal": current_sets["valid_set_equal"],
                    "certified_solution_closed": current_sets[
                        "certified_solution_closed"
                    ],
                    "semantic_branch_set_equal": current_sets[
                        "semantic_branch_set_equal"
                    ],
                    "deployed_grid_exact": current_deployed_exact,
                    "deployed_class_equal": deployed_class == base_deployed_class,
                    "deployed_index_equal": deployed_index == base_deployed_index,
                    "transformed_deployed_class": deployed_class,
                    "roundtrip": roundtrip,
                    "inverse_class_equal": inverse_classification_equal[destination],
                }
            )

    family_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    stages, candidate_slots = candidate_coordinates(candidate_count, slots)
    for family_index, family in enumerate(FAMILIES):
        family_exact = candidate_exact[family_index].reshape(
            args.repeats * len(dataset), candidate_count
        )
        family_class_equal = candidate_class_equal[family_index].reshape(
            args.repeats * len(dataset), candidate_count
        )
        family_valid_equal = candidate_valid_equal[family_index].reshape(
            args.repeats * len(dataset), candidate_count
        )
        family_residual = preference_residual[family_index].reshape(
            args.repeats * len(dataset), candidate_count
        )
        collection = family_collections[family]
        merged_sets = {
            key: np.concatenate([item[key] for item in collection])
            for key in (
                "candidate_set_jaccard",
                "valid_set_jaccard",
                "valid_set_equal",
                "certified_solution_closed",
                "semantic_branch_set_equal",
            )
        }
        merged_deployed_class = np.concatenate(
            [item["transformed_deployed_class"] for item in collection]
        )
        repeated_base_deployed = np.tile(base_deployed_class, args.repeats)
        family_rows.append(
            summarize_slice(
                family=family,
                repeat="all",
                candidate_exact=family_exact,
                candidate_class_equal=family_class_equal,
                candidate_valid_equal=family_valid_equal,
                base_candidate_valid=np.tile(base["valid"], (args.repeats, 1)),
                transformed_candidate_valid=np.concatenate(
                    [item["transformed_candidate_valid"] for item in collection]
                ),
                preference_residual=family_residual,
                set_values=merged_sets,
                deployed_grid_exact=np.concatenate(
                    [item["deployed_grid_exact"] for item in collection]
                ),
                deployed_class_equal=np.concatenate(
                    [item["deployed_class_equal"] for item in collection]
                ),
                deployed_index_equal=np.concatenate(
                    [item["deployed_index_equal"] for item in collection]
                ),
                base_deployed_class=repeated_base_deployed,
                transformed_deployed_class=merged_deployed_class,
                roundtrip=np.concatenate(
                    [item["roundtrip"] for item in collection]
                ),
                inverse_class_equal=np.concatenate(
                    [item["inverse_class_equal"] for item in collection]
                ),
            )
        )
        for candidate_index in range(candidate_count):
            residual = family_residual[:, candidate_index].astype(np.float64)
            candidate_rows.append(
                {
                    "family": family,
                    "candidate_index": candidate_index,
                    "stage": int(stages[candidate_index]),
                    "slot": int(candidate_slots[candidate_index]),
                    "n": int(len(family_exact)),
                    "exact_equivariance_rate": safe_rate(
                        family_exact[:, candidate_index]
                    ),
                    "semantic_class_rate": safe_rate(
                        family_class_equal[:, candidate_index]
                    ),
                    "validity_rate": safe_rate(
                        family_valid_equal[:, candidate_index]
                    ),
                    "preference_residual_mean": safe_float(residual.mean()),
                    "preference_residual_mean_abs": safe_float(
                        np.abs(residual).mean()
                    ),
                    "preference_residual_rms": safe_float(
                        np.sqrt(np.mean(residual**2))
                    ),
                    "preference_residual_max_abs": safe_float(
                        np.abs(residual).max()
                    ),
                }
            )

    details_path = output_dir / "equivariance_details.npz"
    np.savez_compressed(
        details_path,
        source_index=dataset.source_index,
        family_names=np.asarray(FAMILIES, dtype="U16"),
        base_candidate_predictions=base["predictions"],
        base_candidate_hashes=base_hashes,
        base_candidate_class=base["classes"],
        base_candidate_valid=base["valid"],
        base_preference_delta=base["preference_delta"],
        base_deployed_class=base_deployed_class,
        base_deployed_index=base_deployed_index,
        base_deployed_prediction=base_deployed_grid,
        candidate_exact_equivariance=candidate_exact,
        candidate_semantic_class_equal=candidate_class_equal,
        candidate_validity_equal=candidate_valid_equal,
        transformed_candidate_class=transformed_candidate_class,
        transformed_candidate_valid=transformed_candidate_valid,
        inverse_candidate_hashes=inverse_hashes,
        preference_delta_residual=preference_residual,
        candidate_set_jaccard=candidate_set_jaccard,
        valid_set_jaccard=valid_set_jaccard,
        valid_set_equal=valid_set_equal,
        certified_solution_closed=certified_solution_closed,
        semantic_branch_set_equal=semantic_branch_set_equal,
        deployed_grid_exact=deployed_grid_exact,
        deployed_semantic_class_equal=deployed_class_equal,
        deployed_candidate_index_equal=deployed_index_equal,
        transformed_deployed_class=transformed_deployed_class,
        transformed_deployed_index=transformed_deployed_index,
        inverse_deployed_predictions=inverse_deployed_predictions,
        strict_roundtrip=strict_roundtrip,
        inverse_classification_equal=inverse_classification_equal,
        transformation_seed=transformation_seeds,
        row_order=row_orders,
        col_order=col_orders,
        digit_map=digit_maps,
        transpose=transpose_flags,
        candidate_stage=stages,
        candidate_slot=candidate_slots,
    )
    repeat_csv = output_dir / "repeat_summary.csv"
    family_csv = output_dir / "family_summary.csv"
    candidate_csv = output_dir / "candidate_index_summary.csv"
    write_csv(repeat_csv, repeat_rows)
    write_csv(family_csv, family_rows)
    write_csv(candidate_csv, candidate_rows)

    overall_residual = preference_residual.astype(np.float64)
    overall_both_branch = np.isin(
        np.broadcast_to(
            base_deployed_class,
            (family_count, args.repeats, len(dataset)),
        ),
        (CLASS_A, CLASS_B),
    ) & np.isin(transformed_deployed_class, (CLASS_A, CLASS_B))
    overall_both_valid = (
        np.broadcast_to(
            base_deployed_class,
            (family_count, args.repeats, len(dataset)),
        )
        > CLASS_INVALID
    ) & (transformed_deployed_class > CLASS_INVALID)
    summary = {
        "experiment": "hybrid_c5_external_isomorphism_equivariance",
        "scope": (
            "External standard-Sudoku isomorphisms g:P->gP; this is not an "
            "instance-stabilizer or Aut(P) experiment."
        ),
        "n_puzzles": len(dataset),
        "families": list(FAMILIES),
        "repeats_per_family": args.repeats,
        "candidate_count": candidate_count,
        "slots": slots,
        "cycles": cycles,
        "overall": {
            "candidate_index_exact_rate": safe_rate(candidate_exact),
            "all_41_candidates_exact_rate": safe_rate(
                candidate_exact.all(axis=-1)
            ),
            "candidate_semantic_class_rate": safe_rate(candidate_class_equal),
            "candidate_validity_rate": safe_rate(candidate_valid_equal),
            "mean_candidate_set_jaccard": safe_float(
                candidate_set_jaccard.mean()
            ),
            "mean_valid_set_jaccard": safe_float(valid_set_jaccard.mean()),
            "valid_set_equal_rate": safe_rate(valid_set_equal),
            "certified_solution_closure_rate": safe_rate(
                certified_solution_closed
            ),
            "semantic_branch_set_equal_rate": safe_rate(
                semantic_branch_set_equal
            ),
            "deployed_grid_exact_rate": safe_rate(deployed_grid_exact),
            "deployed_grid_exact_rate_both_valid": safe_rate(
                deployed_grid_exact, overall_both_valid
            ),
            "deployed_semantic_class_rate": safe_rate(deployed_class_equal),
            "deployed_candidate_index_rate": safe_rate(deployed_index_equal),
            "ab_semantic_branch_preservation": safe_rate(
                np.broadcast_to(
                    base_deployed_class,
                    transformed_deployed_class.shape,
                )
                == transformed_deployed_class,
                overall_both_branch,
            ),
            "preference_residual_mean": safe_float(overall_residual.mean()),
            "preference_residual_mean_abs": safe_float(
                np.abs(overall_residual).mean()
            ),
            "preference_residual_rms": safe_float(
                np.sqrt(np.mean(overall_residual**2))
            ),
            "preference_residual_max_abs": safe_float(
                np.abs(overall_residual).max()
            ),
            "strict_roundtrip_rate": safe_rate(strict_roundtrip),
            "inverse_classification_rate": safe_rate(
                inverse_classification_equal
            ),
        },
        "by_family": family_rows,
        "class_encoding": {
            "invalid": CLASS_INVALID,
            "solution_a": CLASS_A,
            "solution_b": CLASS_B,
            "other_valid": CLASS_OTHER_VALID,
        },
        "definitions": {
            "identity_control": (
                "A second untransformed GPU forward pass; calibrates numerical "
                "repeatability separately from external-isomorphism error"
            ),
            "candidate_index_exact": (
                "g^{-1}(candidate_i(gP)) equals candidate_i(P) in all 81 cells"
            ),
            "candidate_set_jaccard": (
                "Jaccard of unordered unique predicted-grid sets after inverse mapping"
            ),
            "valid_set_equal": (
                "unordered clue-valid Sudoku candidate-grid sets are exactly equal"
            ),
            "certified_solution_closed": (
                "every inverse-mapped valid candidate is certified A or B"
            ),
            "deployed_grid_exact": (
                "first-valid deployed grid agrees after inverse mapping"
            ),
            "preference_delta_residual": (
                "Delta_AB(gP,candidate_i) - Delta_AB(P,candidate_i), with A/B "
                "transformed semantically"
            ),
        },
    }
    # Keep the repeated identity pass as a numerical-repeatability control,
    # but do not dilute the headline external-isomorphism estimates with it.
    external_candidate_exact = candidate_exact[1:]
    external_class_equal = candidate_class_equal[1:]
    external_valid_equal = candidate_valid_equal[1:]
    external_transformed_valid = transformed_candidate_valid[1:]
    external_base_valid = np.broadcast_to(
        base["valid"], external_transformed_valid.shape
    )
    external_union_valid = external_base_valid | external_transformed_valid
    external_deployed_class = transformed_deployed_class[1:]
    external_base_deployed = np.broadcast_to(
        base_deployed_class, external_deployed_class.shape
    )
    external_both_branch = np.isin(
        external_base_deployed, (CLASS_A, CLASS_B)
    ) & np.isin(external_deployed_class, (CLASS_A, CLASS_B))
    external_both_valid = (external_base_deployed > CLASS_INVALID) & (
        external_deployed_class > CLASS_INVALID
    )
    external_residual = preference_residual[1:].astype(np.float64)
    external_overall = {
        "candidate_index_exact_rate": safe_rate(external_candidate_exact),
        "all_41_candidates_exact_rate": safe_rate(
            external_candidate_exact.all(axis=-1)
        ),
        "candidate_semantic_class_rate": safe_rate(external_class_equal),
        "candidate_validity_rate": safe_rate(external_valid_equal),
        "candidate_exact_rate_union_valid": safe_rate(
            external_candidate_exact, external_union_valid
        ),
        "candidate_semantic_class_rate_union_valid": safe_rate(
            external_class_equal, external_union_valid
        ),
        "candidate_valid_to_invalid_rate": safe_rate(
            ~external_transformed_valid, external_base_valid
        ),
        "candidate_invalid_to_valid_rate": safe_rate(
            external_transformed_valid, ~external_base_valid
        ),
        "mean_candidate_set_jaccard": safe_float(
            candidate_set_jaccard[1:].mean()
        ),
        "mean_valid_set_jaccard": safe_float(valid_set_jaccard[1:].mean()),
        "valid_set_equal_rate": safe_rate(valid_set_equal[1:]),
        "certified_solution_closure_rate": safe_rate(
            certified_solution_closed[1:]
        ),
        "semantic_branch_set_equal_rate": safe_rate(
            semantic_branch_set_equal[1:]
        ),
        "deployed_grid_exact_rate": safe_rate(deployed_grid_exact[1:]),
        "deployed_grid_exact_rate_both_valid": safe_rate(
            deployed_grid_exact[1:], external_both_valid
        ),
        "deployed_semantic_class_rate": safe_rate(deployed_class_equal[1:]),
        "deployed_candidate_index_rate": safe_rate(deployed_index_equal[1:]),
        "ab_semantic_branch_preservation": safe_rate(
            external_base_deployed == external_deployed_class,
            external_both_branch,
        ),
        "preference_residual_mean": safe_float(external_residual.mean()),
        "preference_residual_mean_abs": safe_float(
            np.abs(external_residual).mean()
        ),
        "preference_residual_rms": safe_float(
            np.sqrt(np.mean(external_residual**2))
        ),
        "preference_residual_max_abs": safe_float(
            np.abs(external_residual).max()
        ),
        "strict_roundtrip_rate": safe_rate(strict_roundtrip[1:]),
        "inverse_classification_rate": safe_rate(
            inverse_classification_equal[1:]
        ),
    }
    summary["overall_including_identity_control"] = summary["overall"]
    summary["identity_control"] = family_rows[0]
    summary["overall"] = external_overall
    summary["overall_scope"] = (
        "Five non-identity external-isomorphism families; identity is reported "
        "separately as a numerical repeatability control."
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    artifact_hashes = {
        path.name: sha256_file(path)
        for path in (details_path, repeat_csv, family_csv, candidate_csv, summary_path)
    }
    manifest = {
        "experiment": summary["experiment"],
        "scope": summary["scope"],
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "base_checkpoint": checkpoint.get("base_checkpoint"),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "seed": args.seed,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "n_puzzles": len(dataset),
        "families": list(FAMILIES),
        "transform_convention": (
            "optional transpose -> legal row indexing -> legal column indexing "
            "-> digit permutation; parameters vary by sample and repeat"
        ),
        "transform_seed_formula": (
            "seed + 100000007*(family+1) + 10000019*(repeat+1) + "
            "1000003*(source_index+1) + 97003*(sample_index+1), modulo 2^63-25"
        ),
        "strict_roundtrip_verified": bool(strict_roundtrip.all()),
        "inverse_classification_verified": bool(
            inverse_classification_equal.all()
        ),
        "candidate_order": (
            "0=anchor; 1+(cycle_zero_based*8)+slot_zero_based"
        ),
        "artifacts": artifact_hashes,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "n": len(dataset),
                "candidate_exact": summary["overall"][
                    "candidate_index_exact_rate"
                ],
                "deployed_exact": summary["overall"][
                    "deployed_grid_exact_rate"
                ],
                "roundtrip": summary["overall"]["strict_roundtrip_rate"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="External-isomorphism equivariance audit for current Hybrid+C5"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
