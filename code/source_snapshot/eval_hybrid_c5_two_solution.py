"""Generate and evaluate controlled exactly-two-solution Sudoku puzzles.

The generator starts from a unique Kaggle puzzle and a four-cell Sudoku trade
in its supplied solution.  It removes the (normally single) clue intersecting
that trade, then exhaustively enumerates up to three solutions.  A sample is
kept only when the exhaustive solution set is exactly the source solution and
the traded solution.

The evaluator deliberately bypasses active/first-valid halting.  It collects
the anchor plus every cycle/slot candidate from Hybrid+C5, classifies all of
them as solution A, solution B, other valid, or invalid, and only afterwards
reconstructs the deployed first-valid choice.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from eval_hybrid_hyper_rrn_restarts import hard_violation_count_batch
from eval_symbolic_active_reflection import load_symbolic
from sudoku_cache_utils import load_sudoku_dataset
from sudoku_dlx_solver import build_links, rows_to_grid
from sudoku_exchange_experiment import set_seed, set_torch_threads


SPLITS = {"train": 0, "val": 1, "test": 2}
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
class EnumerationResult:
    solutions: tuple[np.ndarray, ...]
    nodes: int
    exhausted: bool
    node_limit_hit: bool
    solution_cap_hit: bool


def enumerate_dlx_solutions(
    puzzle: Iterable[int] | np.ndarray,
    *,
    max_solutions: int = 3,
    max_nodes: int = 0,
) -> EnumerationResult:
    """Enumerate solutions and report whether the search tree was exhausted."""
    values = np.asarray(list(puzzle), dtype=np.int64).reshape(-1)
    if values.size != 81:
        raise ValueError(f"Expected 81 cells, got {values.size}")
    links = build_links(values.tolist())
    partial = []
    solutions: list[np.ndarray] = []
    nodes = 0
    node_limit_hit = False
    solution_cap_hit = False

    def search() -> bool:
        nonlocal nodes, node_limit_hit, solution_cap_hit
        if links.root.right is links.root:
            solutions.append(np.asarray(rows_to_grid(partial), dtype=np.uint8))
            if len(solutions) >= max_solutions:
                solution_cap_hit = True
                return True
            return False

        if max_nodes and nodes >= max_nodes:
            node_limit_hit = True
            return True

        column = links.choose_column()
        if column.size == 0:
            return False

        links.cover(column)
        row = column.down
        while row is not column:
            nodes += 1
            partial.append(row)
            node = row.right
            while node is not row:
                links.cover(node.column)
                node = node.right

            stop = search()

            node = row.left
            while node is not row:
                links.uncover(node.column)
                node = node.left
            partial.pop()
            if stop:
                links.uncover(column)
                return True
            row = row.down
        links.uncover(column)
        return False

    stopped_early = search()
    return EnumerationResult(
        solutions=tuple(solutions),
        nodes=nodes,
        exhausted=not stopped_early,
        node_limit_hit=node_limit_hit,
        solution_cap_hit=solution_cap_hit,
    )


def compact_grid(grid: np.ndarray) -> bytes:
    return np.asarray(grid, dtype=np.uint8).reshape(81).tobytes()


def valid_grid_numpy(grid: np.ndarray) -> bool:
    grid = np.asarray(grid).reshape(9, 9)
    target = np.arange(1, 10)
    for row in range(9):
        if not np.array_equal(np.sort(grid[row]), target):
            return False
    for col in range(9):
        if not np.array_equal(np.sort(grid[:, col]), target):
            return False
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            values = grid[box_row : box_row + 3, box_col : box_col + 3]
            if not np.array_equal(np.sort(values.reshape(-1)), target):
                return False
    return True


def find_four_cell_trades(solution: np.ndarray) -> list[tuple[int, ...]]:
    """Find a/b;b/a rectangles that occupy exactly two boxes."""
    grid = np.asarray(solution).reshape(9, 9)
    trades = []
    for row_a in range(9):
        for row_b in range(row_a + 1, 9):
            for col_a in range(9):
                for col_b in range(col_a + 1, 9):
                    digit_a = int(grid[row_a, col_a])
                    digit_b = int(grid[row_a, col_b])
                    if digit_a == digit_b:
                        continue
                    if int(grid[row_b, col_a]) != digit_b:
                        continue
                    if int(grid[row_b, col_b]) != digit_a:
                        continue
                    boxes = {
                        (row_a // 3, col_a // 3),
                        (row_a // 3, col_b // 3),
                        (row_b // 3, col_a // 3),
                        (row_b // 3, col_b // 3),
                    }
                    if len(boxes) == 2:
                        trades.append(
                            (row_a, row_b, col_a, col_b, digit_a, digit_b)
                        )
    return trades


def trade_positions(trade: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    row_a, row_b, col_a, col_b, _, _ = trade
    return (
        (row_a, col_a),
        (row_a, col_b),
        (row_b, col_a),
        (row_b, col_b),
    )


def apply_trade(solution: np.ndarray, trade: tuple[int, ...]) -> np.ndarray:
    row_a, row_b, col_a, col_b, digit_a, digit_b = trade
    result = np.asarray(solution, dtype=np.uint8).reshape(9, 9).copy()
    result[row_a, col_a] = digit_b
    result[row_a, col_b] = digit_a
    result[row_b, col_a] = digit_a
    result[row_b, col_b] = digit_b
    if not valid_grid_numpy(result):
        raise AssertionError("Trade did not preserve Sudoku validity")
    return result


def exact_solution_set(result: EnumerationResult, expected: tuple[np.ndarray, ...]) -> bool:
    if not result.exhausted or result.node_limit_hit:
        return False
    actual_keys = {compact_grid(value) for value in result.solutions}
    expected_keys = {compact_grid(value) for value in expected}
    return actual_keys == expected_keys and len(result.solutions) == len(expected)


def generate_dataset(args) -> None:
    set_torch_threads()
    rng = np.random.default_rng(args.seed)
    dataset = load_sudoku_dataset(args.cache_path, split=SPLITS[args.split], limit=0)
    source_order = rng.permutation(len(dataset))
    if args.scan_limit:
        source_order = source_order[: args.scan_limit]

    records: dict[str, list] = {
        "puzzles": [],
        "solutions_a": [],
        "solutions_b": [],
        "controls_a": [],
        "controls_b": [],
        "diff_masks": [],
        "trade": [],
        "pivot": [],
        "source_index": [],
        "source_rating": [],
        "source_clues": [],
        "base_clues": [],
        "removed_trade_clues": [],
        "base_enum_nodes": [],
        "control_a_enum_nodes": [],
        "control_b_enum_nodes": [],
    }
    seen_puzzles: set[bytes] = set()
    sources_seen = 0
    trades_tested = 0
    rejected_three_plus = 0
    rejected_wrong_pair = 0
    started = time.perf_counter()

    for source_index in source_order:
        sources_seen += 1
        puzzle = np.asarray(dataset.puzzles[source_index], dtype=np.uint8).reshape(9, 9)
        solution_a = np.asarray(
            dataset.solutions[source_index], dtype=np.uint8
        ).reshape(9, 9)
        trades = find_four_cell_trades(solution_a)
        rng.shuffle(trades)
        accepted = False
        for trade in trades:
            positions = trade_positions(trade)
            clue_positions = [(row, col) for row, col in positions if puzzle[row, col] > 0]
            if args.require_single_removed_clue and len(clue_positions) != 1:
                continue
            if not clue_positions:
                # A genuinely unique source puzzle must intersect every trade.
                continue
            base = puzzle.copy()
            for row, col in positions:
                base[row, col] = 0
            base_clues = int((base > 0).sum())
            if base_clues < args.min_clues or base_clues > args.max_clues:
                continue
            puzzle_key = compact_grid(base)
            if puzzle_key in seen_puzzles:
                continue

            solution_b = apply_trade(solution_a, trade)
            trades_tested += 1
            enumeration = enumerate_dlx_solutions(
                base.reshape(-1), max_solutions=3, max_nodes=args.max_enum_nodes
            )
            if enumeration.solution_cap_hit or len(enumeration.solutions) >= 3:
                rejected_three_plus += 1
                continue
            if not exact_solution_set(enumeration, (solution_a, solution_b)):
                rejected_wrong_pair += 1
                continue

            # Choose the removed source clue as the paired counterfactual pivot.
            pivot = clue_positions[0]
            control_a = base.copy()
            control_b = base.copy()
            control_a[pivot] = solution_a[pivot]
            control_b[pivot] = solution_b[pivot]
            enum_a = enumerate_dlx_solutions(
                control_a.reshape(-1), max_solutions=2, max_nodes=args.max_enum_nodes
            )
            enum_b = enumerate_dlx_solutions(
                control_b.reshape(-1), max_solutions=2, max_nodes=args.max_enum_nodes
            )
            if not exact_solution_set(enum_a, (solution_a,)):
                raise AssertionError("A control was not uniquely solved by solution A")
            if not exact_solution_set(enum_b, (solution_b,)):
                raise AssertionError("B control was not uniquely solved by solution B")

            diff_mask = solution_a != solution_b
            if int(diff_mask.sum()) != 4:
                raise AssertionError("Expected a four-cell solution difference")
            seen_puzzles.add(puzzle_key)
            records["puzzles"].append(base)
            records["solutions_a"].append(solution_a)
            records["solutions_b"].append(solution_b)
            records["controls_a"].append(control_a)
            records["controls_b"].append(control_b)
            records["diff_masks"].append(diff_mask)
            records["trade"].append(trade)
            records["pivot"].append(pivot)
            records["source_index"].append(int(source_index))
            records["source_rating"].append(float(dataset.ratings[source_index]))
            records["source_clues"].append(int((puzzle > 0).sum()))
            records["base_clues"].append(base_clues)
            records["removed_trade_clues"].append(len(clue_positions))
            records["base_enum_nodes"].append(enumeration.nodes)
            records["control_a_enum_nodes"].append(enum_a.nodes)
            records["control_b_enum_nodes"].append(enum_b.nodes)
            accepted = True
            break

        if args.progress_every and sources_seen % args.progress_every == 0:
            print(
                f"[generate] sources={sources_seen} accepted={len(records['puzzles'])}/"
                f"{args.target_n} trades_tested={trades_tested} "
                f"three_plus={rejected_three_plus} elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
        if accepted and len(records["puzzles"]) >= args.target_n:
            break

    if len(records["puzzles"]) < args.target_n:
        raise RuntimeError(
            f"Generated only {len(records['puzzles'])}/{args.target_n} samples "
            f"after scanning {sources_seen} source puzzles"
        )

    array_payload = {
        "puzzles": np.asarray(records["puzzles"], dtype=np.uint8),
        "solutions_a": np.asarray(records["solutions_a"], dtype=np.uint8),
        "solutions_b": np.asarray(records["solutions_b"], dtype=np.uint8),
        "controls_a": np.asarray(records["controls_a"], dtype=np.uint8),
        "controls_b": np.asarray(records["controls_b"], dtype=np.uint8),
        "diff_masks": np.asarray(records["diff_masks"], dtype=bool),
        "trade": np.asarray(records["trade"], dtype=np.int16),
        "pivot": np.asarray(records["pivot"], dtype=np.int8),
        "source_index": np.asarray(records["source_index"], dtype=np.int64),
        "source_rating": np.asarray(records["source_rating"], dtype=np.float32),
        "source_clues": np.asarray(records["source_clues"], dtype=np.int8),
        "base_clues": np.asarray(records["base_clues"], dtype=np.int8),
        "removed_trade_clues": np.asarray(
            records["removed_trade_clues"], dtype=np.int8
        ),
        "base_enum_nodes": np.asarray(records["base_enum_nodes"], dtype=np.int32),
        "control_a_enum_nodes": np.asarray(
            records["control_a_enum_nodes"], dtype=np.int32
        ),
        "control_b_enum_nodes": np.asarray(
            records["control_b_enum_nodes"], dtype=np.int32
        ),
    }
    target = Path(args.output_npz)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **array_payload)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    elapsed = time.perf_counter() - started
    summary = {
        "experiment": "controlled_exactly_two_solution_sudoku",
        "n": len(records["puzzles"]),
        "source_split": args.split,
        "seed": args.seed,
        "sources_seen": sources_seen,
        "trades_tested": trades_tested,
        "rejected_three_plus": rejected_three_plus,
        "rejected_wrong_pair": rejected_wrong_pair,
        "mean_base_clues": float(array_payload["base_clues"].mean()),
        "min_base_clues": int(array_payload["base_clues"].min()),
        "max_base_clues": int(array_payload["base_clues"].max()),
        "mean_base_enum_nodes": float(array_payload["base_enum_nodes"].mean()),
        "sha256": digest,
        "elapsed_seconds": elapsed,
        "certification": "exhaustive DLX ended with exactly solutions_a and solutions_b before cap=3",
    }
    summary_path = target.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[generated] {summary}", flush=True)
    print(f"[save] dataset={target} summary={summary_path}", flush=True)


def transform_grid(
    grid: np.ndarray,
    *,
    row_order: np.ndarray,
    col_order: np.ndarray,
    digit_map: np.ndarray,
    transpose: bool,
) -> np.ndarray:
    result = np.asarray(grid).reshape(9, 9)
    if transpose:
        result = result.T
    result = result[np.asarray(row_order)][:, np.asarray(col_order)]
    return digit_map[result]


def random_symmetry_batch(
    puzzles: np.ndarray,
    solutions_a: np.ndarray,
    solutions_b: np.ndarray,
    source_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = [[], [], []]
    for puzzle, solution_a, solution_b, source_index in zip(
        puzzles, solutions_a, solutions_b, source_indices
    ):
        rng = np.random.default_rng(seed + 1_000_003 * int(source_index))
        bands = rng.permutation(3)
        rows = np.concatenate([3 * band + rng.permutation(3) for band in bands])
        stacks = rng.permutation(3)
        cols = np.concatenate([3 * stack + rng.permutation(3) for stack in stacks])
        digit_perm = rng.permutation(9) + 1
        digit_map = np.zeros(10, dtype=np.uint8)
        digit_map[1:] = digit_perm
        transpose = bool(rng.integers(0, 2))
        for target, grid in zip(transformed, (puzzle, solution_a, solution_b)):
            target.append(
                transform_grid(
                    grid,
                    row_order=rows,
                    col_order=cols,
                    digit_map=digit_map,
                    transpose=transpose,
                )
            )
    return tuple(np.asarray(values, dtype=np.uint8) for values in transformed)


def candidate_logits(output: dict[str, torch.Tensor]) -> torch.Tensor:
    anchor = output["anchor_logits"][:, None]
    cycles = output["cycle_logits"]
    batch = cycles.shape[0]
    cycles = cycles.reshape(batch, -1, 9, 9, 9)
    return torch.cat([anchor, cycles], dim=1)


def classify_predictions(
    predictions: torch.Tensor,
    puzzles: torch.Tensor,
    solutions_a: torch.Tensor,
    solutions_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify BxCx9x9 predictions and return class plus valid mask."""
    batch, candidates = predictions.shape[:2]
    flat = predictions.reshape(batch * candidates, 9, 9)
    valid = (hard_violation_count_batch(flat) == 0).reshape(batch, candidates)
    clue_ok = (
        (puzzles[:, None] == 0) | (predictions == puzzles[:, None])
    ).reshape(batch, candidates, -1).all(dim=-1)
    valid = valid & clue_ok
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
    return classes, valid


@torch.no_grad()
def evaluate_variant(
    model,
    puzzles_np: np.ndarray,
    solutions_a_np: np.ndarray,
    solutions_b_np: np.ndarray,
    diff_masks_np: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    progress_label: str,
    progress_every: int,
) -> dict[str, np.ndarray | str]:
    model.eval()
    outputs: dict[str, list[np.ndarray]] = {
        "candidate_class": [],
        "deployed_class": [],
        "deployed_index": [],
        "valid_candidate_count": [],
        "unique_grid_count": [],
        "preference_delta": [],
    }
    prediction_digest = hashlib.sha256()
    started = time.perf_counter()
    n = len(puzzles_np)
    for offset in range(0, n, batch_size):
        end = min(offset + batch_size, n)
        puzzles = torch.as_tensor(
            puzzles_np[offset:end], dtype=torch.long, device=device
        )
        solutions_a = torch.as_tensor(
            solutions_a_np[offset:end], dtype=torch.long, device=device
        )
        solutions_b = torch.as_tensor(
            solutions_b_np[offset:end], dtype=torch.long, device=device
        )
        diff_masks = torch.as_tensor(
            diff_masks_np[offset:end], dtype=torch.bool, device=device
        )
        output = model(puzzles, include_continuation=False)
        logits = candidate_logits(output)
        predictions = logits.argmax(dim=-1) + 1
        classes, valid = classify_predictions(
            predictions, puzzles, solutions_a, solutions_b
        )
        has_valid = valid.any(dim=1)
        first_valid = valid.to(torch.int64).argmax(dim=1)
        rows = torch.arange(len(puzzles), device=device)
        deployed_class = classes[rows, first_valid]
        deployed_class = torch.where(
            has_valid, deployed_class, torch.zeros_like(deployed_class)
        )
        deployed_index = torch.where(
            has_valid, first_valid, torch.full_like(first_valid, -1)
        )

        log_prob = F.log_softmax(logits, dim=-1)
        log_a = log_prob.gather(
            -1, (solutions_a[:, None, :, :, None] - 1)
        ).squeeze(-1)
        log_b = log_prob.gather(
            -1, (solutions_b[:, None, :, :, None] - 1)
        ).squeeze(-1)
        preference_delta = ((log_a - log_b) * diff_masks[:, None]).sum(
            dim=(-1, -2)
        )

        predictions_cpu = predictions.to(torch.uint8).cpu().numpy()
        prediction_digest.update(predictions_cpu.tobytes())
        unique_grid_count = np.asarray(
            [
                np.unique(sample.reshape(sample.shape[0], 81), axis=0).shape[0]
                for sample in predictions_cpu
            ],
            dtype=np.int16,
        )
        outputs["candidate_class"].append(classes.cpu().numpy())
        outputs["deployed_class"].append(deployed_class.cpu().numpy())
        outputs["deployed_index"].append(deployed_index.cpu().numpy())
        outputs["valid_candidate_count"].append(
            valid.sum(dim=1).to(torch.int16).cpu().numpy()
        )
        outputs["unique_grid_count"].append(unique_grid_count)
        outputs["preference_delta"].append(
            preference_delta.float().cpu().numpy()
        )
        if progress_every and end % progress_every < batch_size:
            print(
                f"[eval:{progress_label}] seen={end}/{n} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    result = {key: np.concatenate(values) for key, values in outputs.items()}
    result["prediction_sha256"] = prediction_digest.hexdigest()
    result["elapsed_seconds"] = np.asarray(
        time.perf_counter() - started, dtype=np.float64
    )
    return result


def safe_rate(mask: np.ndarray) -> float:
    return float(np.asarray(mask, dtype=np.float64).mean()) if len(mask) else float("nan")


def summarize_variant(result: dict[str, np.ndarray | str], slots: int, cycles: int) -> dict:
    classes = np.asarray(result["candidate_class"])
    deployed = np.asarray(result["deployed_class"])
    a_any = (classes == CLASS_A).any(axis=1)
    b_any = (classes == CLASS_B).any(axis=1)
    valid_any = classes > CLASS_INVALID
    summary = {
        "n": int(len(classes)),
        "candidate_count": int(classes.shape[1]),
        "deployed_valid": safe_rate(deployed > CLASS_INVALID),
        "deployed_a": safe_rate(deployed == CLASS_A),
        "deployed_b": safe_rate(deployed == CLASS_B),
        "deployed_other_valid": safe_rate(deployed == CLASS_OTHER_VALID),
        "candidate_a_any": safe_rate(a_any),
        "candidate_b_any": safe_rate(b_any),
        "candidate_both_any": safe_rate(a_any & b_any),
        "candidate_valid_any": safe_rate(valid_any.any(axis=1)),
        "candidate_other_valid_any": safe_rate(
            (classes == CLASS_OTHER_VALID).any(axis=1)
        ),
        "mean_valid_candidate_count": float(
            np.asarray(result["valid_candidate_count"]).mean()
        ),
        "mean_unique_grid_count": float(
            np.asarray(result["unique_grid_count"]).mean()
        ),
        "anchor_a": safe_rate(classes[:, 0] == CLASS_A),
        "anchor_b": safe_rate(classes[:, 0] == CLASS_B),
        "anchor_valid": safe_rate(classes[:, 0] > CLASS_INVALID),
        "mean_anchor_preference_delta": float(
            np.asarray(result["preference_delta"])[:, 0].mean()
        ),
        "prediction_sha256": result["prediction_sha256"],
        "elapsed_seconds": float(np.asarray(result["elapsed_seconds"])),
    }
    cumulative_a = classes[:, 0] == CLASS_A
    cumulative_b = classes[:, 0] == CLASS_B
    summary["cumulative"] = [
        {
            "stage": "anchor",
            "a_any": safe_rate(cumulative_a),
            "b_any": safe_rate(cumulative_b),
            "both_any": safe_rate(cumulative_a & cumulative_b),
        }
    ]
    for cycle in range(cycles):
        start = 1 + cycle * slots
        end = start + slots
        cumulative_a = cumulative_a | (classes[:, start:end] == CLASS_A).any(axis=1)
        cumulative_b = cumulative_b | (classes[:, start:end] == CLASS_B).any(axis=1)
        summary["cumulative"].append(
            {
                "stage": f"cycle{cycle + 1}",
                "a_any": safe_rate(cumulative_a),
                "b_any": safe_rate(cumulative_b),
                "both_any": safe_rate(cumulative_a & cumulative_b),
            }
        )
    return summary


def checkpoint_label_and_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = value
        return Path(path).stem, path
    label, path = value.split("=", 1)
    return label.strip(), path.strip()


def write_summary_csv(rows: list[dict], path: Path) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_models(args) -> None:
    set_torch_threads()
    set_seed(args.seed)
    device = torch.device(args.device)
    data = np.load(args.dataset_npz, allow_pickle=False)
    puzzles = np.asarray(data["puzzles"], dtype=np.uint8)
    solutions_a = np.asarray(data["solutions_a"], dtype=np.uint8)
    solutions_b = np.asarray(data["solutions_b"], dtype=np.uint8)
    diff_masks = np.asarray(data["diff_masks"], dtype=bool)
    controls_a = np.asarray(data["controls_a"], dtype=np.uint8)
    controls_b = np.asarray(data["controls_b"], dtype=np.uint8)
    source_indices = np.asarray(data["source_index"], dtype=np.int64)
    if args.limit:
        take = slice(0, args.limit)
        puzzles = puzzles[take]
        solutions_a = solutions_a[take]
        solutions_b = solutions_b[take]
        diff_masks = diff_masks[take]
        controls_a = controls_a[take]
        controls_b = controls_b[take]
        source_indices = source_indices[take]

    symmetry_values = None
    if args.run_symmetry:
        symmetry_values = random_symmetry_batch(
            puzzles,
            solutions_a,
            solutions_b,
            source_indices,
            args.symmetry_seed,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    deployed_by_model = []
    model_labels = []
    for checkpoint_value in args.checkpoint:
        label, checkpoint_path = checkpoint_label_and_path(checkpoint_value)
        print(f"[model] loading label={label} checkpoint={checkpoint_path}", flush=True)
        model, cfg, checkpoint = load_symbolic(checkpoint_path, device)
        ambiguous = evaluate_variant(
            model,
            puzzles,
            solutions_a,
            solutions_b,
            diff_masks,
            device=device,
            batch_size=args.batch_size,
            progress_label=f"{label}:ambiguous",
            progress_every=args.progress_every,
        )
        ambiguous_summary = summarize_variant(ambiguous, cfg.slots, cfg.cycles)
        repeat_digest = None
        repeat_match = None
        repeat_candidate_class_match = None
        repeat_candidate_class_disagreement = None
        repeat_puzzle_any_candidate_class_change = None
        repeat_deployed_class_match = None
        repeat_deployed_branch_flip = None
        repeat = None
        if args.determinism_repeat:
            repeat = evaluate_variant(
                model,
                puzzles,
                solutions_a,
                solutions_b,
                diff_masks,
                device=device,
                batch_size=args.batch_size,
                progress_label=f"{label}:repeat",
                progress_every=args.progress_every,
            )
            repeat_digest = repeat["prediction_sha256"]
            base_candidate_class = np.asarray(ambiguous["candidate_class"])
            repeat_candidate_class = np.asarray(repeat["candidate_class"])
            candidate_changed = base_candidate_class != repeat_candidate_class
            base_deployed = np.asarray(ambiguous["deployed_class"])
            repeat_deployed = np.asarray(repeat["deployed_class"])
            both_branches = np.isin(base_deployed, (CLASS_A, CLASS_B)) & np.isin(
                repeat_deployed, (CLASS_A, CLASS_B)
            )
            repeat_candidate_class_match = bool(not candidate_changed.any())
            repeat_candidate_class_disagreement = safe_rate(candidate_changed.reshape(-1))
            repeat_puzzle_any_candidate_class_change = safe_rate(
                candidate_changed.any(axis=1)
            )
            repeat_deployed_class_match = safe_rate(base_deployed == repeat_deployed)
            repeat_deployed_branch_flip = (
                safe_rate(base_deployed[both_branches] != repeat_deployed[both_branches])
                if both_branches.any()
                else float("nan")
            )
            repeat_match = bool(
                repeat_digest == ambiguous["prediction_sha256"]
                and repeat_candidate_class_match
            )

        controls = {}
        if args.run_controls:
            controls["a"] = evaluate_variant(
                model,
                controls_a,
                solutions_a,
                solutions_b,
                diff_masks,
                device=device,
                batch_size=args.batch_size,
                progress_label=f"{label}:control_a",
                progress_every=args.progress_every,
            )
            controls["b"] = evaluate_variant(
                model,
                controls_b,
                solutions_a,
                solutions_b,
                diff_masks,
                device=device,
                batch_size=args.batch_size,
                progress_label=f"{label}:control_b",
                progress_every=args.progress_every,
            )

        symmetry = None
        symmetry_summary = None
        symmetry_branch_agreement = None
        if symmetry_values is not None:
            sym_puzzles, sym_a, sym_b = symmetry_values
            sym_diff = sym_a != sym_b
            symmetry = evaluate_variant(
                model,
                sym_puzzles,
                sym_a,
                sym_b,
                sym_diff,
                device=device,
                batch_size=args.batch_size,
                progress_label=f"{label}:symmetry",
                progress_every=args.progress_every,
            )
            symmetry_summary = summarize_variant(symmetry, cfg.slots, cfg.cycles)
            base_deployed = np.asarray(ambiguous["deployed_class"])
            sym_deployed = np.asarray(symmetry["deployed_class"])
            comparable = np.isin(base_deployed, (CLASS_A, CLASS_B)) & np.isin(
                sym_deployed, (CLASS_A, CLASS_B)
            )
            symmetry_branch_agreement = (
                safe_rate(base_deployed[comparable] == sym_deployed[comparable])
                if comparable.any()
                else float("nan")
            )

        summary = {
            "label": label,
            "checkpoint": checkpoint_path,
            "checkpoint_step": int(checkpoint.get("step", -1)),
            "checkpoint_experiment": checkpoint.get("experiment", ""),
            "checkpoint_data_mode": checkpoint.get("data_mode", ""),
            "checkpoint_initialization_seed": checkpoint.get("metadata", {}).get(
                "initialization_seed"
            ),
            "checkpoint_data_seed": checkpoint.get("metadata", {}).get("data_seed"),
            "checkpoint_initial_fingerprint": checkpoint.get("metadata", {}).get(
                "initial_trainable_fingerprint"
            ),
            "checkpoint_train_pool_fingerprint": checkpoint.get("metadata", {}).get(
                "train_pool_fingerprint"
            ),
            "slots": int(cfg.slots),
            "cycles": int(cfg.cycles),
            "ambiguous": ambiguous_summary,
            "determinism_repeat_sha256": repeat_digest,
            "determinism_repeat_match": repeat_match,
            "repeat_candidate_class_match": repeat_candidate_class_match,
            "repeat_candidate_class_disagreement": repeat_candidate_class_disagreement,
            "repeat_puzzle_any_candidate_class_change": (
                repeat_puzzle_any_candidate_class_change
            ),
            "repeat_deployed_class_match": repeat_deployed_class_match,
            "repeat_deployed_branch_flip": repeat_deployed_branch_flip,
            "symmetry": symmetry_summary,
            "symmetry_deployed_branch_agreement": symmetry_branch_agreement,
        }
        if controls:
            control_a_classes = np.asarray(controls["a"]["deployed_class"])
            control_b_classes = np.asarray(controls["b"]["deployed_class"])
            summary["control_a"] = summarize_variant(
                controls["a"], cfg.slots, cfg.cycles
            )
            summary["control_b"] = summarize_variant(
                controls["b"], cfg.slots, cfg.cycles
            )
            summary["paired_controls"] = {
                "a_deployed_correct": safe_rate(control_a_classes == CLASS_A),
                "b_deployed_correct": safe_rate(control_b_classes == CLASS_B),
                "both_deployed_correct": safe_rate(
                    (control_a_classes == CLASS_A) & (control_b_classes == CLASS_B)
                ),
                "a_minus_b_accuracy": safe_rate(control_a_classes == CLASS_A)
                - safe_rate(control_b_classes == CLASS_B),
            }

        detail_payload = {
            "source_index": source_indices,
            "ambiguous_candidate_class": ambiguous["candidate_class"],
            "ambiguous_deployed_class": ambiguous["deployed_class"],
            "ambiguous_deployed_index": ambiguous["deployed_index"],
            "ambiguous_preference_delta": ambiguous["preference_delta"],
            "ambiguous_valid_candidate_count": ambiguous["valid_candidate_count"],
            "ambiguous_unique_grid_count": ambiguous["unique_grid_count"],
        }
        if repeat is not None:
            detail_payload.update(
                {
                    "repeat_candidate_class": repeat["candidate_class"],
                    "repeat_deployed_class": repeat["deployed_class"],
                }
            )
        if controls:
            detail_payload.update(
                {
                    "control_a_candidate_class": controls["a"]["candidate_class"],
                    "control_a_deployed_class": controls["a"]["deployed_class"],
                    "control_b_candidate_class": controls["b"]["candidate_class"],
                    "control_b_deployed_class": controls["b"]["deployed_class"],
                }
            )
        if symmetry is not None:
            detail_payload.update(
                {
                    "symmetry_candidate_class": symmetry["candidate_class"],
                    "symmetry_deployed_class": symmetry["deployed_class"],
                }
            )
        np.savez_compressed(output_dir / f"{label}_details.npz", **detail_payload)
        (output_dir / f"{label}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[summary:{label}] {json.dumps(summary, ensure_ascii=False)}", flush=True)

        flat_row = {
            "label": label,
            "n": ambiguous_summary["n"],
            "deployed_valid": ambiguous_summary["deployed_valid"],
            "deployed_a": ambiguous_summary["deployed_a"],
            "deployed_b": ambiguous_summary["deployed_b"],
            "candidate_a_any": ambiguous_summary["candidate_a_any"],
            "candidate_b_any": ambiguous_summary["candidate_b_any"],
            "candidate_both_any": ambiguous_summary["candidate_both_any"],
            "candidate_other_valid_any": ambiguous_summary[
                "candidate_other_valid_any"
            ],
            "mean_valid_candidate_count": ambiguous_summary[
                "mean_valid_candidate_count"
            ],
            "mean_unique_grid_count": ambiguous_summary["mean_unique_grid_count"],
            "anchor_valid": ambiguous_summary["anchor_valid"],
            "anchor_a": ambiguous_summary["anchor_a"],
            "anchor_b": ambiguous_summary["anchor_b"],
            "determinism_repeat_match": repeat_match,
            "repeat_candidate_class_disagreement": (
                repeat_candidate_class_disagreement
            ),
            "repeat_puzzle_any_candidate_class_change": (
                repeat_puzzle_any_candidate_class_change
            ),
            "repeat_deployed_class_match": repeat_deployed_class_match,
            "repeat_deployed_branch_flip": repeat_deployed_branch_flip,
            "symmetry_branch_agreement": symmetry_branch_agreement,
        }
        if controls:
            flat_row.update(summary["paired_controls"])
        flat_rows.append(flat_row)
        deployed_by_model.append(np.asarray(ambiguous["deployed_class"]))
        model_labels.append(label)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cross_model = None
    if len(deployed_by_model) >= 2:
        matrix = np.stack(deployed_by_model, axis=1)
        branch_only = np.isin(matrix, (CLASS_A, CLASS_B))
        all_branch = branch_only.all(axis=1)
        unanimous = (matrix == matrix[:, :1]).all(axis=1)
        cross_model = {
            "labels": model_labels,
            "n": int(len(matrix)),
            "all_models_returned_a_or_b": safe_rate(all_branch),
            "unanimous_branch_given_all_solved": (
                safe_rate(unanimous[all_branch]) if all_branch.any() else float("nan")
            ),
            "disagreement_given_all_solved": (
                safe_rate(~unanimous[all_branch]) if all_branch.any() else float("nan")
            ),
        }
        (output_dir / "cross_model_summary.json").write_text(
            json.dumps(cross_model, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            output_dir / "cross_model_deployed.npz",
            source_index=source_indices,
            labels=np.asarray(model_labels),
            deployed_class=matrix,
        )
        print(f"[cross-model] {cross_model}", flush=True)

    write_summary_csv(flat_rows, output_dir / "model_summary.csv")
    run_summary = {
        "dataset_npz": args.dataset_npz,
        "n": int(len(puzzles)),
        "models": model_labels,
        "cross_model": cross_model,
        "class_names": CLASS_NAMES,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[save] output_dir={output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument(
        "--cache_path",
        default="data/cache_full_3m.npz",
    )
    generate.add_argument("--split", choices=sorted(SPLITS), default="test")
    generate.add_argument("--target_n", type=int, default=2000)
    generate.add_argument("--scan_limit", type=int, default=0)
    generate.add_argument("--seed", type=int, default=20260801)
    generate.add_argument("--min_clues", type=int, default=17)
    generate.add_argument("--max_clues", type=int, default=31)
    generate.add_argument("--require_single_removed_clue", type=int, default=1)
    generate.add_argument("--max_enum_nodes", type=int, default=0)
    generate.add_argument("--progress_every", type=int, default=500)
    generate.add_argument("--output_npz", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--dataset_npz", required=True)
    evaluate.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Repeat label=/path/checkpoint.pt for each model.",
    )
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch_size", type=int, default=16)
    evaluate.add_argument("--limit", type=int, default=0)
    evaluate.add_argument("--seed", type=int, default=20260801)
    evaluate.add_argument("--run_controls", type=int, default=1)
    evaluate.add_argument("--run_symmetry", type=int, default=1)
    evaluate.add_argument("--symmetry_seed", type=int, default=20260802)
    evaluate.add_argument("--determinism_repeat", type=int, default=1)
    evaluate.add_argument("--progress_every", type=int, default=256)
    evaluate.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "generate":
        generate_dataset(args)
    else:
        evaluate_models(args)


if __name__ == "__main__":
    main()
