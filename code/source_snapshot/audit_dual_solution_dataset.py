#!/usr/bin/env python3
"""Audit a two-solution Sudoku dataset against the mathematical certificates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from dual_solution_theory import (
    SudokuSpec,
    audit_instance,
    compact_grid_key,
    enumerate_backtracking_solutions,
    exact_solution_pair,
    standard_pair_automorphism_audit,
)
from sudoku_dlx_solver import compact_grid as compact_dlx_grid
from sudoku_dlx_solver import solve_dlx


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--box_size", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--recertify_backtracking",
        type=int,
        default=1,
        help="run the independent MRV enumerator with cap=3",
    )
    parser.add_argument(
        "--backtracking_seeds",
        default="0,1",
        help="comma-separated randomized value-order seeds; empty means deterministic",
    )
    parser.add_argument(
        "--recertify_dlx",
        type=int,
        default=1,
        help=(
            "independently enumerate with the existing 324-column "
            "Algorithm-X/DLX solver at cap=3"
        ),
    )
    parser.add_argument(
        "--audit_standard_stabilizer",
        type=int,
        default=1,
        help="exhaustively search the documented standard symmetry group",
    )
    parser.add_argument("--progress_every", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    spec = SudokuSpec(args.box_size)
    dlx_enabled = bool(args.recertify_dlx) and spec.n == 9
    if args.recertify_dlx and not dlx_enabled:
        print(
            "[audit] fixed 324-column DLX applies only to 9x9; "
            "the finite 4x4 atlas and generic MRV certifier remain available",
            flush=True,
        )
    data = np.load(dataset_path, allow_pickle=False)
    dataset_digest = sha256_file(dataset_path)
    required = {"puzzles", "solutions_a", "solutions_b"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"Dataset is missing keys: {sorted(missing)}")
    n = len(data["puzzles"])
    if args.limit:
        n = min(n, args.limit)
    seeds = [
        int(item.strip())
        for item in args.backtracking_seeds.split(",")
        if item.strip()
    ]
    if not seeds:
        seeds = [None]

    jsonl_path = output / "instance_certificates.jsonl"
    rows: list[dict[str, object]] = []
    symmetry_counts: dict[str, int] = {}
    total_backtracking_nodes = 0
    total_dlx_nodes = 0
    started = time.perf_counter()
    seen_puzzles: set[bytes] = set()
    seen_pairs: set[str] = set()
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for index in range(n):
            puzzle = np.asarray(data["puzzles"][index], dtype=np.uint8)
            solution_a = np.asarray(data["solutions_a"][index], dtype=np.uint8)
            solution_b = np.asarray(data["solutions_b"][index], dtype=np.uint8)
            puzzle_key = compact_grid_key(puzzle)
            raw_duplicate = puzzle_key in seen_puzzles
            seen_puzzles.add(puzzle_key)

            recertified = not bool(args.recertify_backtracking)
            seed_nodes: list[int] = []
            solution_order_hashes: list[str] = []
            if args.recertify_backtracking:
                for seed in seeds:
                    enumeration = enumerate_backtracking_solutions(
                        puzzle,
                        spec,
                        max_solutions=3,
                        random_seed=seed,
                    )
                    if not exact_solution_pair(enumeration, solution_a, solution_b):
                        raise RuntimeError(
                            f"Independent exact-two certification failed at index={index}, seed={seed}"
                        )
                    seed_nodes.append(enumeration.nodes)
                    total_backtracking_nodes += enumeration.nodes
                    ordered = b"|".join(
                        compact_grid_key(solution) for solution in enumeration.solutions
                    )
                    solution_order_hashes.append(hashlib.sha256(ordered).hexdigest())
                recertified = True

            dlx_nodes: int | None = None
            dlx_first_solution_sha256: str | None = None
            if dlx_enabled:
                dlx = solve_dlx(
                    puzzle.reshape(-1).astype(int).tolist(),
                    max_solutions=3,
                )
                if dlx.status != "solved" or dlx.solutions_found != 2:
                    raise RuntimeError(
                        "Independent DLX exact-two certification failed at "
                        f"index={index}: status={dlx.status}, "
                        f"solutions_found={dlx.solutions_found}"
                    )
                if dlx.solution is None:
                    raise RuntimeError(
                        f"DLX reported solved without a grid at index={index}"
                    )
                first_dlx = np.asarray(dlx.solution, dtype=np.uint8)
                if not (
                    np.array_equal(first_dlx, solution_a)
                    or np.array_equal(first_dlx, solution_b)
                ):
                    raise RuntimeError(
                        f"DLX first solution was outside certified A/B at index={index}"
                    )
                dlx_nodes = int(dlx.nodes)
                total_dlx_nodes += dlx_nodes
                dlx_first_solution_sha256 = hashlib.sha256(
                    compact_dlx_grid(dlx.solution).encode("ascii")
                ).hexdigest()

            symmetry = None
            if args.audit_standard_stabilizer:
                symmetry = standard_pair_automorphism_audit(
                    puzzle, solution_a, solution_b, spec
                )
            provenance = {
                "dataset_index": index,
                "source_index": (
                    int(data["source_index"][index])
                    if "source_index" in data.files
                    else None
                ),
                "dataset_sha256": dataset_digest,
            }
            certificate = audit_instance(
                puzzle,
                solution_a,
                solution_b,
                spec,
                exact_two_certified=recertified,
                symmetry=symmetry,
                provenance=provenance,
            )
            pair_hash = str(certificate["unordered_pair_sha256"])
            pair_duplicate = pair_hash in seen_pairs
            seen_pairs.add(pair_hash)
            certificate["solution_certification"].update(
                {
                    "independent_solver": (
                        "mrv_bitmask_backtracking"
                        if args.recertify_backtracking
                        else None
                    ),
                    "value_order_seeds": seeds if args.recertify_backtracking else [],
                    "nodes_by_seed": seed_nodes,
                    "enumeration_order_sha256": solution_order_hashes,
                    "independent_dlx_solver": (
                        "knuth_algorithm_x_dancing_links_324_columns"
                        if dlx_enabled
                        else None
                    ),
                    "dlx_max_solutions": 3 if dlx_enabled else None,
                    "dlx_solutions_found": 2 if dlx_enabled else None,
                    "dlx_nodes": dlx_nodes,
                    "dlx_first_solution_sha256": dlx_first_solution_sha256,
                }
            )
            certificate["deduplication"] = {
                "raw_puzzle_duplicate": raw_duplicate,
                "unordered_pair_duplicate": pair_duplicate,
            }
            jsonl.write(json.dumps(certificate, ensure_ascii=False) + "\n")

            regime = str(certificate["symmetry"]["regime"])
            symmetry_counts[regime] = symmetry_counts.get(regime, 0) + 1
            difference = certificate["difference"]
            trade = certificate["exact_cover_trade"]
            orientation = certificate["orientations"]
            rows.append(
                {
                    "dataset_index": index,
                    "sample_id": certificate["sample_id"],
                    "source_index": provenance["source_index"],
                    "difference_size": difference["size"],
                    "trade_components": trade["component_count"],
                    "trade_connected": trade["connected"],
                    "minimal_unavoidable": trade["minimal_unavoidable_certified"],
                    "four_cell_rectangle": difference["four_cell_rectangle"] is not None,
                    "symmetry_regime": regime,
                    "stabilizer_size": certificate["symmetry"].get("stabilizer_size"),
                    "rho_image_size": certificate["symmetry"].get("rho_image_size"),
                    "swap_exists": certificate["symmetry"].get("swap_exists"),
                    "a_is_low": orientation["a_is_low_at_first_difference"],
                    "a_is_lex_first": orientation["a_is_lexicographically_first"],
                    "raw_puzzle_duplicate": raw_duplicate,
                    "unordered_pair_duplicate": pair_duplicate,
                    "backtracking_nodes_mean": (
                        float(np.mean(seed_nodes)) if seed_nodes else None
                    ),
                    "dlx_solutions_found": 2 if dlx_enabled else None,
                    "dlx_nodes": dlx_nodes,
                }
            )
            if args.progress_every and (index + 1) % args.progress_every == 0:
                print(
                    f"[audit] {index + 1}/{n} regimes={symmetry_counts} "
                    f"elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )

    write_csv(output / "instance_summary.csv", rows)
    summary = {
        "schema_version": "dual_solution_dataset_audit/v1",
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": dataset_digest,
        "n": n,
        "box_size": args.box_size,
        "independent_backtracking_recertified": bool(args.recertify_backtracking),
        "backtracking_seeds": seeds if args.recertify_backtracking else [],
        "total_backtracking_nodes": total_backtracking_nodes,
        "independent_dlx_recertified": dlx_enabled,
        "dlx_algorithm": (
            "knuth_algorithm_x_dancing_links_324_columns"
            if dlx_enabled
            else None
        ),
        "dlx_max_solutions": 3 if dlx_enabled else None,
        "total_dlx_nodes": total_dlx_nodes,
        "standard_stabilizer_audited": bool(args.audit_standard_stabilizer),
        "symmetry_regimes": symmetry_counts,
        "difference_size_counts": {
            str(value): int(sum(row["difference_size"] == value for row in rows))
            for value in sorted({int(row["difference_size"]) for row in rows})
        },
        "trade_connected_all": all(bool(row["trade_connected"]) for row in rows),
        "minimal_unavoidable_all": all(
            bool(row["minimal_unavoidable"]) for row in rows
        ),
        "raw_puzzle_duplicates": int(sum(bool(row["raw_puzzle_duplicate"]) for row in rows)),
        "unordered_pair_duplicates": int(
            sum(bool(row["unordered_pair_duplicate"]) for row in rows)
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
