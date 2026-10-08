"""Resumable dataset and scoring audit; never changes the source cache.

Example (Linux, from the repository root)::
    python3 review_experiments_20260929/audit_dataset.py \
      --cache data/cache_full_3m.npz \
      --output reproduction/data_audit \
      --seed-candidates 0,1,42 --workers 8 --timeout-seconds 5

Exact duplicate grouping uses complete 81-byte keys, not hash equality. Digit
normalization covers digit permutations ONLY. No claim about the full Sudoku
symmetry group (row/band/column/stack permutations and transpose) is made.
Uniqueness means exhaustive enumeration stopped after at most two solutions;
any node/time limit is UNKNOWN, including a limit reached after finding a first
solution. SQLite commits make both stages resumable. Reuse the same output
directory and identical semantic options to resume. --max-new-unique is only a
per-invocation work cap and may be changed on resume; 0 means the full test set.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sudoku_dlx_solver as dlx

VERSION = 1
REQUIRED = ("puzzles", "solutions", "splits")
OPTIONAL = ("clues", "ratings")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(pending, path)


def load_cache(path):
    path = Path(path).resolve()
    if path.is_dir():
        paths = [path / f"{name}.npy" for name in REQUIRED + OPTIONAL if (path / f"{name}.npy").is_file()]
        arrays = {item.stem: np.load(item, mmap_mode="r", allow_pickle=False) for item in paths}
    else:
        paths = [path]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in REQUIRED + OPTIONAL if name in archive.files}
    missing = set(REQUIRED) - arrays.keys()
    if missing:
        raise ValueError(f"Missing cache arrays: {sorted(missing)}")
    n = len(arrays["puzzles"])
    for name in ("puzzles", "solutions"):
        if arrays[name].shape not in ((n, 81), (n, 9, 9)):
            raise ValueError(f"{name}: expected (N,81) or (N,9,9), got {arrays[name].shape}")
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"{name} must contain integers, got {arrays[name].dtype}")
        arrays[name] = arrays[name].reshape(n, 81)
    for name in ("splits", "clues", "ratings"):
        if name in arrays and arrays[name].shape != (n,):
            raise ValueError(f"{name}: expected (N,), got {arrays[name].shape}")
    if not np.issubdtype(arrays["splits"].dtype, np.integer):
        raise ValueError("splits must contain integers")
    metadata = {
        "cache_path": str(path),
        "files": [{"path": str(item), "size": item.stat().st_size, "sha256": sha256_file(item)} for item in paths],
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
    }
    return arrays, metadata


def split_from_puzzle(puzzle, seed):
    """The exact historical split algorithm, copied without importing torch."""
    text = "".join(str(int(x)) for x in puzzle)
    return split_from_text(text, seed)


def split_from_text(text, seed):
    """Hash an already serialized puzzle; identical to the historical split."""
    digest = hashlib.blake2b((str(seed) + text).encode("ascii"), digest_size=2).digest()
    bucket = int.from_bytes(digest, "little") % 10
    return 0 if bucket < 8 else 1 if bucket == 8 else 2


def canonical_digits(puzzle):
    """Complete canonical key for S9 digit relabeling; zero stays zero."""
    mapping = {0: 0}
    result = bytearray()
    for digit in puzzle:
        digit = int(digit)
        if not 0 <= digit <= 9:
            raise ValueError("canonical_digits expects digits 0..9")
        if digit not in mapping:
            mapping[digit] = len(mapping)
        result.append(mapping[digit])
    return bytes(result)


def solution_valid(solution):
    values = np.asarray(solution).reshape(9, 9)
    expected = np.arange(1, 10)
    boxes = values.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
    return bool(np.all(np.sort(values, axis=1) == expected) and np.all(np.sort(values.T, axis=1) == expected) and np.all(np.sort(boxes, axis=1) == expected))


def row_flags(puzzle, solution, split, clue_count=None):
    flags = []
    if not np.all((puzzle >= 0) & (puzzle <= 9)):
        flags.append("puzzle_digit_range")
    if not solution_valid(solution):
        flags.append("invalid_target_solution")
    if np.any((puzzle != 0) & (puzzle != solution)):
        flags.append("clue_target_mismatch")
    if int(split) not in (0, 1, 2):
        flags.append("invalid_split")
    if clue_count is not None and (not np.isfinite(clue_count) or clue_count != np.count_nonzero(puzzle)):
        flags.append("clue_count_mismatch")
    return flags


class AuditTimeout(Exception):
    pass


def audit_unique_task(task):
    """A process-local deadline hook leaves the original solver file untouched.

    DLX invokes choose_column once per nonterminal search node. The hook checks
    wall time there, so the deadline is cooperative (not a hard OS kill). This
    function must not be executed concurrently in threads within one process.
    The CLI uses separate worker processes and always restores the method.
    """
    index, puzzle, target, max_nodes, timeout_seconds = task
    started = time.perf_counter()
    original = dlx.DancingLinks.choose_column

    def choose_with_deadline(self):
        if timeout_seconds > 0 and time.perf_counter() - started >= timeout_seconds:
            raise AuditTimeout()
        return original(self)

    dlx.DancingLinks.choose_column = choose_with_deadline
    result = {"row_index": index, "status": "unknown", "reason": "", "solutions_found": None,
              "nodes": None, "first_solution_matches_target": None, "seconds": None}
    try:
        if not np.all((np.asarray(puzzle) >= 0) & (np.asarray(puzzle) <= 9)):
            result.update(status="invalid_input", reason="puzzle_digit_range")
        else:
            solved = dlx.solve_dlx(puzzle, max_nodes=max_nodes, max_solutions=2)
            first = None if solved.solution is None else np.asarray(solved.solution).reshape(81)
            result.update(solutions_found=solved.solutions_found, nodes=solved.nodes,
                          first_solution_matches_target=None if first is None else int(np.array_equal(first, target)))
            if solved.status == "limit":
                result.update(status="unknown", reason="node_limit")
            elif solved.solutions_found >= 2:
                result.update(status="multiple", reason="two_solutions_found")
            elif solved.solutions_found == 1:
                result.update(status="unique", reason="exhaustive_one_solution")
            else:
                result.update(status="unsat", reason="exhaustive_zero_solutions")
    except AuditTimeout:
        result.update(status="unknown", reason="wall_timeout_cooperative")
    except Exception as exc:
        result.update(status="error", reason=f"{type(exc).__name__}: {exc}")
    finally:
        dlx.DancingLinks.choose_column = original
    result["seconds"] = time.perf_counter() - started
    return result


def create_db(path):
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript("""
      CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS rows (
        row_index INTEGER PRIMARY KEY, split INTEGER NOT NULL, clues INTEGER,
        rating REAL, flags TEXT NOT NULL, exact_sha256 TEXT, digit_sha256 TEXT);
      CREATE TABLE IF NOT EXISTS exact_keys (
        key BLOB PRIMARY KEY, first_index INTEGER, count INTEGER, split_mask INTEGER,
        first_solution BLOB, label_disagreement INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS digit_keys (
        key BLOB PRIMARY KEY, first_index INTEGER, count INTEGER, split_mask INTEGER) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS seeds (seed INTEGER PRIMARY KEY, checked INTEGER, mismatches INTEGER);
      CREATE TABLE IF NOT EXISTS uniqueness (
        row_index INTEGER PRIMARY KEY, status TEXT NOT NULL, reason TEXT,
        solutions_found INTEGER, nodes INTEGER, first_solution_matches_target INTEGER,
        seconds REAL);
      CREATE INDEX IF NOT EXISTS rows_split ON rows(split);
    """)
    return connection


def state_get(connection, key, default=None):
    item = connection.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return default if item is None else json.loads(item[0])


def state_set(connection, key, value):
    connection.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value, sort_keys=True)))


def run_static(connection, arrays, seeds, chunk_size):
    start = state_get(connection, "static_next_index", 0)
    n = len(arrays["puzzles"])
    for offset in range(start, n, chunk_size):
        end = min(offset + chunk_size, n)
        counts = {seed: [0, 0] for seed in seeds}
        with connection:
            for index in range(offset, end):
                puzzle, solution = arrays["puzzles"][index], arrays["solutions"][index]
                split = int(arrays["splits"][index])
                clue_count = arrays.get("clues", [])[index] if "clues" in arrays else None
                flags = row_flags(puzzle, solution, split, clue_count)
                key = None if "puzzle_digit_range" in flags else np.asarray(puzzle, dtype=np.uint8).tobytes()
                digit_key = None if key is None else canonical_digits(puzzle)
                rating = float(arrays["ratings"][index]) if "ratings" in arrays else None
                if rating is not None and not np.isfinite(rating):
                    rating = None
                connection.execute("INSERT INTO rows VALUES (?,?,?,?,?,?,?)", (index, split, int(np.count_nonzero(puzzle)), rating, ";".join(flags),
                    None if key is None else hashlib.sha256(key).hexdigest(), None if digit_key is None else hashlib.sha256(digit_key).hexdigest()))
                if key is not None:
                    split_mask = (1 << split) if split in (0, 1, 2) else 8
                    connection.execute("""INSERT INTO exact_keys VALUES (?,?,?,?,?,0)
                      ON CONFLICT(key) DO UPDATE SET count=count+1,
                      split_mask=split_mask|excluded.split_mask,
                      label_disagreement=label_disagreement OR first_solution!=excluded.first_solution""",
                      (key, index, 1, split_mask, solution.tobytes()))
                    connection.execute("""INSERT INTO digit_keys VALUES (?,?,?,?)
                      ON CONFLICT(key) DO UPDATE SET count=count+1, split_mask=split_mask|excluded.split_mask""", (digit_key, index, 1, split_mask))
                    puzzle_text = "".join(str(int(digit)) for digit in puzzle)
                    for seed in seeds:
                        counts[seed][0] += 1
                        counts[seed][1] += int(split_from_text(puzzle_text, seed) != split)
            for seed, (checked, mismatches) in counts.items():
                connection.execute("""INSERT INTO seeds VALUES (?,?,?) ON CONFLICT(seed)
                    DO UPDATE SET checked=checked+excluded.checked,mismatches=mismatches+excluded.mismatches""", (seed, checked, mismatches))
            state_set(connection, "static_next_index", end)
        print(json.dumps({"stage": "static", "completed": end, "total": n}), flush=True)
    with connection:
        state_set(connection, "static_complete", True)


def uniqueness_pending(connection, split, max_new):
    query = "SELECT r.row_index FROM rows r LEFT JOIN uniqueness u ON u.row_index=r.row_index WHERE r.split=? AND u.row_index IS NULL ORDER BY r.row_index"
    if max_new:
        query += f" LIMIT {int(max_new)}"
    return [row[0] for row in connection.execute(query, (split,))]


def run_uniqueness(connection, arrays, args):
    pending = uniqueness_pending(connection, args.test_split, args.max_new_unique)
    def tasks():
        for index in pending:
            yield (index, arrays["puzzles"][index].tolist(), arrays["solutions"][index].tolist(), args.max_nodes, args.timeout_seconds)
    pool = None
    if args.workers > 1:
        pool = ProcessPoolExecutor(max_workers=args.workers)
    # Bound submitted work to a commit chunk: avoids enqueuing 300k futures.
    iterator = iter(tasks())
    completed = 0
    import itertools
    try:
        while chunk := list(itertools.islice(iterator, args.commit_every)):
            results = map(audit_unique_task, chunk) if pool is None else pool.map(audit_unique_task, chunk, chunksize=max(1, min(16, len(chunk) // args.workers)))
            with connection:
                for result in results:
                    connection.execute("INSERT INTO uniqueness VALUES (?,?,?,?,?,?,?)", tuple(result[key] for key in (
                        "row_index", "status", "reason", "solutions_found", "nodes", "first_solution_matches_target", "seconds")))
                    completed += 1
            print(json.dumps({"stage": "uniqueness", "new_completed": completed, "scheduled_this_run": len(pending)}), flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def write_query_csv(connection, path, query):
    temporary = Path(str(path) + ".tmp")
    cursor = connection.execute(query)
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([column[0] for column in cursor.description])
        writer.writerows(cursor)
    os.replace(temporary, path)


def summarize(connection, n, test_split):
    audited = connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
    splits = {str(k): v for k, v in connection.execute("SELECT split,COUNT(*) FROM rows GROUP BY split")}
    flags = {flag: count for flag, count in connection.execute("SELECT flags,COUNT(*) FROM rows WHERE flags!='' GROUP BY flags")}
    duplicate_summary = {}
    # Bits 0..2 are train/validation/test. Invalid split metadata is never
    # misreported as a cross-split leakage event, and has a separate flag.
    cross = "((split_mask&7)&((split_mask&7)-1))!=0"
    for table in ("exact_keys", "digit_keys"):
        groups, rows = connection.execute(f"SELECT COUNT(*),COALESCE(SUM(count),0) FROM {table} WHERE count>1").fetchone()
        cross_groups, cross_rows = connection.execute(f"SELECT COUNT(*),COALESCE(SUM(count),0) FROM {table} WHERE {cross}").fetchone()
        duplicate_summary[table] = {"duplicate_groups": groups, "rows_in_duplicate_groups": rows,
            "cross_split_groups": cross_groups, "rows_in_cross_split_groups": cross_rows}
    seeds = [{"seed": seed, "checked": checked, "mismatches": mismatches,
              "matches_all_audited_rows": checked == audited and mismatches == 0} for seed, checked, mismatches in connection.execute("SELECT * FROM seeds ORDER BY seed")]
    statuses = {status: count for status, count in connection.execute("SELECT status,COUNT(*) FROM uniqueness GROUP BY status")}
    expected = splits.get(str(test_split), 0)
    return {"dataset_rows": n, "static_audited_rows": audited, "static_complete": audited == n,
        "split_counts": splits, "row_flag_combinations": flags, "seed_candidates": seeds,
        "duplicates": duplicate_summary,
        "exact_duplicate_groups_with_disagreeing_labels": connection.execute("SELECT COUNT(*) FROM exact_keys WHERE label_disagreement=1").fetchone()[0],
        "uniqueness": {"test_split": test_split, "expected_rows": expected, "audited_rows": sum(statuses.values()),
            "pending_rows": expected - sum(statuses.values()), "status_counts": statuses,
            "all_test_rows_certified_unique": audited == n and expected > 0 and statuses.get("unique", 0) == expected},
        "symmetry_coverage": {"exact_puzzle": "complete", "digit_relabeling_S9": "complete",
            "full_sudoku_symmetry_group": "NOT AUDITED", "warning": "Digit-normalized duplicates are a subset of full Sudoku symmetry equivalence; zero matches do not exclude row/column/band/stack/transpose leakage."},
        "scoring_contract": {"valid": "complete Sudoku rules AND all original clues preserved",
            "exact": "all 81 predicted digits equal the supplied target",
            "other_legal_completion": "valid=1, exact=0; do not reject it as an invalid Sudoku solution",
            "unsolved_output": "valid=0, exact=0; denominator includes every test puzzle",
            "budget_exhaustion": "UNSOLVED, never proof of UNSAT",
            "uniqueness_timeout_or_node_limit": "UNKNOWN, never a uniqueness certificate"}}


def export(connection, output, n, test_split):
    summary = summarize(connection, n, test_split)
    atomic_json(output / "summary.json", summary)
    write_query_csv(connection, output / "row_audit.csv", """SELECT r.*,u.status AS uniqueness_status,u.reason AS uniqueness_reason,
        u.solutions_found,u.nodes,u.first_solution_matches_target,u.seconds AS uniqueness_seconds
        FROM rows r LEFT JOIN uniqueness u ON r.row_index=u.row_index ORDER BY r.row_index""")
    for table, filename in (("exact_keys", "exact_duplicate_groups.csv"), ("digit_keys", "digit_duplicate_groups.csv")):
        extra = ",label_disagreement" if table == "exact_keys" else ""
        write_query_csv(connection, output / filename, f"SELECT lower(hex(key)) AS complete_key_hex,first_index,count,split_mask{extra} FROM {table} WHERE count>1 ORDER BY first_index")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("static", "uniqueness", "all"), default="all")
    parser.add_argument("--seed-candidates", default="0,1,42")
    parser.add_argument("--test-split", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--max-new-unique", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--commit-every", type=int, default=256)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.workers, args.chunk_size, args.commit_every) < 1 or min(args.timeout_seconds, args.max_nodes, args.max_new_unique) < 0:
        raise ValueError("Counts must be positive; limits must be nonnegative")
    seeds = sorted({int(item) for item in args.seed_candidates.split(",") if item.strip()})
    if not seeds:
        raise ValueError("At least one seed candidate is required")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays, cache_meta = load_cache(args.cache)
    n = len(arrays["puzzles"])
    semantic = {"version": VERSION, "input": cache_meta, "seed_candidates": seeds, "test_split": args.test_split,
        "max_solutions": 2, "max_nodes": args.max_nodes, "timeout_seconds": args.timeout_seconds,
        "code_sha256": {"audit_dataset.py": sha256_file(__file__), "sudoku_dlx_solver.py": sha256_file(dlx.__file__)}}
    connection = create_db(output / "audit.sqlite3")
    previous = state_get(connection, "semantic_manifest")
    if previous is not None and previous != semantic:
        connection.close()
        raise ValueError("Resume refused: input content, script, solver, or semantic settings changed. Use a new output directory.")
    with connection:
        state_set(connection, "semantic_manifest", semantic)
    metadata = {**semantic, "last_invocation": {"utc": datetime.now(timezone.utc).isoformat(), "argv": vars(args),
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "pid": os.getpid()},
        "timeout_method": "cooperative wall deadline checked at each nonterminal DLX node; process-local method wrapper",
        "storage": "SQLite WAL; chunk transactions; full keys used for duplicate grouping"}
    atomic_json(output / "metadata.json", metadata)
    try:
        if args.stage in ("static", "all"):
            run_static(connection, arrays, seeds, args.chunk_size)
        if args.stage in ("uniqueness", "all"):
            if not state_get(connection, "static_complete", False):
                raise ValueError("Run --stage static before --stage uniqueness")
            run_uniqueness(connection, arrays, args)
        summary = export(connection, output, n, args.test_split)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary
    finally:
        connection.close()


if __name__ == "__main__":
    main()
