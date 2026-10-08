"""A05 DLX/Minisat22 baseline on a frozen, source-row-keyed evaluation set.

No neural weights, cached solutions, or reference labels are passed to a solver.
Solver construction, clue ingestion, search, decoding and independent verification
are inside per-puzzle pipeline timing. Constant Sudoku CNF template generation,
cache reads and artifact writes are outside it. Parallel wall timing includes
process startup/IPC and result collection; worker latency excludes queue time.

--mode pilot permits a small pipeline check under shared hardware. Formal timing
requires --mode timing and refuses non-idle/unavailable nvidia-smi status. Each
repetition is a fresh solve, with no reused solver state between puzzles.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

# Numeric helper libraries must not turn one solver process into a BLAS pool.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sudoku_dlx_solver as dlx

_SAT_CLASS = None
_BASE_CNF = None


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def variable(row, col, digit):
    return row * 81 + col * 9 + digit


def sudoku_cnf():
    """Standard pairwise exactly-one encoding, equivalent to the legacy ABL CNF."""
    clauses = []
    for row in range(9):
        for col in range(9):
            atoms = [variable(row, col, digit) for digit in range(1, 10)]
            clauses.append(atoms)
            clauses.extend([-a, -b] for a, b in itertools.combinations(atoms, 2))
    units = [[(r, c) for c in range(9)] for r in range(9)]
    units += [[(r, c) for r in range(9)] for c in range(9)]
    units += [[(br + dr, bc + dc) for dr in range(3) for dc in range(3)] for br in (0, 3, 6) for bc in (0, 3, 6)]
    for unit in units:
        for digit in range(1, 10):
            atoms = [variable(r, c, digit) for r, c in unit]
            clauses.extend([-a, -b] for a, b in itertools.combinations(atoms, 2))
    return clauses


def verify_completion(prediction, puzzle):
    """Independent checker, including given clues; no target answer is used."""
    if prediction is None:
        return False
    values = np.asarray(prediction)
    given = np.asarray(puzzle)
    if values.size != 81 or given.size != 81:
        return False
    values, given = values.reshape(9, 9), given.reshape(9, 9)
    if not np.all((values >= 1) & (values <= 9)) or not np.all((given == 0) | (given == values)):
        return False
    digits = set(range(1, 10))
    for i in range(9):
        if set(values[i].tolist()) != digits or set(values[:, i].tolist()) != digits:
            return False
    return all(set(values[r:r+3, c:c+3].ravel().tolist()) == digits for r in (0, 3, 6) for c in (0, 3, 6))


def peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        return None


def gpu_status():
    try:
        output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_memory", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10)
        lines = [line.strip() for line in output.stdout.splitlines() if line.strip()]
        return {"available": output.returncode == 0, "idle": output.returncode == 0 and not lines,
                "processes": lines, "stderr": output.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "idle": False, "error": str(exc)}


def require_timing_context(mode):
    status = gpu_status()
    if mode == "timing" and not status["idle"]:
        raise RuntimeError(f"Formal timing requires confirmed idle GPUs; observed {status}")
    return status


def load_sat(pysat_path=""):
    global _SAT_CLASS, _BASE_CNF
    if pysat_path:
        path = str(Path(pysat_path).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        module = importlib.import_module("pysat.solvers")
        package = importlib.import_module("pysat")
    except ImportError as exc:
        raise RuntimeError("PySAT is unavailable. Use the existing isolated --pysat-path; this script never installs packages.") from exc
    _SAT_CLASS = module.Minisat22
    _BASE_CNF = sudoku_cnf()
    extension = importlib.import_module("pysolvers")
    paths = [Path(module.__file__), Path(extension.__file__)]
    return {"backend": "Minisat22", "package_version": getattr(package, "__version__", None),
            "files": [{"path": str(path), "sha256": sha256_file(path)} for path in paths],
            "solver_threads": 1, "base_clause_count": len(_BASE_CNF),
            "timeout_api": "threading.Timer -> interrupt; solve_limited(expect_interrupt=True)",
            "api_reference": "https://pysathq.github.io/docs/html/api/solvers.html"}


class SolveTimeout(Exception):
    pass


def dlx_once(puzzle, timeout, max_nodes):
    started = time.perf_counter()
    original = dlx.DancingLinks.choose_column
    def with_deadline(self):
        if timeout and time.perf_counter() - started >= timeout:
            raise SolveTimeout()
        return original(self)
    dlx.DancingLinks.choose_column = with_deadline
    try:
        answer = dlx.solve_dlx(puzzle, max_nodes=max_nodes, max_solutions=1)
        if timeout and time.perf_counter() - started >= timeout:
            return None, "unknown", "wall_timeout_cooperative", answer.nodes, None
        status = "unknown" if answer.status == "limit" else answer.status
        return answer.solution, status, "node_limit" if status == "unknown" else "", answer.nodes, None
    except SolveTimeout:
        return None, "unknown", "wall_timeout_cooperative", None, None
    finally:
        dlx.DancingLinks.choose_column = original


def sat_once(puzzle, timeout, conflict_budget):
    if _SAT_CLASS is None:
        raise RuntimeError("SAT worker has not been initialized")
    started = time.perf_counter()
    # A new solver for every puzzle: no learning or phase state crosses rows.
    with _SAT_CLASS(bootstrap_with=_BASE_CNF) as solver:
        for index, digit in enumerate(puzzle):
            if digit:
                solver.add_clause([index * 9 + int(digit)])
        if conflict_budget:
            solver.conf_budget(conflict_budget)
        remaining = timeout - (time.perf_counter() - started)
        if timeout and remaining <= 0:
            return None, "unknown", "wall_timeout_during_setup", None, None
        timer = threading.Timer(remaining, solver.interrupt) if timeout else None
        if timer is not None:
            timer.daemon = True
            timer.start()
        try:
            satisfiable = solver.solve_limited(expect_interrupt=bool(timer))
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()  # Never delete a solver while its interrupt callback runs.
        stats = solver.accum_stats()
        conflicts = int(stats.get("conflicts", 0))
        if timeout and time.perf_counter() - started >= timeout:
            return None, "unknown", "wall_timeout", None, conflicts
        if satisfiable is None:
            return None, "unknown", "sat_budget_or_interrupt", None, conflicts
        if not satisfiable:
            return None, "unsat", "proved_unsatisfiable", None, conflicts
        positives = {literal for literal in solver.get_model() if 1 <= literal <= 729}
        result = [next((digit for digit in range(1, 10) if index * 9 + digit in positives), 0) for index in range(81)]
        return result, "solved", "", None, conflicts


def solve_task(task):
    source_row, puzzle, solver_name, timeout, max_nodes, conflict_budget = task
    started = time.perf_counter()
    prediction = None
    try:
        values = np.asarray(puzzle)
        if values.size != 81 or not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values > 9)):
            raise ValueError("Invalid puzzle: expected 81 integer digits in 0..9")
        if solver_name == "dlx":
            prediction, status, reason, nodes, conflicts = dlx_once(puzzle, timeout, max_nodes)
        else:
            prediction, status, reason, nodes, conflicts = sat_once(puzzle, timeout, conflict_budget)
        valid = verify_completion(prediction, puzzle)
        if status == "solved" and not valid:
            status, reason = "invalid_solver_output", "independent_verifier_failed"
        if status != "solved":
            prediction, valid = None, False
    except Exception as exc:
        status, reason, nodes, conflicts, valid = "error", f"{type(exc).__name__}: {exc}", None, None, False
    seconds = time.perf_counter() - started
    return {"source_row": int(source_row), "status": status, "reason": reason, "valid": int(valid),
            "pipeline_seconds": seconds, "nodes": nodes, "conflicts": conflicts,
            "prediction": "" if prediction is None else "".join(str(int(x)) for x in np.asarray(prediction).ravel()),
            "worker_pid": os.getpid(), "worker_peak_rss_bytes": peak_rss_bytes()}


def initialize_worker(pysat_path, solver_name, cpu_affinity):
    if cpu_affinity and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpu_affinity))
    if solver_name == "sat":
        load_sat(pysat_path)


def select_data(args):
    with np.load(args.cache_path, allow_pickle=False) as cache:
        splits = np.asarray(cache["splits"])
        rows = np.flatnonzero(splits == 2)
        manifest = None
        if args.test_global_ids:
            allowed = np.load(args.test_global_ids, allow_pickle=False)
            if allowed.ndim != 1 or not np.issubdtype(allowed.dtype, np.integer) or len(np.unique(allowed)) != len(allowed):
                raise ValueError("Manifest must contain unique integer global indices")
            if np.any(allowed < 0) or np.any(allowed >= len(splits)) or np.any(splits[allowed] != 2):
                raise ValueError("Manifest contains out-of-range or non-test indices")
            rows = rows[np.isin(rows, allowed)]
            manifest = {"path": str(Path(args.test_global_ids).resolve()), "sha256": sha256_file(args.test_global_ids), "n": len(allowed)}
        ratings_all = np.asarray(cache["ratings"])
        if args.rating_min_exclusive is not None:
            rows = rows[ratings_all[rows] > args.rating_min_exclusive]
        rows = rows[args.offset:]
        if args.limit:
            rows = rows[:args.limit]
        puzzles = np.asarray(cache["puzzles"])[rows].reshape(-1, 81)
        targets = np.asarray(cache["solutions"])[rows].reshape(-1, 81)
        ratings = ratings_all[rows]
    if not len(rows):
        raise ValueError("Empty evaluation set")
    return rows, puzzles, targets, ratings, manifest


def latency_summary(seconds):
    values = np.asarray(seconds)
    return {"n": len(values), "mean_seconds": float(np.mean(values)), "p50_seconds": float(np.quantile(values, .5)),
            "p95_seconds": float(np.quantile(values, .95)), "max_seconds": float(np.max(values))}


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory; timing runs are never overwritten or silently resumed")
    rows, puzzles, targets, ratings, manifest = select_data(args)
    context = require_timing_context(args.mode)
    sat_meta = load_sat(args.pysat_path) if "sat" in args.solvers else None
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"utc": datetime.now(timezone.utc).isoformat(), "args": vars(args), "n": len(rows),
        "cache_sha256": sha256_file(args.cache_path), "test_manifest": manifest,
        "source_rows_sha256": hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "cpu_affinity_before": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__), str(Path(dlx.__file__).resolve()): sha256_file(dlx.__file__)},
        "gpu_before": context, "sat": sat_meta, "solver_threads_per_process": 1, "parallelism_unit": "independent worker processes",
        "native_thread_env": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "pipeline_timing_scope": "input validation + per-puzzle fresh solver construction + clues + solve + decode + independent clue/rule verification; excludes target comparison, cache load, CNF template generation and output files",
        "parallel_wall_scope": "whole evaluation including process startup, IPC and result collection; excludes cache load, parent warmup and output files",
        "warning": "workers>1 latencies are in-worker pipeline latencies, not single-request queue-inclusive latency; primary batch1 result uses workers=1",
        "timeout_boundary": "budget covers solver setup/search; cooperative DLX deadline or Minisat limited-call interrupt; independent verifier may add small extra time"}
    write_json(output / "metadata.json", metadata)
    np.save(output / "source_rows.npy", rows)
    summaries = []
    for solver_name in args.solvers:
        initialize_worker(args.pysat_path, solver_name, args.cpu_affinity)
        for index in range(min(args.warmup, len(rows))):
            result = solve_task((int(rows[index]), puzzles[index].tolist(), solver_name, args.timeout_seconds, args.max_nodes, args.sat_conflict_budget))
            if result["status"] in {"error", "invalid_solver_output"}:
                raise RuntimeError(f"Warmup failed: {result}")
        for repeat in range(args.repeats):
            before = require_timing_context(args.mode)
            tasks = ((int(row), puzzle.tolist(), solver_name, args.timeout_seconds, args.max_nodes, args.sat_conflict_budget) for row, puzzle in zip(rows, puzzles))
            start = time.perf_counter()
            if args.workers == 1:
                results = list(map(solve_task, tasks))
            else:
                with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                         initargs=(args.pysat_path, solver_name, args.cpu_affinity)) as executor:
                    results = list(executor.map(solve_task, tasks, chunksize=args.chunksize))
            wall_seconds = time.perf_counter() - start
            after = gpu_status()
            uncontended = before.get("idle", False) and after.get("idle", False)
            counts = {}
            exact_count = valid_not_exact = 0
            for result, target, rating in zip(results, targets, ratings):
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                exact = result["valid"] and result["prediction"] == "".join(str(int(x)) for x in target)
                result["exact"] = int(exact)
                result["rating"] = float(rating) if np.isfinite(rating) else ""
                result["solver"] = solver_name
                result["repeat"] = repeat
                exact_count += int(exact)
                valid_not_exact += int(result["valid"] and not exact)
            path = output / f"{solver_name}_repeat{repeat}.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
            valid_count = sum(result["valid"] for result in results)
            rss = [result["worker_peak_rss_bytes"] for result in results if result["worker_peak_rss_bytes"] is not None]
            row = {"solver": solver_name, "repeat": repeat, "n": len(rows), "statuses": counts,
                "valid_count": valid_count, "valid_rate": valid_count / len(rows), "exact_count": exact_count,
                "exact_rate": exact_count / len(rows), "valid_not_exact_count": valid_not_exact,
                "pipeline_latency": latency_summary([result["pipeline_seconds"] for result in results]),
                "worker_pipeline_seconds_sum": sum(result["pipeline_seconds"] for result in results),
                "wall_seconds": wall_seconds, "attempted_puzzles_per_second": len(rows) / wall_seconds,
                "valid_solutions_per_second": valid_count / wall_seconds,
                "max_worker_peak_rss_bytes": max(rss) if rss else None,
                "rss_scope": "largest process lifetime ru_maxrss, not simultaneous aggregate RSS or incremental solver-only memory",
                "gpu_before": before, "gpu_after": after,
                "eligible_for_formal_timing": args.mode == "timing" and uncontended and not any(counts.get(name) for name in ("error", "invalid_solver_output")),
                "output_sha256": sha256_file(path)}
            summaries.append(row)
            write_json(output / "summary.json", {"runs": summaries, "mode": args.mode, "status": "running"})
            print(json.dumps({"solver": solver_name, "repeat": repeat, "n": len(rows), "statuses": counts, "wall_seconds": wall_seconds}), flush=True)
            if args.mode == "timing" and not uncontended:
                raise RuntimeError("GPU activity changed during this repetition; artifact retained but excluded from formal timing")
    write_json(output / "summary.json", {"runs": summaries, "mode": args.mode, "status": "complete"})
    return summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-global-ids", default="")
    parser.add_argument("--solvers", nargs="+", choices=("dlx", "sat"), default=["dlx", "sat"])
    parser.add_argument("--pysat-path", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cpu-affinity", type=lambda value: [int(item) for item in value.split(",")], default=[])
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=5.)
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--sat-conflict-budget", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rating-min-exclusive", type=float, default=None)
    parser.add_argument("--mode", choices=("pilot", "timing"), default="pilot")
    args = parser.parse_args(argv)
    if min(args.workers, args.repeats, args.chunksize) < 1 or min(args.warmup, args.offset, args.limit, args.max_nodes, args.sat_conflict_budget, args.timeout_seconds) < 0:
        parser.error("Counts must be positive; limits nonnegative")
    if len(args.solvers) != len(set(args.solvers)):
        parser.error("Do not repeat a solver in --solvers")
    return args


if __name__ == "__main__":
    run(parse_args())
