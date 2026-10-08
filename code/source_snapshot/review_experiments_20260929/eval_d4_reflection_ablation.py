"""A03: full-D4 inference ablations using the unchanged pilot interventions.

Every condition starts from one shared parent state for each batch. Candidate
order is anchor, cycle, slot; the first valid candidate is selected without
using the target. Exact scores that selected candidate; unresolved = zero.
The ablations have different computation and are not equal-budget baselines.
"""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONDITIONS = ("full", "no_recovery", "recovery_only", "no_cell")
DEFINITIONS = {
    "full": "Unchanged pilot symbolic_reflect_intervened(full), followed by configured propagation.",
    "no_recovery": "Unchanged pilot reflection writebacks; skip all propagation after each reflection event.",
    "recovery_only": "Unchanged pilot recovery_only: skip base and symbolic cell/unit writebacks; retain propagation. Residual/dual/attention diagnostic computation still runs.",
    "no_cell": "Unchanged pilot no_cell: disable base and symbolic direct cell writebacks; unit writebacks and propagation remain. Cell proposals are still computed.",
}
COMPUTE_CHANGES = {
    "full": "Reference; actual work also depends on first-valid stopping.",
    "no_recovery": "Removes slots*recovery_steps propagation updates per active cycle; reflection still executes.",
    "recovery_only": "Keeps propagation cap, skips learned state-writeback bodies; diagnostic calculations remain; total arithmetic changes.",
    "no_cell": "Keeps propagation cap and computes proposal bodies, suppresses cell writeback arithmetic; stopping changes actual work.",
}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_array(array, dtype):
    value = np.ascontiguousarray(array, dtype=dtype)
    digest = hashlib.sha256(json.dumps(list(value.shape)).encode())
    digest.update(value.dtype.str.encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def source_manifest():
    pending, seen = [Path(__file__).resolve()], {}
    allowed = {ROOT, ROOT.parent}

    def resolve(name):
        try:
            spec = importlib.util.find_spec(name.split(".")[0])
        except (ImportError, ValueError, AttributeError):
            return None
        if not spec or not spec.origin or spec.origin in {"built-in", "frozen"}:
            return None
        path = Path(spec.origin).resolve()
        return path if path.suffix == ".py" and path.parent in allowed else None

    for name in ("hybrid_c5_explainability_pilot", "eval_symbolic_active_reflection"):
        path = resolve(name)
        if path is None:
            raise ImportError(f"Required local source is unavailable: {name}")
        pending.append(path)
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen[path] = sha_file(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = ([node.module] if isinstance(node, ast.ImportFrom) else
                     [entry.name for entry in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                candidate = resolve(name) if name else None
                if candidate is not None:
                    pending.append(candidate)
    return {str(path): digest for path, digest in sorted(seen.items())}


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numpy_valid(grids, puzzle):
    """Independent output verification, including clues and the digit domain."""
    grids = np.asarray(grids).reshape(-1, 9, 9)
    puzzle = np.broadcast_to(np.asarray(puzzle), grids.shape)
    target = np.arange(1, 10)
    rows = (np.sort(grids, axis=2) == target).all(axis=(1, 2))
    columns = (np.sort(grids.transpose(0, 2, 1), axis=2) == target).all(axis=(1, 2))
    boxes = grids.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    boxes_ok = (np.sort(boxes, axis=2) == target).all(axis=(1, 2))
    clues = ((puzzle == 0) | (grids == puzzle)).all(axis=(1, 2))
    return rows & columns & boxes_ok & clues


def paired_ci(delta, repeats, seed):
    """Paired puzzle bootstrap with a bounded NumPy allocation."""
    delta = np.asarray(delta, dtype=np.float64)
    if not len(delta) or repeats < 1:
        raise ValueError("Nonempty paired observations and positive repeats are required")
    rng = np.random.default_rng(seed)
    means = np.empty(repeats)
    chunk = max(1, min(256, 1_000_000 // len(delta)))
    for start in range(0, repeats, chunk):
        end = min(start + chunk, repeats)
        means[start:end] = rng.choice(delta, (end - start, len(delta)), replace=True).mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def pair_statistics(full, other, repeats, seed):
    full, other = np.asarray(full, dtype=bool), np.asarray(other, dtype=bool)
    if full.shape != other.shape:
        raise ValueError("Paired outcomes must have the same shape")
    delta = other.astype(int) - full.astype(int)
    low, high = paired_ci(delta, repeats, seed)
    return dict(n=len(full), full_count=int(full.sum()), condition_count=int(other.sum()),
                difference_pp=float(delta.mean() * 100), ci_low_pp=float(low * 100), ci_high_pp=float(high * 100),
                both_success=int((full & other).sum()), lost=int((full & ~other).sum()),
                gained=int((~full & other).sum()), neither_success=int((~full & ~other).sum()),
                bootstrap_repeats=repeats, bootstrap_seed=seed)


def torch_candidate_flags(pred, puzzle):
    import torch
    from eval_hybrid_hyper_rrn_restarts import hard_violation_count_batch
    clue_ok = ((pred == puzzle) | (puzzle == 0)).reshape(len(pred), -1).all(dim=1)
    domain = ((pred >= 1) & (pred <= 9)).reshape(len(pred), -1).all(dim=1)
    valid = (hard_violation_count_batch(pred) == 0) & clue_ok & domain
    if not bool(clue_ok.all()):
        raise AssertionError("A generated candidate changed a given clue")
    return valid, clue_ok


def keep_slots(tensor, keep, old_batch, slots):
    return tensor.reshape(old_batch, slots, *tensor.shape[1:])[keep].reshape(
        int(keep.sum()) * slots, *tensor.shape[1:]).contiguous()


def active_ablation_batch(model, puzzle, solution, parent, condition):
    """First-valid active halting around the unchanged legacy intervention."""
    import torch
    from hybrid_c5_explainability_pilot import symbolic_reflect_intervened
    if condition not in CONDITIONS:
        raise ValueError(condition)
    batch, slots, cycles = len(puzzle), int(model.cfg.slots), int(model.cfg.cycles)
    recovery_steps = 0 if condition == "no_recovery" else int(model.cfg.recovery_steps)
    # Clone all shared parent tensors so no arm can mutate another arm's prefix.
    x0, unit_x0, parent_h, parent_u, anchor_logits = [value.clone() for value in parent]
    anchor_pred = anchor_logits.argmax(dim=-1) + 1
    anchor_valid, _ = torch_candidate_flags(anchor_pred, puzzle)
    device = puzzle.device
    selected = torch.zeros_like(puzzle)
    selected[anchor_valid] = anchor_pred[anchor_valid]
    first_cycle = torch.full((batch,), -1, dtype=torch.int16, device=device)
    first_cycle[anchor_valid] = 0
    first_slot = torch.full((batch,), -1, dtype=torch.int16, device=device)
    selected_exact = torch.zeros(batch, dtype=torch.bool, device=device)
    anchor_exact = (anchor_pred == solution).reshape(batch, -1).all(1)
    selected_exact[anchor_valid] = anchor_exact[anchor_valid]
    propagated = torch.full((batch,), int(model.cfg.parent_steps), dtype=torch.int32, device=device)
    reflected = torch.zeros(batch, dtype=torch.int16, device=device)
    candidates = torch.ones(batch, dtype=torch.int16, device=device)
    active = torch.nonzero(~anchor_valid, as_tuple=False).squeeze(-1)
    if len(active):
        state = {"h": model._expand_slots(parent_h[active], slots),
                 "unit_h": model._expand_slots(parent_u[active], slots),
                 "x0": model._expand_slots(x0[active], slots),
                 "unit_x0": model._expand_slots(unit_x0[active], slots),
                 "puzzle": model._expand_slots(puzzle[active], slots)}
        state["dual"] = state["h"].new_zeros(len(active) * slots, 27, 9)
    for cycle in range(cycles):
        count = len(active)
        if not count:
            break
        reflected[active] += 1
        propagated[active] += slots * recovery_steps
        candidates[active] += slots
        state["h"], state["unit_h"], state["dual"], _ = symbolic_reflect_intervened(
            model, state["h"], state["unit_h"], state["x0"], state["unit_x0"],
            state["puzzle"], state["dual"], source_batch=count, slots=slots,
            cycle_index=cycle, ablation=condition)
        for _ in range(recovery_steps):
            state["h"], state["unit_h"] = model.backbone.step(
                state["h"], state["unit_h"], state["x0"], state["unit_x0"])
        logits = model.backbone.logits_from_state(state["h"], state["puzzle"])
        pred = logits.argmax(dim=-1) + 1
        valid, _ = torch_candidate_flags(pred, state["puzzle"])
        valid = valid.reshape(count, slots)
        solved = valid.any(dim=1)
        if bool(solved.any()):
            # Lowest valid slot is fixed before any target comparisons.
            slot = valid.to(torch.int64).argmax(dim=1)
            chosen = pred.reshape(count, slots, 9, 9)[torch.arange(count, device=device), slot]
            chosen_ids = active[solved]
            selected[chosen_ids] = chosen[solved]
            first_cycle[chosen_ids] = cycle + 1
            first_slot[chosen_ids] = slot[solved].to(torch.int16)
            selected_exact[chosen_ids] = (chosen[solved] == solution[chosen_ids]).reshape(len(chosen_ids), -1).all(1)
        keep = ~solved
        active = active[keep]
        if len(active):
            state = {key: keep_slots(value, keep, count, slots) for key, value in state.items()}
    return {"first_valid_cycle": first_cycle, "first_valid_slot": first_slot,
            "first_valid": first_cycle >= 0, "selected_exact": selected_exact,
            "selected_grid": selected, "propagation_slot_steps": propagated,
            "reflection_events": reflected, "candidates_generated": candidates}


def self_test():
    """Five data/metric boundary checks requiring NumPy only."""
    solved = np.asarray([[((r * 3 + r // 3 + c) % 9) + 1 for c in range(9)] for r in range(9)])
    alternate = solved % 9 + 1
    puzzle = np.zeros((9, 9), dtype=int)
    assert numpy_valid([solved, alternate], puzzle).all()
    clue_puzzle = puzzle.copy()
    clue_puzzle[0, 0] = solved[0, 0]
    assert numpy_valid([solved, alternate], clue_puzzle).tolist() == [True, False]
    assert not numpy_valid(np.zeros((1, 9, 9), dtype=int), puzzle)[0]
    # A first legal alternative must be selected even when a later slot is exact.
    proposals = np.stack([alternate, solved])
    first = int(np.flatnonzero(numpy_valid(proposals, puzzle))[0])
    assert first == 0 and not np.array_equal(proposals[first], solved)
    stats = pair_statistics([True, False], [False, True], 200, 17)
    assert stats["difference_pp"] == 0 and stats["lost"] == 1 and stats["gained"] == 1
    np.testing.assert_array_equal(paired_ci([-1, -1], 200, 17), [-1, -1])
    print("A03 NumPy self-test: 5 boundary groups passed", flush=True)


def torch_self_test():
    """CPU integration check with a tiny random model and real pilot functions."""
    import torch
    from kaggle_sudoku_hyper_rrn_experiment import HyperRRNCfg, SudokuHyperRRN
    from train_symbolic_primal_dual_reflection import SymbolicPrimalDualCfg, SymbolicPrimalDualReflector
    torch.manual_seed(7)
    backbone = SudokuHyperRRN(HyperRRNCfg(model_type="hybrid", D=8, msg_hidden=16,
                                         train_T=2, eval_T=2, dropout=0., force_clues=True)).eval()
    cfg = SymbolicPrimalDualCfg(mode="global8", D=8, hidden=16, slots=8, cycles=2,
                                recovery_steps=1, parent_steps=2, max_slots=8, dropout=0.)
    model = SymbolicPrimalDualReflector(backbone, cfg).eval()
    solved = torch.tensor([[((r * 3 + r // 3 + c) % 9) + 1 for c in range(9)] for r in range(9)])
    solution = solved.unsqueeze(0).repeat(2, 1, 1)
    puzzle = solution.clone()
    puzzle[1] = 0
    with torch.no_grad():
        parent = model.encode_parent(puzzle)
        snapshots = [v.clone() for v in parent]
        for condition in CONDITIONS:
            values = active_ablation_batch(model, puzzle, solution, parent, condition)
            assert values["first_valid_cycle"][0] == 0 and bool(values["selected_exact"][0])
            assert values["reflection_events"][0] == 0 and values["candidates_generated"][0] == 1
            if condition == "no_recovery":
                assert bool((values["propagation_slot_steps"] == cfg.parent_steps).all())
            for before, after in zip(snapshots, parent):
                assert torch.equal(before, after), "Shared parent was mutated"
            grids = values["selected_grid"].numpy()
            valid = values["first_valid"].numpy()
            np.testing.assert_array_equal(numpy_valid(grids, puzzle.numpy()), valid)
    print("A03 CPU model/pilot integration self-test passed", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--cache-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-checkpoint-sha256", default="")
    parser.add_argument("--expected-count", type=int, default=6543)
    parser.add_argument("--expected-selection-sha256", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--tf32", choices=["off", "on"], default="off")
    parser.add_argument("--determinism", choices=["off", "warn", "error"], default="off")
    parser.add_argument("--progress-every", type=int, default=512)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-torch", action="store_true")
    args = parser.parse_args()
    if not (args.self_test or args.self_test_torch) and not all((args.checkpoint, args.cache_path, args.output_dir)):
        parser.error("--checkpoint, --cache-path and --output-dir are required for evaluation")
    return args


def main():
    args = parse_args()
    if args.self_test or args.self_test_torch:
        self_test()
        if args.self_test_torch:
            torch_self_test()
        return
    if args.batch_size < 1 or args.limit < 0 or args.bootstrap_repeats < 1:
        raise ValueError("Invalid batch, limit or bootstrap size")
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new/empty output directory: {out}")
    checkpoint_sha = sha_file(args.checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_sha != args.expected_checkpoint_sha256.lower():
        raise ValueError("Checkpoint SHA-256 mismatch")
    if args.determinism != "off":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from eval_symbolic_active_reflection import load_symbolic
    from sudoku_cache_utils import load_sudoku_dataset, load_sudoku_cache
    from sudoku_exchange_experiment import set_seed, set_torch_threads
    set_torch_threads()
    set_seed(args.seed)
    torch.use_deterministic_algorithms(args.determinism != "off", warn_only=args.determinism == "warn")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = args.determinism != "off"
    torch.backends.cuda.matmul.allow_tf32 = args.tf32 == "on"
    torch.backends.cudnn.allow_tf32 = args.tf32 == "on"
    torch.set_float32_matmul_precision("high" if args.tf32 == "on" else "highest")
    device = torch.device(args.device)
    model, cfg, checkpoint = load_symbolic(args.checkpoint, device)
    model = model.float().eval()
    if cfg.mode != "global8" or not model.backbone.cfg.force_clues:
        raise ValueError("This legacy-pilot audit requires global8 and force_clues=True")
    dataset = load_sudoku_dataset(args.cache_path, split=2, limit=0)
    indices = np.flatnonzero(np.nan_to_num(dataset.ratings, nan=-np.inf) > 4.0).astype(np.int64)
    total_d4 = len(indices)
    if args.expected_count and total_d4 != args.expected_count:
        raise ValueError(f"D4 has {total_d4} puzzles, expected {args.expected_count}")
    if args.limit:
        indices = indices[:args.limit]
    if not len(indices):
        raise ValueError("Empty D4 selection")
    cache = load_sudoku_cache(args.cache_path)
    test_ids = np.flatnonzero(np.asarray(cache["splits"]) == 2)
    if hasattr(cache, "close"):
        cache.close()
    if len(test_ids) != len(dataset):
        raise AssertionError("Test dataset/cache ID mismatch")
    puzzles = np.asarray(dataset.puzzles[indices], dtype=np.uint8)
    solutions = np.asarray(dataset.solutions[indices], dtype=np.uint8)
    ratings = np.asarray(dataset.ratings[indices], dtype=np.float32)
    if not numpy_valid(solutions, puzzles).all():
        raise AssertionError("A supplied target is invalid or changes a clue")
    selection = dict(split="test", criterion="rating > 4.0; original split-position order; no outcome selection",
                     total_D4=total_d4, evaluated=len(indices), split_ids_sha256=sha_array(test_ids, "<i8"),
                     positions_sha256=sha_array(indices, "<i8"), raw_ids_sha256=sha_array(test_ids[indices], "<i8"),
                     puzzle_sha256=sha_array(puzzles, "u1"), target_sha256=sha_array(solutions, "u1"),
                     ratings_sha256=sha_array(ratings, "<f4"))
    selection_sha = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    if args.expected_selection_sha256 and args.expected_selection_sha256 != selection_sha:
        raise ValueError("Selection SHA-256 mismatch")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "locked_selection.npz", test_position=indices, raw_cache_id=test_ids[indices],
                        puzzle=puzzles, target=solutions, rating=ratings)
    metadata = dict(status="running", args=vars(args), checkpoint_sha256=checkpoint_sha,
                    base_checkpoint=checkpoint["base_checkpoint"], base_checkpoint_sha256=sha_file(checkpoint["base_checkpoint"]),
                    checkpoint_step=checkpoint.get("step"), cfg=asdict(cfg), source_sha256=source_manifest(),
                    selection=selection, selection_sha256=selection_sha, definitions=DEFINITIONS, computation=COMPUTE_CHANGES,
                    python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    device=str(device), parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}), amp=False,
                    tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    candidate_order="anchor; then ascending cycle and slot index; first Valid, without target access",
                    shared_prefix="one encode_parent per batch; all five parent tensors cloned per condition",
                    full_runner="legacy pilot intervention function, including full; does not assume bitwise equivalence to native full-test runner",
                    inference_only=True, independently_trained_seeds=1,
                    time_scope="parent separately; per-condition totals exclude shared parent and are audit timing, not equal-budget throughput",
                    max_propagation_slot_steps={name: int(cfg.parent_steps + cfg.cycles * cfg.slots * (0 if name == "no_recovery" else cfg.recovery_steps)) for name in CONDITIONS})
    save_json(out / "metadata.json", metadata)
    collected = {name: {} for name in CONDITIONS}
    elapsed = {name: 0.0 for name in CONDITIONS}
    parent_seconds = 0.0

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        for offset in range(0, len(indices), args.batch_size):
            end = min(offset + args.batch_size, len(indices))
            puzzle = torch.as_tensor(puzzles[offset:end], dtype=torch.long, device=device)
            solution = torch.as_tensor(solutions[offset:end], dtype=torch.long, device=device)
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                sync()
                started = time.perf_counter()
                parent = model.encode_parent(puzzle)
                sync()
                parent_seconds += time.perf_counter() - started
                for condition in CONDITIONS:
                    started = time.perf_counter()
                    values = active_ablation_batch(model, puzzle, solution, parent, condition)
                    sync()
                    elapsed[condition] += time.perf_counter() - started
                    for key, value in values.items():
                        collected[condition].setdefault(key, []).append(value.cpu().numpy())
            if args.progress_every and (end % args.progress_every < end - offset or end == len(indices)):
                print(f"[A03] seen={end}/{len(indices)}", flush=True)
        results, summary, per_puzzle = {}, [], []
        for condition in CONDITIONS:
            values = {key: np.concatenate(parts) for key, parts in collected[condition].items()}
            valid = values["first_valid"].astype(bool)
            exact = values["selected_exact"].astype(bool)
            grids = values["selected_grid"]
            np.testing.assert_array_equal(numpy_valid(grids, puzzles), valid)
            np.testing.assert_array_equal((grids == solutions).all(axis=(1, 2)) & valid, exact)
            if np.any(grids[~valid]):
                raise AssertionError("Unsolved output must be the zero sentinel")
            values["selected_clue_preserving"] = valid & ((grids == puzzles) | (puzzles == 0)).all(axis=(1, 2))
            results[condition] = values
            np.savez_compressed(out / f"{condition}_details.npz", test_position=indices, raw_cache_id=test_ids[indices], rating=ratings, **values)
            row = dict(condition=condition, n=len(indices), first_valid_count=int(valid.sum()), selected_exact_count=int(exact.sum()),
                       valid_not_exact_count=int((valid & ~exact).sum()), unsolved_count=int((~valid).sum()),
                       first_valid_rate=float(valid.mean()), selected_exact_rate=float(exact.mean()),
                       all_generated_candidates_clue_preserving=True, selected_clue_preserving_count=int(values["selected_clue_preserving"].sum()),
                       propagation_slot_steps_total=int(values["propagation_slot_steps"].sum()),
                       reflection_events_total=int(values["reflection_events"].sum()), candidates_generated_total=int(values["candidates_generated"].sum()),
                       condition_seconds_excluding_shared_parent=elapsed[condition])
            for cycle in range(cfg.cycles + 1):
                reached = valid & (values["first_valid_cycle"] <= cycle)
                row[f"cycle{cycle}_cumulative_valid"] = int(reached.sum())
                row[f"cycle{cycle}_cumulative_exact"] = int((reached & exact).sum())
            summary.append(row)
            per_puzzle.extend(dict(condition=condition, test_position=int(indices[i]), raw_cache_id=int(test_ids[indices[i]]),
                                   rating=float(ratings[i]), first_valid_cycle=int(values["first_valid_cycle"][i]),
                                   first_valid_slot=int(values["first_valid_slot"][i]), first_valid=int(valid[i]),
                                   selected_exact=int(exact[i]), selected_clue_preserving=int(values["selected_clue_preserving"][i]),
                                   propagation_slot_steps=int(values["propagation_slot_steps"][i]),
                                   reflection_events=int(values["reflection_events"][i]), candidates_generated=int(values["candidates_generated"][i]))
                              for i in range(len(indices)))
        paired = []
        for condition_index, condition in enumerate(CONDITIONS[1:], 1):
            for metric_index, metric in enumerate(("first_valid", "selected_exact")):
                paired.append(dict(condition=condition, comparator="full", metric=metric,
                                   **pair_statistics(results["full"][metric], results[condition][metric],
                                                     args.bootstrap_repeats, args.seed + 1009 * condition_index + metric_index)))
        save_csv(out / "condition_summary.csv", summary)
        save_csv(out / "paired_summary.csv", paired)
        save_csv(out / "per_puzzle.csv", per_puzzle)
        metadata.update(status="complete", parent_seconds=parent_seconds, condition_seconds=elapsed,
                        output_verification="Independent NumPy row/column/box/domain/clue check matched all selected Valid flags; target match checked separately",
                        ci="Puzzle-level paired percentile bootstrap; 2.5/97.5 percentiles; no retraining-seed uncertainty",
                        output_sha256={p.name: sha_file(p) for p in sorted(out.iterdir()) if p.name != "metadata.json" and p.is_file()})
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(out / "metadata.json", metadata)


if __name__ == "__main__":
    main()
