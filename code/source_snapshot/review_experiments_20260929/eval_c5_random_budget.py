"""A04: C5 budget curves and RMS-normalized random-state controls.

validation evaluates a preregistered small grid and writes a content-locked
selection. test requires that lock and validates the validation files' SHA256.
Equal parent+slot propagation steps and candidate caps do NOT mean equal FLOPs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import eval_d4_reflection_ablation as a03

SCALING = "selected_state_rms"
SUPPORT = "empty_cell_hidden_and_all_unit_hidden"
TIMING = "once_before_each_recovery_cycle"


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_plan(plan):
    if plan.get("protocol") != "a04_budget_v1":
        raise ValueError("Unknown budget protocol")
    if (plan.get("noise_normalization"), plan.get("noise_support"), plan.get("noise_timing")) != (SCALING, SUPPORT, TIMING):
        raise ValueError("Primary noise protocol is fixed; do not choose a normalization on test")
    if int(plan["parent_steps"]) < 1 or int(plan["random_repeats"]) < 1:
        raise ValueError("Positive parent steps/random repeats required")
    sigmas = plan["sigmas"]
    if not sigmas or len(set(sigmas)) != len(sigmas) or any(not np.isfinite(s) or s <= 0 for s in sigmas):
        raise ValueError("Noise sigma grid must contain distinct finite positive values")
    names = set()
    if not plan["schedules"]:
        raise ValueError("At least one schedule is required")
    for schedule in plan["schedules"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", schedule["name"]) or schedule["name"] in names:
            raise ValueError("Schedule names must be unique safe file labels")
        names.add(schedule["name"])
        for key in ("slots", "cycles", "recovery_steps"):
            if not isinstance(schedule[key], int) or schedule[key] < 1:
                raise ValueError(f"Schedule {key} must be a positive integer")


def noise_seed_words(seed, split, raw_id, cycle, slot, repeat, kind):
    """No sigma, schedule, active-batch position, or target enters the RNG key."""
    return [int(seed) & 0xffffffff, int(split), int(raw_id) & 0xffffffff,
            (int(raw_id) >> 32) & 0xffffffff, int(cycle), int(slot), int(repeat), int(kind)]


def noise_array(raw_ids, slots, nodes, dim, seed, split, cycle, repeat, kind):
    values = np.empty((len(raw_ids), slots, nodes, dim), dtype=np.float32)
    for row, raw_id in enumerate(raw_ids):
        for slot in range(slots):
            rng = np.random.default_rng(np.random.SeedSequence(noise_seed_words(seed, split, raw_id, cycle, slot, repeat, kind)))
            values[row, slot] = rng.standard_normal((nodes, dim), dtype=np.float32)
    return values.reshape(len(raw_ids) * slots, nodes, dim)


def perturb_rms(h, unit_h, puzzle, raw_ids, slots, sigma, seed, split, cycle, repeat):
    import torch
    mutable = puzzle.reshape(len(puzzle), 81).eq(0).unsqueeze(-1)
    weights = mutable.to(h.dtype)
    # Separate scale for each puzzle/slot tensor, computed over the actual support.
    denom = (weights.sum(dim=(1, 2)) * h.shape[-1]).clamp_min(1)
    cell_rms = ((h.square() * weights).sum(dim=(1, 2)) / denom).sqrt().clamp_min(1e-6)
    unit_rms = unit_h.square().mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
    cell_noise = torch.from_numpy(noise_array(raw_ids, slots, 81, h.shape[-1], seed, split, cycle, repeat, 0)).to(h.device)
    unit_noise = torch.from_numpy(noise_array(raw_ids, slots, 27, unit_h.shape[-1], seed, split, cycle, repeat, 1)).to(unit_h.device)
    return (h + sigma * cell_rms[:, None, None] * cell_noise * weights,
            unit_h + sigma * unit_rms[:, None, None] * unit_noise)


def evaluate_batch(model, puzzle, target, parent, raw_ids, schedule, method, sigma, repeat, seed, split):
    """One budget schedule with clue-aware first-valid active halting."""
    import torch
    from hybrid_c5_explainability_pilot import symbolic_reflect_intervened
    batch, device = len(puzzle), puzzle.device
    slots, cycles, recovery = (schedule[key] for key in ("slots", "cycles", "recovery_steps"))
    x0, unit_x0, parent_h, parent_u, anchor_logits = [value.clone() for value in parent]
    pred = anchor_logits.argmax(-1) + 1
    valid, _ = a03.torch_candidate_flags(pred, puzzle)
    grid = torch.zeros_like(puzzle)
    grid[valid] = pred[valid]
    stop_cycle = torch.full((batch,), -1, dtype=torch.int16, device=device)
    stop_cycle[valid] = 0
    stop_slot = torch.full((batch,), -1, dtype=torch.int16, device=device)
    exact = valid & (pred == target).reshape(batch, -1).all(1)
    steps = torch.full((batch,), int(model.cfg.parent_steps), dtype=torch.int32, device=device)
    candidates = torch.ones(batch, dtype=torch.int32, device=device)
    reflections = torch.zeros(batch, dtype=torch.int16, device=device)
    perturbations = torch.zeros(batch, dtype=torch.int16, device=device)
    active = torch.nonzero(~valid, as_tuple=False).squeeze(-1)
    if len(active):
        state = {"h": model._expand_slots(parent_h[active], slots), "u": model._expand_slots(parent_u[active], slots),
                 "x0": model._expand_slots(x0[active], slots), "u0": model._expand_slots(unit_x0[active], slots),
                 "puzzle": model._expand_slots(puzzle[active], slots)}
        state["dual"] = state["h"].new_zeros(len(active) * slots, 27, 9)
    for cycle in range(cycles):
        count = len(active)
        if not count:
            break
        steps[active] += slots * recovery
        candidates[active] += slots
        if method == "c5":
            reflections[active] += 1
            state["h"], state["u"], state["dual"], _ = symbolic_reflect_intervened(
                model, state["h"], state["u"], state["x0"], state["u0"], state["puzzle"], state["dual"],
                source_batch=count, slots=slots, cycle_index=cycle, ablation="full")
        elif method == "random":
            perturbations[active] += 1
            state["h"], state["u"] = perturb_rms(state["h"], state["u"], state["puzzle"],
                                                   raw_ids[active.cpu().numpy()], slots, sigma, seed, split, cycle, repeat)
        else:
            raise ValueError(method)
        for _ in range(recovery):
            state["h"], state["u"] = model.backbone.step(state["h"], state["u"], state["x0"], state["u0"])
        pred = model.backbone.logits_from_state(state["h"], state["puzzle"]).argmax(-1) + 1
        valid, _ = a03.torch_candidate_flags(pred, state["puzzle"])
        valid = valid.reshape(count, slots)
        solved = valid.any(1)
        if bool(solved.any()):
            slot = valid.to(torch.int64).argmax(1)
            chosen = pred.reshape(count, slots, 9, 9)[torch.arange(count, device=device), slot]
            ids = active[solved]
            grid[ids] = chosen[solved]
            stop_cycle[ids] = cycle + 1
            stop_slot[ids] = slot[solved].to(torch.int16)
            exact[ids] = (chosen[solved] == target[ids]).reshape(len(ids), -1).all(1)
        keep = ~solved
        active = active[keep]
        if len(active):
            state = {key: a03.keep_slots(value, keep, count, slots) for key, value in state.items()}
    return dict(first_valid=stop_cycle >= 0, selected_exact=exact, first_valid_cycle=stop_cycle,
                first_valid_slot=stop_slot, selected_grid=grid, propagation_slot_steps=steps,
                candidates_generated=candidates, reflection_events=reflections, perturbation_events=perturbations)


def verify_validation_lock(path):
    path = Path(path).resolve()
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("phase") != "validation" or lock.get("protocol") != "a04_budget_v1":
        raise ValueError("Test requires a validation-derived A04 lock")
    required_artifacts = {"metadata.json", "summary.json", "condition_summary.csv", "locked_selection.npz"}
    if {entry["file"] for entry in lock["validation_artifacts"]} != required_artifacts:
        raise ValueError("Incomplete validation artifact manifest")
    for entry in lock["validation_artifacts"]:
        source = (path.parent / entry["file"]).resolve()
        if source.parent != path.parent or a03.sha_file(source) != entry["sha256"]:
            raise ValueError(f"Validation artifact SHA mismatch: {entry['file']}")
    meta = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
    if meta["status"] != "complete" or meta["phase"] != "validation":
        raise ValueError("Validation did not complete successfully")
    if (meta["checkpoint_sha256"] != lock["checkpoint_sha256"] or meta["plan"] != lock["plan"]
            or meta["source_sha256"] != lock["source_sha256"]):
        raise ValueError("Validation metadata and lock do not agree")
    # Recompute the choice from the hashed validation log, not merely trust the lock.
    summaries = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
    selected = choose_sigmas(summaries, lock["plan"])
    if selected != lock["selected_sigma_by_schedule"]:
        raise ValueError("Locked sigma does not match the preregistered validation selection rule")
    return lock, a03.sha_file(path)


def choose_sigmas(summaries, plan):
    choices = {}
    for schedule in plan["schedules"]:
        scores = []
        for sigma in plan["sigmas"]:
            values = [row["selected_exact_rate"] for row in summaries
                      if row["method"] == "random" and row["schedule"] == schedule["name"] and row["sigma"] == sigma]
            if len(values) != plan["random_repeats"]:
                raise ValueError("Incomplete validation repeat grid")
            scores.append((float(np.mean(values)), float(sigma)))
        # Maximize mean Exact; a fixed smaller-sigma tie-breaker.
        choices[schedule["name"]] = sorted(scores, key=lambda item: (-item[0], item[1]))[0][1]
    return choices


def self_test(torch_test=False):
    plan = json.loads((HERE / "a04_budget_plan.json").read_text())
    validate_plan(plan)
    ids = np.array([45, 97])
    noise = noise_array(ids, 2, 3, 4, 7, 1, 0, 0, 0)
    single = noise_array(ids[1:], 2, 3, 4, 7, 1, 0, 0, 0)
    np.testing.assert_array_equal(noise[2:], single)
    prefix = noise_array(ids, 1, 3, 4, 7, 1, 0, 0, 0)
    np.testing.assert_array_equal(noise.reshape(2, 2, 3, 4)[:, 0], prefix)
    assert not np.array_equal(single, noise_array(ids[1:], 2, 3, 4, 7, 1, 0, 1, 0))
    rows = [dict(method="random", schedule=s["name"], sigma=sigma, selected_exact_rate=0.5)
            for s in plan["schedules"] for sigma in plan["sigmas"] for _ in range(plan["random_repeats"])]
    assert set(choose_sigmas(rows, plan).values()) == {min(plan["sigmas"])}
    # A changed validation log must fail before any test set is loaded.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="a04-lock-selftest-") as temp:
        folder = Path(temp)
        metadata = dict(status="complete", phase="validation", checkpoint_sha256="synthetic",
                        plan=plan, source_sha256={})
        a03.save_json(folder / "metadata.json", metadata)
        a03.save_json(folder / "summary.json", rows)
        (folder / "condition_summary.csv").write_text("synthetic\n")
        np.savez(folder / "locked_selection.npz", split_position=[1])
        lock = dict(protocol="a04_budget_v1", phase="validation", checkpoint_sha256="synthetic",
                    plan=plan, source_sha256={}, selected_sigma_by_schedule=choose_sigmas(rows, plan),
                    validation_artifacts=[dict(file=name, sha256=a03.sha_file(folder/name)) for name in
                                          ("metadata.json", "summary.json", "condition_summary.csv", "locked_selection.npz")])
        a03.save_json(folder / "lock.json", lock)
        verify_validation_lock(folder / "lock.json")
        (folder / "summary.json").write_text("[]")
        try:
            verify_validation_lock(folder / "lock.json")
        except ValueError:
            pass
        else:
            raise AssertionError("Changed validation log was accepted")
    a03.self_test()
    print("A04 NumPy tests passed: ID/batch/slot noise invariance, repeat separation, fixed validation tie-break, validation-log hash enforcement", flush=True)
    if torch_test:
        import torch
        from kaggle_sudoku_hyper_rrn_experiment import HyperRRNCfg, SudokuHyperRRN
        from train_symbolic_primal_dual_reflection import SymbolicPrimalDualCfg, SymbolicPrimalDualReflector
        torch.set_num_threads(1)
        torch.manual_seed(7)
        backbone = SudokuHyperRRN(HyperRRNCfg(model_type="hybrid", D=8, msg_hidden=16, train_T=2,
                                             eval_T=2, dropout=0., force_clues=True)).eval()
        cfg = SymbolicPrimalDualCfg(mode="global8", D=8, hidden=16, slots=8, cycles=2,
                                    recovery_steps=1, parent_steps=2, max_slots=8, dropout=0.)
        model = SymbolicPrimalDualReflector(backbone, cfg).eval()
        solved = torch.tensor([[((r*3+r//3+c)%9)+1 for c in range(9)] for r in range(9)])
        target = solved[None].repeat(2, 1, 1)
        puzzle = target.clone()
        puzzle[1] = 0
        schedule = dict(slots=2, cycles=2, recovery_steps=1)
        with torch.no_grad():
            parent = model.encode_parent(puzzle)
            for method in ("c5", "random"):
                result = evaluate_batch(model, puzzle, target, parent, ids, schedule, method, .1, 0, 7, 1)
                assert bool(result["selected_exact"][0]) and int(result["first_valid_cycle"][0]) == 0
                np.testing.assert_array_equal(a03.numpy_valid(result["selected_grid"].numpy(), puzzle.numpy()),
                                              result["first_valid"].numpy())
                assert bool((result["propagation_slot_steps"] <= 6).all())
                assert bool((result["candidates_generated"] <= 5).all())
                if method == "random":
                    assert not bool(result["reflection_events"].any())
        print("A04 tiny CPU C5/random integration check passed", flush=True)
        # Scripted real-tensor check covers anchor success, later success, and
        # exhaustion in the SAME active batch, with exact expected work counts.
        from types import SimpleNamespace

        class ScriptedBackbone:
            def step(self, h, u, x0, u0):
                result = h.clone()
                result[:, 0, 0] += 1
                return result, u

            def logits_from_state(self, h, puzzle):
                identity = torch.div(h[:, 0, 0], 100, rounding_mode="floor").long()
                updates = h[:, 0, 0] - identity * 100
                grids = torch.ones((len(h), 9, 9), dtype=torch.long)
                grids[(identity == 1) & (updates >= 4)] = solved
                return torch.nn.functional.one_hot(grids-1, num_classes=9).float() * 20

        class ScriptedModel:
            cfg = SimpleNamespace(parent_steps=64)
            backbone = ScriptedBackbone()

            def _expand_slots(self, value, slots):
                return value[:, None].expand(-1, slots, *value.shape[1:]).reshape(-1, *value.shape[1:]).clone()

        with torch.no_grad():
            scripted = ScriptedModel()
            h = torch.zeros(3, 81, 1)
            h[:, 0, 0] = torch.tensor([0., 100., 200.])
            u = torch.zeros(3, 27, 1)
            empty = torch.zeros(3, 9, 9, dtype=torch.long)
            anchor_grids = torch.ones_like(empty)
            anchor_grids[0] = solved
            logits = torch.nn.functional.one_hot(anchor_grids-1, num_classes=9).float() * 20
            result = evaluate_batch(scripted, empty, solved[None].repeat(3, 1, 1),
                                    (h*0, u, h, u, logits), np.array([1, 2, 3]),
                                    dict(slots=2, cycles=3, recovery_steps=2), "random", 0., 0, 7, 1)
            expected = {"first_valid_cycle": [0, 2, -1], "selected_exact": [True, True, False],
                        "propagation_slot_steps": [64, 72, 76], "candidates_generated": [1, 5, 7],
                        "reflection_events": [0, 0, 0], "perturbation_events": [0, 2, 3]}
            for key, value in expected.items():
                assert result[key].tolist() == value, (key, result[key].tolist(), value)
            assert not bool(result["selected_grid"][2].any())
        print("A04 scripted active-stop accounting passed: anchor/late/unsolved steps=64/72/76, candidates=1/5/7", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["validation", "test"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--cache-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--plan", default=str(HERE / "a04_budget_plan.json"))
    parser.add_argument("--lock-config")
    parser.add_argument("--expected-checkpoint-sha256", default="")
    parser.add_argument("--selection", choices=["d4", "all"], default="d4")
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-torch", action="store_true")
    args = parser.parse_args()
    if not (args.self_test or args.self_test_torch) and not all((args.phase, args.checkpoint, args.cache_path, args.output_dir)):
        parser.error("phase/checkpoint/cache-path/output-dir required")
    return args


def main():
    args = parse_args()
    if args.self_test or args.self_test_torch:
        self_test(args.self_test_torch)
        return
    if args.batch_size < 1 or args.limit < 0 or args.bootstrap_repeats < 1:
        raise ValueError("Invalid numerical argument")
    if args.smoke and not (1 <= args.limit <= 32):
        raise ValueError("Smoke requires --limit 1..32")
    if args.limit and not args.smoke:
        raise ValueError("Partial data runs are smoke only; use --smoke to prevent selecting on a partial validation set")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory")
    checkpoint_sha = a03.sha_file(args.checkpoint)
    if args.expected_checkpoint_sha256 and args.expected_checkpoint_sha256.lower() != checkpoint_sha:
        raise ValueError("Checkpoint SHA mismatch")
    validation_lock_sha, choices = None, None
    if args.phase == "test":
        if not args.lock_config:
            raise ValueError("Test requires --lock-config produced by validation")
        lock, validation_lock_sha = verify_validation_lock(args.lock_config)
        if lock["smoke_only"] and not args.smoke:
            raise ValueError("A smoke validation lock cannot authorize a full test evaluation")
        for key, value in (("checkpoint_sha256", checkpoint_sha), ("selection", args.selection),
                           ("seed", args.seed), ("batch_size", args.batch_size)):
            if lock[key] != value:
                raise ValueError(f"Test cannot change locked {key}")
        plan, choices = lock["plan"], lock["selected_sigma_by_schedule"]
    else:
        if args.lock_config:
            raise ValueError("Validation does not consume a test/validation lock")
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    validate_plan(plan)
    import torch
    from eval_symbolic_active_reflection import load_symbolic
    from sudoku_cache_utils import load_sudoku_dataset, load_sudoku_cache
    from sudoku_exchange_experiment import set_seed, set_torch_threads
    set_torch_threads()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    model, cfg, checkpoint = load_symbolic(args.checkpoint, device)
    model = model.float().eval()
    if int(cfg.parent_steps) != plan["parent_steps"] or cfg.mode != "global8" or not model.backbone.cfg.force_clues:
        raise ValueError("Checkpoint does not support the predeclared parent/global8/clue policy")
    for schedule in plan["schedules"]:
        if schedule["slots"] > cfg.slots or schedule["cycles"] > cfg.cycles:
            raise ValueError("Cannot extrapolate beyond trained slot/cycle embeddings")
    split = 1 if args.phase == "validation" else 2
    dataset = load_sudoku_dataset(args.cache_path, split=split, limit=0)
    ids = (np.flatnonzero(np.nan_to_num(dataset.ratings, nan=-np.inf) > 4) if args.selection == "d4" else np.arange(len(dataset)))
    if args.expected_count and len(ids) != args.expected_count:
        raise ValueError("Selected dataset count mismatch")
    full_count = len(ids)
    if args.limit:
        ids = ids[:args.limit]
    if not len(ids):
        raise ValueError("Empty selection")
    cache = load_sudoku_cache(args.cache_path)
    raw_split_ids = np.flatnonzero(np.asarray(cache["splits"]) == split)
    if hasattr(cache, "close"):
        cache.close()
    raw_ids = raw_split_ids[ids]
    puzzles = np.asarray(dataset.puzzles[ids], dtype=np.uint8)
    targets = np.asarray(dataset.solutions[ids], dtype=np.uint8)
    ratings = np.asarray(dataset.ratings[ids], dtype=np.float32)
    if not a03.numpy_valid(targets, puzzles).all():
        raise ValueError("Invalid target or clue disagreement")
    configurations = []
    for schedule in plan["schedules"]:
        configurations.append(dict(schedule=schedule, method="c5", sigma=0., repeat=0, label=schedule["name"] + "_c5"))
        for sigma in (plan["sigmas"] if choices is None else [choices[schedule["name"]]]):
            for repeat in range(plan["random_repeats"]):
                configurations.append(dict(schedule=schedule, method="random", sigma=float(sigma), repeat=repeat,
                                           label=f"{schedule['name']}_random_s{sigma}_r{repeat}"))
    output.mkdir(parents=True, exist_ok=True)
    sources = a03.source_manifest()
    sources[str(Path(__file__).resolve())] = a03.sha_file(__file__)
    sources[str(Path(a03.__file__).resolve())] = a03.sha_file(a03.__file__)
    if args.phase == "test" and sources != lock["source_sha256"]:
        raise ValueError("Source code changed since validation; do not silently reuse the lock")
    np.savez_compressed(output / "locked_selection.npz", split_position=ids, raw_cache_id=raw_ids, puzzle=puzzles, target=targets, rating=ratings)
    metadata = dict(status="running", phase=args.phase, args=vars(args), plan=plan, checkpoint_sha256=checkpoint_sha,
                    base_checkpoint=checkpoint["base_checkpoint"], base_checkpoint_sha256=a03.sha_file(checkpoint["base_checkpoint"]),
                    source_sha256=sources, cfg=asdict(cfg), validation_lock_sha256=validation_lock_sha,
                    data=dict(split=split, selected_count=len(ids), full_selected_count=full_count,
                              raw_ids_sha256=a03.sha_array(raw_ids, "<i8"), puzzle_sha256=a03.sha_array(puzzles, "u1"),
                              target_sha256=a03.sha_array(targets, "u1")),
                    python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                    device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    precision="FP32, AMP off, TF32 off", deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    interpretation="Propagation slot-step and candidate caps matched per schedule, not FLOPs, parameters, wall time, or realized active work.",
                    c5_slots="prefix of trained slot embeddings; changing K changes cross-slot means; inference intervention, no retraining",
                    noise="For each trajectory separately, Gaussian sigma*RMS over empty-cell H support and separately all U; RMS floor 1e-6. Once before each recovery cycle; input/clue logits unchanged.",
                    noise_key="NumPy SeedSequence(seed, split, raw_ID_low, raw_ID_high, cycle, slot, repeat, cell_or_unit); independent of sigma/schedule/batch position; common random numbers across sigma and slot prefixes",
                    timing="shared parent timed separately; condition time includes RNG/CPU noise transfer for random and excludes shared parent; unsuitable as production throughput benchmark",
                    budget_caps={s["name"]: dict(propagation_slot_steps=cfg.parent_steps+s["slots"]*s["cycles"]*s["recovery_steps"],
                                                  candidates=1+s["slots"]*s["cycles"], c5_reflection_events=s["cycles"],
                                                  random_reflection_events=0) for s in plan["schedules"]})
    a03.save_json(output / "metadata.json", metadata)
    collected = {c["label"]: {} for c in configurations}
    elapsed = {c["label"]: 0.0 for c in configurations}
    reflection_calls = {c["label"]: 0 for c in configurations}
    propagation_calls = {c["label"]: 0 for c in configurations}
    parent_seconds = 0.0

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        for start in range(0, len(ids), args.batch_size):
            end = min(start+args.batch_size, len(ids))
            puzzle = torch.as_tensor(puzzles[start:end], dtype=torch.long, device=device)
            target = torch.as_tensor(targets[start:end], dtype=torch.long, device=device)
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                sync()
                begin = time.perf_counter()
                parent = model.encode_parent(puzzle)
                sync()
                parent_seconds += time.perf_counter()-begin
                for config in configurations:
                    begin = time.perf_counter()
                    values = evaluate_batch(model, puzzle, target, parent, raw_ids[start:end], config["schedule"],
                                            config["method"], config["sigma"], config["repeat"], args.seed, split)
                    sync()
                    elapsed[config["label"]] += time.perf_counter()-begin
                    reflection_calls[config["label"]] += int(values["reflection_events"].max())
                    active_cycles = int(torch.maximum(values["reflection_events"], values["perturbation_events"]).max())
                    propagation_calls[config["label"]] += active_cycles * config["schedule"]["recovery_steps"]
                    for key, value in values.items():
                        collected[config["label"]].setdefault(key, []).append(value.cpu().numpy())
            if end % 512 < end-start or end == len(ids):
                print(f"[A04] phase={args.phase} seen={end}/{len(ids)} configurations={len(configurations)}", flush=True)
        summaries, results = [], {}
        for config in configurations:
            values = {key: np.concatenate(parts) for key, parts in collected[config["label"]].items()}
            valid, exact, grid = values["first_valid"], values["selected_exact"], values["selected_grid"]
            np.testing.assert_array_equal(a03.numpy_valid(grid, puzzles), valid)
            np.testing.assert_array_equal((grid == targets).all(axis=(1, 2)) & valid, exact)
            if np.any(grid[~valid]):
                raise AssertionError("Unresolved output must be zero")
            results[config["label"]] = values
            np.savez_compressed(output / (config["label"] + "_details.npz"), split_position=ids, raw_cache_id=raw_ids, **values)
            summaries.append(dict(label=config["label"], schedule=config["schedule"]["name"], method=config["method"],
                                  sigma=config["sigma"], repeat=config["repeat"], n=len(ids),
                                  selected_exact_count=int(exact.sum()), selected_exact_rate=float(exact.mean()),
                                  first_valid_count=int(valid.sum()), first_valid_rate=float(valid.mean()),
                                  valid_not_exact_count=int((valid & ~exact).sum()), unsolved_count=int((~valid).sum()),
                                  actual_propagation_slot_steps=int(values["propagation_slot_steps"].sum()),
                                  actual_candidates_generated=int(values["candidates_generated"].sum()),
                                  actual_reflection_events=int(values["reflection_events"].sum()),
                                  actual_reflection_slot_events=int(values["reflection_events"].sum())*config["schedule"]["slots"],
                                  actual_reflection_function_calls=reflection_calls[config["label"]],
                                  actual_recovery_step_function_calls=propagation_calls[config["label"]],
                                  actual_perturbation_events=int(values["perturbation_events"].sum()),
                                  condition_seconds_excluding_parent=elapsed[config["label"]]))
        a03.save_csv(output / "condition_summary.csv", summaries)
        a03.save_json(output / "summary.json", summaries)
        if args.phase == "test":
            pairs = []
            for index, schedule in enumerate(plan["schedules"]):
                c5 = results[schedule["name"] + "_c5"]
                random_configs = [c for c in configurations if c["method"] == "random" and c["schedule"]["name"] == schedule["name"]]
                for metric_index, metric in enumerate(("selected_exact", "first_valid")):
                    matrix = np.stack([results[c["label"]][metric] for c in random_configs]).astype(float)
                    delta = c5[metric].astype(float) - matrix.mean(axis=0)
                    low, high = a03.paired_ci(delta, args.bootstrap_repeats, args.seed+1009*index+metric_index)
                    pairs.append(dict(schedule=schedule["name"], metric=metric, sigma=choices[schedule["name"]],
                                      n=len(ids), c5_rate=float(np.mean(c5[metric])), random_repeat_mean=float(matrix.mean()),
                                      random_repeat_rate_std=float(matrix.mean(axis=1).std(ddof=1)) if len(matrix)>1 else 0.,
                                      c5_minus_random_pp=float(delta.mean()*100), ci_low_pp=float(low*100), ci_high_pp=float(high*100),
                                      ci_scope="paired puzzle bootstrap conditional on these inference-noise repeats; not training-seed uncertainty"))
            a03.save_csv(output / "paired_budget_comparison.csv", pairs)
        metadata.update(status="complete", parent_seconds=parent_seconds, condition_seconds=elapsed,
                        shared_parent_forward_calls=(len(ids)+args.batch_size-1)//args.batch_size,
                        count_definitions="reflection_events sums active puzzle-cycles; reflection_slot_events multiplies by K; function_calls counts actual batched API invocations; propagation_slot_steps counts parent once per puzzle plus every active trajectory update")
        a03.save_json(output / "metadata.json", metadata)
        if args.phase == "validation":
            # Hash the completed validation log and metadata before writing the lock.
            lock = dict(protocol="a04_budget_v1", phase="validation", smoke_only=args.smoke,
                        checkpoint_sha256=checkpoint_sha, source_sha256=sources, selection=args.selection,
                        seed=args.seed, batch_size=args.batch_size, plan=plan,
                        selection_rule="highest validation mean selected Exact across noise repeats; ties choose smaller sigma",
                        selected_sigma_by_schedule=choose_sigmas(summaries, plan),
                        validation_artifacts=[dict(file=name, sha256=a03.sha_file(output/name)) for name in
                                              ("metadata.json", "summary.json", "condition_summary.csv", "locked_selection.npz")])
            a03.save_json(output / "selected_validation_lock.json", lock)
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        a03.save_json(output / "metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()
