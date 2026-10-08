"""Read-only evaluation of legacy RRN/Hyper-RRN best and final checkpoints.

Outputs source-row keyed predictions and separate Exact/Valid, plus explicit
timing scopes. Trusted project checkpoints only: legacy pickle loading is needed
for __main__ dataclass configurations. Does not choose a checkpoint using test.
"""

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from e2_metrics import aggregate_metrics, board_metrics


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filter_test_global_ids(test_rows, manifest_path):
    """Intersect an outcome-independent row manifest, preserving cache order."""
    allowed = np.load(manifest_path, allow_pickle=False)
    if allowed.ndim != 1 or not np.issubdtype(allowed.dtype, np.integer):
        raise ValueError("Test global IDs must be a one-dimensional integer .npy array")
    if np.any(allowed < 0) or len(np.unique(allowed)) != len(allowed):
        raise ValueError("Test global IDs must be nonnegative and unique")
    test_rows = np.asarray(test_rows)
    selected = test_rows[np.isin(test_rows, allowed)]
    metadata = {
        "path": str(Path(manifest_path).resolve()),
        "sha256": sha256_file(manifest_path),
        "original_test_count": len(test_rows), "manifest_count": len(allowed),
        "retained_test_count_before_rating_or_limit": len(selected),
        "excluded_test_count": len(test_rows) - len(selected),
        "manifest_ids_outside_test_count": len(allowed) - len(selected),
        "ordering": "original cache split-2 order, not manifest order",
        "selection_basis": "caller-supplied data-audit manifest; no model outcomes used",
    }
    return selected, metadata


def load_model(path, device):
    import __main__
    import torch
    from kaggle_sudoku_hyper_rrn_experiment import HyperRRNCfg, SudokuHyperRRN
    from kaggle_sudoku_rrn_paper_experiment import RRNPaperCfg, SudokuRRNPaper

    __main__.HyperRRNCfg = HyperRRNCfg
    __main__.RRNPaperCfg = RRNPaperCfg
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_cfg = checkpoint["cfg"]
    cfg = asdict(saved_cfg) if is_dataclass(saved_cfg) else dict(saved_cfg)
    state = checkpoint["model"]
    architecture = checkpoint.get("architecture")
    if architecture is not None:
        # Controlled E2 checkpoints retain their exact architecture identity.
        # In particular the pure hypergraph has no pair topology buffers.
        from e2_architectures import build_architecture
        model = build_architecture(
            architecture, D=cfg["D"], msg_hidden=cfg["msg_hidden"],
            train_T=cfg["train_T"], eval_T=cfg["eval_T"], dropout=cfg["dropout"],
        )
        if asdict(model.cfg) != cfg:
            raise ValueError("Controlled checkpoint architecture/configuration mismatch")
        construction_protocol = "e2_controlled_architecture_factory"
    elif "unit_gru.weight_ih" in state:
        model = SudokuHyperRRN(HyperRRNCfg(**cfg))
        construction_protocol = "legacy_hyper_rrn"
    elif "gru.weight_ih" in state and state["gru.weight_ih"].shape[1] == 2 * cfg["D"]:
        model = SudokuRRNPaper(RRNPaperCfg(**cfg))
        construction_protocol = "legacy_paper_rrn"
    else:
        raise ValueError("Unsupported checkpoint: expected paper-style RRN or Hyper-RRN backbone")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    metadata = {
        "checkpoint_path": str(Path(path).resolve()),
        "checkpoint_sha256": sha256_file(path),
        "model_type": type(model).__name__, "cfg": cfg,
        "architecture": architecture, "construction_protocol": construction_protocol,
        "checkpoint_format": checkpoint.get("checkpoint_format", "legacy_or_unversioned"),
        "saved_step": checkpoint.get("step"),
        "saved_val_exact": checkpoint.get("val_exact"),
        "training_args": checkpoint.get("args"),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    return model, metadata


def sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timing_summary(values):
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"n": 0}
    return {"n": len(array), "mean_seconds": float(array.mean()),
            "median_seconds": float(np.median(array)),
            "p95_seconds": float(np.quantile(array, .95)),
            "min_seconds": float(array.min()), "max_seconds": float(array.max())}


def forward_logits(model, puzzle, steps, readout_protocol="legacy"):
    if readout_protocol == "common_final":
        from e2_rollout import rollout_logits
        return rollout_logits(model, puzzle, steps)
    return model(puzzle, steps=steps)[0]


def evaluate_steps(model, puzzles, solutions, device, steps, batch_size, warmup, readout_protocol="legacy"):
    import torch
    probe = torch.as_tensor(puzzles[:min(batch_size, len(puzzles))], dtype=torch.long, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            forward_logits(model, probe, steps, readout_protocol)
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        predictions = np.empty_like(puzzles, dtype=np.uint8)
        compute_seconds, pipeline_seconds, batch_sizes = [], [], []
        for start in range(0, len(puzzles), batch_size):
            stop = min(start + batch_size, len(puzzles))
            sync(device)
            begin = time.perf_counter()
            batch = torch.as_tensor(puzzles[start:stop], dtype=torch.long, device=device)
            sync(device)
            before_forward = time.perf_counter()
            logits = forward_logits(model, batch, steps, readout_protocol)
            pred = logits.argmax(dim=-1) + 1
            sync(device)
            after_forward = time.perf_counter()
            array = pred.cpu().numpy().astype(np.uint8)
            # Include verifier and output transfer in pipeline timing; cache
            # loading and artifact serialization are deliberately outside it.
            board_metrics(array, puzzles[start:stop], solutions[start:stop])
            after_pipeline = time.perf_counter()
            predictions[start:stop] = array
            compute_seconds.append(after_forward - before_forward)
            pipeline_seconds.append(after_pipeline - begin)
            batch_sizes.append(stop - start)
            if start == 0 or stop % (batch_size * 100) == 0 or stop == len(puzzles):
                print(f"[eval] T={steps} n={stop}/{len(puzzles)} elapsed={sum(pipeline_seconds):.2f}s", flush=True)
    memory = {}
    if device.type == "cuda":
        memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    timing = {
        "forward_scope": "device-resident rollout + argmax, synchronized; readout protocol=" + readout_protocol,
        "pipeline_scope": "host-to-device input, forward, argmax, device-to-host prediction, CPU clue-aware verification; excludes cache loading and output-file writing",
        "batch_size": batch_size, "batch_sizes": batch_sizes,
        "forward_batch_seconds": compute_seconds,
        "pipeline_batch_seconds": pipeline_seconds,
        "forward_total_seconds": sum(compute_seconds),
        "pipeline_total_seconds": sum(pipeline_seconds),
        "pipeline_puzzles_per_second": len(puzzles) / sum(pipeline_seconds),
        "batch_latency": timing_summary(pipeline_seconds),
        "warning": "Batch latency is not a distribution of single-puzzle latency; use isolated_batch1_latency separately.",
        "memory": memory,
    }
    return predictions, timing


def benchmark_batch1(model, puzzles, solutions, device, steps, samples, warmup, readout_protocol="legacy"):
    import torch
    count = min(samples, len(puzzles))
    if count <= 0:
        return {"n": 0}
    positions = np.linspace(0, len(puzzles) - 1, count, dtype=np.int64)
    durations = []
    with torch.inference_mode():
        probe = torch.as_tensor(puzzles[:1], dtype=torch.long, device=device)
        for _ in range(warmup):
            forward_logits(model, probe, steps, readout_protocol)
        for pos in positions:
            sync(device)
            start = time.perf_counter()
            batch = torch.as_tensor(puzzles[pos:pos+1], dtype=torch.long, device=device)
            logits = forward_logits(model, batch, steps, readout_protocol)
            pred = (logits.argmax(dim=-1) + 1).cpu().numpy()
            board_metrics(pred, puzzles[pos:pos+1], solutions[pos:pos+1])
            sync(device)
            durations.append(time.perf_counter() - start)
    result = timing_summary(durations)
    result.update({"selection": "deterministically evenly spaced positions in selected evaluation subset",
                   "positions": positions.tolist(), "seconds": durations,
                   "scope": "same pipeline boundary as batched evaluation, batch_size=1"})
    return result


def run(args):
    import torch
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.cache_path, allow_pickle=False) as data:
        all_ratings = np.asarray(data["ratings"])
        rows = np.flatnonzero(np.asarray(data["splits"]) == args.split)
        test_manifest = None
        manifest_path = getattr(args, "test_global_ids", None)
        if manifest_path:
            if args.split != 2:
                raise ValueError("--test-global-ids applies only to split 2")
            rows, test_manifest = filter_test_global_ids(rows, manifest_path)
        if args.rating_min_exclusive is not None:
            rows = rows[all_ratings[rows] > args.rating_min_exclusive]
        rows = rows[args.offset:]
        if args.limit:
            rows = rows[:args.limit]
        puzzles = np.asarray(data["puzzles"])[rows].reshape(-1, 9, 9)
        solutions = np.asarray(data["solutions"])[rows].reshape(-1, 9, 9)
        ratings = all_ratings[rows]
    if len(rows) == 0:
        raise ValueError("Empty selected evaluation set")
    provenance = {
        "args": vars(args), "cache_sha256": sha256_file(args.cache_path),
        "source_rows_sha256": hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
        "n": len(rows), "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "threads": torch.get_num_threads(), "tf32": False, "precision": "float32",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "selection_rule": "Caller-provided best/final paths; no test-based checkpoint selection",
        "candidate_policy": "one final candidate after exactly T steps; no early stopping",
        "test_global_ids_manifest": test_manifest,
    }
    for checkpoint_path in args.checkpoints:
        model, checkpoint_info = load_model(checkpoint_path, device)
        tag = Path(checkpoint_path).stem + "_" + checkpoint_info["checkpoint_sha256"][:8]
        for steps in args.steps:
            dest = output / tag / f"T{steps}"
            if dest.exists() and any(dest.iterdir()):
                raise FileExistsError(f"Refusing to overwrite existing evaluation: {dest}")
            dest.mkdir(parents=True, exist_ok=True)
            readout_protocol = getattr(args, "readout_protocol", "legacy")
            pred, timing = evaluate_steps(model, puzzles, solutions, device, steps, args.batch_size, args.warmup, readout_protocol)
            metrics = board_metrics(pred, puzzles, solutions)
            summaries = aggregate_metrics(metrics, ratings)
            timing["isolated_batch1_latency"] = benchmark_batch1(
                model, puzzles, solutions, device, steps, args.timing_samples, args.warmup, readout_protocol)
            np.savez_compressed(dest / "per_puzzle.npz", source_row=rows, ratings=ratings,
                                predictions=pred, clues=(puzzles > 0).sum(axis=(1, 2)), **metrics)
            with (dest / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
            with (dest / "per_puzzle.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["source_row", "rating", "exact", "valid", "clue_ok", "prediction"])
                for i, row in enumerate(rows):
                    writer.writerow([int(row), float(ratings[i]), int(metrics["exact"][i]),
                                     int(metrics["valid"][i]), int(metrics["clue_ok"][i]),
                                     "".join(map(str, pred[i].ravel()))])
            result = {"provenance": provenance, "checkpoint": checkpoint_info,
                      "steps": steps, "metrics": summaries, "timing": timing}
            (dest / "summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            print(json.dumps({"output": str(dest), "all": summaries[0]}, default=str), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--cache_path", default="data/cache_full_3m.npz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[64, 128, 256, 384])
    parser.add_argument("--split", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--test-global-ids", dest="test_global_ids",
                        help="Optional data-audit .npy global-ID allowlist; intersects split 2 in cache order")
    parser.add_argument("--rating_min_exclusive", type=float)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timing_samples", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--readout-protocol", choices=["legacy", "common_final"], default="legacy")
    args = parser.parse_args(argv)
    if min(args.steps) <= 0 or args.batch_size <= 0 or args.threads <= 0:
        parser.error("steps, batch_size, and threads must be positive")
    if min(args.offset, args.limit, args.warmup, args.timing_samples) < 0:
        parser.error("offset, limit, warmup, and timing_samples must be nonnegative")
    if args.test_global_ids and args.split != 2:
        parser.error("--test-global-ids requires --split 2")
    return args


if __name__ == "__main__":
    run(parse_args())
