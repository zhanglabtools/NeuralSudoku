"""Resumable controlled A07 training, with no test-based selection.

State includes optimizer, constant scheduler, all RNGs, shuffled data order and
cursor, completed optimizer step, validation phase, and durable CSV position.
SIGINT/SIGTERM save at the next completed update boundary. Unexpected errors do
not overwrite the last committed resume state with a possibly partial update.
"""
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import signal
import shutil
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from e2_architectures import build_architecture, describe
from e2_eval_checkpoints import sha256_file, sync
from e2_rollout import rollout_logits
from e2_metrics import board_metrics
from kaggle_sudoku_experiment import KaggleSudokuDataset
from kaggle_sudoku_rrn_paper_experiment import stepwise_loss


@torch.no_grad()
def validate(model, loader, device, steps):
    model.eval()
    count = exact = valid = 0
    for batch in loader:
        puzzle = batch["puzzle"].to(device)
        pred = (rollout_logits(model, puzzle, steps).argmax(-1) + 1).cpu().numpy()
        values = board_metrics(pred, batch["puzzle"].numpy(), batch["solution"].numpy())
        count += len(pred)
        exact += int(values["exact"].sum())
        valid += int(values["valid"].sum())
    return dict(n=count, exact_count=exact, valid_count=valid, exact=exact/count, valid=valid/count)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)


def atomic_torch_save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class StatefulShuffle:
    """Single-process sampler; no worker prefetch can advance its saved cursor."""
    def __init__(self, n, batch_size, seed):
        self.n, self.batch_size = n, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.permutation = torch.empty(0, dtype=torch.int64)
        self.cursor, self.epoch = 0, -1

    def next_batch(self):
        if not len(self.permutation) or self.cursor + self.batch_size > self.n:
            self.permutation = torch.randperm(self.n, generator=self.generator)
            self.cursor = 0
            self.epoch += 1
        result = self.permutation[self.cursor:self.cursor + self.batch_size].numpy()
        self.cursor += self.batch_size
        return result

    def state_dict(self):
        return dict(n=self.n, batch_size=self.batch_size, generator=self.generator.get_state(),
                    permutation=self.permutation, cursor=self.cursor, epoch=self.epoch,
                    permutation_sha256=hashlib.sha256(self.permutation.numpy().tobytes()).hexdigest())

    def load_state_dict(self, state):
        if state["n"] != self.n or state["batch_size"] != self.batch_size:
            raise ValueError("Resume sampler data size/batch mismatch")
        permutation = state["permutation"].cpu()
        if permutation.dtype != torch.int64 or len(permutation) not in (0, self.n):
            raise ValueError("Malformed saved shuffle permutation")
        if hashlib.sha256(permutation.numpy().tobytes()).hexdigest() != state["permutation_sha256"]:
            raise ValueError("Saved shuffle permutation checksum mismatch")
        if not 0 <= state["cursor"] <= self.n or state["cursor"] % self.batch_size:
            raise ValueError("Malformed saved data cursor")
        self.generator.set_state(state["generator"].cpu())
        self.permutation, self.cursor, self.epoch = permutation, state["cursor"], state["epoch"]


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if device.type != "cuda" or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA RNG topology changed")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def run(args):
    output = Path(args.output_dir)
    resume_path = Path(args.resume) if args.resume else output / "latest.pt"
    do_resume = bool(args.resume) or (args.resume_if_exists and resume_path.is_file())
    if not do_resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("Nonempty output without a resume checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    if args.workers:
        raise ValueError("Production resume uses a synchronous sampler; workers must be zero")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with np.load(args.cache_path, allow_pickle=False) as cache:
        train_ds = KaggleSudokuDataset(cache, 0, args.train_limit)
        val_ds = KaggleSudokuDataset(cache, 1, 0)
        val_rows = np.flatnonzero(cache["splits"] == 1)
        if args.validation_global_ids:
            ids = np.load(args.validation_global_ids, allow_pickle=False)
            if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
                raise ValueError("Invalid validation IDs")
            if not np.all(np.isin(ids, val_rows)):
                raise ValueError("Validation selection contains a non-validation row")
            selected = np.flatnonzero(np.isin(val_rows, ids))
            if args.val_limit:
                raise ValueError("Cannot combine validation manifest and val_limit")
        else:
            selected = np.arange(len(val_rows))[:args.val_limit or None]
        val_rows = val_rows[selected]
        for key in ("puzzles", "solutions", "clues", "ratings"):
            setattr(val_ds, key, getattr(val_ds, key)[selected])
    if len(train_ds) < args.batch_size or not len(val_ds):
        raise ValueError("Insufficient train/validation data")
    sampler = StatefulShuffle(len(train_ds), args.batch_size, args.data_seed)
    val_generator = torch.Generator().manual_seed(args.data_seed + 100000)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=0, generator=val_generator)
    model = build_architecture(args.architecture, args.D, args.msg_hidden, args.train_T,
                               args.eval_T, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    semantic_keys = ("architecture", "D", "msg_hidden", "train_T", "eval_T", "steps", "seed", "data_seed",
                     "train_limit", "val_limit", "batch_size", "eval_batch_size", "lr", "weight_decay",
                     "grad_clip", "empty_weight", "dropout", "eval_every", "checkpoint_steps", "microbatch_size")
    semantics = {k: getattr(args, k) for k in semantic_keys}
    semantics.update(cache_sha256=sha256_file(args.cache_path), train_n=len(train_ds), val_n=len(val_ds),
                     device_type=device.type, torch_version=torch.__version__)
    semantics.update(validation_global_ids_sha256=sha256_file(args.validation_global_ids) if args.validation_global_ids else None,
                     validation_rows_sha256=hashlib.sha256(val_rows.astype("<i8").tobytes()).hexdigest())
    budget_step = 0
    if args.compute_calibration:
        calibration = json.loads(Path(args.compute_calibration).read_text(encoding="utf-8"))
        profile = calibration["profiles"][args.architecture]
        plan = calibration["training_configuration"]
        if calibration["status"] != "complete" or (args.D, args.msg_hidden, args.train_T, args.batch_size, args.steps) != (profile["D"], profile["msg_hidden"], plan["train_T"], plan["batch_size"], plan["steps"]):
            raise ValueError("Training configuration differs from compute calibration")
        budget_step = int(profile["training_budget_steps"])
        if not 1 <= budget_step <= args.steps:
            raise ValueError("Invalid compute budget prefix")
    semantics.update(compute_budget_step=budget_step,
                     compute_calibration_sha256=sha256_file(args.compute_calibration) if args.compute_calibration else None)
    source_paths = [Path(__file__), Path(inspect.getfile(build_architecture)),
                    Path(inspect.getfile(type(model))), Path(inspect.getfile(rollout_logits)),
                    Path(inspect.getfile(stepwise_loss)), Path(inspect.getfile(validate)),
                    Path(inspect.getfile(board_metrics)),Path(inspect.getfile(KaggleSudokuDataset))]
    source_hashes = {str(p.resolve()): sha256_file(p) for p in source_paths}
    state = dict(step=0, last_validation_step=0, best_exact=-1.0, best_step=None,
                 last_validation=None, elapsed_seconds=0.0, validation_seconds=0.0,
                 curve_bytes=0, resume_count=0, first_training_batch_sha256=None, compute_budget_done=False,
                 compute_budget_best_step=None, compute_budget_best_exact=None,
                 best_snapshot=None, best_snapshot_sha256=None)
    state["training_update_seconds"] = 0.0
    curve_path = output / "training_curve.csv"
    if do_resume:
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("resume_format") != "e2_exact_state_v1":
            raise ValueError("Not a production resume checkpoint")
        if saved["semantics"] != semantics:
            raise ValueError("Resume training/data semantics differ")
        if saved["source_sha256"] != source_hashes:
            raise ValueError("Resume source code changed; review/version explicitly instead of silently continuing")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        sampler.load_state_dict(saved["sampler"])
        val_generator.set_state(saved["val_generator"].cpu())
        state = saved["training_state"]
        state["resume_count"] += 1
        if state["best_snapshot"]:
            snapshot = output / state["best_snapshot"]
            if sha256_file(snapshot) != state["best_snapshot_sha256"]:
                raise ValueError("Committed validation-best snapshot is missing or changed")
            # A crash after writing a new best but before committing latest must
            # not leave best.pt pointing to an uncommitted training future.
            shutil.copyfile(snapshot, output / "best.pt.tmp")
            os.replace(output / "best.pt.tmp", output / "best.pt")
        if not curve_path.is_file() or curve_path.stat().st_size < state["curve_bytes"]:
            raise ValueError("Training curve is missing/truncated relative to checkpoint")
        with curve_path.open("r+b") as stream:
            stream.truncate(state["curve_bytes"])
        restore_rng(saved["rng"], device)
    metadata = dict(status="running", args=vars(args), semantics=semantics, model=describe(model, args.architecture),
                    source_sha256=source_hashes, test_used=False, reflection=False,
                    scheduler="constant LambdaLR; state saved on every resume commit", torch=torch.__version__,
                    cuda_runtime=torch.version.cuda, precision="float32", tf32=False,
                    device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu",
                    validation_selection="fixed validation manifest Exact at fixed eval_T; strict maximum, earliest tie",
                    validation_global_ids=args.validation_global_ids,
                    effective_batch_size=args.batch_size, microbatch_size=args.microbatch_size,
                    numerical_resume="CPU deterministic tests require exact parity; CUDA atomic reductions may remain nondeterministic",
                    gradient_checkpointing="each recurrent step, non-reentrant, RNG-preserving" if args.checkpoint_steps else "disabled",
                    data_protocol="independent stateful torch.Generator shuffle; saved permutation and cursor; drop incomplete epoch tail",
                    resume_count=state["resume_count"])
    atomic_json(output / "metadata.json", metadata)
    curve_stream = curve_path.open("a" if do_resume else "w", newline="", encoding="utf-8")
    curve = csv.DictWriter(curve_stream, fieldnames=["event", "step", "loss", "grad_norm", "update_seconds", "elapsed_seconds", "validation_exact", "validation_valid"])
    if not do_resume:
        curve.writeheader()
        curve_stream.flush()
    previous_elapsed = state["elapsed_seconds"]
    invocation_started = time.perf_counter()
    invocation_updates = 0
    stop = {"requested": False, "signal": None}
    old_handlers = {}
    def request_stop(signum, frame):
        stop.update(requested=True, signal=signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, request_stop)
    def elapsed():
        return previous_elapsed + time.perf_counter() - invocation_started
    def payload():
        return dict(checkpoint_format="e2_controlled_backbone_v1", model=model.state_dict(), cfg=asdict(model.cfg),
                    model_type=type(model).__name__, architecture=args.architecture, step=state["step"],
                    val_exact=None if state["last_validation"] is None else state["last_validation"]["exact"],
                    validation_metrics=state["last_validation"], args=vars(args), cache_sha256=semantics["cache_sha256"])
    last_committed_step = state["step"]
    def save_resume():
        nonlocal last_committed_step
        sync(device)
        curve_stream.flush()
        state.update(elapsed_seconds=elapsed(), curve_bytes=curve_stream.tell())
        record = payload()
        record.update(resume_format="e2_exact_state_v1", semantics=semantics, source_sha256=source_hashes,
                      optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                      sampler=sampler.state_dict(), val_generator=val_generator.get_state(),
                      rng=rng_state(device), training_state=dict(state))
        atomic_torch_save(output / "latest.pt", record)
        last_committed_step = state["step"]
    result = None
    try:
        if not do_resume:
            save_resume()
        while True:
            if stop["requested"] or (args.max_new_updates and invocation_updates >= args.max_new_updates):
                save_resume()
                result = dict(status="paused", step=state["step"], reason="signal" if stop["requested"] else "invocation_update_limit",
                              signal=stop["signal"], resume_path=str(output / "latest.pt"))
                break
            due = state["step"] > 0 and (state["step"] % args.eval_every == 0 or state["step"] == args.steps)
            if due and state["last_validation_step"] < state["step"]:
                sync(device)
                started = time.perf_counter()
                val = validate(model, val_loader, device, args.eval_T)
                sync(device)
                state["validation_seconds"] += time.perf_counter() - started
                state.update(last_validation=val, last_validation_step=state["step"])
                if val["exact"] > state["best_exact"]:
                    state.update(best_exact=val["exact"], best_step=state["step"])
                    snapshot_name = f"validation_best_step{state['step']}.pt"
                    atomic_torch_save(output / snapshot_name, payload())
                    state.update(best_snapshot=snapshot_name, best_snapshot_sha256=sha256_file(output / snapshot_name))
                    shutil.copyfile(output / snapshot_name, output / "best.pt.tmp")
                    os.replace(output / "best.pt.tmp", output / "best.pt")
                if state["step"] == args.steps:
                    atomic_torch_save(output / "final.pt", payload())
                row = dict(event="validation", step=state["step"], elapsed_seconds=elapsed(),
                           validation_exact=val["exact"], validation_valid=val["valid"])
                curve.writerow(row)
                print(json.dumps(row), flush=True)
                save_resume()
            if budget_step and state["step"] == budget_step and not state["compute_budget_done"]:
                # A budget-specific validation does not enter the 50k main best pool.
                if state["last_validation_step"] == state["step"]:
                    budget_val = state["last_validation"]
                else:
                    sync(device)
                    started = time.perf_counter()
                    budget_val = validate(model, val_loader, device, args.eval_T)
                    sync(device)
                    state["validation_seconds"] += time.perf_counter() - started
                prefix = payload()
                prefix.update(val_exact=budget_val["exact"], validation_metrics=budget_val,
                              budget_protocol="training forward dense-MAC proxy; no test selection")
                atomic_torch_save(output / "compute_budget.pt", prefix)
                if budget_val["exact"] > state["best_exact"]:
                    atomic_torch_save(output / "compute_budget_best.pt", prefix)
                    selected_step, selected_exact = state["step"], budget_val["exact"]
                else:
                    shutil.copyfile(output / "best.pt", output / "compute_budget_best.pt.tmp")
                    os.replace(output / "compute_budget_best.pt.tmp", output / "compute_budget_best.pt")
                    selected_step, selected_exact = state["best_step"], state["best_exact"]
                state.update(compute_budget_done=True, compute_budget_best_step=selected_step,
                             compute_budget_best_exact=selected_exact)
                curve.writerow(dict(event="budget_validation", step=state["step"], elapsed_seconds=elapsed(),
                                    validation_exact=budget_val["exact"], validation_valid=budget_val["valid"]))
                save_resume()
            if state["step"] == args.steps:
                result = dict(status="complete", step=state["step"], best_step=state["best_step"],
                              best_validation_exact=state["best_exact"], best_sha256=sha256_file(output / "best.pt"),
                              final_sha256=sha256_file(output / "final.pt"))
                break
            sync(device)
            update_started = time.perf_counter()
            model.train()
            batch_ids = sampler.next_batch()
            puzzle_cpu = train_ds.puzzles[batch_ids].astype(np.int64)
            if state["first_training_batch_sha256"] is None:
                state["first_training_batch_sha256"] = hashlib.sha256(puzzle_cpu.astype(np.uint8).tobytes()).hexdigest()
            puzzle = torch.from_numpy(puzzle_cpu).to(device)
            target = torch.from_numpy(train_ds.solutions[batch_ids].astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for start in range(0, len(puzzle), args.microbatch_size):
                micro_puzzle = puzzle[start:start + args.microbatch_size]
                micro_target = target[start:start + args.microbatch_size]
                logits = rollout_logits(model, micro_puzzle, args.train_T, return_all=True, checkpoint_steps=args.checkpoint_steps)
                loss = stepwise_loss(logits, micro_puzzle, micro_target, args.empty_weight, "all", args.train_T)
                loss = loss * (len(micro_puzzle) / len(puzzle))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                loss_value += float(loss.detach())
                del logits, loss
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            sync(device)
            update_seconds = time.perf_counter() - update_started
            state["training_update_seconds"] += update_seconds
            state["step"] += 1
            invocation_updates += 1
            if state["step"] == 1 or state["step"] % args.log_every == 0 or state["step"] % args.eval_every == 0:
                row = dict(event="train", step=state["step"], loss=loss_value, grad_norm=float(grad), update_seconds=update_seconds, elapsed_seconds=elapsed())
                curve.writerow(row)
                curve_stream.flush()
                print(json.dumps(row), flush=True)
            if state["step"] % args.checkpoint_every == 0 or state["step"] == args.steps:
                save_resume()
        result.update(seed=args.seed, data_seed=args.data_seed, architecture=args.architecture,
                      completed_training_exposure=state["step"] * args.batch_size,
                      planned_training_exposure=args.steps * args.batch_size,
                      wall_seconds_including_validation_and_checkpoints=elapsed(),
                      validation_seconds=state["validation_seconds"], resume_count=state["resume_count"],
                      training_update_seconds=state["training_update_seconds"],
                      first_training_batch_sha256=state["first_training_batch_sha256"],
                      cfg=asdict(model.cfg), parameter_count=sum(p.numel() for p in model.parameters()))
        result.update(validation_manifest_sha256=semantics["validation_global_ids_sha256"],
                      validation_rows_sha256=semantics["validation_rows_sha256"], validation_n=len(val_ds),
                      validation_rule=metadata["validation_selection"], eval_every=args.eval_every, eval_T=args.eval_T)
        if budget_step and state["compute_budget_done"]:
            result.update(compute_budget_step=budget_step, compute_budget_best_step=state["compute_budget_best_step"],
                          compute_budget_best_exact=state["compute_budget_best_exact"],
                          compute_budget_best_sha256=sha256_file(output / "compute_budget_best.pt"),
                          compute_calibration_sha256=semantics["compute_calibration_sha256"])
        atomic_json(output / "finished.json", result)
        metadata.update(status=result["status"], first_training_batch_sha256=state["first_training_batch_sha256"],
                        completed_step=state["step"])
        atomic_json(output / "metadata.json", metadata)
        return result
    except BaseException as error:
        atomic_json(output / "failure.json", dict(status="failed", error=f"{type(error).__name__}: {error}",
                    last_committed_step=last_committed_step, resume_path=str(output / "latest.pt")))
        raise
    finally:
        curve_stream.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=["graph", "hypergraph", "joint"])
    parser.add_argument("--cache_path", default="data/cache_full_3m.npz")
    parser.add_argument("--output_dir", required=True)
    for key, default in dict(D=128, msg_hidden=256, train_T=32, eval_T=64, steps=50000,
                             seed=0, data_seed=0, train_limit=0, val_limit=0, batch_size=64,
                             eval_batch_size=128, workers=0, threads=1, eval_every=5000,
                             log_every=500, checkpoint_every=1000, max_new_updates=0, microbatch_size=16).items():
        parser.add_argument("--" + key, type=int, default=default)
    for key, default in dict(lr=.001, weight_decay=.0001, grad_clip=1., empty_weight=2., dropout=0.).items():
        parser.add_argument("--" + key, type=float, default=default)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default="")
    parser.add_argument("--validation-global-ids", dest="validation_global_ids", default="")
    parser.add_argument("--compute-calibration", default="")
    parser.add_argument("--resume-if-exists", action="store_true")
    parser.add_argument("--no-checkpoint-steps", dest="checkpoint_steps", action="store_false")
    parser.set_defaults(checkpoint_steps=True)
    args = parser.parse_args(argv)
    if min(args.D, args.msg_hidden, args.train_T, args.eval_T, args.steps, args.batch_size,
           args.eval_batch_size, args.threads, args.eval_every, args.log_every, args.checkpoint_every, args.microbatch_size) < 1:
        parser.error("dimensions, steps, batches and intervals must be positive")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "complete" else 75)
