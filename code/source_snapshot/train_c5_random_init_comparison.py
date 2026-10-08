"""Train randomly initialized C5 reflectors on two controlled train pools.

The Hybrid Hyper-RRN backbone is loaded from one fixed checkpoint and remains
frozen.  The complete five-cycle, eight-slot primal-dual reflector is randomly
initialized (with the existing zero-initialized correction output layers) and
trained from scratch in one of two data modes:

* hybrid_failures: the fixed Hybrid T64 failures inside train rating>4;
* rating4plus: every train puzzle with rating>4, regardless of Hybrid success.

Both modes share the same architecture, initialization seed, optimizer, losses,
training budget, and validation protocol.  No C3 reflector checkpoint is loaded.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch

from eval_hybrid_hyper_rrn_restarts import load_hybrid
from sudoku_cache_utils import load_sudoku_dataset
from sudoku_exchange_experiment import set_seed, set_torch_threads
from train_symbolic_primal_dual_reflection import (
    SymbolicPrimalDualCfg,
    SymbolicPrimalDualReflector,
    evaluate_validation,
    total_loss,
)


def array_fingerprint(values):
    values = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def trainable_fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(
            parameter.detach().cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def load_failure_indices(path):
    payload = np.load(path, allow_pickle=False)
    if isinstance(payload, np.ndarray):
        return np.asarray(payload, dtype=np.int64)
    if "split" not in payload.files:
        raise ValueError("Failure bank must record its source split")
    source_split = int(np.asarray(payload["split"]).item())
    if source_split != 0:
        raise ValueError(
            f"Failure bank must come from train split=0, got split={source_split}"
        )
    return np.asarray(payload["indices"], dtype=np.int64)


def choose_train_pool(args, train_ds):
    hard = np.flatnonzero(
        np.nan_to_num(train_ds.ratings, nan=-np.inf) > 4.0
    ).astype(np.int64)
    if args.data_mode == "rating4plus":
        pool = hard
    else:
        pool = load_failure_indices(args.failure_indices)
        if pool.size == 0:
            raise ValueError("Hybrid failure pool is empty")
        if pool.min() < 0 or pool.max() >= len(train_ds):
            raise ValueError("Hybrid failure indices fall outside the train split")
        hard_membership = np.zeros(len(train_ds), dtype=np.bool_)
        hard_membership[hard] = True
        if not hard_membership[pool].all():
            bad = int((~hard_membership[pool]).sum())
            raise ValueError(
                f"Expected rating>4 Hybrid failures, found {bad} non-hard indices"
            )
    if pool.size == 0:
        raise ValueError(f"Training pool for {args.data_mode!r} is empty")
    return np.asarray(pool, dtype=np.int64), hard


def build_model(args):
    cpu = torch.device("cpu")
    backbone, backbone_cfg, source = load_hybrid(args.checkpoint, cpu)
    cfg = SymbolicPrimalDualCfg(
        mode="global8",
        D=int(backbone_cfg.D),
        hidden=args.hidden,
        slots=8,
        cycles=5,
        recovery_steps=8,
        parent_steps=64,
        correction_scale=0.35,
        unit_correction_scale=0.25,
        dropout=0.0,
        max_slots=8,
        dual_decay=0.80,
        symbolic_cell_scale=0.20,
        symbolic_unit_scale=0.15,
        variant="primal_dual",
    )
    model = SymbolicPrimalDualReflector(backbone, cfg)
    model.backbone_checkpoint_path = args.checkpoint
    return model, cfg, source


def save_checkpoint(path, model, cfg, args, step, metrics, metadata):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "symbolic_primal_dual_reflection",
            "variant": cfg.variant,
            "experiment": "random_init_c5_comparison",
            "data_mode": args.data_mode,
            "base_checkpoint": args.checkpoint,
            "init_reflection_checkpoint": None,
            "reflection_cfg": asdict(cfg),
            "model_state": model.state_dict(),
            "args": vars(args),
            "step": int(step),
            "metrics": metrics,
            "metadata": metadata,
        },
        target,
    )
    print(
        f"[save] path={target} step={step} score={metrics['score']:.6f} "
        f"all={metrics['all']['symbolic_exact']:.6f} "
        f"hard={metrics['hard']['symbolic_exact']:.6f}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/"
        "model_hybrid_hyper_rrn_d128_t32_eval64_50k_20260717_best.pt",
    )
    parser.add_argument(
        "--cache_path",
        default="data/cache_full_3m.npz",
    )
    parser.add_argument(
        "--failure_indices",
        default="data/"
        "hybrid_t64_train_gt4_failure_bank_20260724.npz",
    )
    parser.add_argument(
        "--data_mode",
        choices=["hybrid_failures", "rating4plus"],
        required=True,
    )
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--final_path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--data_seed",
        type=int,
        default=-1,
        help="Independent sampling seed; -1 preserves the historical seed+offset behavior.",
    )
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--best_ce_coef", type=float, default=1.0)
    parser.add_argument("--mean_ce_coef", type=float, default=0.10)
    parser.add_argument("--selected_ce_coef", type=float, default=0.20)
    parser.add_argument("--critic_coef", type=float, default=0.15)
    parser.add_argument("--cycle_ce_coef", type=float, default=0.35)
    parser.add_argument("--progress_coef", type=float, default=0.0)
    parser.add_argument("--progress_margin", type=float, default=0.002)
    parser.add_argument("--diversity_coef", type=float, default=0.02)
    parser.add_argument("--constraint_coef", type=float, default=0.03)
    parser.add_argument("--correction_reg_coef", type=float, default=0.001)
    parser.add_argument("--critic_temperature", type=float, default=0.25)
    parser.add_argument("--critic_target_temperature", type=float, default=0.05)
    parser.add_argument("--val_hard_n", type=int, default=4096)
    parser.add_argument("--val_all_n", type=int, default=4096)
    parser.add_argument("--eval_batch_size", type=int, default=12)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    set_torch_threads()
    set_seed(args.seed)
    device = torch.device(args.device)

    print("[init] loading fixed Hybrid and constructing random C5 on CPU", flush=True)
    model, cfg, source = build_model(args)
    print("[init] computing random-initialization fingerprint", flush=True)
    initial_fingerprint = trainable_fingerprint(model)
    print(f"[init] moving model to {device}", flush=True)
    model = model.to(device)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"[data] loading train split=0 from {args.cache_path}", flush=True)
    train_ds = load_sudoku_dataset(args.cache_path, split=0, limit=0)
    print(f"[data] loading validation split=1 from {args.cache_path}", flush=True)
    val_ds = load_sudoku_dataset(args.cache_path, split=1, limit=0)
    print("[data] constructing controlled train and validation pools", flush=True)
    train_pool, train_hard = choose_train_pool(args, train_ds)
    val_hard = np.flatnonzero(
        np.nan_to_num(val_ds.ratings, nan=-np.inf) > 4.0
    )[: args.val_hard_n].astype(np.int64)
    effective_data_seed = args.seed if args.data_seed < 0 else args.data_seed
    val_rng = np.random.default_rng(effective_data_seed + 101)
    val_all_n = min(int(args.val_all_n), len(val_ds))
    val_all = np.sort(
        val_rng.choice(len(val_ds), size=val_all_n, replace=False)
    ).astype(np.int64)
    train_rng = np.random.default_rng(effective_data_seed + 202)

    metadata = {
        "initial_trainable_fingerprint": initial_fingerprint,
        "train_pool_fingerprint": array_fingerprint(train_pool),
        "train_pool_n": int(len(train_pool)),
        "train_hard_n": int(len(train_hard)),
        "val_hard_n": int(len(val_hard)),
        "val_all_n": int(len(val_all)),
        "backbone_step": int(source.get("step", -1)),
        "backbone_val_exact": float(source.get("val_exact", float("nan"))),
        "train_split": 0,
        "validation_split": 1,
        "test_used_for_training_or_selection": False,
        "initialization_seed": int(args.seed),
        "data_seed": int(effective_data_seed),
    }
    print(
        f"[config] mode={args.data_mode} random_c5=1 seed={args.seed} "
        f"data_seed={effective_data_seed} "
        f"train_split=0 validation_split=1 test_used=0 "
        f"pool={len(train_pool)} hard_pool={len(train_hard)} "
        f"cycles={cfg.cycles} slots={cfg.slots} recovery={cfg.recovery_steps} "
        f"trainable={sum(parameter.numel() for parameter in trainable)} "
        f"batch={args.batch_size} steps={args.train_steps} "
        f"init_sha256={initial_fingerprint} "
        f"pool_sha256={metadata['train_pool_fingerprint']} device={device}",
        flush=True,
    )

    best_score = -math.inf
    best_step = -1
    started = time.time()
    for step in range(1, args.train_steps + 1):
        take = train_rng.choice(
            train_pool, size=args.batch_size, replace=True
        )
        puzzle = torch.as_tensor(
            train_ds.puzzles[take], dtype=torch.long, device=device
        )
        solution = torch.as_tensor(
            train_ds.solutions[take], dtype=torch.long, device=device
        )

        model.train()
        output = model(puzzle, include_continuation=False)
        loss, pieces = total_loss(model, output, solution, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, args.grad_clip
        )
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            print(
                f"[train] mode={args.data_mode} step={step}/{args.train_steps} "
                f"loss={float(loss.detach()):.4f} "
                + " ".join(
                    f"{name}={float(value):.4f}"
                    for name, value in pieces.items()
                )
                + f" grad={float(grad_norm):.3f} "
                f"elapsed={time.time()-started:.1f}s",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.train_steps:
            hard_metrics = evaluate_validation(
                model,
                val_ds,
                val_hard,
                device,
                args.eval_batch_size,
            )
            all_metrics = evaluate_validation(
                model,
                val_ds,
                val_all,
                device,
                args.eval_batch_size,
            )
            score = (
                float(hard_metrics["symbolic_exact"])
                + float(all_metrics["symbolic_exact"])
            )
            metrics = {
                "score": score,
                "hard": hard_metrics,
                "all": all_metrics,
            }
            print(
                f"[val] mode={args.data_mode} step={step} score={score:.6f} "
                f"hard={hard_metrics} all={all_metrics}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_step = step
                save_checkpoint(
                    args.save_path,
                    model,
                    cfg,
                    args,
                    step,
                    metrics,
                    metadata,
                )

    final_path = args.final_path or str(
        Path(args.save_path).with_name(
            Path(args.save_path).stem + "_final.pt"
        )
    )
    final_metrics = {
        "score": score,
        "hard": hard_metrics,
        "all": all_metrics,
    }
    save_checkpoint(
        final_path,
        model,
        cfg,
        args,
        args.train_steps,
        final_metrics,
        metadata,
    )
    print(
        f"[done] mode={args.data_mode} best_step={best_step} "
        f"best_score={best_score:.6f} elapsed={time.time()-started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
