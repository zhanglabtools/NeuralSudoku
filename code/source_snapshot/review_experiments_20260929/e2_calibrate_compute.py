"""Outcome-free dense MAC accounting and integer-step compute control.

Counts actual executed nn.Linear and GRUCell matrix products, batch=1. One MAC
is one multiply-accumulate. Scatter, embedding lookup, normalization, activation,
softmax, elementwise work, memory traffic and verification are NOT counted.
This is a stated dense-MAC budget, never a claim of equal total FLOPs or latency.
"""
import argparse
import json
from pathlib import Path
import torch
from e2_architectures import build_architecture
from e2_eval_checkpoints import sha256_file
from e2_rollout import rollout_logits

CONFIGS = {"graph": (208, 384), "hypergraph": (128, 432), "joint": (128, 256)}


def dense_macs(model, puzzle, steps, return_all=False):
    counts = {}
    handles = []
    def hook(name, module, inputs, output):
        if isinstance(module, torch.nn.Linear):
            macs = inputs[0].numel() // module.in_features * module.in_features * module.out_features
        else:
            n = inputs[0].numel() // module.input_size
            macs = n * 3 * module.hidden_size * (module.input_size + module.hidden_size)
        counts[name] = counts.get(name, 0) + int(macs)
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.GRUCell)):
            handles.append(module.register_forward_hook(lambda m, i, o, name=name: hook(name, m, i, o)))
    try:
        with torch.inference_mode():
            rollout_logits(model, puzzle, steps, return_all=return_all)
    finally:
        for h in handles:
            h.remove()
    return sum(counts.values()), counts


def run(output_dir, target_steps=64):
    torch.set_num_threads(1)
    torch.manual_seed(20260929)
    puzzle = torch.zeros((1, 9, 9), dtype=torch.long)
    profiles = {}
    for architecture, (D, msg_hidden) in CONFIGS.items():
        model = build_architecture(architecture, D, msg_hidden).eval()
        first, modules = dense_macs(model, puzzle, 1)
        second, _ = dense_macs(model, puzzle, 2)
        third, _ = dense_macs(model, puzzle, 3)
        training, _ = dense_macs(model, puzzle, 32, return_all=True)
        if third-second != second-first:
            raise ValueError("Nonlinear step cost: cannot use affine accounting")
        profiles[architecture] = dict(D=D, msg_hidden=msg_hidden, params=sum(p.numel() for p in model.parameters()),
                                     initialization_plus_final_macs=2*first-second, per_step_macs=second-first,
                                     one_step_module_macs=modules, training_forward_macs_per_update=training*64)
    target = profiles["joint"]["initialization_plus_final_macs"] + target_steps*profiles["joint"]["per_step_macs"]
    for p in profiles.values():
        steps = max(1, round((target-p["initialization_plus_final_macs"])/p["per_step_macs"]))
        p.update(matched_steps=steps, dense_macs=p["initialization_plus_final_macs"]+steps*p["per_step_macs"],
                 fixed_T64_dense_macs=p["initialization_plus_final_macs"]+64*p["per_step_macs"])
        p["relative_deviation_from_target"] = p["dense_macs"]/target-1
    training_target = min(p["training_forward_macs_per_update"] for p in profiles.values())*50000
    for p in profiles.values():
        p["training_budget_steps"] = training_target//p["training_forward_macs_per_update"]
        p["training_budget_used_macs"] = p["training_budget_steps"]*p["training_forward_macs_per_update"]
        p["training_budget_relative_deviation"] = p["training_budget_used_macs"]/training_target-1
    result = dict(status="complete", readout_protocol="common_final", target_architecture="joint", target_steps=target_steps,
                  target_dense_macs=target, training_target_forward_dense_macs=training_target, profiles=profiles,
                  training_protocol="T32 all-step readouts, effective batch64; min architecture cost times 50000; floor integer updates; validation not counted",
                  training_configuration=dict(train_T=32, batch_size=64, steps=50000),
                  training_exclusions="backward, optimizer, gradient-checkpoint recomputation, scatter/norm/activations/memory; forward dense-MAC proxy only, not total training FLOPs",
                  selection_basis="architecture shapes only; no data, labels, or predictions",
                  counted="Linear and GRUCell weight-matrix MACs; batch1; multiply-accumulate is one MAC",
                  excluded="scatter, embedding lookup, normalization, activations, elementwise arithmetic, memory, verification",
                  claims="approximately equal stated dense-MAC inference budget only; not equal total FLOPs, latency or training compute",
                  source_sha256=sha256_file(__file__))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/"summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_steps", type=int, default=64)
    print(json.dumps(run(**vars(p.parse_args()))))
