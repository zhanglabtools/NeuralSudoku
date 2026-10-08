"""Common recurrent execution: one final readout, or all readouts for training."""
import torch
from torch.utils.checkpoint import checkpoint


def rollout_logits(model, puzzle, steps, return_all=False, checkpoint_steps=False):
    if steps < 1:
        raise ValueError("steps must be positive")
    x0 = model.input_features(puzzle)
    h = x0
    hyper = hasattr(model, "unit_gru")
    if hyper:
        unit_x0 = model.initial_unit_features(x0)
        unit_h = unit_x0
    else:
        edge_e = model.edge_emb(model.edge_type).unsqueeze(0).expand(len(puzzle), -1, -1)
    outputs = []
    for _ in range(steps):
        args = (h, unit_h, x0, unit_x0) if hyper else (h, x0, edge_e)
        result = (checkpoint(model.step, *args, use_reentrant=False, preserve_rng_state=True)
                  if checkpoint_steps and torch.is_grad_enabled() else model.step(*args))
        if hyper:
            h, unit_h = result
        else:
            h = result
        if return_all:
            outputs.append(model.logits_from_state(h, puzzle))
    return outputs if return_all else model.logits_from_state(h, puzzle)
