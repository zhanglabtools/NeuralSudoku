"""Function excerpt from eval_symbolic_active_reflection.py; algorithm unchanged.
Place source_snapshot on PYTHONPATH. See README_CODE.txt for packaged data/models.
solution is read only for post-acceptance Exact metrics, not candidate selection.
This file omits the original CLI; the complete dependency is in source_snapshot.
"""
import torch
from eval_hybrid_hyper_rrn_restarts import hard_violation_count_batch

def _expand(model, tensor, slots):
    return model._expand_slots(tensor, slots)


def _keep_slots(tensor, keep, old_batch, slots):
    return tensor.reshape(old_batch, slots, *tensor.shape[1:])[keep].reshape(
        int(keep.sum()) * slots, *tensor.shape[1:]
    ).contiguous()


@torch.no_grad()
def active_batch(model, puzzle, solution, symbolic):
    batch = puzzle.shape[0]
    slots = int(model.cfg.slots)
    cycles = int(model.cfg.cycles)
    x0, unit_x0, parent_h, parent_u, anchor_logits = model.encode_parent(puzzle)
    anchor_pred = anchor_logits.argmax(dim=-1) + 1
    anchor_valid = hard_violation_count_batch(anchor_pred) == 0
    anchor_exact = (anchor_pred == solution).reshape(batch, -1).all(dim=-1)

    solved_step = torch.full(
        (batch,), -1, dtype=torch.int16, device=puzzle.device
    )
    solved_exact = torch.zeros(batch, dtype=torch.bool, device=puzzle.device)
    solved_step[anchor_valid] = 0
    solved_exact[anchor_valid] = anchor_exact[anchor_valid]
    equivalent_updates = torch.full(
        (batch,), int(model.cfg.parent_steps), dtype=torch.int32, device=puzzle.device
    )
    active_local = torch.nonzero(~anchor_valid, as_tuple=False).squeeze(-1)
    active_before = [int(active_local.numel())]
    new_valid = [int(anchor_valid.sum())]

    if active_local.numel() == 0:
        return solved_step, solved_exact, equivalent_updates, active_before, new_valid

    active_puzzle = puzzle.index_select(0, active_local)
    active_x0 = x0.index_select(0, active_local)
    active_unit_x0 = unit_x0.index_select(0, active_local)
    active_h = parent_h.index_select(0, active_local)
    active_u = parent_u.index_select(0, active_local)
    source_batch = int(active_local.numel())
    h = _expand(model, active_h, slots)
    unit_h = _expand(model, active_u, slots)
    slot_x0 = _expand(model, active_x0, slots)
    slot_unit_x0 = _expand(model, active_unit_x0, slots)
    slot_puzzle = _expand(model, active_puzzle, slots)
    dual = h.new_zeros(source_batch * slots, 27, 9) if symbolic else None

    for cycle_index in range(cycles):
        if source_batch == 0:
            active_before.append(0)
            new_valid.append(0)
            continue
        equivalent_updates[active_local] += slots * int(model.cfg.recovery_steps)
        if symbolic:
            h, unit_h, dual, _ = model._symbolic_reflect_once(
                h,
                unit_h,
                slot_x0,
                slot_unit_x0,
                slot_puzzle,
                dual,
                source_batch=source_batch,
                slots=slots,
                cycle_index=cycle_index,
            )
        else:
            h, unit_h, _ = model._reflect_once(
                h,
                unit_h,
                slot_x0,
                slot_unit_x0,
                slot_puzzle,
                source_batch=source_batch,
                slots=slots,
                cycle_index=cycle_index,
            )
        h, unit_h = model._rollout(
            h,
            unit_h,
            slot_x0,
            slot_unit_x0,
            model.cfg.recovery_steps,
        )
        logits = model.backbone.logits_from_state(h, slot_puzzle)
        pred = logits.argmax(dim=-1) + 1
        valid = (hard_violation_count_batch(pred) == 0).reshape(
            source_batch, slots
        )
        puzzle_solved = valid.any(dim=1)
        solved_count = int(puzzle_solved.sum())
        new_valid.append(solved_count)
        if solved_count:
            first_slot = valid.to(torch.int64).argmax(dim=1)
            pred_by_puzzle = pred.reshape(source_batch, slots, 9, 9)
            rows = torch.arange(source_batch, device=puzzle.device)
            chosen = pred_by_puzzle[rows, first_slot]
            active_solution = solution.index_select(0, active_local)
            exact = (chosen == active_solution).reshape(source_batch, -1).all(dim=-1)
            solved_indices = active_local[puzzle_solved]
            solved_step[solved_indices] = cycle_index + 1
            solved_exact[solved_indices] = exact[puzzle_solved]

        keep = ~puzzle_solved
        old_batch = source_batch
        active_local = active_local[keep]
        source_batch = int(keep.sum())
        active_before.append(source_batch)
        if source_batch == 0:
            continue
        h = _keep_slots(h, keep, old_batch, slots)
        unit_h = _keep_slots(unit_h, keep, old_batch, slots)
        slot_x0 = _keep_slots(slot_x0, keep, old_batch, slots)
        slot_unit_x0 = _keep_slots(slot_unit_x0, keep, old_batch, slots)
        slot_puzzle = _keep_slots(slot_puzzle, keep, old_batch, slots)
        if symbolic:
            dual = _keep_slots(dual, keep, old_batch, slots)

    return solved_step, solved_exact, equivalent_updates, active_before, new_valid
