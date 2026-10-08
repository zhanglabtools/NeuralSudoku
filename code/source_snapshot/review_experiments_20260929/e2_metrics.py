"""NumPy-only, clue-aware metrics shared by the E2 checkpoint audit."""

import numpy as np


def board_metrics(predictions, puzzles, solutions):
    pred = np.asarray(predictions).reshape(-1, 9, 9)
    puzzle = np.asarray(puzzles).reshape(-1, 9, 9)
    target = np.asarray(solutions).reshape(-1, 9, 9)
    if not (pred.shape == puzzle.shape == target.shape):
        raise ValueError("Prediction, puzzle, and target shapes differ")
    in_range = ((pred >= 1) & (pred <= 9)).all(axis=(1, 2))
    expected = np.arange(1, 10)
    rows = (np.sort(pred, axis=2) == expected).all(axis=(1, 2))
    cols = (np.sort(pred.transpose(0, 2, 1), axis=2) == expected).all(axis=(1, 2))
    boxes = pred.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    boxes_ok = (np.sort(boxes, axis=2) == expected).all(axis=(1, 2))
    clue_ok = ((puzzle == 0) | (pred == puzzle)).all(axis=(1, 2))
    exact = (pred == target).all(axis=(1, 2))
    return {
        "exact": exact,
        "valid": in_range & rows & cols & boxes_ok & clue_ok,
        "clue_ok": clue_ok,
        "grid_valid_ignoring_clues": in_range & rows & cols & boxes_ok,
        "correct_cells": (pred == target).sum(axis=(1, 2)),
    }


def aggregate_metrics(metrics, ratings):
    ratings = np.asarray(ratings)
    selectors = [
        ("all", np.ones(len(ratings), dtype=bool)),
        ("rating_0", (ratings > -np.inf) & (ratings <= 0)),
        ("rating_0_1", (ratings > 0) & (ratings <= 1)),
        ("rating_1_2", (ratings > 1) & (ratings <= 2)),
        ("rating_2_4", (ratings > 2) & (ratings <= 4)),
        ("rating_4_plus", ratings > 4),
        ("rating_nan", np.isnan(ratings)),
        ("rating_other", np.isneginf(ratings)),
    ]
    result = []
    for name, mask in selectors:
        count = int(mask.sum())
        if count == 0:
            continue
        row = {"bucket": name, "n": count}
        for key in ("exact", "valid", "clue_ok", "grid_valid_ignoring_clues"):
            value = int(np.asarray(metrics[key])[mask].sum())
            row[key + "_count"] = value
            row[key + "_rate"] = value / count
        row["valid_not_exact_count"] = int(
            (np.asarray(metrics["valid"])[mask] & ~np.asarray(metrics["exact"])[mask]).sum()
        )
        row["cell_accuracy"] = float(np.asarray(metrics["correct_cells"])[mask].sum() / (81 * count))
        result.append(row)
    return result
