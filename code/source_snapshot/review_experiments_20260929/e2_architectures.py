"""Strict, independently trainable graph/hypergraph/joint backbone factory.

These are architecture choices at construction, not evaluation-time branch
ablations. Existing implementations supply identical input features, cell GRU
input injection, clue forcing, and readout conventions. No old source is edited.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_sudoku_hyper_rrn_experiment import HyperRRNCfg, SudokuHyperRRN
from kaggle_sudoku_rrn_paper_experiment import RRNPaperCfg, SudokuRRNPaper


def build_architecture(architecture, D=128, msg_hidden=256, train_T=32, eval_T=64, dropout=0.0):
    shared = dict(D=D, msg_hidden=msg_hidden, train_T=train_T, eval_T=eval_T,
                  dropout=dropout, force_clues=True)
    if architecture == "graph":
        model = SudokuRRNPaper(RRNPaperCfg(**shared))
    elif architecture in {"hypergraph", "joint"}:
        kind = "hyper_only" if architecture == "hypergraph" else "hybrid"
        model = SudokuHyperRRN(HyperRRNCfg(model_type=kind, **shared))
        if architecture == "hypergraph":
            # The legacy implementation creates unused pair topology buffers.
            # Remove even those: no pair topology/state is present in this arm.
            for name in ("pair_src", "pair_dst", "pair_type"):
                delattr(model, name)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    assert model.cfg.force_clues
    return model


def describe(model, architecture):
    return {
        "architecture": architecture,
        "model_type": type(model).__name__,
        "cfg": asdict(model.cfg),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "modules": [name for name, _ in model.named_modules() if name],
        "reflection": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["graph", "hypergraph", "joint"], required=True)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--msg_hidden", type=int, default=256)
    args = parser.parse_args()
    model = build_architecture(args.architecture, args.D, args.msg_hidden)
    print(json.dumps(describe(model, args.architecture), indent=2))


if __name__ == "__main__":
    main()
