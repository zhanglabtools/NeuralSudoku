"""Convert the compressed Sudoku cache once for repeated experiment startup."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    started = time.time()
    source = np.load(args.input_path, allow_pickle=False)
    arrays = {}
    for key in source.files:
        arrays[key] = np.asarray(source[key])
        print(
            f"[load] key={key} shape={arrays[key].shape} "
            f"dtype={arrays[key].dtype} elapsed={time.time()-started:.1f}s",
            flush=True,
        )
    target = Path(args.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **arrays)
    print(
        f"[save] {target} bytes={target.stat().st_size} "
        f"elapsed={time.time()-started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
